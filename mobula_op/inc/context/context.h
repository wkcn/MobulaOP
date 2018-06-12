#ifndef _CONTEXT_H_
#define _CONTEXT_H_

#include <cstring>

#if USING_CUDA
#include "context/cuda_ctx.h"
#else
#include "context/cpu_ctx.h"
#endif // USING_CUDA

enum class CTX {
    HOST,
    DEVICE
};

template<typename T>
T* MemcpyHostToHost(T *dst, const T *src, int size) {
    if (dst == src) return dst;
    return static_cast<T*>(memcpy(dst, src, size));
}

template <typename T>
class ctx_pointer{

public:
    ctx_pointer(const int data_size = 0, T *host_pointer = nullptr):_size(data_size), _host_pointer(host_pointer), _dev_pointer(nullptr), _current_pointer(host_pointer), _ctx(CTX::HOST) {
        _host_pointer_owner = false;
        _dev_pointer_owner = false;
        set_ctx(CTX::HOST, true);
    }
    ~ctx_pointer() {
        if (_host_pointer_owner) {
            delete []_host_pointer;
        }
        if (_dev_pointer_owner) {
            xdel(_dev_pointer);
        }
    }
    int size() const {
        return _size;
    }
    ctx_pointer(const ctx_pointer &p) {
        _size = p._size;
        _ctx = p._ctx;
        if (_ctx == CTX::HOST) {
            _host_pointer_owner = true;
            _dev_pointer_owner = false;
            _dev_pointer = nullptr;
            _host_pointer = new T[_size];
            MemcpyHostToHost(_host_pointer, p._host_pointer, sizeof(T) * _size);
        } else {
            _host_pointer_owner = false;
            _dev_pointer_owner = true;
            _host_pointer = nullptr;
            _dev_pointer = xnew<T>(_size);
            MemcpyDevToDev(_dev_pointer, p._dev_pointer, sizeof(T) * _size);
        }
    }
    ctx_pointer<T>& copy() {
        return ctx_pointer<T>(*this);
    }
    void set_ctx(CTX ctx, bool force = false) {
        if (_ctx == ctx && !force) return;
        _ctx = ctx;
        if (_ctx == CTX::HOST) {
            sync_to_host();
            _current_pointer = _host_pointer;
        } else {
            sync_to_dev();
            _current_pointer = _dev_pointer;
        }
    }
    T& operator[](const int index) {
        return _current_pointer[index];
    }
    T& operator[](const int index) const {
        return _current_pointer[index];
    }
    T* pointer() const{
        return _current_pointer;
    }
private:
    void check_host_pointer() {
        if (_host_pointer == nullptr) {
            _host_pointer_owner = true;
            _host_pointer = new T[_size];
            if (_dev_pointer != nullptr) {
                MemcpyDevToHost(_host_pointer, _dev_pointer, sizeof(T) * _size);
            }
        }
    }
    void check_dev_pointer() {
        if (_dev_pointer == nullptr) {
            _dev_pointer_owner = true;
            _dev_pointer = xnew<T>(_size);
            if (_host_pointer != nullptr) {
                MemcpyHostToDev(_dev_pointer, _host_pointer, sizeof(T) * _size);
            }
        }
    }
    void set_host_pointer(T *new_host_pointer) {
        if (_host_pointer_owner) delete []_host_pointer;
        _host_pointer_owner = false;
        _host_pointer = new_host_pointer;
    }
    void set_dev_pointer(T *new_dev_pointer) {
        if (_dev_pointer_owner) xdel(_dev_pointer);
        _dev_pointer_owner = false;
        _dev_pointer = new_dev_pointer;
    }
    void resize(int data_size) {
        _size = data_size;
    }
    void sync_to_host() {
        if (_host_pointer == nullptr) {
            _host_pointer_owner = true;
            _host_pointer = new T[_size];
        }
        if (_dev_pointer != nullptr) {
            MemcpyDevToHost(_host_pointer, _dev_pointer, sizeof(T) * _size);
        }
    }
    void sync_to_dev() {
        if (_dev_pointer == nullptr) {
            _dev_pointer_owner = true;
            _dev_pointer = xnew<T>(_size);
        }
        if (_host_pointer != nullptr) {
            MemcpyHostToDev(_dev_pointer, _host_pointer, sizeof(T) * _size);
        }
    }
    T* get_host_pointer() {
        check_host_pointer();
        return _host_pointer;
    }
    T* get_dev_pointer() {
        check_dev_pointer();
        return _dev_pointer;
    }
private:
    int _size;
    T *_host_pointer;
    T *_dev_pointer;
    T *_current_pointer;
    bool _host_pointer_owner, _dev_pointer_owner;
    CTX _ctx;
};

// C API
extern "C" {

void set_device(const int device_id);

}

#endif
