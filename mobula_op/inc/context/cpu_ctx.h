#ifndef _CPU_CTX_
#define _CPU_CTX_

#include <thread>
#include <mutex>
#include <cmath>
#include <cstring>

namespace mobula {

#define MOBULA_KERNEL void
#define MOBULA_DEVICE

using std::max;
using std::min;
using std::abs;

#if HOST_NUM_THREADS > 1 or USING_OPENMP
constexpr int NUM_MOBULA_ATOMIC_ADD_MUTEXES = HOST_NUM_THREADS * 8;
extern std::mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
inline MOBULA_DEVICE float atomic_add(const float val, float* address) {
    long id = (reinterpret_cast<long>(address) / sizeof(float)) % NUM_MOBULA_ATOMIC_ADD_MUTEXES;
    MOBULA_ATOMIC_ADD_MUTEXES[id].lock();
    *address += val;
    MOBULA_ATOMIC_ADD_MUTEXES[id].unlock();
    return *address;
}
#else
// no lock for single thread mode
inline MOBULA_DEVICE float atomic_add(const float val, float* address) {
    *address += val;
    return *address;
}
#endif

}

template<typename T>
T* xnew(const int size) {
    return new T[size];
}

template<typename T>
void xdel(T *p) {
    delete []p;
}

template<typename T>
T* MemcpyHostToDev(T *dst, const T *src, int size) {
    if (dst == src) return dst;
    return static_cast<T*>(memcpy(dst, src, size));
}

template<typename T>
T* MemcpyDevToHost(T *dst, const T *src, int size) {
    if (dst == src) return dst;
    return static_cast<T*>(memcpy(dst, src, size));
}

template<typename T>
T* MemcpyDevToDev(T *dst, const T *src, int size) {
    if (dst == src) return dst;
    return static_cast<T*>(memcpy(dst, src, size));
}

#if USING_OPENMP
#include "openmp_ctx.h"
#else
#include "naive_ctx.h"
#endif

#endif
