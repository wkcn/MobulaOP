#ifndef _MOBULA_DEFINES_
#define _MOBULA_DEFINES_

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <utility>
#include <typeinfo>
#include <cassert>
#include <cmath>
#include <memory>
#include <thread>
#include <mutex>
#include <cfloat>
#if USING_OPENMP
#include "omp.h"
#endif

namespace mobula {

typedef float DType;

#if USING_CUDA

#include <cuda_runtime.h>
#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(n) ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__
#define KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n);i += blockDim.x * gridDim.x)
#define KERNEL_RUN(a, n) (a)<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

template <typename T>
inline MOBULA_DEVICE T atomic_add(const T val, T* address);

template <>
inline MOBULA_DEVICE float atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

#else

using std::max;
using std::min;

#define MOBULA_KERNEL void
#define MOBULA_DEVICE

#if USING_OPENMP

#define KERNEL_LOOP(i,n) _Pragma("omp parallel for") \
                         for (int i = 0;i < (n);++i)
#define KERNEL_RUN(a, n) a

#else // USING_OPENMP else

extern std::map<std::thread::id, std::pair<int, int> > MOBULA_KERNEL_INFOS;
extern std::mutex MOBULA_KERNEL_MUTEX;

template<typename Func>
class KernelRunner{
public:
	KernelRunner(Func func, int n):_func(func), _n(n <= HOST_NUM_THREADS ? n : HOST_NUM_THREADS){};
	template<typename ...Args>
	void operator()(Args... args){
        std::vector<std::thread> threads(_n);
        MOBULA_KERNEL_MUTEX.lock();
        for (int i = 0;i < _n;++i) {
            threads[i] = std::thread(_func, args...);
            std::thread::id id = threads[i].get_id();
            MOBULA_KERNEL_INFOS[id] = std::make_pair(i, _n);
        }
        MOBULA_KERNEL_MUTEX.unlock();
        for (int i = 0;i < _n;++i) {
            threads[i].join();
        }
    }
private:
	Func _func;
	int _n;
};

#define KERNEL_LOOP(i,n) MOBULA_KERNEL_MUTEX.lock(); \
						 const std::pair<int, int> MOBULA_KERNEL_INFO = MOBULA_KERNEL_INFOS[std::this_thread::get_id()]; \
						 MOBULA_KERNEL_INFOS.erase(std::this_thread::get_id()); \
						 MOBULA_KERNEL_MUTEX.unlock(); \
						 const int MOBULA_KERNEL_START = MOBULA_KERNEL_INFO.first; \
						 const int MOBULA_KERNEL_STEP = MOBULA_KERNEL_INFO.second; \
						 for (int i = MOBULA_KERNEL_START;i < (n);i += MOBULA_KERNEL_STEP)
#define KERNEL_RUN(a, n) (KernelRunner<decltype(&a)>(&a, (n)))

#endif // USING_OPENMP endif

constexpr int NUM_MOBULA_ATOMIC_ADD_MUTEXES = HOST_NUM_THREADS * 8;
extern std::mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
inline MOBULA_DEVICE float atomic_add(const float val, float* address) {
    long id = (reinterpret_cast<long>(address) / sizeof(float)) % NUM_MOBULA_ATOMIC_ADD_MUTEXES;
    MOBULA_ATOMIC_ADD_MUTEXES[id].lock();
    *address += val;
    MOBULA_ATOMIC_ADD_MUTEXES[id].unlock();
    return *address;
}

#endif // USING_CUDA

// Allocate Memory
template<typename T = DType>
T* xnew(int size) {
#if USING_CUDA
	T *p;
	cudaMalloc((void **)&p, sizeof(T) * size);
	return p;
#else
	return new T[size];
#endif
}

template<typename T = DType>
void xdel(T* p) {
#if USING_CUDA
	cudaFree(p);
#else
	delete []p;
#endif
}

template<typename F, typename T = DType>
inline MOBULA_DEVICE void mobula_map(F func, const T *data, const int n, const int stride = 1, T *out = 0) {
    if (out == 0) out = const_cast<T*>(data);
    for (int i = 0, j = 0; i < n; ++i, j += stride) {
        out[j] = func(data[j]);
    }
}

template<typename F, typename T = DType>
inline MOBULA_DEVICE void mobula_reduce(F func, const T *data, const int n, const int stride = 1, T *out = 0) {
    if (out == 0) out = const_cast<T*>(data);
    T &val = out[0];
    val = data[0];
    for (int i = 1, j = stride; i < n; ++i, j += stride) {
        val = func(val, data[j]);
    }
}

inline MOBULA_DEVICE int get_middle_loop_offset(const int i, const int middle_size, const int inner_size) {
    // &a[outer_size][0][inner_size] = &a[j]
    const int inner_i = i % inner_size;
    const int outer_i = i / inner_size;
    return outer_i * middle_size * inner_size + inner_i; // j
}

template<typename T = DType>
MOBULA_DEVICE T ADD_FUNC(const T &a, const T &b) {
    return a + b;
}
template<typename T = DType>
MOBULA_DEVICE T MAX_FUNC(const T &a, const T &b) {
    return a > b ? a : b;
}

} // namespace mobula

// C API
extern "C" {

void set_device(const int device_id);

}

#endif
