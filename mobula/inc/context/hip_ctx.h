#ifndef _HIP_CTX_H_
#define _HIP_CTX_H_

#include <iostream>
#include <hip/hip_runtime.h>

namespace mobula {

#if USING_CBLAS
#endif

const int HIP_MAX_GRID_NUM = 65535;
const int HIP_MAX_NUM_THREADS = 512;
inline int HIP_GET_NUM_THREADS(const int n) {
    return min(HIP_MAX_NUM_THREADS, ((n + 31) / 32) * 32);
}
inline int HIP_GET_BLOCKS(const int n, const int num_threads) {
    return min(HIP_MAX_GRID_NUM, n + num_threads - 1) / num_threads;
}

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__

template<typename Func>
class KernelRunner {
public:
    KernelRunner(Func func, int n):func_(func), n_(n){};
    template<typename ...Args>
    void operator()(Args... args){
        const int nthreads = std::min(n_, HOST_NUM_THREADS);
        const int threadsPerBlock = HIP_GET_NUM_THREADS(nthreads);  
        const int blocks = HIP_GET_BLOCKS(nthreads, threadsPerBlock); 
        hipLaunchKernelGGL(func_, dim3(blocks), dim3(threadsPerBlock), 0, 0, args...);
    }
private:
    Func func_;
    int n_;
};

#define KERNEL_RUN(a, n) (KernelRunner<decltype(&(a))>(&(a), (n)))

#define HIP_CHECK(condition) \
  do { \
    hipError_t error = condition; \
    if (error != hipSuccess) { \
      std::cout << hipGetErrorString(error) << std::endl; \
    } \
  } while (0)

template <typename T>
inline __device__ T atomic_add(const T val, T* address);

template <>
inline __device__ float atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template<typename T>
T* new_array(size_t size) {
	T *p;
	hipMalloc((void **)&p, sizeof(T) * size);
	return p;
}

template<typename T>
void del_array(T *p) {
	hipFree(p);
}

template<typename T>
T* MemcpyHostToDev(T *dst, const T *src, size_t size) {
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    return dst;
}

template<typename T>
T* MemcpyDevToHost(T *dst, const T *src, size_t size) {
    hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    return dst;
}

template<typename T>
T* MemcpyDevToDev(T *dst, const T *src, size_t size) {
    hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
    return dst;
}

inline MOBULA_DEVICE void get_parfor_range(const int n, const int num_threads, const int thread_id, int *start, int *end) {
    const int avg_len = n / num_threads;
    const int rest = n % num_threads;
    // [start, end)
    *start = avg_len * thread_id;
    if (rest > 0) {
        if (thread_id <= rest) {
            *start += thread_id;
        } else {
            *start += rest;
        }
    }
    *end = *start + avg_len + (thread_id < rest);
}

// parfor for hip device should be called in hip kernel.
template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    // [gridDim.x, blockDim.x]
    const int num_threads = hipGridDim_x * hipBlockDim_x;
    const int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;  // thread_id is in [0, num_threads)
    int start, end;
    get_parfor_range(n, num_threads, thread_id, &start, &end);
    for (int i = start; i < end; ++i) {
        F(i);
    }
}

}

#endif
