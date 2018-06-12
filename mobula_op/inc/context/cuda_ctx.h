#ifndef _CUDA_CTX_H_
#define _CUDA_CTX_H_

#include <iostream>
#include <cuda_runtime.h>

namespace mobula {

#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(n) ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__ __host__
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

template<typename T>
T* xnew(const int size) {
	T *p;
	cudaMalloc((void **)&p, sizeof(T) * size);
	return p;
}

template<typename T>
void xdel(T *p) {
	cudaFree(p);
}

template<typename T>
T* MemcpyHostToDev(T *dst, const T *src, int size) {
    return cudaMemcpy(dst, host, size, cudaMemcpyHostToDevice);
}

template<typename T>
T* MemcpyDevToHost(T *dst, const T *src, int size) {
    return cudaMemcpy(dst, host, size, cudaMemcpyDeviceToHost);
}

template<typename T>
T* MemcpyDevToDev(T *dst, const T *src, int size) {
    return cudaMemcpy(dst, host, size, cudaMemcpyDeviceToDevice);
}

// parfor for cuda device should be called in cuda kernel.
template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) {
        F(i);
    }
}


}

#endif
