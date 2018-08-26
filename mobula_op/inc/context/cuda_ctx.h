#ifndef _CUDA_CTX_H_
#define _CUDA_CTX_H_

#include <iostream>
#include <cuda_runtime.h>

#if USING_CBLAS
#include <cublas_v2.h>
static cublasHandle_t CUBLAS_HANDLE;
static struct CUBLAS_INIT {
    CUBLAS_INIT() {
        cublasCreate(&CUBLAS_HANDLE);
    }
} cublas_init_dummy;
inline void blas_gemm(const int axis, const bool tA, const bool tB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc) {
    if (axis == 0)
        // row major
        cublasSgemm(CUBLAS_HANDLE, tB ? CUBLAS_OP_T: CUBLAS_OP_N, tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
    else
        // column major
        cublasSgemm(CUBLAS_HANDLE, tA ? CUBLAS_OP_T: CUBLAS_OP_N, tB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
#endif

namespace mobula {

const int CUDA_MAX_GRID_NUM = 65535;
const int CUDA_NUM_THREADS = 512;
inline int CUDA_GET_BLOCKS(const int n) {
    return min(CUDA_MAX_GRID_NUM, n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__
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
inline __device__ T atomic_add(const T val, T* address);

template <>
inline __device__ float atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template<typename T>
T* xnew(size_t size) {
	T *p;
	cudaMalloc((void **)&p, sizeof(T) * size);
	return p;
}

template<typename T>
void xdel(T *p) {
	cudaFree(p);
}

template<typename T>
T* MemcpyHostToDev(T *dst, const T *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return dst;
}

template<typename T>
T* MemcpyDevToHost(T *dst, const T *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return dst;
}

template<typename T>
T* MemcpyDevToDev(T *dst, const T *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return dst;
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
