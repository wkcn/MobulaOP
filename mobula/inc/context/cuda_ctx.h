#ifndef MOBULA_INC_CONTEXT_CUDA_CTX_H_
#define MOBULA_INC_CONTEXT_CUDA_CTX_H_

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

namespace mobula {

#if USING_CBLAS
#include <cublas_v2.h>
static cublasHandle_t CUBLAS_HANDLE;
static struct CUBLAS_INIT {
  CUBLAS_INIT() { cublasCreate(&CUBLAS_HANDLE); }
} cublas_init_dummy;
inline void blas_gemm(const int axis, const bool tA, const bool tB, const int M,
                      const int N, const int K, const float alpha,
                      const float *A, const int lda, const float *B,
                      const int ldb, const float beta, float *C,
                      const int ldc) {
  if (axis == 0)
    // row major
    cublasSgemm(CUBLAS_HANDLE, tB ? CUBLAS_OP_T : CUBLAS_OP_N,
                tA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda,
                &beta, C, ldc);
  else
    // column major
    cublasSgemm(CUBLAS_HANDLE, tA ? CUBLAS_OP_T : CUBLAS_OP_N,
                tB ? CUBLAS_OP_T : CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb,
                &beta, C, ldc);
}
#endif

const int CUDA_MAX_GRID_NUM = 65535;
const int CUDA_MAX_NUM_THREADS = 512;
inline int CUDA_GET_NUM_THREADS(const int n) {
  return std::min(CUDA_MAX_NUM_THREADS, ((n + 31) / 32) * 32);
}
inline int CUDA_GET_BLOCKS(const int n, const int num_threads) {
  return std::min(CUDA_MAX_GRID_NUM, n + num_threads - 1) / num_threads;
}

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__

#define KERNEL_RUN(a, n)                                    \
  const int __cuda_num_threads__ = CUDA_GET_NUM_THREADS(n); \
  (a)<<<CUDA_GET_BLOCKS(n, __cuda_num_threads__), __cuda_num_threads__>>>

#define CUDA_CHECK(condition)                               \
  /* Code block avoids redefinition of cudaError_t error */ \
  do {                                                      \
    cudaError_t error = condition;                          \
    if (error != cudaSuccess) {                             \
      std::cout << cudaGetErrorString(error) << std::endl;  \
    }                                                       \
  } while (0)

template <typename T>
inline __device__ T atomic_add(const T val, T *address);

template <>
inline __device__ float atomic_add(const float val, float *address) {
  return atomicAdd(address, val);
}

template <typename T>
T *new_array(size_t size) {
  T *p;
  cudaMalloc(reinterpret_cast<void **>(&p), sizeof(T) * size);
  return p;
}

template <typename T>
void del_array(T *p) {
  cudaFree(p);
}

template <typename T>
T *MemcpyHostToDev(T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  return dst;
}

template <typename T>
T *MemcpyDevToHost(T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  return dst;
}

template <typename T>
T *MemcpyDevToDev(T *dst, const T *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  return dst;
}

inline MOBULA_DEVICE void get_parfor_range(const int n, const int num_threads,
                                           const int thread_id, int *start,
                                           int *end) {
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

// parfor for cuda device should be called in cuda kernel.
template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
  // [gridDim.x, blockDim.x]
  const int num_threads = gridDim.x * blockDim.x;
  // thread_id is in [0, num_threads)
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int start, end;
  get_parfor_range(n, num_threads, thread_id, &start, &end);
  for (int i = start; i < end; ++i) {
    F(i);
  }
}

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_CUDA_CTX_H_
