#ifndef MOBULA_INC_CONTEXT_CUDA_CTX_H_
#define MOBULA_INC_CONTEXT_CUDA_CTX_H_

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include "./common.h"

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

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  do {                                                                       \
    cudaError_t e = cudaGetLastError();                                      \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  } while (0)

/*!
 * \brief Check CUDA error.
 * \param condition the return value when calling CUDA function
 */
#define CUDA_CHECK(condition)                                  \
  /* Code block avoids redefinition of cudaError_t error */    \
  do {                                                         \
    cudaError_t error = condition;                             \
    CHECK_EQ(error, cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

template <typename Func>
class KernelRunner {
 public:
  KernelRunner(Func func, int n) : func_(func), n_(n) {}
  template <typename... Args>
  void operator()(Args... args) {
    const int nthreads = std::min(n_, HOST_NUM_THREADS);
    const int threadsPerBlock = CUDA_GET_NUM_THREADS(nthreads);
    const int blocks = CUDA_GET_BLOCKS(nthreads, threadsPerBlock);
    func_<<<blocks, threadsPerBlock>>>(args...);
    CHECK_CUDA_ERROR("Run Kernel");
  }

 private:
  Func func_;
  int n_;
};

#define KERNEL_RUN(a, n) (mobula::KernelRunner<decltype(&(a))>(&(a), (n)))

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
