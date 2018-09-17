#ifndef MOBULA_INC_CONTEXT_HIP_CTX_H_
#define MOBULA_INC_CONTEXT_HIP_CTX_H_

#include <hip/hip_runtime.h>
#include <algorithm>
#include <iostream>

namespace mobula {

#if USING_CBLAS
#include <hipblas.h>
static hipblasHandle_t HIPBLAS_HANDLE;
static struct HIPBLAS_INIT {
  HIPBLAS_INIT() { hipblasCreate(&HIPBLAS_HANDLE); }
} hipblas_init_dummy;
inline void blas_gemm(const int axis, const bool tA, const bool tB, const int M,
                      const int N, const int K, const float alpha,
                      const float *A, const int lda, const float *B,
                      const int ldb, const float beta, float *C,
                      const int ldc) {
  if (axis == 0)
    // row major
    hipblasSgemm(HIPBLAS_HANDLE, tB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                tA ? HIPBLAS_OP_T : HIPBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda,
                &beta, C, ldc);
  else
    // column major
    hipblasSgemm(HIPBLAS_HANDLE, tA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                tB ? HIPBLAS_OP_T : HIPBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb,
                &beta, C, ldc);
}
#endif

const int HIP_MAX_GRID_NUM = 65535;
const int HIP_MAX_NUM_THREADS = 512;
inline int HIP_GET_NUM_THREADS(const int n) {
  return std::min(HIP_MAX_NUM_THREADS, ((n + 31) / 32) * 32);
}
inline int HIP_GET_BLOCKS(const int n, const int num_threads) {
  return std::min(HIP_MAX_GRID_NUM, n + num_threads - 1) / num_threads;
}

#define MOBULA_KERNEL __global__ void
#define MOBULA_DEVICE __device__

template <typename Func>
class KernelRunner {
 public:
  KernelRunner(Func func, int n) : func_(func), n_(n) {}
  template <typename... Args>
  void operator()(Args... args) {
    const int nthreads = std::min(n_, HOST_NUM_THREADS);
    const int threadsPerBlock = HIP_GET_NUM_THREADS(nthreads);
    const int blocks = HIP_GET_BLOCKS(nthreads, threadsPerBlock);
    hipLaunchKernelGGL(func_, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       args...);
  }

 private:
  Func func_;
  int n_;
};

#define KERNEL_RUN(a, n) (mobula::KernelRunner<decltype(&(a))>(&(a), (n)))

#define HIP_CHECK(condition)                              \
  do {                                                    \
    hipError_t error = condition;                         \
    if (error != hipSuccess) {                            \
      std::cout << hipGetErrorString(error) << std::endl; \
    }                                                     \
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
  hipMalloc(reinterpret_cast<void **>(&p), sizeof(T) * size);
  return p;
}

template <typename T>
void del_array(T *p) {
  hipFree(p);
}

template <typename T>
T *MemcpyHostToDev(T *dst, const T *src, size_t size) {
  hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
  return dst;
}

template <typename T>
T *MemcpyDevToHost(T *dst, const T *src, size_t size) {
  hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
  return dst;
}

template <typename T>
T *MemcpyDevToDev(T *dst, const T *src, size_t size) {
  hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
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

// parfor for hip device should be called in hip kernel.
template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
  // [gridDim.x, blockDim.x]
  const int num_threads = hipGridDim_x * hipBlockDim_x;
  // thread_id is in [0, num_threads)
  const int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int start, end;
  get_parfor_range(n, num_threads, thread_id, &start, &end);
  for (int i = start; i < end; ++i) {
    F(i);
  }
}

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_HIP_CTX_H_
