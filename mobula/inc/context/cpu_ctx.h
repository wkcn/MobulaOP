#ifndef MOBULA_INC_CONTEXT_CPU_CTX_H_
#define MOBULA_INC_CONTEXT_CPU_CTX_H_

#define MOBULA_KERNEL void
#define MOBULA_DEVICE

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include "../ctypes.h"
#include "./common.h"

namespace mobula {

#if USING_CBLAS
#include <cblas.h>
inline void blas_gemm(const int axis, const bool tA, const bool tB, const int M,
                      const int N, const int K, const float alpha,
                      const float *A, const int lda, const float *B,
                      const int ldb, const float beta, float *C,
                      const int ldc) {
  cblas_sgemm(axis == 0 ? CblasRowMajor : CblasColMajor,
              tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans, M,
              N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

using std::abs;
using std::max;
using std::min;

#if HOST_NUM_THREADS > 1 || USING_OPENMP
constexpr int NUM_MOBULA_ATOMIC_ADD_MUTEXES = HOST_NUM_THREADS * 8;
static std::mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
inline MOBULA_DEVICE float atomic_add(const float val, float *address) {
  PointerValue id = (reinterpret_cast<PointerValue>(address) / sizeof(float)) %
                    NUM_MOBULA_ATOMIC_ADD_MUTEXES;
  MOBULA_ATOMIC_ADD_MUTEXES[id].lock();
  *address += val;
  MOBULA_ATOMIC_ADD_MUTEXES[id].unlock();
  return *address;
}
#else
// no lock for single thread mode
inline MOBULA_DEVICE float atomic_add(const float val, float *address) {
  *address += val;
  return *address;
}
#endif

template <typename T>
T *new_array(size_t size) {
  return new T[size];
}

template <typename T>
void del_array(T *p) {
  delete[] p;
}

template <typename T>
T *MemcpyHostToDev(T *dst, const T *src, size_t size) {
  if (dst == src) return dst;
  return static_cast<T *>(memcpy(dst, src, size));
}

template <typename T>
T *MemcpyDevToHost(T *dst, const T *src, size_t size) {
  if (dst == src) return dst;
  return static_cast<T *>(memcpy(dst, src, size));
}

template <typename T>
T *MemcpyDevToDev(T *dst, const T *src, size_t size) {
  if (dst == src) return dst;
  return static_cast<T *>(memcpy(dst, src, size));
}

}  // namespace mobula

#define KERNEL_RUN_BEGIN(device_id) \
  {                                 \
    UNUSED_EXPR(device_id)
#define KERNEL_RUN_END(device_id) \
  UNUSED_EXPR(device_id);         \
  }

#if USING_OPENMP
#include "./openmp_ctx.h"
#else
#include "./naive_ctx.h"
#endif

#endif  // MOBULA_INC_CONTEXT_CPU_CTX_H_
