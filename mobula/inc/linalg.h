#ifndef MOBULA_INC_LINALG_H_
#define MOBULA_INC_LINALG_H_

#include "./defines.h"
#include "context/context.h"

namespace mobula {

// out[i, j] = sum(a[i, :] * b[:, j])
template <typename T>
MOBULA_KERNEL linalg_gemm_ff_kernel(const int n, const T *a, const T *b,
                                    const int U, const int J, T *out) {
  parfor(n, [&](int i) {
    for (int u = 0; u < U; ++u) {
      for (int j = 0; j < J; ++j) {
        out[i * J + j] += a[i * U + u] * b[u * J + j];
      }
    }
  });
}

// out[i, j] = sum(a[i, :] * b[j, :])
template <typename T>
MOBULA_KERNEL linalg_gemm_ft_kernel(const int n, const T *a, const T *b,
                                    const int U, const int J, T *out) {
  parfor(n, [&](int index) {
    int i = index / J;
    int j = index % J;
    for (int u = 0; u < U; ++u) {
      out[i * J + j] += a[i * U + u] * b[j * U + u];
    }
  });
}

// out[i, j] = sum(a[:, i] * b[:, j])
template <typename T>
MOBULA_KERNEL linalg_gemm_tf_kernel(const int n, const T *a, const T *b,
                                    const int I, const int U, const int J,
                                    T *out) {
  for (int u = 0; u < U; ++u) {
    parfor(n, [&](int index) {
      int i = index / J;
      int j = index % J;
      out[i * J + j] += a[u * I + i] * b[u * J + j];
    });
  }
}

// out[i, j] = sum(a[:, i] * b[j, :])
template <typename T>
MOBULA_KERNEL linalg_gemm_tt_kernel(const int n, const T *a, const T *b,
                                    const int I, const int U, const int J,
                                    T *out) {
  parfor(n, [&](int j) {
    for (int u = 0; u < U; ++u) {
      for (int i = 0; i < I; ++i) {
        out[i * J + j] += a[u * I + i] * b[j * U + u];
      }
    }
  });
}

}  // namespace mobula

extern "C" {
using namespace mobula;

MOBULA_DLL void linalg_gemm_ff(const DType *a, const DType *b, const int I,
                               const int U, const int J, DType *out) {
#if not USING_CBLAS
  const int N = I;
  KERNEL_RUN(linalg_gemm_ff_kernel<DType>, N)(N, a, b, U, J, out);
#else
  blas_gemm(0, false, false, I, J, U, 1.0f, a, U, b, J, 1.0f, out, J);
#endif
}

MOBULA_DLL void linalg_gemm_ft(const DType *a, const DType *b, const int I,
                               const int U, const int J, DType *out) {
#if not USING_CBLAS
  const int N = I * J;
  KERNEL_RUN(linalg_gemm_ft_kernel<DType>, N)(N, a, b, U, J, out);
#else
  blas_gemm(0, false, true, I, J, U, 1.0f, a, U, b, U, 1.0f, out, J);
#endif
}

MOBULA_DLL void linalg_gemm_tf(const DType *a, const DType *b, const int I,
                               const int U, const int J, DType *out) {
#if not USING_CBLAS
  const int N = I * J;
  KERNEL_RUN(linalg_gemm_tf_kernel<DType>, N)(N, a, b, I, U, J, out);
#else
  blas_gemm(0, true, false, I, J, U, 1.0f, a, I, b, J, 1.0f, out, J);
#endif
}

MOBULA_DLL void linalg_gemm_tt(const DType *a, const DType *b, const int I,
                               const int U, const int J, DType *out) {
#if not USING_CBLAS
  const int N = J;
  KERNEL_RUN(linalg_gemm_tt_kernel<DType>, N)(N, a, b, I, U, J, out);
#else
  blas_gemm(0, true, true, I, J, U, 1.0f, a, I, b, U, 1.0f, out, J);
#endif
}
}

#endif  // MOBULA_INC_LINALG_H_
