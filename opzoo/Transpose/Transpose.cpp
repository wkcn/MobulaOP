#include "mobula_op.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL transpose_2d_ci_kernel(const int N, const T *X, const int R,
                                     const int C, T *Y) {
  // continuous input
  // N = R * C
  // X: (R, C)
  // Y: (C, R)
  // Y[c, r] = X[r, c]
  parfor(N, [&](int i) {
    int r = i / C;
    int c = i % C;
    int t = R * c + r;
    Y[t] = X[i];
  });
}

template <typename T>
MOBULA_KERNEL transpose_2d_co_kernel(const int N, const T *X, const int R,
                                     const int C, T *Y) {
  // continuous input
  // N = R * C
  // X: (R, C)
  // Y: (C, R)
  // Y[r, c] = X[c, r]
  parfor(N, [&](int i) {
    int r = i / R;
    int c = i % R;
    int t = C * c + r;
    Y[i] = X[t];
  });
}

}  // namespace mobula
