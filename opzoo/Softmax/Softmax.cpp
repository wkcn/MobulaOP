#include "mobula_op.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL softmax_channel_forward_kernel(const int C, const int N,
                                             const T *X, T *tmp, T *Y) {
  const T *x = X;
  T *y = Y;
  for (int n = 0; n < N; ++n, x += C, y += C) {
    // max
    Reduce(C, x, tmp, max_func<T>, x[0]);
    const T max_value = tmp[0];
    parfor(C, [&](int i) { y[i] = exp(x[i] - max_value); });
    __syncthreads();
    // sum
    Reduce(C, y, tmp, add_residual_reduce_func<T>, add_residual_merge_func<T>,
           T(0));
    const T sum_value = tmp[0];
    parfor(C, [&](int i) { y[i] /= sum_value; });
    __syncthreads();
  }
}

template <typename T>
MOBULA_KERNEL softmax_batch_forward_kernel(const int N, const int C, const T *X,
                                           T *Y) {
  parfor(N, [&](int i) {
    const T *x = X + i * C;
    T *y = Y + i * C;
    // find maximum
    T max_value = x[0];
    for (int c = 1; c < C; ++c) {
      if (x[c] > max_value) max_value = x[c];
    }
    // compute exp(x - max_value)
    for (int c = 0; c < C; ++c) {
      y[c] = exp(x[c] - max_value);
    }
    // sum
    T sum_value = 0;
    for (int c = 0; c < C; ++c) {
      sum_value += y[c];
    }
    // divide by sum
    for (int c = 0; c < C; ++c) {
      y[c] /= sum_value;
    }
  });
}

template <typename T>
MOBULA_KERNEL softmax_backward_kernel(const int N, const int C, const T *Y,
                                      const T *dY, T *dX) {
  parfor(N, [&](int i) {
    const T *y = Y + i * C;
    const T *dy = dY + i * C;
    T *dx = dX + i * C;
    for (int j = 0; j < C; ++j) {
      T &grad = dx[j];
      for (int k = 0; k < C; ++k) {
        grad -= y[j] * y[k] * dy[k];
      }
      grad += y[j] * dy[j];
    }
  });
}

}  // namespace mobula
