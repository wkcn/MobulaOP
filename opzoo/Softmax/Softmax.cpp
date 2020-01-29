#include "mobula_op.h"

namespace mobula {
template <typename T>
MOBULA_DEVICE void Reduce(const int n, const T *data, T *out,
                          void (*func)(T &, const T &), T init) {
  const int num_threads = get_num_threads();
  const int thread_num = get_thread_num();
  // 1. Reduce to `num_threads` slots
  T res = init;
  int start, end;
  get_parfor_range(n, num_threads, thread_num, &start, &end);
  for (int i = start; i < end; ++i) {
    func(res, data[i]);
  }
  out[thread_num] = res;
  __syncthreads();
  // 2. Reduce `num_threads` slots to the first slot
  int bi = 1;
  while (bi < num_threads) {
    int mask = (bi << 1) - 1;
    if ((thread_num & mask) == 0) {
      // valid thread
      int other_i = thread_num + bi;
      if (other_i < num_threads) func(out[thread_num], out[other_i]);
    }
    bi <<= 1;
    __syncthreads();
  }
}

template <typename T>
MOBULA_DEVICE void Reduce(const int n, const T *data, T *out,
                          void (*func_reduce)(T &, const T &, T &),
                          void (*func_merge)(T &, T &, const T &, const T &),
                          T init) {
  const int num_threads = get_num_threads();
  const int thread_num = get_thread_num();
  // 1. Reduce to `num_threads` slots
  T res = init;
  int start, end;
  T residual = 0;
  get_parfor_range(n, num_threads, thread_num, &start, &end);
  for (int i = start; i < end; ++i) {
    func_reduce(res, data[i], residual);
  }
  out[thread_num] = res;
  __syncthreads();
  // 2. merge the adjacent threads
  if ((thread_num & 1) == 0) {
    func_reduce(out[thread_num], out[thread_num + 1], residual);
  }
  __syncthreads();
  // 3. Record the residual in out
  if (thread_num & 1) {
    out[thread_num] = residual;
  }
  __syncthreads();
  // 4. Reduce `num_threads` slots to the first slot
  int bi = 1 << 1;
  while (bi < num_threads) {
    int mask = (bi << 1) - 1;
    if ((thread_num & mask) == 0) {
      // valid thread
      int other_i = thread_num + bi;
      if (other_i < num_threads)
        func_merge(out[thread_num], out[thread_num + 1], out[other_i],
                   out[other_i + 1]);
    }
    bi <<= 1;
    __syncthreads();
  }
}

template <typename T>
MOBULA_DEVICE void max_func(T &dst, const T &src) {
  if (src > dst) dst = src;
}

template <typename T>
MOBULA_DEVICE void add_func(T &dst, const T &src) {
  dst += src;
}

template <typename T>
MOBULA_DEVICE void add_residual_reduce_func(T &dst, const T &src, T &residual) {
  T y = src - residual;
  T t = dst + y;
  residual = (t - dst) - y;
  dst = t;
}

template <typename T>
MOBULA_DEVICE void add_residual_merge_func(T &dst, T &dst_residual,
                                           const T &src,
                                           const T &src_residual) {
  T t1 = dst + src;
  T e = t1 - dst;
  T t2 = ((src - e) + (dst - (t1 - e))) + dst_residual + src_residual;
  dst = t1 + t2;
  dst_residual = t2 - (dst - t1);
}

template <typename T>
MOBULA_KERNEL softmax_channel_forward_kernel(const int C, const int N,
                                             const T *X, T *tmp, T *Y) {
  const T *x = X;
  T *y = Y;
  for (int n = 0; n < N; ++n, x += C, y += C) {
    // max
    Reduce(C, x, tmp, max_func<T>, x[0]);
    T max_value = tmp[0];
    parfor(C, [&](int i) { y[i] = exp(x[i] - max_value); });
    __syncthreads();
    // sum
    Reduce(C, y, tmp, add_residual_reduce_func<T>, add_residual_merge_func<T>,
           T(0));
    T sum_value = tmp[0];
    parfor(C, [&](int i) { y[i] /= sum_value; });
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
