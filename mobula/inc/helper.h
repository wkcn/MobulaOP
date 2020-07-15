#ifndef MOBULA_INC_HELPER_H_
#define MOBULA_INC_HELPER_H_

#include "defines.h"

namespace mobula {

// Reduce without residual value
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
  /*
   * for example, if num_thread is 5,
   * thread ids: 0000, 0001, 0010, 0011, 0100
   * data index ids: 0000, 0001, 0010, 0011, 0100
   * step1: reduce the rightest 1st,
   *   (0000, 0001) -> 0000 in thread 0000
   *   (0010, 0011) -> 0010 in thread 0010 (+2)
   *   (0100,     ) -> 0100 in thread 0100
   * step2: reduce the rightest 2nd,
   *   (0000, 0010) -> 0000 in thread 0000
   *   (0100,     ) -> 0100 in thread 0100 (+4)
   * step3: reduce the rightest 3rd,
   *   (0000, 0100) -> 0000 in thread 0000
   */
  int bi = 1;
  while (bi < num_threads) {
    int mask = (bi << 1) - 1;
    if ((thread_num & mask) == 0) {
      // valid thread
      int other_i = thread_num | bi;
      if (other_i < num_threads) func(out[thread_num], out[other_i]);
    }
    bi <<= 1;
    __syncthreads();
  }
}

// Reduce with residual value
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
  T residual = T(0);
  get_parfor_range(n, num_threads, thread_num, &start, &end);
  for (int i = start; i < end; ++i) {
    func_reduce(res, data[i], residual);
  }
  out[thread_num] = res;
  __syncthreads();
  // 2. Merge the adjacent threads
  if ((thread_num & 1) == 0) {
    func_reduce(out[thread_num], out[thread_num + 1], residual);
    out[thread_num + 1] = residual;
  }
  __syncthreads();
  // 3. Record the residual in out
  if (thread_num & 1) {
    out[thread_num] += residual;
  }
  __syncthreads();
  // 4. Reduce `num_threads` slots to the first slot
  int bi = 1 << 1;
  while (bi < num_threads) {
    int mask = (bi << 1) - 1;
    if ((thread_num & mask) == 0) {
      // valid thread
      int other_i = thread_num | bi;
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

}  // namespace mobula

#endif  // MOBULA_INC_HELPER_H_
