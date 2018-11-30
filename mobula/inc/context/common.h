#ifndef MOBULA_INC_CONTEXT_COMMON_H_
#define MOBULA_INC_CONTEXT_COMMON_H_

#include <cassert>
#include <climits>
#include "../logging.h"

namespace mobula {

template <typename T>
void UNUSED_EXPR(T &&) {}

#define INDEX_TYPE_SWITCH(N, ...)                                 \
  do {                                                            \
    if (N <= INT32_MAX) {                                         \
      typedef int32_t index_t;                                    \
      { __VA_ARGS__ }                                             \
    } else if (N <= INT64_MAX) {                                  \
      typedef int64_t index_t;                                    \
      { __VA_ARGS__ }                                             \
    } else {                                                      \
      printf("Max number of iteration exceeds: n > INT64_MAX\n"); \
      assert(0);                                                  \
    }                                                             \
  } while (0)

template <typename index_t>
inline MOBULA_DEVICE void get_parfor_range(const size_t n,
                                           const int num_threads,
                                           const int thread_id, index_t *start,
                                           index_t *end) {
  const index_t avg_len = n / num_threads;
  const index_t rest = n % num_threads;
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

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_COMMON_H_
