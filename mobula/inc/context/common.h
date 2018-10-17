#ifndef MOBULA_INC_CONTEXT_COMMON_H_
#define MOBULA_INC_CONTEXT_COMMON_H_

#include "../logging.h"

namespace mobula {

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

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_COMMON_H_
