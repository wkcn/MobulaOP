#include "mobula_op.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL sum_kernel(const int N, const T *X, T *Y) {
  Reduce(N, X, Y, add_residual_reduce_func<T>, add_residual_merge_func<T>,
         T(0));
}
}  // namespace mobula
