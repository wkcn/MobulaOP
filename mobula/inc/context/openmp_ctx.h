#ifndef MOBULA_INC_CONTEXT_OPENMP_CTX_H_
#define MOBULA_INC_CONTEXT_OPENMP_CTX_H_

#include <omp.h>

namespace mobula {

#define KERNEL_RUN(a, n) (a)

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    F(i);
  }
}

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_OPENMP_CTX_H_
