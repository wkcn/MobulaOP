#ifndef _OPENMP_CTX_
#define _OPENMP_CTX_

#include <omp.h>

namespace mobula {

#define KERNEL_LOOP(i,n) _Pragma("omp parallel for") \
                         for (int i = 0;i < (n);++i)
#define KERNEL_RUN(a, n) a

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        F(i);
    }
}

}

#endif
