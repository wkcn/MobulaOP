#ifndef _OPENMP_CTX_
#define _OPENMP_CTX_

namespace mobula {

#include "omp.h"

#define KERNEL_LOOP(i,n) _Pragma("omp parallel for") \
                         for (int i = 0;i < (n);++i)
#define KERNEL_RUN(a, n) a


}

#endif
