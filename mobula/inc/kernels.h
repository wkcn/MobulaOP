#ifndef _MOBULA_KERNELS_
#define _MOBULA_KERNELS_

#include "defines.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL add_kernel(const int n, const T *a, const T *b, T *out);

}

#endif
