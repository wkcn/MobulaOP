#include "mobula_op.h"

namespace mobula {
template <typename T>
MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* c) {
    parfor(n, [&](int i) {
        c[i] = a[i] * b[i];
    });
}

}
