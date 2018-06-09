#ifndef _MOBULA_FUNC_
#define _MOBULA_FUNC_

#include "defines.h"

namespace mobula {


template <typename T, typename UNARY_FUNC>
MOBULA_KERNEL unary_kernel(const int n, const T *a, T *out, UNARY_FUNC func) {
    parfor(n, [&](int i) {
        out[i] = func(a[i]);
    });
}

template <typename T, typename BINARY_FUNC>
MOBULA_KERNEL binary_kernel(const int n, const T *a, const T *b, T *out, BINARY_FUNC func) {
    parfor(n, [&](int i) {
        out[i] = func(a[i], b[i]);
    });
}

}

extern "C" {
using namespace mobula;

#define REGISTER_UNARY_FUNC(func_name, func) \
    using T = DType;\
    void func_name(const int _n, const T *_a, T *_out) {\
        auto _func = func;\
        KERNEL_RUN((unary_kernel<T, decltype(_func)>), _n)(_n, _a, _out, _func);\
    } \

#define REGISTER_BINARY_FUNC(func_name, func) \
    void func_name(const int _n, const DType *_a, const DType *_b, DType *_out) {\
        auto _func = func;\
        KERNEL_RUN((binary_kernel<DType, decltype(_func)>), _n)(_n, _a, _b, _out, _func);\
    } \

REGISTER_UNARY_FUNC(abs_, [](const DType &a){return abs(a);})

REGISTER_BINARY_FUNC(add, [](const DType &a, const DType &b){return a + b;})
REGISTER_BINARY_FUNC(sub, [](const DType &a, const DType &b){return a - b;})
REGISTER_BINARY_FUNC(mul, [](const DType &a, const DType &b){return a * b;})
REGISTER_BINARY_FUNC(div_, [](const DType &a, const DType &b){return a / b;})

}

#endif
