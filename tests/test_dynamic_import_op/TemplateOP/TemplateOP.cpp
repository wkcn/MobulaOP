/*!
 * \file AdditionOP.cpp
 * \brief a C++ addition operator
 */

// include Mobula OP head file
#include "mobula_op.h"
using namespace mobula;

template <typename T>
MOBULA_DEVICE T maximum(const T a, const T b) {
    return a >= b ? a : b;
}

/**
 * \brief the forward kernel implementation of AdditionOP (out_c = in_a + in_b)
 * \param n      the number of elements
 * \param a      input array
 * \param b      input array
 * \param c     output array
 */
// use `MOBULA_KERNEL` macro to declare a kernel function
// the kernel function will be adapted for CPU and GPU
template <typename T>
MOBULA_KERNEL maximum_kernel(const int n, const T* a, const T* b, T* c) {
    // use parallel for-loop
    // `parfor(number-of-iterations, function)`
    // please NOTE Thread Safety in `parfor`
    parfor(n, [&](int i){
        c[i] = maximum(a[i], b[i]);
    });
}
