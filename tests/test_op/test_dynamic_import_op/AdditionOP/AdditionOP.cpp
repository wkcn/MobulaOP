/*!
 * \file AdditionOP.cpp
 * \brief a C++ addition operator
 */

// include Mobula OP head file
#include "mobula_op.h"
using namespace mobula;

/**
 * \brief the forward kernel implementation of AdditionOP (out_c = in_a + in_b)
 * \param n      the number of elements
 * \param a      input array
 * \param b      input array
 * \param c     output array
 */
// use `MOBULA_KERNEL` macro to declare a kernel function
// the kernel function will be adapted for CPU and GPU
MOBULA_KERNEL addition_op_forward_kernel(const int n, const float* a,
                                         const float* b, float* c) {
  // use parallel for-loop
  // `parfor(number-of-iterations, function)`
  // please NOTE Thread Safety in `parfor`
  parfor(n, [&](int i) { c[i] = a[i] + b[i]; });
}
