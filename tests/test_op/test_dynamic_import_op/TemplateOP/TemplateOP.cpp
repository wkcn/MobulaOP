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
  parfor(n, [&](int i) { c[i] = maximum(a[i], b[i]); });
}

template <typename T1, typename T2, typename T3>
MOBULA_KERNEL maximum_3type_kernel(const int n, const T1* a, const T2* b,
                                   T3* c) {
  // type(T1) < type(T2) < type(T3)
  parfor(n, [&](int i) {
    T2 a_t2 = static_cast<T2>(a[i]);
    T2 c_t2 = maximum(a_t2, b[i]);
    c[i] = static_cast<T3>(c_t2);
  });
}
