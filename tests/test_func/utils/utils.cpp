#include "mobula_op.h"

namespace mobula {
template <typename T>
MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* c) {
  parfor(n, [&](int i) { c[i] = a[i] * b[i]; });
}

template <typename T>
MOBULA_KERNEL test_thread_kernel(const int n, T* out_1, T* out_2, T* out_3,
                                 T* out_4, T* out_5) {
  parfor(n, [&](int i) { out_1[i] = i; });
  parfor(n / 2, [&](int i) { out_2[i] = i * 2; });
  parfor(n / 3, [&](int i) { out_3[i] = i * 3; });
  parfor(n * 2, [&](int i) { out_4[i] = i * 2; });
  parfor(n * 3, [&](int i) { out_5[i] = i * 3; });
}

template <typename T1, typename T2>
MOBULA_KERNEL test_const_template_kernel(const int n, const T1 value, T2* out) {
  parfor(n, [&](int i) { out[i] = static_cast<T2>(value); });
}

template <typename T>
MOBULA_KERNEL test_void_pointer_kernel(const int n, const void* p, T* out) {
  parfor(n, [&](int) {});
  *out = reinterpret_cast<T>(p);
}

template <typename T>
MOBULA_KERNEL infer_type_for_const_kernel(const int n, T value, T* out) {
  parfor(n, [&](int i) { out[i] = value; });
}

template <typename T>
MOBULA_FUNC T set_and_return(const T* a, T* b) {
  b[0] = a[0];
  return a[0];
}

}  // namespace mobula
