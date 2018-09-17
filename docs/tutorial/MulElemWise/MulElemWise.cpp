template <typename T>
MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* out) {
  parfor(n, [&](int i) { out[i] = a[i] * b[i]; });
}
