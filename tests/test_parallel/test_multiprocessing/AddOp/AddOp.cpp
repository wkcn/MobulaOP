template <typename T>
MOBULA_KERNEL add_kernel(const int n, const T *a, T b, T *c) {
  parfor(n, [&](int i) { c[i] = a[i] + b; });
}
