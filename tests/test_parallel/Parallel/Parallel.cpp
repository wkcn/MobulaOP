MOBULA_KERNEL test_syncthreads_kernel(const int N, int *out) {
  const int num_threads = get_num_threads();
  const int tid = get_thread_num();
  for (int i = 0; i < N; ++i) {
    if (i % num_threads == tid) {
      ++(*out);
    }
    __syncthreads();
  }
}

MOBULA_KERNEL test_parfor_kernel(const int N, int *out) {
  parfor(N, [&](int i) { out[i] = 0; });
  __syncthreads();
  parfor(N, [&](int i) { out[i] += i; });
}
