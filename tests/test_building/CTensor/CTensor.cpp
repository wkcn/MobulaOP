struct CTensor {
  float *data;
  int size;
};

MOBULA_KERNEL ctensor_inc_kernel(const int n, CTensor *tensor) {
  CHECK_EQ(n, 1);
  parfor(tensor->size, [&](int i) { tensor->data[i] += 1; });
}
