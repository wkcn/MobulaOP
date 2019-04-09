#include "mobula_op.h"

int main() {
  auto lib = mobula::op::load("./AddtionOP");
  auto func = lib.get_function("addtion_op_forward");

  const int n = 3;
  float a[3] = {1, 2, 3};
  float b[3] = {4, 5, 6};
  float *c = new float[3];

  func(n, a, b, c);

  for (int i = 0; i < n; ++i) {
    if (i != 0) cout << ", ";
    cout << c[i];
  }
  cout << endl;
  return 0;
}
