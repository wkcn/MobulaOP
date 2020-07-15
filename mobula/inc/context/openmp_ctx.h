#ifndef MOBULA_INC_CONTEXT_OPENMP_CTX_H_
#define MOBULA_INC_CONTEXT_OPENMP_CTX_H_

#include <omp.h>

#include <algorithm>

#ifndef _MSC_VER
#define __pragma(id) _Pragma(#id)
#endif

namespace mobula {

#if HOST_NUM_THREADS > 1

template <typename Func>
class KernelRunner {
 public:
  explicit KernelRunner(Func func) : func_(func) {}
  template <typename... Args>
  void operator()(const int n, Args... args) {
    const int nthreads = std::min(n, omp_get_max_threads());
#pragma omp parallel num_threads(nthreads)
    { func_(n, args...); }
  }

 private:
  Func func_;
};

MOBULA_DEVICE inline int get_num_threads() { return omp_get_num_threads(); }

MOBULA_DEVICE inline int get_thread_num() { return omp_get_thread_num(); }

template <typename Func>
MOBULA_DEVICE void parfor(const size_t n, Func F) {
  INDEX_TYPE_SWITCH(n, {
    index_t start, end;
    get_parfor_range(n, get_num_threads(), get_thread_num(), &start, &end);
    for (index_t i = start; i < end; ++i) {
      F(i);
    }
  });
}

inline void __syncthreads() { __pragma(omp barrier); }

#define KERNEL_RUN(a) (mobula::KernelRunner<decltype(&(a))>(&(a)))

#else  // HOST_NUM_THREADS > 1 else

MOBULA_DEVICE inline int get_num_threads() { return 1; }

MOBULA_DEVICE inline int get_thread_num() { return 1; }

template <typename Func>
MOBULA_DEVICE void parfor(const size_t n, Func F) {
  INDEX_TYPE_SWITCH(n, {
    for (index_t i = 0; i < static_cast<index_t>(n); ++i) {
      F(i);
    }
  });
}

inline void __syncthreads() {}

#define KERNEL_RUN(a) (a)

#endif  // HOST_NUM_THREADS > 1

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_OPENMP_CTX_H_
