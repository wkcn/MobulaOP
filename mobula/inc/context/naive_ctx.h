#ifndef MOBULA_INC_CONTEXT_NAIVE_CTX_H_
#define MOBULA_INC_CONTEXT_NAIVE_CTX_H_

#include <algorithm>
#include <condition_variable>
#include <thread>
#include <vector>

namespace mobula {

#if HOST_NUM_THREADS > 1

static thread_local int thread_local_i;
static thread_local int thread_local_n;

class Barrier {
 public:
  explicit Barrier(size_t nthreads) : count_(nthreads), nthreads_(nthreads) {}
  void wait() {
    std::unique_lock<std::mutex> lck(mutex_);
    if (--count_ == 0) {
      // set `count` for next barrier
      count_ = nthreads_;
      cv_.notify_all();
    } else {
      cv_.wait(lck);
    }
  }

 private:
  size_t count_;
  size_t nthreads_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

static thread_local Barrier *thread_local_barrier;

template <typename Func, typename... Args>
void thread_func_wrapper(Func func, const int i, const int nthreads,
                         const int n, Barrier *barrier, Args... args) {
  thread_local_i = i;
  thread_local_n = nthreads;
  thread_local_barrier = barrier;
  func(n, args...);
}

template <typename Func>
class KernelRunner {
 public:
  explicit KernelRunner(Func func) : func_(func) {}
  template <typename... Args>
  void operator()(const int n, Args... args) {
    const int nthreads = std::min(n, HOST_NUM_THREADS);
    std::vector<std::thread> threads(nthreads);
    Barrier barrier(nthreads);
    for (int i = 0; i < nthreads; ++i) {
      threads[i] = std::thread(thread_func_wrapper<Func, Args...>, func_, i,
                               nthreads, n, &barrier, args...);
    }
    for (int i = 0; i < nthreads; ++i) {
      threads[i].join();
    }
  }

 private:
  Func func_;
};

MOBULA_DEVICE inline int get_num_threads() { return thread_local_n; }

MOBULA_DEVICE inline int get_thread_num() { return thread_local_i; }

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

inline void __syncthreads() { thread_local_barrier->wait(); }

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

#endif  // MOBULA_INC_CONTEXT_NAIVE_CTX_H_
