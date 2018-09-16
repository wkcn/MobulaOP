#ifndef MOBULA_INC_CONTEXT_NAIVE_CTX_H_
#define MOBULA_INC_CONTEXT_NAIVE_CTX_H_

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <thread>

namespace mobula {

#if HOST_NUM_THREADS > 1

static thread_local int thread_local_i;
static thread_local int thread_local_n;

template<typename Func, typename ...Args>
void thread_func_wrapper(Func func, const int i,
        const int nthreads, Args... args) {
    thread_local_i = i;
    thread_local_n = nthreads;
    func(args...);
}

template<typename Func>
class KernelRunner {
 public:
     KernelRunner(Func func, int n):func_(func), n_(n) {}
     template<typename ...Args>
     void operator()(Args... args) {
         const int nthreads = std::min(n_, HOST_NUM_THREADS);
         std::vector<std::thread> threads(nthreads);
         for (int i = 0; i < nthreads; ++i) {
             threads[i] = std::thread(thread_func_wrapper<Func, Args...>,
                     func_, i, nthreads, args...);
         }
         for (int i = 0; i < nthreads; ++i) {
             threads[i].join();
         }
     }
 private:
     Func func_;
     int n_;
};

inline MOBULA_DEVICE void get_parfor_range(const int n, const int num_threads,
        const int thread_id, int *start, int *end) {
    const int avg_len = n / num_threads;
    const int rest = n % num_threads;
    // [start, end)
    *start = avg_len * thread_id;
    if (rest > 0) {
        if (thread_id <= rest) {
            *start += thread_id;
        } else {
            *start += rest;
        }
    }
    *end = *start + avg_len + (thread_id < rest);
}

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    int start, end;
    get_parfor_range(n, thread_local_n, thread_local_i, &start, &end);
    for (int i = start; i < end; ++i) {
        F(i);
    }
}

#define KERNEL_RUN(a, n) (mobula::KernelRunner<decltype(&(a))>(&(a), (n)))

#else  // HOST_NUM_THREADS > 1 else

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    for (int i = 0; i < n; ++i) {
        F(i);
    }
}

#endif  // HOST_NUM_THREADS > 1

}  // namespace mobula

#endif  // MOBULA_INC_CONTEXT_NAIVE_CTX_H_
