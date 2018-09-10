#ifndef _NAIVE_CTX_
#define _NAIVE_CTX_

#include <vector>
#include <map>
#include <utility>
#include <thread>
#include <mutex>

namespace mobula {

#if HOST_NUM_THREADS > 1
// global_thread_id -> (local_thread_id, num_threads)
static std::map<std::thread::id, std::pair<int, int> > MOBULA_KERNEL_INFOS;
static std::mutex MOBULA_KERNEL_MUTEX;

template<typename Func>
class KernelRunner {
public:
    KernelRunner(Func func, int n):_func(func), _n(n){};
    template<typename ...Args>
    void operator()(Args... args){
        const int nthreads = std::min(_n, HOST_NUM_THREADS);
        std::vector<std::thread> threads(nthreads);
        std::vector<std::thread::id> thread_ids;
        MOBULA_KERNEL_MUTEX.lock();
        for (int i = 0; i < nthreads; ++i) {
            threads[i] = std::thread(_func, args...);
            std::thread::id id = threads[i].get_id();
            thread_ids.push_back(id);
            MOBULA_KERNEL_INFOS[id] = std::make_pair(i, nthreads);
        }
        MOBULA_KERNEL_MUTEX.unlock();
        for (int i = 0;i < nthreads; ++i) {
            threads[i].join();
        }
        // release resource
        MOBULA_KERNEL_MUTEX.lock();
        for (std::thread::id tid : thread_ids) {
            MOBULA_KERNEL_INFOS.erase(tid);
        }
        MOBULA_KERNEL_MUTEX.unlock();
    }
private:
    Func _func;
    int _n;
};

inline MOBULA_DEVICE void get_parfor_range(const int n, const int num_threads, const int thread_id, int *start, int *end) {
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
    MOBULA_KERNEL_MUTEX.lock();
    const std::pair<int, int> kernel_info = MOBULA_KERNEL_INFOS[std::this_thread::get_id()];
    MOBULA_KERNEL_MUTEX.unlock();
    const int thread_id = kernel_info.first;
    const int num_threads = kernel_info.second;
    int start, end;
    get_parfor_range(n, num_threads, thread_id, &start, &end);
    for (int i = start; i < end; ++i) {
        F(i);
    }
}

#define KERNEL_RUN(a, n) (KernelRunner<decltype(&(a))>(&(a), (n)))

#else // HOST_NUM_THREADS > 1 else

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    for (int i = 0; i < n; ++i) {
        F(i);
    }
}

#endif // HOST_NUM_THREADS > 1

} // namespace mobula

#endif
