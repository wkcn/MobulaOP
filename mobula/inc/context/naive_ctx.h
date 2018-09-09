#ifndef _NAIVE_CTX_
#define _NAIVE_CTX_

#include <vector>
#include <map>
#include <utility>
#include <thread>
#include <mutex>

namespace mobula {

#if HOST_NUM_THREADS > 1
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
        const int step = (_n + nthreads - 1) / nthreads;
        int blockBegin = 0;
        int blockEnd;
        for (int i = 0; i < nthreads; ++i) {
            threads[i] = std::thread(_func, args...);
            std::thread::id id = threads[i].get_id();
            thread_ids.push_back(id);
            blockEnd = std::min(blockBegin + step, _n);
            MOBULA_KERNEL_INFOS[id] = std::make_pair(blockBegin, blockEnd);
            blockBegin = blockEnd;
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

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    MOBULA_KERNEL_MUTEX.lock();
    const std::pair<int, int> MOBULA_KERNEL_INFO = MOBULA_KERNEL_INFOS[std::this_thread::get_id()];
    MOBULA_KERNEL_MUTEX.unlock();
    const int MOBULA_KERNEL_START = MOBULA_KERNEL_INFO.first;
    const int MOBULA_KERNEL_END = min(MOBULA_KERNEL_INFO.second, n);
    for (int i = MOBULA_KERNEL_START; i < MOBULA_KERNEL_END; ++i) {
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
