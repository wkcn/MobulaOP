#ifndef _NAIVE_ENGINE_
#define _NAIVE_ENGINE_

namespace mobula {

#if HOST_NUM_THREADS > 1
extern std::map<std::thread::id, std::pair<int, int> > MOBULA_KERNEL_INFOS;
extern std::mutex MOBULA_KERNEL_MUTEX;

template<typename Func>
class KernelRunner{
public:
	KernelRunner(Func func, int n):_func(func), _n(n <= HOST_NUM_THREADS ? n : HOST_NUM_THREADS){};
	template<typename ...Args>
	void operator()(Args... args){
        std::vector<std::thread> threads(_n);
        MOBULA_KERNEL_MUTEX.lock();
        for (int i = 0;i < _n;++i) {
            threads[i] = std::thread(_func, args...);
            std::thread::id id = threads[i].get_id();
            MOBULA_KERNEL_INFOS[id] = std::make_pair(i, _n);
        }
        MOBULA_KERNEL_MUTEX.unlock();
        for (int i = 0;i < _n;++i) {
            threads[i].join();
        }
    }
private:
	Func _func;
	int _n;
};

#define KERNEL_LOOP(i,n) MOBULA_KERNEL_MUTEX.lock(); \
						 const std::pair<int, int> MOBULA_KERNEL_INFO = MOBULA_KERNEL_INFOS[std::this_thread::get_id()]; \
						 MOBULA_KERNEL_INFOS.erase(std::this_thread::get_id()); \
						 MOBULA_KERNEL_MUTEX.unlock(); \
						 const int MOBULA_KERNEL_START = MOBULA_KERNEL_INFO.first; \
						 const int MOBULA_KERNEL_STEP = MOBULA_KERNEL_INFO.second; \
						 for (int i = MOBULA_KERNEL_START;i < (n);i += MOBULA_KERNEL_STEP)
#define KERNEL_RUN(a, n) (KernelRunner<decltype(&a)>(&a, (n)))

#else

// Single Thread Mode
#define KERNEL_LOOP(i,n) for (int i = 0;i < (n);++i)
#define KERNEL_RUN(a, n) a

#endif

}

#endif
