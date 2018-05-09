#include "defines.h"

namespace mobula{

map<thread::id, pair<int, int> > MOBULA_KERNEL_INFOS;
mutex MOBULA_KERNEL_MUTEX;

};
