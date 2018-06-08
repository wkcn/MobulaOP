import os
import sys
import functools
import yaml
from easydict import EasyDict as edict
import multiprocessing
import multiprocessing.queues
import hashlib

def save_code_hash(obj, fname):
    with open(fname, 'w') as f:
        for k, v in obj.items():
            f.write('%s %s\n' % (k, v))

def load_code_hash(fname):
    data = dict()
    try:
        with open(fname, 'r') as f:
            for line in f:
                sp = line.split(' ')
                data[sp[0]] = sp[1].strip()
    except:
        pass
    return data

class Flags:
    def __init__(self, s = ''):
        self.flags = s
    def add_definition(self, key, value):
        if isinstance(value, bool):
            value = int(value)
        self.flags += ' -D %s=%s' % (key, str(value))
        return self
    def add_string(self, s):
        self.flags += ' %s' % str(s)
        return self
    def __str__(self):
        return self.flags

def wildcard(path, ext):
    if isinstance(path, list):
        res = []
        for p in path:
            res.extend(wildcard(p, ext))
        return res
    res = []
    for name in os.listdir(path):
        e = os.path.splitext(name)[1]
        if e == '.' + ext:
            res.append(os.path.join(path, name))
    return res

def change_ext(lst, origin, target):
    res = []
    for name in lst:
        sp = os.path.splitext(name)
        if sp[1] == '.' + origin:
            res.append(sp[0] + '.' + target)
        else:
            res.append(name)
    return res

def run_command(command):
    print (command)
    os.system(command)

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        print ('mkdir -p %s' % dir_name)
        os.makedirs(dir_name)

def rmdir(dir_name):
    if os.path.exists(dir_name):
        command = 'rm -rf %s' % dir_name
        run_command(command)

with open('./config.yaml') as fin:
    config = edict(yaml.load(fin))

num_cpu_core = multiprocessing.cpu_count()
host_num_threads = config.HOST_NUM_THREADS if config.HOST_NUM_THREADS > 0 else num_cpu_core
COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', host_num_threads)
if config.USING_OPTIMIZATION:
    COMMON_FLAGS.add_string('-O3')

CFLAGS = Flags('-std=c++11 -Iinc -fPIC').add_definition('USING_CUDA', 0).add_definition('USING_OPENMP', config.USING_OPENMP).add_string(COMMON_FLAGS)
LDFLAGS = Flags('-lpthread -shared')

CU_FLAGS = Flags('-std=c++11 -Iinc -Wno-deprecated-gpu-targets -dc --compiler-options "-fPIC"').add_definition('USING_CUDA', 1).add_string(COMMON_FLAGS)
CU_LDFLAGS = Flags('-lpthread -shared -Wno-deprecated-gpu-targets -L%s/lib64 -lcuda -lcudart -lcublas' % config.CUDA_DIR)

if config.USING_OPENMP:
    CFLAGS.add_string('-fopenmp')
    LDFLAGS.add_string('-fopenmp')

if config.USING_HIGH_LEVEL_WARNINGS:
    CFLAGS.add_string('-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wundef -fdiagnostics-show-option')

SRCS = wildcard(['src', 'src/operators'], 'cpp')
OBJS = change_ext(SRCS, 'cpp', 'o')

CU_SRCS = change_ext(SRCS, 'cpp', 'cu')
CU_OBJS = change_ext(CU_SRCS, 'cu', 'cu.o')

code_hash = dict()
code_hash_filename = os.path.join(config.BUILD_PATH, 'code.hash')
if os.path.exists(code_hash_filename):
    code_hash = load_code_hash(code_hash_filename)
code_hash_updated = False

def get_file_hash(fname):
    return str(os.path.getmtime(fname))
    m = hashlib.md5()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            m.update(data)
    return m.hexdigest()[:8]

def file_changed(fname):
    global code_hash_updated
    new_hash = get_file_hash(fname)
    if fname not in code_hash or new_hash != code_hash[fname]:
        code_hash_updated = True
        code_hash[fname] = new_hash
        return True
    return False

def file_is_latest(source, target):
    if file_changed(source):
        return False
    return os.path.exists(target)

class ProcessQueue(multiprocessing.queues.Queue):
    def __init__(self):
        if sys.version_info[0] <= 2:
            super(ProcessQueue, self).__init__()
        else:
            super(ProcessQueue, self).__init__(ctx = multiprocessing.get_context())

def run_command_parallel(commands):
    command_queue = ProcessQueue()
    for c in commands:
        command_queue.put(c)
    max_worker_num = min(config.MAX_BUILDING_WORKER_NUM, len(commands))
    def worker(command_queue):
        while not command_queue.empty():
            e = command_queue.get()
            if e is None:
                break
            run_command(e)
    workers = [multiprocessing.Process(target = worker, args = (command_queue,)) for _ in range(max_worker_num)]
    for w in workers:
        w.daemon = True
    for w in workers:
        w.start()
    for w in workers:
        w.join()

def source_to_o(build_path, it, compiler = config.CXX, cflags = CFLAGS):
    mkdir(build_path)
    existed_dirs = set()
    updated = False
    commands = []
    for src, obj in it:
        dir_name = os.path.dirname(obj)
        build_dir_name = os.path.join(build_path, dir_name)
        build_name = os.path.join(build_path, obj)
        if file_is_latest(src, build_name):
            continue
        updated = True
        if build_dir_name not in existed_dirs:
            mkdir(build_dir_name)
            existed_dirs.add(build_dir_name)
        command = '%s %s %s -c -o %s' % (compiler, src, cflags, build_name) 
        commands.append(command)
    run_command_parallel(commands)
    return updated

def o_to_so(target_name, objs, linker, ldflags = LDFLAGS):
    command = "%s %s %s -o %s" % (linker, ' '.join(objs), ldflags, target_name)
    run_command(command)

def link(srcs, tars):
    existed_dirs = set()
    for src, tar in zip(srcs, tars):
        dir_name = os.path.dirname(tar)
        if dir_name not in existed_dirs:
            mkdir(dir_name)
            existed_dirs.add(dir_name)
        run_command('ln -f %s %s' % (src, tar))

def add_path(path, files):
    return list(map(lambda x : os.path.join(path, x), files))

def all_func():
    build_path = os.path.join(config.BUILD_PATH, 'cpu')
    target_name = os.path.join(config.BUILD_PATH, '%s_cpu.so' % config.TARGET)
    if source_to_o(build_path, zip(SRCS, OBJS)) or not os.path.exists(target_name):
        objs = add_path(build_path, OBJS)
        o_to_so(target_name, objs, config.CXX)

def cuda_func():
    build_path = os.path.join(config.BUILD_PATH, 'gpu')
    cu_srcs = add_path(build_path, CU_SRCS)
    link(SRCS, cu_srcs)
    target_name = os.path.join(config.BUILD_PATH, '%s_gpu.so' % config.TARGET)
    if source_to_o(build_path, zip(cu_srcs, CU_OBJS), config.NVCC, CU_FLAGS) or not os.path.exists(target_name):
        objs = add_path(build_path, CU_OBJS)
        o_to_so(target_name, objs, config.NVCC, CU_LDFLAGS)

def clean_func():
    rmdir(config.BUILD_PATH)

RULES = dict(
    all = all_func,
    cuda = cuda_func,
    clean = clean_func,
)
def run_rule(name):
    RULES[name]()

if __name__ == '__main__':
    run_rule(sys.argv[1])
    if code_hash_updated:
        save_code_hash(code_hash, code_hash_filename)
