import os
import sys
import functools
import yaml
from easydict import EasyDict as edict

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

COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', config.HOST_NUM_THREADS)
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

def file_is_latest(source, target):
    return os.path.exists(target)

def source_to_o(build_path, it, compiler = config.CXX, cflags = CFLAGS):
    mkdir(build_path)
    existed_dirs = set()
    updated = False
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
        run_command(command)
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
    source_to_o(build_path, zip(SRCS, OBJS))
    objs = add_path(build_path, OBJS)
    target_name = os.path.join(config.BUILD_PATH, '%s_cpu.so' % config.TARGET)
    o_to_so(target_name, objs, config.CXX) 
def cuda_func():
    build_path = os.path.join(config.BUILD_PATH, 'gpu')
    cu_srcs = add_path(build_path, CU_SRCS)
    link(SRCS, cu_srcs)
    source_to_o(build_path, zip(cu_srcs, CU_OBJS), config.NVCC, CU_FLAGS)
    objs = add_path(build_path, CU_OBJS)
    target_name = os.path.join(config.BUILD_PATH, '%s_gpu.so' % config.TARGET)
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
