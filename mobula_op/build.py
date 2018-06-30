from build_utils import *

num_cpu_core = multiprocessing.cpu_count()
host_num_threads = config.HOST_NUM_THREADS if config.HOST_NUM_THREADS > 0 else num_cpu_core
COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', host_num_threads)
if config.USING_OPTIMIZATION:
    COMMON_FLAGS.add_string('-O3')
COMMON_FLAGS.add_definition('USING_CBLAS', config.USING_CBLAS)

CFLAGS = Flags('-std=c++11 -Iinc -fPIC').add_definition('USING_CUDA', 0).add_definition('USING_OPENMP', config.USING_OPENMP).add_string(COMMON_FLAGS)
LDFLAGS = Flags('-lpthread -shared')
if config.USING_CBLAS:
    LDFLAGS.add_string('-lopenblas')

CU_FLAGS = Flags('-std=c++11 -Iinc -Wno-deprecated-gpu-targets -dc --compiler-options "-fPIC" --expt-extended-lambda').add_definition('USING_CUDA', 1).add_string(COMMON_FLAGS)
CU_LDFLAGS = Flags('-lpthread -shared -Wno-deprecated-gpu-targets -L%s/lib64 -lcuda -lcudart -lcublas' % config.CUDA_DIR)

if config.USING_OPENMP:
    CFLAGS.add_string('-fopenmp')
    LDFLAGS.add_string('-fopenmp')

if config.USING_HIGH_LEVEL_WARNINGS:
    CFLAGS.add_string('-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wundef -fdiagnostics-show-option')

SRCS = wildcard(['src', 'src/op'], 'cpp')
OBJS = change_ext(SRCS, 'cpp', 'o')

CU_SRCS = change_ext(SRCS, 'cpp', 'cu')
CU_OBJS = change_ext(CU_SRCS, 'cu', 'cu.o')

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
