"""Building Implementation"""
import sys
import multiprocessing
try:
    from .build_utils import *
except:
    from build_utils import *

NUM_CPU_CORE = multiprocessing.cpu_count()
HOST_NUM_THREADS = config.HOST_NUM_THREADS if config.HOST_NUM_THREADS > 0 else NUM_CPU_CORE
COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', HOST_NUM_THREADS)
if config.USING_OPTIMIZATION:
    COMMON_FLAGS.add_string('-O3')
COMMON_FLAGS.add_definition('USING_CBLAS', config.USING_CBLAS)
INC_PATHS.append('inc')
for path in INC_PATHS:
    p = os.path.join(ENV_PATH, path)
    if p:
        COMMON_FLAGS.add_string('-I{}'.format(p))

CFLAGS = Flags('-std=c++11').add_definition('USING_CUDA', 0).\
    add_definition('USING_HIP', 0).add_definition('USING_OPENMP', config.USING_OPENMP).\
    add_string(COMMON_FLAGS)
if not OS_IS_WINDOWS:
    CFLAGS.add_string('-fPIC')
LDFLAGS = Flags('-lpthread -shared')
if config.USING_CBLAS:
    LDFLAGS.add_string('-lopenblas')

CU_FLAGS = Flags('-std=c++11 -x cu -Wno-deprecated-gpu-targets -dc \
--expt-extended-lambda').\
    add_definition('USING_CUDA', 1).\
    add_definition('USING_HIP', 0).\
    add_string(COMMON_FLAGS)
if not OS_IS_WINDOWS:
    CU_FLAGS.add_string('--compiler-options "-fPIC"')
CU_LDFLAGS = Flags('-shared -Wno-deprecated-gpu-targets \
-L%s/lib64 -lcuda -lcudart' % config.CUDA_DIR)
if config.USING_CBLAS:
    CU_LDFLAGS.add_string('-lcublas')

HIP_FLAGS = Flags('-std=c++11 -Wno-deprecated-gpu-targets -Wno-deprecated-declarations -dc \
--expt-extended-lambda').\
    add_definition('USING_CUDA', 0).\
    add_definition('USING_HIP', 1).\
    add_string(COMMON_FLAGS)
if not OS_IS_WINDOWS:
    HIP_FLAGS.add_string('--compiler-options "-fPIC"')
HIP_LDFLAGS = Flags('-shared -Wno-deprecated-gpu-targets')
if config.USING_CBLAS:
    HIP_LDFLAGS.add_string('-lhipblas')

if config.USING_OPENMP:
    CFLAGS.add_string('-fopenmp')
    LDFLAGS.add_string('-fopenmp')

if config.USING_HIGH_LEVEL_WARNINGS:
    CFLAGS.add_string('-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual \
-Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs \
-Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow \
-Wsign-promo -Wundef -fdiagnostics-show-option')


def source_to_o(build_path, src_obj, compiler=config.CXX, cflags=CFLAGS):
    mkdir(build_path)
    existed_dirs = set()
    updated = False
    commands = []
    for src, obj in src_obj:
        dir_name = os.path.dirname(obj)
        build_dir_name = os.path.join(build_path, dir_name)
        build_name = os.path.join(build_path, obj)
        if file_is_latest(src) and os.path.exists(build_name):
            continue
        updated = True
        if build_dir_name not in existed_dirs:
            mkdir(build_dir_name)
            existed_dirs.add(build_dir_name)
        if OS_IS_WINDOWS and not command_exists(compiler):
            inc_flags = Flags()
            for path in INC_PATHS:
                p = os.path.join(ENV_PATH, path)
                inc_flags.add_string('-I{}'.format(p))
            cflags_sp = str(cflags).split()
            def_flags = ' '.join(
                [s for s in cflags_sp if len(s) > 2 and s[:2] == '-D'])
            command = 'cl /O2 %s %s -c %s -Fo%s' % (
                def_flags, inc_flags, src, build_name)
        else:
            command = '%s %s %s -c -o %s' % (compiler, src, cflags, build_name)
        commands.append(command)
    run_command_parallel(commands)
    return updated


def o_to_so(target_name, objs, linker, ldflags=LDFLAGS):
    if OS_IS_WINDOWS and not command_exists(linker):
        command = 'link -DLL %s -out:%s' % (' '.join(objs), target_name)
    else:
        command = '%s %s %s -o %s' % (linker,
                                      ' '.join(objs), ldflags, target_name)
    run_command(command)


def source_to_so(build_path, srcs, target_name, compiler, cflags, ldflags, buildin_o=None):
    objs = change_exts(srcs, [('cpp', 'o')])

    if source_to_o(build_path, zip(srcs, objs), compiler, cflags) or\
            not os.path.exists(target_name):
        if buildin_o is not None:
            objs.extend(buildin_o)
        abs_objs = add_path(build_path, objs)
        o_to_so(target_name, abs_objs, compiler, ldflags)


BUILD_FLAGS = dict(
    cpu=(config.CXX, CFLAGS, LDFLAGS),
    cuda=(config.NVCC, CU_FLAGS, CU_LDFLAGS),
    hip=(config.HIPCC, HIP_FLAGS, HIP_LDFLAGS)
)


def source_to_so_ctx(build_path, srcs, target_name, ctx_name, buildin_cpp=None):
    assert ctx_name in BUILD_FLAGS, ValueError(
        'Unsupported Context: {} -('.format(ctx_name))

    buildin_o = []
    if buildin_cpp is not None:
        buildin_path = os.path.join(ENV_PATH, config.BUILD_PATH, ctx_name)
        buildin_o = [os.path.join(buildin_path, fname) for fname in
                     change_exts(buildin_cpp, [('cpp', 'o')])]
        for fname in buildin_o:
            assert os.path.exists(fname),\
                Exception(
                    'File {} not found, please rebuild MobulaOP :-('.format(fname))

    flags = BUILD_FLAGS[ctx_name] + (buildin_o, )
    source_to_so(build_path, srcs, target_name, *flags)


def cpu_func():
    # cpu
    build_path = os.path.join(config.BUILD_PATH, 'cpu')
    target_name = os.path.join(config.BUILD_PATH, '%s_cpu.so' % config.TARGET)
    source_to_so_ctx(build_path, SRCS, target_name, 'cpu')


def cuda_func():
    build_path = os.path.join(config.BUILD_PATH, 'cuda')
    target_name = os.path.join(config.BUILD_PATH, '%s_cuda.so' % config.TARGET)
    source_to_so_ctx(build_path, SRCS, target_name, 'cuda')


def hip_func():
    build_path = os.path.join(config.BUILD_PATH, 'hip')
    target_name = os.path.join(config.BUILD_PATH, '%s_hip.so' % config.TARGET)
    source_to_so_ctx(build_path, SRCS, target_name, 'hip')


def clean_func():
    rmdir(config.BUILD_PATH)


def all_func():
    cpu_func()
    if command_exists(config.NVCC):
        cuda_func()
    elif command_exists(config.HIPCC):
        hip_func()


RULES = dict(
    all=all_func,
    cpu=cpu_func,
    cuda=cuda_func,
    hip=hip_func,
    clean=clean_func,
)


def run_rule(name):
    assert name in RULES, ValueError(
        "No rule to make target '{}'".format(name))
    RULES[name]()


if __name__ == '__main__':
    SRCS = wildcard(['src'], 'cpp')
    with build_context():
        run_rule(sys.argv[1])
