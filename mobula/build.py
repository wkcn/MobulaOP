"""Building Implementation"""
import os
import multiprocessing
try:
    from .build_utils import *
except Exception:
    from build_utils import *

NUM_CPU_CORE = multiprocessing.cpu_count()
HOST_NUM_THREADS = config.HOST_NUM_THREADS if config.HOST_NUM_THREADS > 0 else NUM_CPU_CORE
COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', HOST_NUM_THREADS)

if config.USING_OPTIMIZATION:
    COMMON_FLAGS.add_string('-O3')
if config.DEBUG:
    COMMON_FLAGS.add_string('-g')
COMMON_FLAGS.add_definition('USING_CBLAS', config.USING_CBLAS)
INC_PATHS.extend(['./inc', '../3rdparty/dlpack/include',
                  '../3rdparty/tvm_packed_func'])
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
-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --expt-extended-lambda').\
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
        dir_name, obj_name = os.path.split(obj)
        build_dir_name = os.path.join(build_path, dir_name)
        build_name = os.path.join(build_path, dir_name, obj_name)
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
            command = 'cl /EHsc /O2 %s %s -c %s -Fo%s' % (
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


def source_to_so_ctx(build_path, srcs, target_name, ctx_name):
    assert ctx_name in BUILD_FLAGS, ValueError(
        'Unsupported Context: {} -('.format(ctx_name))
    flags = BUILD_FLAGS[ctx_name]
    compiler, cflags, ldflags = flags[:3]

    buildin_path = os.path.join(ENV_PATH, config.BUILD_PATH, ctx_name)
    buildin_o = []
    buildin_cpp = []
    for src in ['defines.cpp', 'context.cpp']:
        fname = os.path.join('src', src)
        buildin_o.append(os.path.join(buildin_path, fname))
        buildin_cpp.append(os.path.join(ENV_PATH, fname))
    buildin_o = change_exts(buildin_o, [('cpp', 'o')])

    for fname in buildin_o:
        if not os.path.exists(fname):
            with build_context():
                source_to_o(build_path, zip(
                    buildin_cpp, buildin_o), compiler, cflags)
    flags += (buildin_o, )
    source_to_so(build_path, srcs, target_name, *flags)
