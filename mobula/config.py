from .utils import with_metaclass


class ConfigMeta(type):
    def __setattr__(cls, name, value):
        cdict = super(ConfigMeta, cls).__dict__
        if name not in cdict:
            raise AttributeError('Config has no attribute `{}`.'.format(name))
        data = cdict[name]
        target_type = type(data)
        value_type = type(value)
        if target_type is not value_type:
            raise TypeError('The type of config attribute `{}` is not consistent, target {} vs value {}.'.format(
                name, target_type, value_type))
        super(ConfigMeta, cls).__setattr__(name, value)


class Config(with_metaclass(ConfigMeta)):
    TARGET = 'mobula_op'
    BUILD_PATH = 'build'
    MAX_BUILDING_WORKER_NUM = 8

    DEBUG = False
    USING_OPENMP = True
    USING_CBLAS = False
    HOST_NUM_THREADS = 0  # 0 : auto
    USING_HIGH_LEVEL_WARNINGS = True
    USING_OPTIMIZATION = True
    USING_ASYNC_EXEC = True
    GPU_BACKEND = 'cuda'

    CXX = 'g++'
    NVCC = 'nvcc'
    HIPCC = 'hipcc'
    CUDA_DIR = '/opt/cuda'
    HIP_DIR = '/opt/rocm/hip'


# alias
config = Config
