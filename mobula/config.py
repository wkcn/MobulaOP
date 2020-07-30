from .utils import with_metaclass


class DefaultConfig:
    TARGET = 'mobula_op'
    BUILD_PATH = 'build'
    MAX_BUILDING_WORKER_NUM = 8

    DEBUG = False
    USING_OPENMP = True
    USING_CBLAS = False
    HOST_NUM_THREADS = 0  # 0 : auto
    USING_HIGH_LEVEL_WARNINGS = False
    USING_OPTIMIZATION = True
    USING_ASYNC_EXEC = True
    GPU_BACKEND = 'cuda'

    CXX = 'g++'
    NVCC = 'nvcc'
    HIPCC = 'hipcc'
    CUDA_DIR = '/opt/cuda'
    HIP_DIR = '/opt/rocm/hip'


class Config:
    def __init__(self):
        for name in dir(DefaultConfig):
            if not name.startswith('_'):
                self.__dict__[name] = getattr(DefaultConfig, name)

    def __setattr__(self, name, value):
        data = self.__dict__.get(name, None)
        if data is None:
            raise AttributeError("Config has no attribute '{}'".format(name))
        target_type = type(data)
        value_type = type(value)
        if target_type is not value_type:
            raise TypeError('The type of config attribute `{}` is not consistent, target {} vs value {}.'.format(
                name, target_type, value_type))
        self.__dict__[name] = value


config = Config()
