class Config:
    TARGET = 'mobula_op'
    BUILD_PATH = 'build'
    MAX_BUILDING_WORKER_NUM = 8

    DEBUG = False
    USING_OPENMP = False
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
