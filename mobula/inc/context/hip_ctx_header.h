#ifndef MOBULA_INC_CONTEXT_HIP_CTX_HEADER_H_
#define MOBULA_INC_CONTEXT_HIP_CTX_HEADER_H_

#include <algorithm>
#include <iostream>

#include "./common.h"

#if USING_HIP

// HIP
#include <hip/hip_runtime.h>

#if USING_CBLAS
#include <hipblas.h>
#endif  // USING_CBLAS
#include <cuda_runtime.h>

#else  // USING_HIP

// CUDA
#if USING_CBLAS
#include <cublas_v2.h>
using hipblasHandle_t = cublasHandle_t;
#define hipblasCreate cublasCreate
#define hipblasSgemm cublasSgemm
#define HIPBLAS_OP_T CUBLAS_OP_T
#define HIPBLAS_OP_N CUBLAS_OP_N
#endif  // USING_CBLAS

using hipStream_t = cudaStream_t;
using hipError_t = cudaError_t;

// basic
#define hipGetLastError cudaGetLastError
#define hipGetErrorString cudaGetErrorString
#define hipSuccess cudaSuccess

// index
#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

// device
#define hipSetDevice cudaSetDevice
#define hipGetDevice cudaGetDevice

// memory
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

// stream
#define hipStreamCreate cudaStreamCreate
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamDestroy cudaStreamDestroy

#endif  // USING_HIP

#endif  // MOBULA_INC_CONTEXT_HIP_CTX_HEADER_H_
