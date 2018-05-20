#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "api_cuda.h"
#include "interface.h"

#define CUDA_ERROR_CHECK(fun)                                                           \
do{                                                                                     \
    cudaError_t err = fun;                                                              \
    char str[FTI_BUFS];                                                                 \
    sprintf(str, "Cuda error %d %s:: %s", __LINE__, __func__, cudaGetErrorString(err)); \
    if(err != cudaSuccess)                                                              \
    {                                                                                   \
      FTI_Print(str, FTI_EROR);                                                         \
      return FTI_NSCS;                                                                  \
    }                                                                                   \
}while(0);

int FTI_determine_pointer_type(const void *ptr, int *pointer_type)
{
  *pointer_type = CPU_POINTER;
  struct cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  if(err == cudaErrorInvalidDevice){
    return FTI_NSCS;
  }

  char str[FTI_BUFS];

  if(attributes.memoryType == cudaMemoryTypeDevice){
    *pointer_type = GPU_POINTER;
  }

  if(attributes.isManaged == 1){
    *pointer_type = CPU_POINTER;
  }

  sprintf(str, "Pointer type: %s", (*pointer_type == CPU_POINTER) ? "CPU Pointer" : "GPU Pointer"); 
  FTI_Print(str, FTI_DBUG);

  return FTI_SCES;
}

int FTI_copy_from_device(void* dst, void* src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying data from GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  return FTI_SCES;
}

int FTI_copy_to_device(void *dst, void *src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying data to GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
  return FTI_SCES;
}
