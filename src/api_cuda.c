#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "api_cuda.h"
#include "interface.h"

#define CUDA_ERROR_CHECK(fun)                                                                   \
do{                                                                                             \
    cudaError_t err = fun;                                                                      \
    if(err != cudaSuccess)                                                                      \
    {                                                                                           \
      fprintf(stderr, "Cuda error %d %s:: %s\n", __LINE__, __func__, cudaGetErrorString(err));  \
      return FTI_NSCS;                                                                          \
    }                                                                                           \
}while(0);

int host_accessible_pointer(const void *ptr)
{
  int host_accessible = 1;
  struct cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, ptr);

  char str[FTI_BUFS];

  if(attributes.memoryType == cudaMemoryTypeDevice)
  {
    host_accessible = 0;
  }
  if(attributes.isManaged == 1)
  {
    host_accessible = 1;
  }

  sprintf(str, "Host accessible pointer found: %s\n", (host_accessible == 1) ? "true" : "false"); 
  FTI_Print(str, FTI_DBUG);

  return host_accessible;
}

int copy_from_device(void* dst, void* src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying data from GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
  return FTI_SCES;
}

int copy_to_device(void *dst, void *src, long count)
{
  char str[FTI_BUFS];
  sprintf(str, "Copying to GPU");
  FTI_Print(str, FTI_DBUG);
  CUDA_ERROR_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
  return FTI_SCES;
}
