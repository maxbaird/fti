#ifndef _API_CUDA_H
#define _API_CUDA_H

#define CPU_POINTER 1
#define GPU_POINTER 0

int FTI_determine_pointer_type(const void *ptr);
int FTI_copy_from_device(void* dst, void* src, long count);
int FTI_copy_to_device(void* dst, void* src, long count);

#endif
