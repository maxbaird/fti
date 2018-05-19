#ifndef _API_CUDA_H
#define _API_CUDA_H

int copy_from_device(void* dst, void* src, long count);
int copy_to_device(void* dst, void* src, long count);
int host_accessible_pointer(const void *ptr);

#endif
