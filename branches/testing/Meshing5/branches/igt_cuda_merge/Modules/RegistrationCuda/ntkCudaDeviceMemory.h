#ifndef NTKCUDADEVICEMEMORY_H
#define NTKCUDADEVICEMEMORY_H

#include <stdio.h>

#include "cuda_runtime_api.h"
#include <cutil.h>

class ntkCudaDeviceMemory{
 protected:
  __device__ void* m_d_buffer;
  int m_size;
 public:
  ntkCudaDeviceMemory();
  ~ntkCudaDeviceMemory();
  
  void AllocateBytes(int size);
  template<typename T> void Allocate(int count){
    this->AllocateBytes(count* sizeof(T));
  }

  void copyFromHost(void * buffer);
  void copyToHost(void * buffer);
  void* getDeviceBuffer(void);
  void setDeviceBuffer(void *buffer);
  int getSize(void);
};

#endif
