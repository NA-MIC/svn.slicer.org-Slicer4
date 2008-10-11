#include "ntkCudaDeviceMemory.h"
#include <stdlib.h>

ntkCudaDeviceMemory::ntkCudaDeviceMemory(){
  m_size=0;
  CUDA_SAFE_CALL( cudaMalloc((void**)&m_d_buffer, 0));
}

ntkCudaDeviceMemory::~ntkCudaDeviceMemory(){
  CUDA_SAFE_CALL(cudaFree(m_d_buffer));
}

void ntkCudaDeviceMemory::AllocateBytes(int size){
  if(size!=m_size){
    CUDA_SAFE_CALL( cudaFree(m_d_buffer));
    CUDA_SAFE_CALL( cudaMalloc((void**)&m_d_buffer, size));
    m_size=size;
  }
}

void ntkCudaDeviceMemory::copyFromHost(void * buffer){
  CUDA_SAFE_CALL (cudaMemcpy(m_d_buffer, buffer, m_size, cudaMemcpyHostToDevice));
}

void ntkCudaDeviceMemory::copyToHost(void * buffer){
  CUDA_SAFE_CALL (cudaMemcpy(buffer, m_d_buffer, m_size, cudaMemcpyDeviceToHost));
}

void *ntkCudaDeviceMemory::getDeviceBuffer(){
  return m_d_buffer;
}

void ntkCudaDeviceMemory::setDeviceBuffer(void* buffer){
  m_d_buffer=buffer;
}

int ntkCudaDeviceMemory::getSize(){
  return m_size;
}
