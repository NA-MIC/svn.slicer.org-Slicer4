#include "CudappOpenGLBufferObject.h"

#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

#include "CudappBase.h"

CudappOpenGLBufferObject::CudappOpenGLBufferObject()
{
  this->BufferObject = 0;
  this->DevPointer = NULL;
}

CudappOpenGLBufferObject::~CudappOpenGLBufferObject()
{
  this->Unmap();
  this->Unregister();
}

void CudappOpenGLBufferObject::Register(GLuint bufferObject)
{
  cudaError_t error = 
    cudaGLRegisterBufferObject(bufferObject);
  if (error != cudaSuccess)
        CudappBase::PrintError(error);
  else    
    this->BufferObject = bufferObject;
}

void CudappOpenGLBufferObject::Unregister()
{
  cudaError_t error = 
    cudaGLUnregisterBufferObject(this->BufferObject);
  if (error != cudaSuccess)
        CudappBase::PrintError(error);
}

/**
 * @brief maps a GLBufferObject to a memory space. 
 * @returns a pointer to the mapped area
 * 
 * @note Any prior mappings of this Bufferobject will be removed.
 * If the BufferObject has not yet been registered NULL will be returned.
 */
void* CudappOpenGLBufferObject::Map()
{
  this->Unmap();
  if (this->BufferObject == 0)
    return NULL;
  cudaError_t error = cudaGLMapBufferObject(&this->DevPointer, this->BufferObject);
  if (error != cudaSuccess) 
    CudappBase::PrintError(error);
  return this->DevPointer;
}

/**
 * @brief unmaps a BufferObject's memory point. sets DevPointer to NULL.
 */
void CudappOpenGLBufferObject::Unmap()
{
  if (this->DevPointer != NULL)
  {
    cudaError_t error = 
      cudaGLUnregisterBufferObject(this->BufferObject);
      if (error != cudaSuccess)
        CudappBase::PrintError(error);
        
    this->DevPointer = NULL;
  }
}

void CudappOpenGLBufferObject::PrintSelf(ostream &os)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->GetBufferObject() == 0)
    os << "Not yet initialized Buffer Object";
}

