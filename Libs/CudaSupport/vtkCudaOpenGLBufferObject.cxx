#include "vtkCudaOpenGLBufferObject.h"

#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

#include "vtkCudaBase.h"
#include "vtkObjectFactory.h"

vtkCxxRevisionMacro(vtkCudaOpenGLBufferObject, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaOpenGLBufferObject);

vtkCudaOpenGLBufferObject::vtkCudaOpenGLBufferObject()
{
  this->BufferObject = 0;
  this->DevPointer = NULL;
}

vtkCudaOpenGLBufferObject::~vtkCudaOpenGLBufferObject()
{
  this->Unmap();
  this->Unregister();
}

void vtkCudaOpenGLBufferObject::Register(GLuint bufferObject)
{
  cudaError_t error = 
    cudaGLRegisterBufferObject(bufferObject);
  if (error != cudaSuccess)
        vtkCudaBase::PrintError(error);
  else    
    this->BufferObject = bufferObject;
}

void vtkCudaOpenGLBufferObject::Unregister()
{
  cudaError_t error = 
    cudaGLUnregisterBufferObject(this->BufferObject);
  if (error != cudaSuccess)
        vtkCudaBase::PrintError(error);
}

/**
 * @brief maps a GLBufferObject to a memory space. 
 * @returns a pointer to the mapped area
 * 
 * @note Any prior mappings of this Bufferobject will be removed.
 * If the BufferObject has not yet been registered NULL will be returned.
 */
void* vtkCudaOpenGLBufferObject::Map()
{
  this->Unmap();
  if (this->BufferObject == 0)
    return NULL;
  cudaError_t error = cudaGLMapBufferObject(&this->DevPointer, this->BufferObject);
  if (error != cudaSuccess) 
    vtkCudaBase::PrintError(error);
  return this->DevPointer;
}

/**
 * @brief unmaps a BufferObject's memory point. sets DevPointer to NULL.
 */
void vtkCudaOpenGLBufferObject::Unmap()
{
  if (this->DevPointer != NULL)
  {
    cudaError_t error = 
      cudaGLUnregisterBufferObject(this->BufferObject);
      if (error != cudaSuccess)
        vtkCudaBase::PrintError(error);
        
    this->DevPointer = NULL;
  }
}

void vtkCudaOpenGLBufferObject::PrintSelf(ostream &os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  if (this->GetBufferObject() == 0)
    os << "Not yet initialized Buffer Object";
}

