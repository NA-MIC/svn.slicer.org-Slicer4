#ifndef CUDAPPOPENGLBUFFEROBJECT_H_
#define CUDAPPOPENGLBUFFEROBJECT_H_

#include "vtkObject.h"

#include <GL/gl.h> // OpenGL headers used for the Buffer Reference
#include "CudappSupportModule.h"

class CUDA_SUPPORT_EXPORT CudappOpenGLBufferObject : public vtkObject
{
public:
  CudappOpenGLBufferObject();
  virtual ~CudappOpenGLBufferObject();
  
  void Register(GLuint bufferObject);
  void Unregister();
  void* Map();
  void Unmap();
  
  GLuint GetBufferObject() const { return this->BufferObject; }
  void* GetDevPointer() const { return this->DevPointer; } 

  virtual void PrintSelf(std::ostream &os);
protected:
  
  GLuint BufferObject; //!< The BufferObject
  void* DevPointer;   //!< Pointer to the Data in this Bufferobject
};

#endif /*CUDAPPOPENGLBUFFEROBJECT_H_*/
