#ifndef VTKCUDAOPENGLBUFFEROBJECT_H_
#define VTKCUDAOPENGLBUFFEROBJECT_H_

#include <GL/gl.h>

class vtkCudaOpenGLBufferObject
{
  public:
  static vtkCudaOpenGLBufferObject* New();
  
  void Register(GLuint bufferObject);
  void Unregister();
  void* Map();
  void Unmap();
  
  GLuint GetBufferObject() const { return this->BufferObject; }
  void* GetDevPointer() const { return this->DevPointer; } 
  
/// TODO make protected, public for now  
  virtual ~vtkCudaOpenGLBufferObject();
protected:
  vtkCudaOpenGLBufferObject();
  
  GLuint BufferObject; //!< The BufferObject
  void* DevPointer;   //!< Pointer to the Data in this Bufferobject
};

#endif /*VTKCUDAOPENGLBUFFEROBJECT_H_*/
