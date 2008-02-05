#ifndef VTKCUDAOPENGLBUFFEROBJECT_H_
#define VTKCUDAOPENGLBUFFEROBJECT_H_

#include "vtkObject.h"

#include <GL/gl.h> // OpenGL headers used for the Buffer Reference
#include "CudappSupportModule.h"

class VTK_CUDASUPPORT_EXPORT CudappOpenGLBufferObject : public vtkObject
{
  vtkTypeRevisionMacro(CudappOpenGLBufferObject, vtkObject);
public:
  static CudappOpenGLBufferObject* New();
  
  void Register(GLuint bufferObject);
  void Unregister();
  void* Map();
  void Unmap();
  
  GLuint GetBufferObject() const { return this->BufferObject; }
  void* GetDevPointer() const { return this->DevPointer; } 

  virtual void PrintSelf(ostream &os, vtkIndent indent);
protected:
  CudappOpenGLBufferObject();
  virtual ~CudappOpenGLBufferObject();
  
  GLuint BufferObject; //!< The BufferObject
  void* DevPointer;   //!< Pointer to the Data in this Bufferobject
};

#endif /*VTKCUDAOPENGLBUFFEROBJECT_H_*/
