#ifndef VTKCUDABASE_H_
#define VTKCUDABASE_H_

#include "vtkObject.h"
#include "driver_types.h"
#include "vtkCudaSupportModule.h"

class VTK_CUDASUPPORT_EXPORT vtkCudaBase : public vtkObject
{
  vtkTypeRevisionMacro(vtkCudaBase, vtkObject);
  public:
  //BTX
  typedef enum {
    Success,
    NotReadyError,
    InvalidValueError,
  } State;
  //ETX
  
  
   static vtkCudaBase* New();
  
   static cudaError_t GetLastError();
   static const char* GetLastErrorString();
   static const char* GetErrorString(cudaError_t error);
   static void PrintError(cudaError_t error);
  
private:
  virtual ~vtkCudaBase();
  vtkCudaBase();
};

#endif /*VTKCUDABASE_H_*/
