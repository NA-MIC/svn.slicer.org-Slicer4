#ifndef VTKCUDABASE_H_
#define VTKCUDABASE_H_

#include "driver_types.h"

class vtkCudaBase
{
  public:
   static cudaError_t GetLastError();
    static const char* GetLastErrorString();
      static const char* GetErrorString(cudaError_t error);
      static void PrintError(cudaError_t error);
  
private:
  virtual ~vtkCudaBase();
  vtkCudaBase();
};

#endif /*VTKCUDABASE_H_*/
