#include "vtkCudaImageData.h"

vtkCxxRevisionMacro(vtkCudaImageData, "$ Revision: 1.0 $");

vtkCudaImageData* vtkCudaImageData::New()
{
  return new vtkCudaImageData();
}


vtkCudaImageData::vtkCudaImageData()
{
  this->Data = NULL
}

vtkCudaImageData::~vtkCudaImageData()
{
  if (this->Data != NULL)
    this->Data->Delete();
}

void vtkCudaImageData::PrintSelf (ostream &os, vtkIndent indent)
{
    
}

