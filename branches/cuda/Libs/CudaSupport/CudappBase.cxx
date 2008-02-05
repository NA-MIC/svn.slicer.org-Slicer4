#include "CudappBase.h"

#include "cuda_runtime_api.h"
#include "vtkSetGet.h"

vtkCxxRevisionMacro(CudappBase, "$Revision: 1.0 $");

cudaError_t CudappBase::GetLastError()
{
    return cudaGetLastError();
}

const char* CudappBase::GetLastErrorString()
{
    return CudappBase::GetErrorString(CudappBase::GetLastError());
}

const char* CudappBase::GetErrorString(cudaError_t error)
{
    return cudaGetErrorString(error);
}

void CudappBase::PrintError(cudaError_t error)
{
    printf(CudappBase::GetErrorString(error));
}

CudappBase* CudappBase::New() 
{
    return NULL;
}

CudappBase::~CudappBase()
{
}

CudappBase::CudappBase()
{
}
