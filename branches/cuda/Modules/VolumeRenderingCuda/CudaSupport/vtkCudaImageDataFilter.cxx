#include "vtkCudaImageDataFilter.h"
#include "vtkCudaImageData.h"

#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"


vtkCxxRevisionMacro(vtkCudaImageDataFilter, "$Revision: 1.0 $");

vtkCudaImageDataFilter* vtkCudaImageDataFilter::New()
{
  return new vtkCudaImageDataFilter();  
}

vtkCudaImageDataFilter::vtkCudaImageDataFilter()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);

  // by default process active point scalars
  this->SetInputArrayToProcess(0,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS,
                               vtkDataSetAttributes::SCALARS);
}

vtkCudaImageDataFilter::~vtkCudaImageDataFilter()
{
}


//----------------------------------------------------------------------------
void vtkCudaImageDataFilter::PrintSelf(ostream& os, vtkIndent indent)
{  
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
// This is the superclasses style of Execute method.  Convert it into
// an imaging style Execute method.
int vtkCudaImageDataFilter::RequestData(
  vtkInformation* request,
  vtkInformationVector** vtkNotUsed( inputVector ),
  vtkInformationVector* outputVector)
{
  // the default implimentation is to do what the old pipeline did find what
  // output is requesting the data, and pass that into ExecuteData

  // which output port did the request come from
  int outputPort = 
    request->Get(vtkDemandDrivenPipeline::FROM_OUTPUT_PORT());

  // if output port is negative then that means this filter is calling the
  // update directly, in that case just assume port 0
  if (outputPort == -1)
      {
      outputPort = 0;
      }
  
  // get the data object
  vtkInformation *outInfo = 
    outputVector->GetInformationObject(outputPort);
  // call ExecuteData
  if (outInfo)
    {
    this->ExecuteData( outInfo->Get(vtkDataObject::DATA_OBJECT()) );
    }
  else
    {
    this->ExecuteData(NULL);
    }
  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaImageDataFilter::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  // do nothing except copy scalar type info
  this->CopyInputArrayAttributesToOutput(request,inputVector,outputVector);
  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaImageDataFilter::AllocateOutputData(vtkCudaImageData *output, 
                                           int *uExtent)
{ 
  // set the extent to be the update extent
  //output->SetExtent(uExtent);
  //output->AllocateScalars();
}

//----------------------------------------------------------------------------
/*
vtkCudaImageData *vtkCudaImageDataFilter::AllocateOutputData(vtkDataObject *output)
{ 
  // set the extent to be the update extent
  vtkCudaImageData *out = vtkCudaImageData::SafeDownCast(output);

  if (out)
    {
    // this needs to be fixed -Ken
    vtkStreamingDemandDrivenPipeline *sddp = 
      vtkStreamingDemandDrivenPipeline::SafeDownCast(this->GetExecutive());
    if (sddp)
      {
      int extent[6];
      sddp->GetOutputInformation(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),extent);
      out->SetExtent(extent);
      }
    out->AllocateScalars();
    }
  return out;
}
*/
//----------------------------------------------------------------------------
vtkCudaImageData* vtkCudaImageDataFilter::GetOutput()
{
  return this->GetOutput(0);
}

//----------------------------------------------------------------------------
vtkCudaImageData* vtkCudaImageDataFilter::GetOutput(int port)
{
  return vtkCudaImageData::SafeDownCast(this->GetOutputDataObject(port));
}

//----------------------------------------------------------------------------
void vtkCudaImageDataFilter::SetOutput(vtkCudaImageData* d)
{
  this->GetExecutive()->SetOutputData(0, d);
}

//----------------------------------------------------------------------------
int vtkCudaImageDataFilter::FillOutputPortInformation(
  int vtkNotUsed(port), vtkInformation* info)
{
  // now add our info
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkCudaImageData");
  
  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaImageDataFilter::FillInputPortInformation(
  int vtkNotUsed(port), vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return 1;
}
