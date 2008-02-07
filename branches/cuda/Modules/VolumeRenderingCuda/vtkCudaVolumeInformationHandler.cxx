#include "vtkCudaVolumeInformationHandler.h"

#include "vtkObjectFactory.h"

//Volume and Property
#include "vtkVolumeProperty.h"
#include "vtkVolume.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkImageData.h"

//CUDA
#include "vector_types.h"

vtkCxxRevisionMacro(vtkCudaVolumeInformationHandler, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaVolumeInformationHandler);

vtkCudaVolumeInformationHandler::vtkCudaVolumeInformationHandler()
{
    this->VolumeInfo.FunctionSize = 0;
    this->ResizeTransferFunction(256);

    this->Volume = NULL;
    this->InputData = NULL;

    this->SetThreshold(90, 255);
}

vtkCudaVolumeInformationHandler::~vtkCudaVolumeInformationHandler()
{
    this->SetVolume(NULL);
    this->SetInputData(NULL);
}

void vtkCudaVolumeInformationHandler::ResizeTransferFunction(unsigned int size)
{
    if (size != this->VolumeInfo.FunctionSize)
    {
        this->VolumeInfo.FunctionSize = size;

        this->LocalAlphaTransferFunction.Allocate<float>(this->VolumeInfo.FunctionSize);
        this->CudaAlphaTransferFunction.Allocate<float>(this->VolumeInfo.FunctionSize);
        this->LocalColorTransferFunction.Allocate<float3>(this->VolumeInfo.FunctionSize);
        this->CudaColorTransferFunction.Allocate<float3>(this->VolumeInfo.FunctionSize);

        this->Modified();
    }
}

void vtkCudaVolumeInformationHandler::SetInputData(vtkImageData* inputData)
{
    if (inputData == NULL)
    {
        this->CudaInputBuffer.Free();
    }
    else if (inputData != this->InputData)
    {
        this->CudaInputBuffer.AllocateBytes(inputData->GetActualMemorySize() * 1024);
        // We do this automatically
        this->CudaInputBuffer.CopyFrom(inputData->GetScalarPointer(), inputData->GetActualMemorySize() * 1024);
    }
    this->InputData = inputData;
}

/**
 * @brief sets the threshold to min and max.
 */
void vtkCudaVolumeInformationHandler::SetThreshold(unsigned int min, unsigned int max)
{
    this->VolumeInfo.MinThreshold = min;
    this->VolumeInfo.MaxThreshold = max;
}

/**
 * @brief Updates the transfer functions on local and global memory.
 * @param property: The property that holds the transfer function information.
 */
void vtkCudaVolumeInformationHandler::UpdateVolumeProperties(vtkVolumeProperty *property)
{
    //FILE *fp;
    //  unsigned char transferFunction[256*6];
    //
    //fp=fopen("C:\\color.map","r");
    //  fread(transferFunction, sizeof(unsigned char), 256*6, fp);
    //  fclose(fp);
    //
    //  float colorTransferFunction[256*3];
    //  float alphaTransferFunction[256];
    //  float zBuffer[1024*768];
    //
    //  int i;
    //  /*
    //  for(i=0;i<256;i++){
    //    colorTransferFunction[i*3]=i/255.0;
    //    colorTransferFunction[i*3+1]=0.7;
    //    colorTransferFunction[i*3+2]=(255-i)/255.0;
    //    alphaTransferFunction[i]=0.1;
    //  }
    //  */
    //
    //  for(i=0;i<256;i++){
    //    colorTransferFunction[i*3]=transferFunction[i*3]/255.0;
    //    colorTransferFunction[i*3+1]=transferFunction[i*3+1]/255.0;
    //    colorTransferFunction[i*3+2]=transferFunction[i*3+2]/255.0;
    //    alphaTransferFunction[i]=transferFunction[i+256*3]/255.0;
    //  }
    //  this->CudaColorTransferFunction->CopyFrom(colorTransferFunction, 256*3*sizeof(float));
    //  this->CudaAlphaTransferFunction->CopyFrom(alphaTransferFunction, 256 * sizeof(float));
    //

    double range[2];
    property->GetRGBTransferFunction()->GetRange(range);
    property->GetRGBTransferFunction()->GetTable(range[0], range[1], 256, this->LocalColorTransferFunction.GetMemPointerAs<float>());

    this->LocalColorTransferFunction.CopyTo(&this->CudaColorTransferFunction);

    property->GetScalarOpacity()->GetTable(range[0], range[1], 256, this->LocalAlphaTransferFunction.GetMemPointerAs<float>());
    LocalAlphaTransferFunction.CopyTo(&CudaAlphaTransferFunction);

    this->VolumeInfo.AlphaTransferFunction = this->CudaAlphaTransferFunction.GetMemPointerAs<float>();
    this->VolumeInfo.ColorTransferFunction = this->CudaColorTransferFunction.GetMemPointerAs<float>();
}

/**
 * @brief Updates the volume information that is being sent to the Cuda Card.
 */
void vtkCudaVolumeInformationHandler::Update()
{
    if (this->Volume != NULL && this->InputData != NULL)
    {
        this->UpdateVolumeProperties(this->Volume->GetProperty());
        int* dims = this->InputData->GetDimensions();

        this->VolumeInfo.SourceData = this->CudaInputBuffer.GetMemPointer();
        this->VolumeInfo.InputDataType = this->InputData->GetScalarType();

        this->VolumeInfo.VoxelSize[0] = 1;
        this->VolumeInfo.VoxelSize[1] = 1;
        this->VolumeInfo.VoxelSize[2] = 1;

        this->VolumeInfo.VolumeTransformation[0] = 0.0f;
        this->VolumeInfo.VolumeTransformation[1] = 0.0f;
        this->VolumeInfo.VolumeTransformation[2] = 0.0f;
        
        this->VolumeInfo.VolumeSize[0] = dims[0];
        this->VolumeInfo.VolumeSize[1] = dims[1];
        this->VolumeInfo.VolumeSize[2] = dims[2];
        
        this->VolumeInfo.SteppingSize = 1.0;// nothing yet!!

        int* extent = InputData->GetExtent();
        this->VolumeInfo.MinMaxValue[0] = this->VolumeInfo.MinValueX = (float)extent[0];
        this->VolumeInfo.MinMaxValue[1] = this->VolumeInfo.MaxValueX = (float)extent[1];
        this->VolumeInfo.MinMaxValue[2] = this->VolumeInfo.MinValueY = (float)extent[2];
        this->VolumeInfo.MinMaxValue[3] = this->VolumeInfo.MaxValueY = (float)extent[3];
        this->VolumeInfo.MinMaxValue[4] = this->VolumeInfo.MinValueZ = (float)extent[4];
        this->VolumeInfo.MinMaxValue[5] = this->VolumeInfo.MaxValueZ = (float)extent[5];
    }
}


void vtkCudaVolumeInformationHandler::PrintSelf(std::ostream& os, vtkIndent indent)
{

}

