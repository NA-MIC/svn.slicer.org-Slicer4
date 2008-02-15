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
    this->SetSteppingSize(1.0);
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
    this->Modified();
}

/**
 * @brief sets the threshold to min and max.
 */
void vtkCudaVolumeInformationHandler::SetThreshold(unsigned int min, unsigned int max)
{
    this->VolumeInfo.MinThreshold = min;
    this->VolumeInfo.MaxThreshold = max;
    this->Modified();
}

void vtkCudaVolumeInformationHandler::SetSteppingSize(float steppingSize)
{ 
    if (steppingSize <= 0.0f)
        steppingSize = .1f;
    else
        this->VolumeInfo.SteppingSize = steppingSize; 
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
    //  fp=fopen("C:\\color.map","r");
    //  fread(transferFunction, sizeof(unsigned char), 256*6, fp);
    //  fclose(fp);

    //  int i;
    //  //for (i = 0; i < 256*6; i++)
    //  //    cout << transferFunction[i] << " "; 
    //  /*
    //  for(i=0;i<256;i++){
    //    colorTransferFunction[i*3]=i/255.0;
    //    colorTransferFunction[i*3+1]=0.7;
    //    colorTransferFunction[i*3+2]=(255-i)/255.0;
    //    alphaTransferFunction[i]=0.1;
    //  }
    //  */
    //  for(i = 0; i < 256; i++){
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3] = ((float)transferFunction[i*3])/255.0;
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+1] = ((float)transferFunction[i*3+1])/255.0;
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+2] = ((float)transferFunction[i*3+2])/255.0;
    //    this->LocalAlphaTransferFunction.GetMemPointerAs<float>()[i] = ((float)transferFunction[i+256*3])/255.0;


    //    cout << i << " " << this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3] << "x"  <<
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+1] << "x" << 
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+2] << std::endl;

    //  }
    //  this->VolumeInfo.FunctionSize = 12;

    //for (i = 0; i < 256 ; i++)
    //    cout << this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3] << "x"  <<
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+1] << "x" << 
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[i*3+2] << std::endl;

    double range[2];
    property->GetRGBTransferFunction()->GetRange(range);
    property->GetRGBTransferFunction()->GetTable(range[0], range[1], this->VolumeInfo.FunctionSize, this->LocalColorTransferFunction.GetMemPointerAs<float>());

    this->VolumeInfo.FunctionRange[0] = range[0];
    this->VolumeInfo.FunctionRange[1] = range[1];

    property->GetScalarOpacity()->GetTable(range[0], range[1], this->VolumeInfo.FunctionSize, this->LocalAlphaTransferFunction.GetMemPointerAs<float>());

    this->LocalColorTransferFunction.CopyTo(&this->CudaColorTransferFunction);
    this->LocalAlphaTransferFunction.CopyTo(&this->CudaAlphaTransferFunction);
    this->VolumeInfo.AlphaTransferFunction = this->CudaAlphaTransferFunction.GetMemPointerAs<float>();
    this->VolumeInfo.ColorTransferFunction = this->CudaColorTransferFunction.GetMemPointerAs<float>();
}


#include "vtkKWHistogram.h"
#include "vtkPointData.h"

/**
 * @brief Updates the volume information that is being sent to the Cuda Card.
 */
void vtkCudaVolumeInformationHandler::Update()
{
    if (this->Volume != NULL && this->InputData != NULL)
    {
        this->CudaInputBuffer.CopyFrom(this->InputData->GetScalarPointer(),
                                        this->InputData->GetActualMemorySize() * 1024);


        this->UpdateVolumeProperties(this->Volume->GetProperty());
        int* dims = this->InputData->GetDimensions();
        double* spacing = this->InputData->GetSpacing();

        this->VolumeInfo.SourceData = this->CudaInputBuffer.GetMemPointer();
        this->VolumeInfo.InputDataType = this->InputData->GetScalarType();

        this->VolumeInfo.Spacing.x = (float)spacing[0];
        this->VolumeInfo.Spacing.y = (float)spacing[1];
        this->VolumeInfo.Spacing.z = (float)spacing[2];

        this->VolumeInfo.VolumeTransformation.x= 0.0f;
        this->VolumeInfo.VolumeTransformation.y = 0.0f;
        this->VolumeInfo.VolumeTransformation.z = 0.0f;
        
        this->VolumeInfo.VolumeSize.x = dims[0];
        this->VolumeInfo.VolumeSize.y = dims[1];
        this->VolumeInfo.VolumeSize.z = dims[2];
        
        int* extent = InputData->GetExtent();
        this->VolumeInfo.MinMaxValue[0] = (float)extent[0];
        this->VolumeInfo.MinMaxValue[1] = (float)extent[1];
        this->VolumeInfo.MinMaxValue[2] = (float)extent[2];
        this->VolumeInfo.MinMaxValue[3] = (float)extent[3];
        this->VolumeInfo.MinMaxValue[4] = (float)extent[4];
        this->VolumeInfo.MinMaxValue[5] = (float)extent[5];
    }
}

void vtkCudaVolumeInformationHandler::PrintSelf(std::ostream& os, vtkIndent indent)
{

}

