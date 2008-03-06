#include "vtkCudaVolumeInformationHandler.h"
#include "vtkObjectFactory.h"

//Volume and Property
#include "vtkVolumeProperty.h"
#include "vtkVolume.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"
#include "vtkImageData.h"
#include "vtkPointData.h"

//Nicholas
//#include "ntkColorTransferFunction.h"

vtkCxxRevisionMacro(vtkCudaVolumeInformationHandler, "$Revision: 1.0 $");
vtkStandardNewMacro(vtkCudaVolumeInformationHandler);

vtkCudaVolumeInformationHandler::vtkCudaVolumeInformationHandler()
{
    this->SetSampleDistance(1.0);
    this->VolumeInfo.FunctionSize = 0;
    this->ResizeTransferFunction(256);

    this->Volume = NULL;
    this->InputData = NULL;

    this->SetThreshold(0, 255);
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

void vtkCudaVolumeInformationHandler::SetVolume(vtkVolume* volume)
{
    this->Volume = volume;
    if (Volume != NULL)
        this->UpdateVolume();
    this->Modified();
}

void vtkCudaVolumeInformationHandler::SetInputData(vtkImageData* inputData)
{
    if (inputData == NULL)
    {
        this->CudaInputBuffer.Free();
        this->InputData = NULL;
    }
    else if (inputData != this->InputData)
    {
        this->InputData = inputData;

        double range[2];
        //    property->GetRGBTransferFunction()->GetRange(range);
        inputData->GetPointData()->GetScalars()->GetRange(range);
        this->VolumeInfo.FunctionRange[0] = range[0];
        this->VolumeInfo.FunctionRange[1] = range[1];
        //HACK
        this->VolumeInfo.MaxThreshold = range[1];

        this->UpdateImageData();
        this->Modified();
    }
}

/**
* @brief sets the threshold to min and max.
*/
void vtkCudaVolumeInformationHandler::SetThreshold(unsigned int min, unsigned int max)
{
    this->VolumeInfo.MinThreshold = min;
    this->VolumeInfo.MaxThreshold = max;
}

void vtkCudaVolumeInformationHandler::SetSampleDistance(float sampleDistance)
{ 
    if (sampleDistance <= 0.0f)
        sampleDistance = .1f;
    else
        this->VolumeInfo.SampleDistance = sampleDistance; 
}

/**
* @brief Updates the transfer functions on local and global memory.
* @param property: The property that holds the transfer function information.
*/
void vtkCudaVolumeInformationHandler::UpdateVolumeProperties(vtkVolumeProperty *property)
{
    //ntkColorTransferFunction colorMap;
    //colorMap.addKeyColorPoint(210, 255, 255, 255);
    //colorMap.addKeyColorPoint(120, 255, 124, 140);
    //colorMap.addKeyColorPoint(60, 100, 124, 240);

    //colorMap.addKeyAlphaPoint(60, 2);
    //colorMap.addKeyAlphaPoint(120, 4);
    //colorMap.addKeyAlphaPoint(210, 100);

    //for (unsigned int i = 0; i < 256; i++)
    //{
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[3*i] = colorMap.getColorBuffer()[3*i] /256.0;
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[3*i+1] = colorMap.getColorBuffer()[3*i+1] /256.0;
    //    this->LocalColorTransferFunction.GetMemPointerAs<float>()[3*i+2] = colorMap.getColorBuffer()[3*i+2]/256.0;
    //    this->LocalAlphaTransferFunction.GetMemPointerAs<float>()[i] = colorMap.getAlphaBuffer()[i]/256.0;
    //}

    property->GetRGBTransferFunction()->GetTable(this->VolumeInfo.FunctionRange[0], this->VolumeInfo.FunctionRange[1],
        this->VolumeInfo.FunctionSize, this->LocalColorTransferFunction.GetMemPointerAs<float>());


    property->GetScalarOpacity()->GetTable(this->VolumeInfo.FunctionRange[0], this->VolumeInfo.FunctionRange[1], 
        this->VolumeInfo.FunctionSize, this->LocalAlphaTransferFunction.GetMemPointerAs<float>());

    this->LocalColorTransferFunction.CopyTo(&this->CudaColorTransferFunction);
    this->LocalAlphaTransferFunction.CopyTo(&this->CudaAlphaTransferFunction);
    this->VolumeInfo.AlphaTransferFunction = this->CudaAlphaTransferFunction.GetMemPointerAs<float>();
    this->VolumeInfo.ColorTransferFunction = this->CudaColorTransferFunction.GetMemPointerAs<float>();

    this->VolumeInfo.Ambient = property->GetAmbient();
    this->VolumeInfo.Diffuse = property->GetDiffuse();
    this->VolumeInfo.Specular = property->GetSpecular();
    this->VolumeInfo.SpecularPower = property->GetSpecularPower();
    
}

#include "vtkMatrix4x4.h"
void vtkCudaVolumeInformationHandler::UpdateVolume()
{
  //  if (this->Volume->GetProperty()->GetMTime() > this->GetMTime())
        this->UpdateVolumeProperties(this->Volume->GetProperty());


    this->VolumeInfo.VolumeTransformation.x= 0.0f;
    this->VolumeInfo.VolumeTransformation.y = 0.0f;
    this->VolumeInfo.VolumeTransformation.z = 0.0f;


    // HACK EREI
    vtkMatrix4x4* mat = vtkMatrix4x4::New();
    if (this->Volume->GetUserMatrix() != NULL)
        mat->DeepCopy(this->Volume->GetUserMatrix());
    else
        mat->Identity();

    mat->Invert();

    for (unsigned int i = 0; i < 4 ; i++)
        for (unsigned int j = 0; j < 4; j++)
            this->VolumeInfo.Transform[i][j] = mat->GetElement(i,j);
    mat->Delete();
}

void vtkCudaVolumeInformationHandler::UpdateImageData()
{
    int* dims = this->InputData->GetDimensions();
    double* spacing = this->InputData->GetSpacing();
    int* extent = this->InputData->GetExtent();
    this->VolumeInfo.MinMaxValue[0] = (float)extent[0];
    this->VolumeInfo.MinMaxValue[1] = (float)extent[1];
    this->VolumeInfo.MinMaxValue[2] = (float)extent[2];
    this->VolumeInfo.MinMaxValue[3] = (float)extent[3];
    this->VolumeInfo.MinMaxValue[4] = (float)extent[4];
    this->VolumeInfo.MinMaxValue[5] = (float)extent[5];

    this->VolumeInfo.Spacing.x = (float)spacing[0];
    this->VolumeInfo.Spacing.y = (float)spacing[1];
    this->VolumeInfo.Spacing.z = (float)spacing[2];
    this->VolumeInfo.VolumeSize.x = dims[0];
    this->VolumeInfo.VolumeSize.y = dims[1];
    this->VolumeInfo.VolumeSize.z = dims[2];

    // needs precalculated data from above
    unsigned long size = this->InputData->GetScalarSize() * 
        this->VolumeInfo.VolumeSize.x *
        this->VolumeInfo.VolumeSize.y *
        this->VolumeInfo.VolumeSize.z *
        this->InputData->GetNumberOfScalarComponents();

    if (size != this->CudaInputBuffer.GetSize())
        this->CudaInputBuffer.AllocateBytes(size);

    this->CudaInputBuffer.CopyFrom(this->InputData->GetScalarPointer(),
        size);

    this->VolumeInfo.SourceData = this->CudaInputBuffer.GetMemPointer();
    this->VolumeInfo.InputDataType = this->InputData->GetScalarType();
}

/**
* @brief Updates the volume information that is being sent to the Cuda Card.
*/
void vtkCudaVolumeInformationHandler::Update()
{
    if (this->Volume != NULL && this->InputData != NULL)
    {
        if ((this->Volume->GetMTime() > this->GetMTime() || this->InputData->GetMTime() > this->GetMTime()))
        {
            if (this->Volume->GetMTime() > this->GetMTime())
                this->UpdateVolume();

            if (this->InputData->GetMTime() > this->GetMTime())
                this->UpdateImageData();

            this->Modified();
        }
    }
}

void vtkCudaVolumeInformationHandler::PrintSelf(std::ostream& os, vtkIndent indent)
{

}
