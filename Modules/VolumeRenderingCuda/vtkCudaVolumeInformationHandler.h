#ifndef vtkCudaVolumeInformationHandler_H_
#define vtkCudaVolumeInformationHandler_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkVolume;
class vtkVolumeProperty;
class vtkImageData;
//BTX
#include "cudaVolumeInformation.h"
#include "CudappHostMemory.h"
#include "CudappDeviceMemory.h"
//ETX

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaVolumeInformationHandler : public vtkObject
{
    vtkTypeRevisionMacro(vtkCudaVolumeInformationHandler, vtkObject);
public:
    static vtkCudaVolumeInformationHandler* New();

    //BTX
    vtkGetMacro(Volume, vtkVolume*);
    vtkSetMacro(Volume, vtkVolume*);
    void SetInputData(vtkImageData* inputData);
    const cudaVolumeInformation& GetVolumeInfo() const { return VolumeInfo; }
    //ETX

    void SetThreshold(unsigned int min, unsigned int max);

    void ResizeTransferFunction(unsigned int size);
    virtual void Update();

protected:
    vtkCudaVolumeInformationHandler();
    ~vtkCudaVolumeInformationHandler();

    void UpdateVolumeProperties(vtkVolumeProperty *property);

    virtual void PrintSelf(ostream& os, vtkIndent indent);

private:
    vtkCudaVolumeInformationHandler& operator=(const vtkCudaVolumeInformationHandler&); // not implemented
    vtkCudaVolumeInformationHandler(const vtkCudaVolumeInformationHandler&); // not implemented


private:
    vtkImageData*           InputData;
    vtkVolume*              Volume;
    //BTX
    cudaVolumeInformation   VolumeInfo;

    Cudapp::DeviceMemory    CudaInputBuffer;

    Cudapp::HostMemory      LocalColorTransferFunction;
    Cudapp::DeviceMemory    CudaColorTransferFunction;
    Cudapp::HostMemory      LocalAlphaTransferFunction;
    Cudapp::DeviceMemory    CudaAlphaTransferFunction;
    //ETX
};

#endif /* vtkCudaVolumeInformationHandler_H_ */
