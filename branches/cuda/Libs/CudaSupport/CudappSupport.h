#ifndef CUDAPPSUPPORT_H_
#define CUDAPPSUPPORT_H_

#include "vtkObject.h"
#include "CudappSupportModule.h"
#include <vector>

class CudappDevice;

class CUDA_SUPPORT_EXPORT CudappSupport : public vtkObject
{
public:
    CudappSupport();
    virtual ~CudappSupport();

    bool IsSupported() { return (this->GetDeviceCount() > 0); }
    bool IsSupported(const char* cudaVersion);

    //BTX
    int GetDeviceCount() const { return this->Devices.size(); }        
    const vtkstd::vector<CudappDevice*> GetDevices() { return this->Devices; }
    CudappDevice* operator[](int deviceNumber) const { return this->Devices[deviceNumber]; }
    //ETX

    void PrintSelf(std::ostream&  os);

protected:
    typedef std::vector<CudappDevice*> DeviceList;

    int CheckSupportedCudaVersion();
    //BTX
    DeviceList Devices;
    //ETX
};

#endif /*CUDAPPSUPPORT_H_*/
