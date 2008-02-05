#ifndef VTKCUDASUPPORT_H_
#define VTKCUDASUPPORT_H_

#include "vtkObject.h"
#include "CudappSupportModule.h"
#include "vtkstd/vector"

class CudappDevice;

class VTK_CUDASUPPORT_EXPORT CudappSupport : public vtkObject
{
public:
    vtkTypeRevisionMacro(CudappSupport, vtkObject);
    static CudappSupport *New();

    bool IsSupported() { return (this->GetDeviceCount() > 0); }
    bool IsSupported(const char* cudaVersion);

    //BTX
    int GetDeviceCount() const { return this->Devices.size(); }        
    const vtkstd::vector<CudappDevice*> GetDevices() { return this->Devices; }
    CudappDevice* operator[](int deviceNumber) const { return this->Devices[deviceNumber]; }
    //ETX

    void PrintSelf(ostream& os, vtkIndent indent);

protected:
    CudappSupport();
    virtual ~CudappSupport();

    int CheckSupportedCudaVersion();
    //BTX
    vtkstd::vector<CudappDevice*> Devices;
    //ETX
};

#endif /*VTKCUDASUPPORT_H_*/
