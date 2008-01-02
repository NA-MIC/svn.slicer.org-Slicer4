#ifndef VTKCUDASUPPORT_H_
#define VTKCUDASUPPORT_H_

#include "vtkObject.h"
#include "vtkCudaSupportModule.h"
#include "vtkstd/vector"

class vtkCudaDevice;

class VTK_CUDASUPPORT_EXPORT vtkCudaSupport : public vtkObject
{
    vtkTypeRevisionMacro(vtkCudaSupport, vtkObject);
    static vtkCudaSupport *New();
        
//BTX
    int GetDeviceCount() const { return this->Devices.size(); }        
    const vtkstd::vector<vtkCudaDevice*> GetDevices() { return this->Devices; }
    vtkCudaDevice* operator[](int deviceNumber) const { return this->Devices[deviceNumber]; }
//ETX

        void PrintSelf(ostream& os, vtkIndent indent);

protected:
        vtkCudaSupport();
        virtual ~vtkCudaSupport();

        int CheckSupportedCudaVersion();
//BTX
        vtkstd::vector<vtkCudaDevice*> Devices;
//ETX
};

#endif /*VTKCUDASUPPORT_H_*/
