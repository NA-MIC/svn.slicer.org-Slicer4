#ifndef VTKCUDASUPPORT_H_
#define VTKCUDASUPPORT_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"
#include "vtkstd/vector"
class vtkCudaDevice;


class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaSupport : public vtkObject
{
        vtkTypeRevisionMacro(vtkCudaSupport, vtkObject);
//BTX
        typedef vtkObject SuperClass;
//ETX
        static vtkCudaSupport *New();
        
        int GetDeviceCount() const { return Devices.size(); }
//BTX
    const vtkstd::vector<vtkCudaDevice*> GetDevices() { return Devices; }
    vtkCudaDevice* operator[](int deviceNumber) const { return Devices[deviceNumber]; }
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
