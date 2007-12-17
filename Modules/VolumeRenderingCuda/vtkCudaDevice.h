#ifndef VTKCUDADEVICE_H_
#define VTKCUDADEVICE_H_

#include "vtkObject.h"
#include "vtkVolumeRenderingCudaModule.h"

#include "driver_types.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaDevice : public vtkObject
{
  vtkTypeRevisionMacro(vtkCudaDevice, vtkObject);
  
  static vtkCudaDevice* New();

/// Device Information
  vtkGetMacro(Initialized, bool);
  void SetDeviceNumber(int deviceNumber);
  vtkGetMacro(DeviceNumber, int);
  const char* GetName() const { return DeviceProp.name; }
  size_t GetTotalGlobalMem() const { return DeviceProp.totalGlobalMem; }
  size_t GetSharedMemPerBlock() const { return DeviceProp. sharedMemPerBlock; }
  int GetRegsPerBlock() const { return DeviceProp.regsPerBlock; }
  int GetWrapSize() const { return DeviceProp.warpSize; }
  size_t GetMemPitch() const { return DeviceProp.memPitch; }
  int GetMaxThreadsPerBlock() const { return DeviceProp.maxThreadsPerBlock; }
  const int* GetMaxThreadsDim() const { return DeviceProp.maxThreadsDim; } // [3]
  const int* GetMaxGridSize() const { return DeviceProp.maxGridSize; }  // [3]
  size_t GetTotalConstMem() const { return DeviceProp.totalConstMem; }
  int GetMajor() const { return DeviceProp.major; }
  int GetMinor() const { return DeviceProp.minor; }
  int GetClockRate() const { return DeviceProp.clockRate; }
  size_t GetTextureAlignment() const { return DeviceProp.textureAlignment; }
  
  
  
  void MakeActive();

  /// Memory Management
  bool AllocateMemory();
        
  void PrintSelf(ostream& os, vtkIndent indent);
        
  protected:
    vtkCudaDevice();
    virtual ~vtkCudaDevice();

  void LoadDeviceProperties();

    bool Initialized;
    int DeviceNumber;
    cudaDeviceProp DeviceProp;    
};

#endif /*VTKCUDADEVICE_H_*/
