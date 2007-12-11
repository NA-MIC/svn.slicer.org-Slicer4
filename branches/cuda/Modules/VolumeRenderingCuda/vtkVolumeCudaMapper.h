#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
  public:

    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    void PrintSelf(ostream& os, vtkIndent indent);

    static vtkVolumeCudaMapper *New();

    virtual void Render(vtkRenderer *, vtkVolume *) {};

//BTX
    enum
    {
      CUDA_VERSION_SUPPORT=0,
      CUDA_VERSION_1_0=1,
      CUDA_VERSION_1_1=2,
      CUDA_NO_SUPPORT=3,
    };

//ETX
    // int GetSupportedCudaVersion();
    // int SetPreferedCudaSupport();

  protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);

    int CheckSupportedCudaVersion(int cudaVersion = 0);

  private:
    int CudaVersion;
    int PreferedCudaVersion;
};

#endif /* __vtkVolumeCudaMapper_h */
