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

    virtual void Render(vtkRenderer *, vtkVolume *);


  protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void PrepareRender();
    
    unsigned char* inputBuffer;
    unsigned char* outputBuffer;

  private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
