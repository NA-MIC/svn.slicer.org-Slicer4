#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkImageData;
class vtkVolumeProperty;
class vtkCudaRendererInformationHandler;
class vtkCudaVolumeInformationHandler;
class vtkCudaMemoryTexture;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();

    virtual void SetInput( vtkImageData * );
    virtual void Render(vtkRenderer *, vtkVolume *);

    // Should be in Property??
    void SetThreshold(unsigned int min, unsigned int max);
    void SetThreshold(double* range) { SetThreshold((unsigned int)range[0], (unsigned int)range[1]); }

   //BTX
   void SetRenderMode(int mode);
   int GetCurrentRenderMode() const;// { return this->CurrentRenderMode; }
   //ETX

   vtkImageData* GetOutput() { return this->LocalOutputImage; }

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

    vtkImageData* LocalOutputImage;

    vtkCudaRendererInformationHandler* RendererInfoHandler;
    vtkCudaVolumeInformationHandler* VolumeInfoHandler;

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
