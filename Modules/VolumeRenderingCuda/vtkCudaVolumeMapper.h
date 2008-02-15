#ifndef __vtkCudaVolumeMapper_h
#define __vtkCudaVolumeMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkVolumeProperty;

class vtkCudaRendererInformationHandler;
class vtkCudaVolumeInformationHandler;
class vtkCudaMemoryTexture;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaVolumeMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkCudaVolumeMapper, vtkVolumeMapper);
    static vtkCudaVolumeMapper *New();

    virtual void SetInput( vtkImageData * );
    virtual void Render(vtkRenderer *, vtkVolume *);

    // Should be in Property??
    void SetThreshold(unsigned int min, unsigned int max);
    void SetThreshold(double* range) { SetThreshold((unsigned int)range[0], (unsigned int)range[1]); }
    void SetSteppingSize(float steppingSize);

   //BTX
   void SetRenderMode(int mode);
   int GetCurrentRenderMode() const;// { return this->CurrentRenderMode; }
   //ETX

   vtkImageData* GetOutput() { return NULL; /*this->LocalOutputImage;*/ }

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkCudaVolumeMapper();
    virtual ~vtkCudaVolumeMapper();

    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

    vtkCudaRendererInformationHandler* RendererInfoHandler;
    vtkCudaVolumeInformationHandler* VolumeInfoHandler;

private:
    vtkCudaVolumeMapper operator=(const vtkCudaVolumeMapper&);
    vtkCudaVolumeMapper(const vtkCudaVolumeMapper&);
};

#endif /* __vtkCudaVolumeMapper_h */
