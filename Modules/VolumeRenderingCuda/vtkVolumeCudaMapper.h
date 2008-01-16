#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaMemory;
class vtkImageData;
class vtkCudaHostMemory;
class vtkCudaLocalMemory;

class vtkCamera;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();

    virtual void SetInput( vtkImageData * );

    virtual void Update();
    virtual void Render(vtkRenderer *, vtkVolume *);

    void SetColor(double r, double g, double b) { Color[0] = r; Color[1] = g; Color[2] = b; this->Modified(); }
    void SetColor(const double c[3]) { this->SetColor(c[0], c[1], c[2]); this->Modified();};
    vtkGetVector3Macro(Color,double);

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void UpdateOutputResolution(unsigned int width, unsigned int height, unsigned int colors);

    void UpdateRenderPlane(vtkRenderer *, vtkVolume *);

    unsigned int OutputDataSize[2];

    vtkImageData* LocalOutputImage;

    vtkCudaMemory* CudaInputBuffer;
    vtkCudaMemory* CudaOutputBuffer;

    double Color[3];

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
