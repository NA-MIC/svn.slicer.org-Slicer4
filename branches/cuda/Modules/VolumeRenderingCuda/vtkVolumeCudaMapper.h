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
    void PrintSelf(ostream& os, vtkIndent indent);

    virtual void SetInput(vtkCudaLocalMemory*, unsigned int x, unsigned int y, unsigned int z);
    virtual void SetInput( vtkImageData * );
    //virtual void SetInput( vtkDataSet * );
    vtkImageData *GetInput() { return this->LocalOutputImage; }

    static vtkVolumeCudaMapper *New();

    virtual void Render(vtkRenderer *, vtkVolume *);

    void SetColor(double r, double g, double b) { Color[0] = r; Color[1] = g; Color[2] = b; }
    void SetColor(const double c[3]) { this->SetColor(c[0], c[1], c[2]); };
    vtkGetVector3Macro(Color,double);

protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void InitializeInternal();
    //void UpdateInputResolution(unsigned int x, unsigned int y, unsigned int z);
    void UpdateOutputResolution(unsigned int width, unsigned int height, unsigned int colors);

    unsigned int  InputDataSize[3];
    unsigned int OutputDataSize[2];

    vtkCudaLocalMemory* LocalInputBuffer;
    vtkCudaHostMemory* LocalOutputBuffer;

    vtkImageData* LocalInputImage;
    vtkImageData* LocalOutputImage;

    vtkCudaMemory* CudaInputBuffer;
    vtkCudaMemory* CudaOutputBuffer;

    double Color[3];

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
