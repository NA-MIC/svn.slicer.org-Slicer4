#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaMemory;
class vtkImageData;
class vtkCudaHostMemory;
class vtkCudaLocalMemory;
class vtkCudaMemoryArray;

class vtkVolumeProperty;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();

    virtual void SetInput( vtkImageData * );

    virtual void Render(vtkRenderer *, vtkVolume *);

    void SetColor(double r, double g, double b) { Color[0] = r; Color[1] = g; Color[2] = b; this->Modified(); }
    void SetColor(const double c[3]) { this->SetColor(c[0], c[1], c[2]); this->Modified();};
    vtkGetVector3Macro(Color,double);

    void SetThreshold(unsigned int min, unsigned int max) { this->Threshold[0] = min; this->Threshold[1] = max; }
    void SetThreshold(double* range) { SetThreshold(range[0], range[1]); }

   //BTX
   typedef enum 
   {
     RenderToTexture,
     RenderToMemory,
   } RenderMode;
   void SetRenderMode(RenderMode mode);
   RenderMode GetCurrentRenderMode() const { return this->CurrentRenderMode; }
   //ETX


   void PrintSelf(ostream& os, vtkIndent indent);

    vtkImageData* MultiInput[5];


protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void UpdateVolumeProperties(vtkVolumeProperty* property);
    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

    unsigned int OutputDataSize[2];


    vtkCudaMemory* CudaInputBuffer;
    vtkCudaMemory* CudaOutputBuffer;
    double Color[3];
//BTX
    RenderMode CurrentRenderMode;
//ETX
    vtkImageData* LocalOutputImage;

    vtkCudaHostMemory* LocalColorTransferFunction;
    vtkCudaMemory* CudaColorTransferFunction;
    vtkCudaHostMemory* LocalAlphaTransferFunction;
    vtkCudaMemory* CudaAlphaTransferFunction;


    unsigned int Threshold[2];

    unsigned int BufferObject;
    unsigned int Texture;

    bool GLBufferObjectsAvailiable;

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
