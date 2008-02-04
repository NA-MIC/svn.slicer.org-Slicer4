#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaDeviceMemory;
class vtkImageData;
class vtkCudaHostMemory;
class vtkCudaLocalMemory;

class vtkVolumeProperty;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    static vtkVolumeCudaMapper *New();

    virtual void SetInput( vtkImageData * );

    virtual void Render(vtkRenderer *, vtkVolume *);

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

   vtkImageData* GetOutput() { return this->LocalOutputImage; }

   void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();

    void UpdateVolumeProperties(vtkVolumeProperty* property);
    void UpdateOutputResolution(unsigned int width, unsigned int height, bool TypeChanged = false);

    unsigned int OutputDataSize[2];


    vtkCudaDeviceMemory* CudaInputBuffer;
    vtkCudaDeviceMemory* CudaOutputBuffer;
//BTX
    RenderMode CurrentRenderMode;
//ETX
    vtkImageData* LocalOutputImage;

    vtkCudaHostMemory* LocalColorTransferFunction;
    vtkCudaDeviceMemory* CudaColorTransferFunction;
    vtkCudaHostMemory* LocalAlphaTransferFunction;
    vtkCudaDeviceMemory* CudaAlphaTransferFunction;

    vtkCudaDeviceMemory* CudaZBuffer;

    unsigned int Threshold[2];

    unsigned int BufferObject;
    unsigned int Texture;

    bool GLBufferObjectsAvailiable;

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
