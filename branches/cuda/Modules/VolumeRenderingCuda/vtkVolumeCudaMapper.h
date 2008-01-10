#ifndef __vtkVolumeCudaMapper_h
#define __vtkVolumeCudaMapper_h

#include "vtkVolumeMapper.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaMemory;
class vtkImageData;
class vtkCudaHostMemory;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeCudaMapper : public vtkVolumeMapper
{
public:
    vtkTypeRevisionMacro(vtkVolumeCudaMapper,vtkVolumeMapper);
    void PrintSelf(ostream& os, vtkIndent indent);

    virtual void SetInput(unsigned char*, unsigned int x, unsigned int y, unsigned int z);
    //virtual void SetInput( vtkImageData * );
    //virtual void SetInput( vtkDataSet * );
    vtkImageData *GetInput() { return this->LocalOutputImage; }

    static vtkVolumeCudaMapper *New();

    virtual void Render(vtkRenderer *, vtkVolume *);


protected:
    vtkVolumeCudaMapper();
    virtual ~vtkVolumeCudaMapper();


    void InitializeInternal();
    //void UpdateInputResolution(unsigned int x, unsigned int y, unsigned int z);
    void UpdateOutputResolution(unsigned int width, unsigned int height, unsigned int colors);

    unsigned int  InputDataSize[3];
    unsigned int OutputDataSize[2];

    unsigned char* LocalInputBuffer;
    vtkCudaHostMemory* LocalOutputBuffer;

    vtkImageData* LocalOutputImage;

    vtkCudaMemory* CudaInputBuffer;
    vtkCudaMemory* CudaOutputBuffer;

private:
    vtkVolumeCudaMapper operator=(const vtkVolumeCudaMapper&);
    vtkVolumeCudaMapper(const vtkVolumeCudaMapper&);
};

#endif /* __vtkVolumeCudaMapper_h */
