#ifndef VTKCUDAIMAGEDATAFILTER_H_
#define VTKCUDAIMAGEDATAFILTER_H_

#include "vtkImageAlgorithm.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkCudaImageData;

class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkCudaImageDataFilter : public vtkImageAlgorithm
{
public:
  vtkTypeRevisionMacro(vtkCudaImageDataFilter, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);
  static vtkCudaImageDataFilter* New();
  
  vtkCudaImageData* GetOutput ();
  vtkCudaImageData* GetOutput (int port);
  virtual void       SetOutput (vtkCudaImageData *d);  
  
protected:
  vtkCudaImageDataFilter();
  virtual ~vtkCudaImageDataFilter();
  
  // convenience method
  virtual int RequestData(vtkInformation* request,
                  vtkInformationVector** vtkNotUsed( inputVector ),
                  vtkInformationVector* outputVector);

  virtual int RequestInformation(vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
                                
  // just allocate the output data
  virtual void AllocateOutputData(vtkCudaImageData *out, 
                                  int *uExtent);
  //virtual vtkCudaImageData *AllocateOutputData(vtkDataObject *out);

  virtual int FillOutputPortInformation(int port, vtkInformation* info);
  virtual int FillInputPortInformation(int port, vtkInformation* info);
  
private:
  vtkCudaImageDataFilter(const vtkCudaImageDataFilter&);
  vtkCudaImageDataFilter& operator=(const vtkCudaImageDataFilter&) const;
 };

#endif /*VTKCUDAIMAGEDATAFILTER_H_*/
