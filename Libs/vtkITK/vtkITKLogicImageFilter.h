// .NAME vtkITKLogicalImageFilter - Wrapper class around itk Logic filters to implement 
// logical operations between images
// .SECTION Description
//

#ifndef __vtkITKLogicImageFilter_h
#define __vtkITKLogicImageFilter_h

#include "vtkITK.h" 
#include "vtkImageTwoInputFilter.h"


#ifndef VTK_AND
#define VTK_AND 0
#endif

#ifndef VTK_OR
#define VTK_OR 1
#endif

#ifndef VTK_XOR
#define VTK_XOR 2
#endif

#ifndef VTK_NOP
#define VTK_NOP 6
#endif

// Usage: SetInput1 is a label image 
// SetInput2 is another label image
// GetOutput produces the output label image


class VTK_EXPORT vtkITKLogicImageFilter : public vtkImageTwoInputFilter
{
public:

  static vtkITKLogicImageFilter *New();
  vtkTypeRevisionMacro(vtkITKLogicImageFilter,vtkImageTwoInputFilter);
  void PrintSelf(ostream& os, vtkIndent indent);

  
  // Methods to set/get the logical operation to perform
  vtkSetMacro(Operation, int);
  vtkGetMacro(Operation, int);

  void SetOperationToOr()
  {
    SetOperation(VTK_OR);
  }

  void SetOperationToXor()
  {
    SetOperation(VTK_XOR);
  }

  void SetOperationToAnd()
  {
    SetOperation(VTK_AND);
  }

public:
  // member variables
  int Operation;

protected:
  vtkITKLogicImageFilter();
  ~vtkITKLogicImageFilter() {};
  
  virtual void ExecuteData(vtkDataObject *outData);
  
private:
  vtkITKLogicImageFilter(const vtkITKLogicImageFilter&);  // Not implemented.
  void operator=(const vtkITKLogicImageFilter&);  // Not implemented.
  
};

#endif
