#ifndef __vtkCameraBasedROILogic_h
#define __vtkCameraBasedROILogic_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

// MRML includes
#include "vtkMRMLScene.h"

// VTKSYS includes
#include <vtksys/SystemTools.hxx>

// VTK includes
#include "vtkObject.h"
#include "vtkMatrix4x4.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkMRMLCameraBasedROINode.h"

// STL includes
#include <string>
#include <vector>
#include <map>
#include <iterator>

#include "vtkSlicerCameraBasedROIModuleLogicExport.h"

// TODO Node registration needs to be done in the Logic. See RegisterNodes

class VTK_SLICER_CAMERABASEDROI_MODULE_LOGIC_EXPORT vtkCameraBasedROILogic : public vtkSlicerModuleLogic
{
 public:
  static vtkCameraBasedROILogic *New();
  vtkTypeMacro(vtkCameraBasedROILogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent){};

  void UpdateROI(vtkMRMLCameraBasedROINode *paramNode);

 protected:
  vtkCameraBasedROILogic();
  ~vtkCameraBasedROILogic();
  vtkCameraBasedROILogic(const vtkCameraBasedROILogic&);
  void operator=(const vtkCameraBasedROILogic&);  
};


#endif

