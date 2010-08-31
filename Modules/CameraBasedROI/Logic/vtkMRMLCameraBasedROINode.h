#ifndef __vtkMRMLCameraBasedROINode_h
#define __vtkMRMLCameraBasedROINode_h

// MRML includes
#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLCameraNode.h"
#include "vtkMRMLROINode.h"
#include "vtkMRMLTransformNode.h"

// VTK includes
#include "vtkStringArray.h"

// STL includes
#include <string>
#include <vector>

#include "vtkSlicerCameraBasedROIModuleLogicExport.h"

class VTK_SLICER_CAMERABASEDROI_MODULE_LOGIC_EXPORT vtkMRMLCameraBasedROINode : public vtkMRMLNode
{
  public:
  static vtkMRMLCameraBasedROINode *New();
  vtkTypeRevisionMacro(vtkMRMLCameraBasedROINode, vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Create instance 
  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Set node attributes from name/value pairs
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);
  
  // Description:
  // Get unique node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "CameraBasedROINode"; };
  
  vtkGetStringMacro ( CameraNodeID );
  vtkSetStringMacro ( CameraNodeID );

  vtkGetStringMacro ( ROINodeID );
  vtkSetStringMacro ( ROINodeID );

  vtkGetMacro ( ROIDistanceToCamera, double );
  vtkSetMacro ( ROIDistanceToCamera, double );

  vtkGetMacro ( ROISize, double );
  vtkSetMacro ( ROISize, double );
  
 protected:
  vtkMRMLCameraBasedROINode();
  ~vtkMRMLCameraBasedROINode();
  vtkMRMLCameraBasedROINode(const vtkMRMLCameraBasedROINode&);
  void operator=(const vtkMRMLCameraBasedROINode&);

  double   ROIDistanceToCamera;
  double   ROISize;
  char *CameraNodeID;
  char *ROINodeID;
};

#endif

