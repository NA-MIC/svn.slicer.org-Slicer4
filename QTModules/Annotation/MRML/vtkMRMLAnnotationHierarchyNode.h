// .NAME vtkMRMLAnnotationHierarchyNode - MRML node to represent hierarchy of Annotations.
// .SECTION Description
// n/a
//

#ifndef __vtkMRMLAnnotationHierarchyNode_h
#define __vtkMRMLAnnotationHierarchyNode_h

#include "vtkSlicerAnnotationModuleMRMLExport.h"
#include "vtkMRMLDisplayableHierarchyNode.h"
#include "vtkMRMLNode.h"

/// \ingroup Slicer_QtModules_Annotation
class  VTK_SLICER_ANNOTATION_MODULE_MRML_EXPORT vtkMRMLAnnotationHierarchyNode : public vtkMRMLDisplayableHierarchyNode
{
public:
  static vtkMRMLAnnotationHierarchyNode *New();
  vtkTypeMacro(vtkMRMLAnnotationHierarchyNode,vtkMRMLDisplayableHierarchyNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  virtual const char* GetIcon() {return ":/Icons/SlicerHierarchy.png";};

  // Description:
  // Read node attributes from XML file
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Get node XML tag name (like Volume, Annotation)
  virtual const char* GetNodeTagName();

  // Description:
  // Get all children associated to this node
  virtual void GetDirectChildren(vtkCollection *children);

  // Description:
  // Delete all children of this node
  // If a child is another hierarchyNode, the parent of it gets set to this' parent
  virtual void DeleteDirectChildren();

protected:
  vtkMRMLAnnotationHierarchyNode();
  ~vtkMRMLAnnotationHierarchyNode();
  vtkMRMLAnnotationHierarchyNode(const vtkMRMLAnnotationHierarchyNode&);
  void operator=(const vtkMRMLAnnotationHierarchyNode&);

};

#endif
