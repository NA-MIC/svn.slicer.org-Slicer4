// .NAME vtkMRMLAnnotationControlPointsStorageNode - MRML node for representing a volume storage
// .SECTION Description
// vtkMRMLAnnotationControlPointsStorageNode nodes describe the annotation storage
// node that allows to read/write point data from/to file.

#ifndef __vtkMRMLAnnotationControlPointsStorageNode_h
#define __vtkMRMLAnnotationControlPointsStorageNode_h

#include "vtkSlicerAnnotationsModuleMRMLExport.h"
#include "vtkMRMLAnnotationStorageNode.h"

class vtkMRMLAnnotationPointDisplayNode;
class vtkMRMLAnnotationControlPointsNode;

/// \ingroup Slicer_QtModules_Annotation
class  VTK_SLICER_ANNOTATIONS_MODULE_MRML_EXPORT vtkMRMLAnnotationControlPointsStorageNode : public vtkMRMLAnnotationStorageNode
{
public:
  static vtkMRMLAnnotationControlPointsStorageNode *New();
  vtkTypeMacro(vtkMRMLAnnotationControlPointsStorageNode,vtkMRMLAnnotationStorageNode);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  virtual vtkMRMLNode* CreateNodeInstance() VTK_OVERRIDE;

  // Description:
  // Get node XML tag name (like Storage, Model)
  virtual const char* GetNodeTagName() VTK_OVERRIDE {return "AnnotationControlPointsStorage";}

  // Initialize all the supported write file types
  virtual bool CanReadInReferenceNode(vtkMRMLNode* refNode) VTK_OVERRIDE;

protected:
  vtkMRMLAnnotationControlPointsStorageNode();
  ~vtkMRMLAnnotationControlPointsStorageNode();
  vtkMRMLAnnotationControlPointsStorageNode(const vtkMRMLAnnotationControlPointsStorageNode&);
  void operator=(const vtkMRMLAnnotationControlPointsStorageNode&);

  const char* GetAnnotationStorageType() { return "point"; }

  int WriteAnnotationPointDisplayProperties(fstream & of, vtkMRMLAnnotationPointDisplayNode *refNode, std::string preposition);
  int WriteAnnotationControlPointsProperties(fstream & of, vtkMRMLAnnotationControlPointsNode *refNode);
  int WriteAnnotationControlPointsData(fstream& of, vtkMRMLAnnotationControlPointsNode *refNode);

  int ReadAnnotation(vtkMRMLAnnotationControlPointsNode *refNode);
  int ReadAnnotationControlPointsData(vtkMRMLAnnotationControlPointsNode *refNode, char line[1024], int typeColumn, int xColumn, int yColumn, int zColumn,
                      int selColumn,  int visColumn, int numColumns);
  int ReadAnnotationPointDisplayProperties(vtkMRMLAnnotationPointDisplayNode *refNode, std::string lineString, std::string preposition);
  int ReadAnnotationControlPointsProperties(vtkMRMLAnnotationControlPointsNode *refNode, char line[1024], int &typeColumn,
                        int& xColumn,    int& yColumn,     int& zColumn, int& selColumn, int& visColumn, int& numColumns);

  /// Read data and set it in the referenced node
  virtual int ReadDataInternal(vtkMRMLNode *refNode) VTK_OVERRIDE;

  virtual int WriteAnnotationDataInternal(vtkMRMLNode *refNode, fstream &of) VTK_OVERRIDE;
};

#endif
