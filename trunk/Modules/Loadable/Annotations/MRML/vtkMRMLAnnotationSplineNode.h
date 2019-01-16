#ifndef __vtkMRMLAnnotationSplineNode_h
#define __vtkMRMLAnnotationSplineNode_h

#include "vtkMRMLAnnotationLinesNode.h"

class vtkMRMLScene;

/// \ingroup Slicer_QtModules_Annotation
class  VTK_SLICER_ANNOTATIONS_MODULE_MRML_EXPORT vtkMRMLAnnotationSplineNode : public vtkMRMLAnnotationLinesNode
{
public:
  static vtkMRMLAnnotationSplineNode *New();
  vtkTypeMacro(vtkMRMLAnnotationSplineNode, vtkMRMLAnnotationLinesNode);
  // Description:
  // Just prints short summary
  void PrintAnnotationInfo(ostream& os, vtkIndent indent, int titleFlag = 1) VTK_OVERRIDE;

  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance() VTK_OVERRIDE;
  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() VTK_OVERRIDE {return "AnnotationRuler";}

  // Description:
  // Read node attributes from XML file
  virtual void ReadXMLAttributes( const char** atts) VTK_OVERRIDE;

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent) VTK_OVERRIDE;


  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node) VTK_OVERRIDE;

  void UpdateScene(vtkMRMLScene *scene) VTK_OVERRIDE;

  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/,
                                   unsigned long /*event*/,
                                   void * /*callData*/ ) VTK_OVERRIDE;


  // Legacy code
  // Description:
  // get/set the first point position
  double* GetPosition1() {return this->GetControlPointCoordinates(0);}

  // Description:
  // get/set the distance annotation format, it's in standard sprintf notation
  vtkGetStringMacro(DistanceAnnotationFormat);
  vtkSetStringMacro(DistanceAnnotationFormat);

  // Description:
  // KP Define - should be part of AnnotationRulerDisplayNode
  double GetDistanceAnnotationScale();
  void SetDistanceAnnotationScale(double init);

  // Description:
  // get/set the distance annotation visbility
  int GetDistanceAnnotationVisibility();
  void SetDistanceAnnotationVisibility(int flag);

  int SetRuler(vtkIdType line1Id, int sel, int vis);

  // Description:
  // get/set the resolution (number of subdivisions) of the line.
  vtkGetMacro(Resolution, int);
  vtkSetMacro(Resolution, int);

  // Description:
  // get/set the point representation colour
  double *GetPointColour();
  void SetPointColour( double initColor[3]);

  // Description:
  // get/set the line representation colour
  double *GetLineColour();
  void SetLineColour(double newColor[3]);

  // Description:
  // get/set the distance annotation text colour
  double *GetDistanceAnnotationTextColour();
  void SetDistanceAnnotationTextColour(double initColor[3]);

  void Initialize(vtkMRMLScene* mrmlScene) VTK_OVERRIDE;

  double GetSplineMeasurement();
  void SetSplineMeasurement(double val);

  int SetControlPoint(double newControl[3], int id);

  enum
  {
      SplineNodeAddedEvent = 0,
      ValueModifiedEvent,
  };


protected:
  vtkMRMLAnnotationSplineNode();
  ~vtkMRMLAnnotationSplineNode();
  vtkMRMLAnnotationSplineNode(const vtkMRMLAnnotationSplineNode&);
  void operator=(const vtkMRMLAnnotationSplineNode&);

  // Description:
  // number of subdivisions on the line
  int Resolution;
  char* DistanceAnnotationFormat;

  int AddControlPoint(double newControl[3],int selectedFlag, int visibleFlag);

  double splineMeasurement;

};

#endif
