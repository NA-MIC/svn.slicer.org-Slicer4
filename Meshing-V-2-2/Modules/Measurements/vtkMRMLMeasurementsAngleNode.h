#ifndef __vtkMRMLMeasurementsAngleNode_h
#define __vtkMRMLMeasurementsAngleNode_h

#include "vtkMeasurementsWin32Header.h"

#include "vtkMRMLMeasurementsNode.h"
class VTK_MEASUREMENTS_EXPORT vtkMRMLMeasurementsAngleNode : public vtkMRMLMeasurementsNode
{
  public:
  static vtkMRMLMeasurementsAngleNode *New();
  vtkTypeRevisionMacro(vtkMRMLMeasurementsAngleNode, vtkMRMLMeasurementsNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // get/set the first point position
  vtkGetVector3Macro(Position1, double);
  vtkSetVector3Macro(Position1, double);

  // Description:
  // get/set the second point position
  vtkGetVector3Macro(Position2, double);
  vtkSetVector3Macro(Position2, double);

  // Description:
  // get/set the center point position
  vtkGetVector3Macro(PositionCenter, double);
  vtkSetVector3Macro(PositionCenter, double);

  // Description:
  // get/set the distance annotation format, it's in standard sprintf notation
  vtkGetStringMacro(LabelFormat);
  vtkSetStringMacro(LabelFormat);
  
  // Description:
  // get/set the distance annotation scale
  vtkGetVector3Macro(LabelScale, double);
  vtkSetVector3Macro(LabelScale, double);
  
  // Description:
  // get/set the distance annotation visbility
  vtkGetMacro(LabelVisibility, int);
  vtkSetMacro(LabelVisibility, int);
  vtkBooleanMacro(LabelVisibility, int);

  // Description:
  // get/set the visibility of the rays and arc
  vtkGetMacro(Ray1Visibility, int);
  vtkSetMacro(Ray1Visibility, int);
  vtkBooleanMacro(Ray1Visibility, int);
  vtkGetMacro(Ray2Visibility, int);
  vtkSetMacro(Ray2Visibility, int);
  vtkBooleanMacro(Ray2Visibility, int);
  vtkGetMacro(ArcVisibility, int);
  vtkSetMacro(ArcVisibility, int);
  vtkBooleanMacro(ArcVisibility, int);
  
  // Description:
  // get/set the resolution (number of subdivisions) of the line.
  vtkGetMacro(Resolution, int);
  vtkSetMacro(Resolution, int);
  
  // Description:
  // get/set the point representation colour
  vtkGetVector3Macro(PointColour, double);
  vtkSetVector3Macro(PointColour, double);

  // Description:
  // get/set the line representation colour
  vtkGetVector3Macro(LineColour, double);
  vtkSetVector3Macro(LineColour, double);

  // Description:
  // get/set the distance annotation text colour
  vtkGetVector3Macro(LabelTextColour, double);
  vtkSetVector3Macro(LabelTextColour, double);

  // Description:
  // get/set the id of the model the ends of the widget are constrained upon
  vtkGetStringMacro(ModelID1);
  vtkSetStringMacro(ModelID1);
  vtkGetStringMacro(ModelID2);
  vtkSetStringMacro(ModelID2);
  vtkGetStringMacro(ModelIDCenter);
  vtkSetStringMacro(ModelIDCenter);
  
  
  // Description:
  // Create instance of a measurements angle node.
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
  virtual const char* GetNodeTagName() {return "MeasurementsAngle"; };

  // Description:
  // transform utility functions
  virtual bool CanApplyNonLinearTransforms() { return true; }
  virtual void ApplyTransform(vtkMatrix4x4* transformMatrix);
  virtual void ApplyTransform(vtkAbstractTransform* transform);

protected:
  vtkMRMLMeasurementsAngleNode();
  ~vtkMRMLMeasurementsAngleNode();
  vtkMRMLMeasurementsAngleNode(const vtkMRMLMeasurementsAngleNode&);
  void operator=(const vtkMRMLMeasurementsAngleNode&);

  // Description:
  // first point location
  double Position1[3];

  // Description:
  // second point location
  double Position2[3];

  // Description:
  // center point location
  double PositionCenter[3];

  // Description:
  // the distance text properties
  char *LabelFormat;
  double LabelScale[3];
  int LabelVisibility;

  // Descritpion:
  // visibility of sub components of the widget
  int Ray1Visibility;
  int Ray2Visibility;
  int ArcVisibility;

  // Description:
  // colours of the actors representing the end points and the line and the text
  double PointColour[3];
  double LineColour[3];
  double LabelTextColour[3];

  // Description:
  // number of subdivisions on the line
  int Resolution;

  // Description:
  // the model ids for the models that the ends of the widget are constrained
  // to
  char *ModelID1;
  char *ModelID2;
  char *ModelIDCenter;
};

#endif

