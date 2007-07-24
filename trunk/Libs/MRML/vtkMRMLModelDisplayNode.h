/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLModelDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

=========================================================================auto=*/
// .NAME vtkMRMLModelDisplayNode - MRML node to represent a dispaly property of 3D surface model.
// .SECTION Description
// vtkMRMLModelDisplayNode nodes stores display property of a 3D surface model
// including reference to ColorNode, texture, opacity, etc.

#ifndef __vtkMRMLModelDisplayNode_h
#define __vtkMRMLModelDisplayNode_h

#include <string>

#include "vtkMRML.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLStorageNode.h"
#include "vtkMRMLDisplayNode.h"
#include "vtkMRMLColorNode.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"

class VTK_MRML_EXPORT vtkMRMLModelDisplayNode : public vtkMRMLDisplayNode
{
public:
  static vtkMRMLModelDisplayNode *New();
  vtkTypeMacro(vtkMRMLModelDisplayNode,vtkMRMLDisplayNode);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Read node attributes from XML file
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);


  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);
  
  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "ModelDisplay";};

  // Description:
  // Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

  // Description:
  // Model's color (r,g,b)
  vtkSetVector3Macro(Color, double);
  vtkGetVector3Macro(Color, double);
  
  // Description:
  // Opacity of the surface expressed as a number from 0 to 1
  vtkSetMacro(Opacity, double);
  vtkGetMacro(Opacity, double);

  // Description:
  // Ambient of the surface expressed as a number from 0 to 100
  vtkSetMacro(Ambient, double);
  vtkGetMacro(Ambient, double);
  
  // Description:
  // Diffuse of the surface expressed as a number from 0 to 100
  vtkSetMacro(Diffuse, double);
  vtkGetMacro(Diffuse, double);
  
  // Description:
  // Specular of the surface expressed as a number from 0 to 100
  vtkSetMacro(Specular, double);
  vtkGetMacro(Specular, double);

  // Description:
  // Power of the surface expressed as a number from 0 to 100
  vtkSetMacro(Power, double);
  vtkGetMacro(Power, double);

  // Description:
  // Indicates if the surface is visible
  vtkBooleanMacro(Visibility, int);
  vtkGetMacro(Visibility, int);
  vtkSetMacro(Visibility, int);

  // Description:
  // Specifies whether to clip the surface with the slice planes
  vtkBooleanMacro(Clipping, int);
  vtkGetMacro(Clipping, int);
  vtkSetMacro(Clipping, int);

  // Description:
  // Indicates whether to cull (not render) the backface of the surface
  vtkBooleanMacro(BackfaceCulling, int);
  vtkGetMacro(BackfaceCulling, int);
  vtkSetMacro(BackfaceCulling, int);

  // Description:
  // Indicates whether to render the scalar value associated with each polygon vertex
  vtkBooleanMacro(ScalarVisibility, int);
  vtkGetMacro(ScalarVisibility, int);
  vtkSetMacro(ScalarVisibility, int);

  // Description:
  // Indicates whether to render the vector value associated with each polygon vertex
  vtkBooleanMacro(VectorVisibility, int);
  vtkGetMacro(VectorVisibility, int);
  vtkSetMacro(VectorVisibility, int);

  // Description:
  // Indicates whether to render the tensor value associated with each polygon vertex
  vtkBooleanMacro(TensorVisibility, int);
  vtkGetMacro(TensorVisibility, int);
  vtkSetMacro(TensorVisibility, int);

  // Description:
  // Range of scalar values to render rather than the single color designated by colorName
  vtkSetVector2Macro(ScalarRange, double);
  vtkGetVector2Macro(ScalarRange, double);


  // Description:
  // Updates this node if it depends on other nodes 
  // when the node is deleted in the scene
  virtual void UpdateReferences();

  // Description:
  // Finds the storage node and read the data
  virtual void UpdateScene(vtkMRMLScene *scene);

  // Description:
  // Associated ImageData
  vtkGetObjectMacro(TextureImageData, vtkImageData);
  void SetAndObserveTextureImageData(vtkImageData *ImageData);

  // Description:
  // Set a default color node
//  void SetDefaultColorMap();
  
  // Description:
  // String ID of the color MRML node
  void SetAndObserveColorNodeID(const char *ColorNodeID);
  vtkGetStringMacro(ColorNodeID);

  // Description:
  // Get associated color MRML node
  vtkMRMLColorNode* GetColorNode();

  
  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                   unsigned long /*event*/, 
                                   void * /*callData*/ );
  // Description:
  // the name of the currently active scalar field for this model
  vtkGetStringMacro(ActiveScalarName);
  // Description:
  // set the active scalar field name, and update the color table if necessary
  void SetActiveScalarName(const char *scalarName);
  
protected:
  vtkMRMLModelDisplayNode();
  ~vtkMRMLModelDisplayNode();
  vtkMRMLModelDisplayNode(const vtkMRMLModelDisplayNode&);
  void operator=(const vtkMRMLModelDisplayNode&);

  vtkSetObjectMacro(TextureImageData, vtkImageData);
  
  vtkImageData    *TextureImageData;
  
  char *ColorNodeID;

  char *ActiveScalarName;
  
  vtkSetReferenceStringMacro(ColorNodeID);

  vtkMRMLColorNode *ColorNode;

  // Numbers
  double Opacity;
  double Ambient;
  double Diffuse;
  double Specular;
  double Power;
  

  // Booleans
  int Visibility;
  int Clipping;
  int BackfaceCulling;
  int ScalarVisibility;
  int VectorVisibility;
  int TensorVisibility;

  // Arrays
  double ScalarRange[2];
  double Color[3];
};

#endif
