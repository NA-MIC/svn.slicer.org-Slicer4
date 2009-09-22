/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLVolumeRenderingParametersNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
// .NAME vtkMRMLVolumeRenderingParametersNode - MRML node for storing a slice through RAS space
// .SECTION Description
// This node stores the information about the currently selected volume
// 
//

#ifndef __vtkMRMLVolumeRenderingParametersNode_h
#define __vtkMRMLVolumeRenderingParametersNode_h

#include "vtkMRML.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLROINode.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLVolumePropertyNode.h"

#include "vtkVolumeRendering.h"

#include "vtkMatrix4x4.h"

class VTK_SLICERVOLUMERENDERING_EXPORT vtkMRMLVolumeRenderingParametersNode : public vtkMRMLNode
{
  public:
  static vtkMRMLVolumeRenderingParametersNode *New();
  vtkTypeMacro(vtkMRMLVolumeRenderingParametersNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "VolumeRenderingSelection";};

  // Description:
  // Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

  // Description:
  // Updates this node if it depends on other nodes 
  // when the node is deleted in the scene
  virtual void UpdateReferences();

  // Description:
  // Observe the reference transform node
  virtual void UpdateScene(vtkMRMLScene *scene);

  virtual void ProcessMRMLEvents ( vtkObject *caller, 
                                   unsigned long event, 
                                   void *callData);

  // Description:
  // the ID of a MRMLVolumeNode
  vtkGetStringMacro (VolumeNodeID);
  void SetAndObserveVolumeNodeID(const char *volumeNodeID);

  // Description:
  // Associated transform MRML node
  vtkMRMLVolumeNode* GetVolumeNode();

  // Description:
  // the ID of a parameter MRMLVolumePropertyNode
  vtkGetStringMacro (VolumePropertyNodeID);
  void SetAndObserveVolumePropertyNodeID(const char *volumePropertyNodeID);
  
  // Description:
  // Associated transform MRML node
  vtkMRMLVolumePropertyNode* GetVolumePropertyNode();

  // Description:
  // the ID of a parameter MRMLROINode
  vtkGetStringMacro (ROINodeID);
  void SetAndObserveROINodeID(const char *rOINodeID);

  // Description:
  // Associated transform MRML node
  vtkMRMLROINode* GetROINode();

  // Description:
  // Is cropping enabled?
  vtkSetMacro(CroppingEnabled,int);
  vtkGetMacro(CroppingEnabled,int);
  vtkBooleanMacro(CroppingEnabled,int);

  // Description:
  // Estimated Sample Distance
  vtkSetMacro(EstimatedSampleDistance,double);
  vtkGetMacro(EstimatedSampleDistance,double);

  // Description:
  // Expected FPS
  vtkSetMacro(ExpectedFPS,int);
  vtkGetMacro(ExpectedFPS,int);

  vtkGetStringMacro (CurrentVolumeMapper);
  vtkSetStringMacro (CurrentVolumeMapper);

  
protected:
  vtkMRMLVolumeRenderingParametersNode();
  ~vtkMRMLVolumeRenderingParametersNode();
  vtkMRMLVolumeRenderingParametersNode(const vtkMRMLVolumeRenderingParametersNode&);
  void operator=(const vtkMRMLVolumeRenderingParametersNode&);

  char *VolumeNodeID;
  vtkSetReferenceStringMacro(VolumeNodeID);
  vtkMRMLVolumeNode* VolumeNode;

  char *VolumePropertyNodeID;
  vtkSetReferenceStringMacro(VolumePropertyNodeID);
  vtkMRMLVolumePropertyNode* VolumePropertyNode;

  char *ROINodeID;
  vtkSetReferenceStringMacro(ROINodeID);
  vtkMRMLROINode* ROINode;

  int CroppingEnabled;

  double  EstimatedSampleDistance;
  int     ExpectedFPS;

  char *CurrentVolumeMapper;
};

#endif

