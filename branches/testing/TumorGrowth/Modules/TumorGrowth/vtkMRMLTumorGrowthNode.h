/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLTumorGrowthNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkMRMLTumorGrowthNode_h
#define __vtkMRMLTumorGrowthNode_h

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLStorageNode.h"

#include "vtkImageData.h"
#include "vtkTumorGrowth.h"

class vtkImageData;

class VTK_TUMORGROWTH_EXPORT vtkMRMLTumorGrowthNode : public vtkMRMLNode
{
  public:
  static vtkMRMLTumorGrowthNode *New();
  vtkTypeMacro(vtkMRMLTumorGrowthNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Create instance of a GAD node.
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
  virtual const char* GetNodeTagName() {return "TGParameters";};

  // Description:
  // Get/Set Conductance (module parameter)
  vtkGetMacro(Conductance, double);
  vtkSetMacro(Conductance, double);

  // Description:
  // Get/Set time step (module parameter)
  vtkGetMacro(TimeStep, double);
  vtkSetMacro(TimeStep, double);

 
  // Description:
  // Get/Set input volume MRML Id
  vtkGetStringMacro(FirstScanRef);
  vtkSetStringMacro(FirstScanRef);
  
  // Description:
  // Get/Set for SegmenterClass
  void SetROIMin(int index, int val) {this->ROIMin[index] = val ; }
  int GetROIMin(int index) {return this->ROIMin[index]; } 

  void SetROIMax(int index, int val) {this->ROIMax[index] = val ; }
  int GetROIMax(int index) {return this->ROIMax[index]; } 

  // Description:
  // Get/Set output volume MRML Id
  vtkGetStringMacro(SecondScanRef);
  vtkSetStringMacro(SecondScanRef);

  // Description:
  // Update the stored reference to another node in the scene
  // virtual void UpdateReferenceID(const char *oldID, const char *newID);

 
protected:
  vtkMRMLTumorGrowthNode();
  ~vtkMRMLTumorGrowthNode();
  vtkMRMLTumorGrowthNode(const vtkMRMLTumorGrowthNode&);
  void operator=(const vtkMRMLTumorGrowthNode&);

  double Conductance;
  double TimeStep;
  
  char* FirstScanRef;
  char* SecondScanRef;


  //BTX
  vtkstd::vector<int>  ROIMin; 
  vtkstd::vector<int>  ROIMax; 
  //ETX 
};

#endif

