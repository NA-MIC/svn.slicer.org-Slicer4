/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiberBundleNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

=========================================================================auto=*/
// .NAME vtkMRMLFiberBundleNode - MRML node to represent a fiber bundle from tractography in DTI data.
// .SECTION Description
// FiberBundle nodes contain trajectories ("fibers") from tractography, internally represented as vtkPolyData.
// A FiberBundle node contains many fibers and forms the smallest logical unit of tractography 
// that MRML will manage/read/write. Each fiber has accompanying tensor data.  
// Visualization parameters for these nodes are controlled by the vtkMRMLFiberBundleDisplayNode class.
//

#ifndef __vtkMRMLFiberBundleNode_h
#define __vtkMRMLFiberBundleNode_h

#include <string>

#include "vtkPolyData.h" 

#include "vtkMRML.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLStorageNode.h"
#include "vtkMRMLFiberBundleDisplayNode.h"


class vtkCallbackCommand;

class VTK_MRML_EXPORT vtkMRMLFiberBundleNode : public vtkMRMLModelNode
{
public:
  static vtkMRMLFiberBundleNode *New();
  vtkTypeMacro(vtkMRMLFiberBundleNode,vtkMRMLModelNode);
  //BTX
  vtkMRMLNodeInheritanceMacro(vtkMRMLFiberBundleNode);
  //ETX
  void PrintSelf(ostream& os, vtkIndent indent);
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "FiberBundle";};


protected:
  vtkMRMLFiberBundleNode(){};
  ~vtkMRMLFiberBundleNode(){};
  vtkMRMLFiberBundleNode(const vtkMRMLFiberBundleNode&);
  void operator=(const vtkMRMLFiberBundleNode&);

};

#endif
