/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkOpenIGTLinkDaemonLogic.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkOpenIGTLinkDaemonLogic_h
#define __vtkOpenIGTLinkDaemonLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkMRMLScene.h"

#include "vtkOpenIGTLinkDaemon.h"
#include "vtkMRMLOpenIGTLinkDaemonNode.h"


class vtkITKOpenIGTLinkDaemon;

class VTK_SLICERDAEMON_EXPORT vtkOpenIGTLinkDaemonLogic : public vtkSlicerModuleLogic
{
  public:
  static vtkOpenIGTLinkDaemonLogic *New();
  vtkTypeMacro(vtkOpenIGTLinkDaemonLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData ){};

  // Description: Get/Set MRML node
  vtkGetObjectMacro (OpenIGTLinkDaemonNode, vtkMRMLOpenIGTLinkDaemonNode);
  vtkSetObjectMacro (OpenIGTLinkDaemonNode, vtkMRMLOpenIGTLinkDaemonNode);
  
  void Apply();
  
protected:
  vtkOpenIGTLinkDaemonLogic();
  ~vtkOpenIGTLinkDaemonLogic();
  vtkOpenIGTLinkDaemonLogic(const vtkOpenIGTLinkDaemonLogic&);
  void operator=(const vtkOpenIGTLinkDaemonLogic&);

  vtkMRMLOpenIGTLinkDaemonNode* OpenIGTLinkDaemonNode;
  vtkITKOpenIGTLinkDaemon* OpenIGTLinkDaemon;


};

#endif

