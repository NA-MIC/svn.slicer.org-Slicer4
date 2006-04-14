/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerLogic.cxx,v $
  Date:      $Date: 2006/01/06 17:56:48 $
  Version:   $Revision: 1.58 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkSlicerLogic.h"

#include "vtkCallbackCommand.h"

vtkCxxRevisionMacro(vtkSlicerLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerLogic);

//----------------------------------------------------------------------------
vtkSlicerLogic::vtkSlicerLogic()
{
  this->MRMLScene = NULL;

  this->MRMLCallbackCommand = vtkCallbackCommand::New();
  this->MRMLCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
  this->MRMLCallbackCommand->SetCallback(vtkSlicerLogic::MRMLCallback);

  this->LogicCallbackCommand = vtkCallbackCommand::New();
  this->LogicCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
  this->LogicCallbackCommand->SetCallback(vtkSlicerLogic::LogicCallback);
}

//----------------------------------------------------------------------------
vtkSlicerLogic::~vtkSlicerLogic()
{
  if (this->MRMLScene)
    {
    this->MRMLScene->Delete();
    this->MRMLScene = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);

  os << indent << "SlicerLogic:             " << this->GetClassName() << "\n";
  os << indent << "MRMLScene: " << this->GetMRMLScene() << "\n";
}


//----------------------------------------------------------------------------
// Description:
// the MRMLCallback is a static function to relay modified events from the 
// observed mrml node back into the gui layer for further processing
//
void 
vtkSlicerLogic::MRMLCallback(vtkObject *__mrmlnode, 
            unsigned long eid, void *__clientData, void *callData)
{
  static int inMRMLCallback = 0;

  vtkSlicerLogic *self = reinterpret_cast<vtkSlicerLogic *>(__clientData);

  if (inMRMLCallback)
    {
    vtkErrorWithObjectMacro(self, "In vtkSlicerLogic *********MRMLCallback called recursively?");
    return;
    }

  vtkDebugWithObjectMacro(self, "In vtkSlicerLogic MRMLCallback");

  inMRMLCallback = 1;
  self->ProcessMRMLEvents();
  inMRMLCallback = 0;
}

//----------------------------------------------------------------------------
// Description:
// the LogicCallback is a static function to relay modified events from the 
// observed mrml node back into the gui layer for further processing
//
void 
vtkSlicerLogic::LogicCallback(vtkObject *__mrmlnode, 
            unsigned long eid, void *__clientData, void *callData)
{
  static int inLogicCallback = 0;

  vtkSlicerLogic *self = reinterpret_cast<vtkSlicerLogic *>(__clientData);

  if (inLogicCallback)
    {
    vtkErrorWithObjectMacro(self, "In vtkSlicerLogic *********LogicCallback called recursively?");
    return;
    }

  vtkDebugWithObjectMacro(self, "In vtkSlicerLogic LogicCallback");

  inLogicCallback = 1;
  self->ProcessLogicEvents();
  inLogicCallback = 0;
}
