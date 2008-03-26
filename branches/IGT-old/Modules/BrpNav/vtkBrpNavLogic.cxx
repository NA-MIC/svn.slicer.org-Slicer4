/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkBrpNavLogic.h"

#include "vtkLandmarkTransform.h"
#include "vtkCylinderSource.h"

#ifndef IGSTK_OFF
#include "igstkAuroraTracker.h"
#endif

vtkCxxRevisionMacro(vtkBrpNavLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkBrpNavLogic);

vtkBrpNavLogic::vtkBrpNavLogic()
{
#ifndef IGSTK_OFF
  igstk::RealTimeClock::Initialize();
#endif
}



vtkBrpNavLogic::~vtkBrpNavLogic()
{

}



void vtkBrpNavLogic::PrintSelf(ostream& os, vtkIndent indent)
{
    this->vtkObject::PrintSelf(os, indent);

    os << indent << "vtkBrpNavLogic:             " << this->GetClassName() << "\n";

}

