/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

// .NAME vtkProstateNavLogic - slicer logic class for Locator module 
// .SECTION Description
// This class manages the logic associated with tracking device for
// IGT. 


#ifndef __vtkProstateNavLogic_h
#define __vtkProstateNavLogic_h

#include "vtkProstateNavWin32Header.h"
#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"


class VTK_PROSTATENAV_EXPORT vtkProstateNavLogic : public vtkSlicerLogic 
{
public:

    // The Usual vtk class functions
    static vtkProstateNavLogic *New();
    vtkTypeRevisionMacro(vtkProstateNavLogic,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);


protected:

    vtkProstateNavLogic();
    ~vtkProstateNavLogic();
    vtkProstateNavLogic(const vtkProstateNavLogic&);
    void operator=(const vtkProstateNavLogic&);

};

#endif


  
