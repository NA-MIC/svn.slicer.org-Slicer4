/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

// .NAME vtkBrpNavLogic - slicer logic class for Locator module 
// .SECTION Description
// This class manages the logic associated with tracking device for
// IGT. 


#ifndef __vtkBrpNavLogic_h
#define __vtkBrpNavLogic_h

#include "vtkBrpNavWin32Header.h"
#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"


class VTK_BRPNAV_EXPORT vtkBrpNavLogic : public vtkSlicerLogic 
{
public:

    // The Usual vtk class functions
    static vtkBrpNavLogic *New();
    vtkTypeRevisionMacro(vtkBrpNavLogic,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);


protected:

    vtkBrpNavLogic();
    ~vtkBrpNavLogic();
    vtkBrpNavLogic(const vtkBrpNavLogic&);
    void operator=(const vtkBrpNavLogic&);

};

#endif


  
