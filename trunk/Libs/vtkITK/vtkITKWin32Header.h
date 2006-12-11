/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   vtkITK
  Module:    $RCSfile: vtkITKWin32Header.h,v $
  Date:      $Date: 2006/01/06 17:56:51 $
  Version:   $Revision: 1.4 $

=========================================================================auto=*/
// .NAME vtkITKWin32Header - manage Windows system differences
// .SECTION Description
// The vtkITKWin32Header captures some system differences between Unix
// and Windows operating systems. 

#ifndef __vtkITKWin32Header_h
#define __vtkITKWin32Header_h

#include <vtkITKConfigure.h>

#if defined(WIN32) && !defined(VTKITK_STATIC)
#if defined(vtkITK_EXPORTS)
#define VTK_ITK_EXPORT __declspec( dllexport ) 
#else
#define VTK_ITK_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_ITK_EXPORT
#endif

#endif
