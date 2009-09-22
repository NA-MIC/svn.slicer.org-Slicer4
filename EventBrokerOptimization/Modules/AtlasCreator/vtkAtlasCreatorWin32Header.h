/*=auto=======================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights
  Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkAtlasCreatorWin32Header.h,v $
  Date:      $Date: 2006/01/06 17:56:51 $
  Version:   $Revision: 1.4 $
  Author:    $Sylvain Jaume (MIT)$

=======================================================================auto=*/

// .NAME vtkAtlasCreatorWin32Header - manage Windows system
// differences
// .SECTION Description
// The vtkAtlasCreatorWin32Header captures some system
// differences between Unix and Windows operating systems.

#ifndef __vtkAtlasCreatorWin32Header_h
#define __vtkAtlasCreatorWin32Header_h

#include <vtkAtlasCreatorConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(AtlasCreator_EXPORTS)
#define VTK_AtlasCreator_EXPORT __declspec( dllexport )
#else
#define VTK_AtlasCreator_EXPORT __declspec( dllimport )
#endif
#else
#define VTK_AtlasCreator_EXPORT
#endif

#endif

