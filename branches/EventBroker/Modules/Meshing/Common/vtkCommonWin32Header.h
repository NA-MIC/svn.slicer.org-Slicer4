/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkCommonWin32Header.h,v $
  Date:      $Date: 2006/01/06 17:56:51 $
  Version:   $Revision: 1.4 $

=========================================================================auto=*/
// .NAME vtkCommonWin32Header - manage Windows system differences
// .SECTION Description
// The vtkCommonWin32Header captures some system differences between Unix
// and Windows operating systems. 

#ifndef __vtkCommonWin32Header_h
#define __vtkCommonWin32Header_h

#include <vtkCommonConfigure.h>

#if defined(WIN32) && !defined(VTKMIMXCommon_STATIC)
#if defined(mimxCommon_EXPORTS)
#define VTK_MIMXCOMMON_EXPORT __declspec( dllexport ) 
#else
#define VTK_MIMXCOMMON_EXPORT __declspec( dllimport ) 
#endif
#else
#define VTK_MIMXCOMMON_EXPORT 
#endif

#endif
