/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkOpenIGTLinkDaemonWin32Header.h,v $
  Date:      $Date: 2006/01/06 17:56:51 $
  Version:   $Revision: 1.4 $

=========================================================================auto=*/
// .NAME vtkOpenIGTLinkDaemonWin32Header - manage Windows system differences
// .SECTION Description
// The vtkOpenIGTLinkDaemonWin32Header captures some system differences between Unix
// and Windows operating systems. 

#ifndef __vtkOpenIGTLinkDaemonWin32Header_h
#define __vtkOpenIGTLinkDaemonWin32Header_h

#include <vtkOpenIGTLinkDaemonConfigure.h>

#if defined(WIN32) && !defined(VTKSLICER_STATIC)
#if defined(OpenIGTLinkDaemon_EXPORTS)
#define VTK_OPENIGTLINKDAEMON_EXPORT __declspec( dllexport ) 
#else
#define VTK_OPENIGTLINKDAEMON __declspec( dllimport ) 
#endif
#else
#define VTK_OPENIGTLINKDAEMON_EXPORT 
#endif

#endif
