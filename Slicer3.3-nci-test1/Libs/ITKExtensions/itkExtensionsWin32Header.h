/*=========================================================================

  Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   ITKExtensions
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/NCI/Slicer3.3-nci-test1/Libs/ITKExtensions/itkExtensionsWin32Header.h $
  Date:      $Date: 2006-12-21 07:33:20 -0500 (Thu, 21 Dec 2006) $
  Version:   $Revision: 1901 $

==========================================================================*/

// .NAME itkExtensionsWin32Header - manage Windows system differences
// .SECTION Description
// The itkExtensionsWin32Header captures some system differences between Unix
// and Windows operating systems. 

#ifndef __itkExtensionsWin32Header_h
#define __itkExtensionsWin32Header_h

#include <itkExtensionsConfigure.h>

#if defined(WIN32) && !defined(ITKExtensions_STATIC)
#if defined(Extensions_EXPORTS)
#define Extensions_EXPORT __declspec( dllexport ) 
#else
#define Extensions_EXPORT __declspec( dllimport ) 
#endif
#else
#define Extensions_EXPORT
#endif

#endif
