/*=========================================================================

  Library:   qCTK

  Copyright (c) Kitware Inc. 
  All rights reserved.
  Distributed under a BSD License. See LICENSE.txt file.

  This software is distributed "AS IS" WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the above copyright notice for more information.

=========================================================================*/
/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   Module Description Parser

=========================================================================auto=*/

#ifndef __qCTKBinaryFileDescriptor_h
#define __qCTKBinaryFileDescriptor_h

// QT includes
#include <QString>

// qCTK includes
#include "qCTKPimpl.h"

#include "qCTKWidgetsExport.h"

/// Allows to resolve global symbols contained into an executable.
/// Implementation valid only for unix-like systems (Linux, Mac, ...)

class qCTKBinaryFileDescriptorPrivate;

class QCTK_WIDGETS_EXPORT qCTKBinaryFileDescriptor
{
public:
  qCTKBinaryFileDescriptor();
  qCTKBinaryFileDescriptor(const QString& _fileName);
  virtual ~qCTKBinaryFileDescriptor();

  QString fileName()const;
  void setFileName(const QString& _fileName);

  /// Load the object file containing the symbols
  bool load();

  /// Unload / close the object file
  bool unload();

  bool isLoaded() const;

  /// Get the address of a symbol in memory
  void* resolve(const char * symbol);

private:
  QCTK_DECLARE_PRIVATE(qCTKBinaryFileDescriptor);

};

#endif
