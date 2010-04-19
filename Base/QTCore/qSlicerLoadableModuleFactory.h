/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/


#ifndef __qSlicerLoadableModuleFactory_h
#define __qSlicerLoadableModuleFactory_h

/// SlicerQT includes
#include "qSlicerAbstractModule.h"

/// CTK includes
#include <qCTKPimpl.h>
#include <ctkAbstractPluginFactory.h>

#include "qSlicerBaseQTCoreExport.h"

class qSlicerLoadableModuleFactoryPrivate;

class Q_SLICER_BASE_QTCORE_EXPORT qSlicerLoadableModuleFactory :
  public ctkAbstractPluginFactory<qSlicerAbstractModule>
{
public:

  typedef ctkAbstractPluginFactory<qSlicerAbstractModule> Superclass;
  qSlicerLoadableModuleFactory();
  virtual ~qSlicerLoadableModuleFactory(){}

  virtual void registerItems();

  /// Extract module name given a library name
  /// See qSlicerUtils::extractModuleNameFromLibraryName
  static QString extractModuleName(const QString& libraryName);


private:
  QCTK_DECLARE_PRIVATE(qSlicerLoadableModuleFactory);
};

#endif
