/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/


#ifndef __qSlicerCoreModuleFactory_h
#define __qSlicerCoreModuleFactory_h

/// SlicerQT includes
#include "qSlicerAbstractModule.h"
#include "qSlicerBaseQTCoreModulesExport.h"

/// CTK includes
#include <qCTKPimpl.h>
#include <ctkAbstractQObjectFactory.h>

class qSlicerCoreModuleFactoryPrivate;

class Q_SLICER_BASE_QTCOREMODULES_EXPORT qSlicerCoreModuleFactory :
  public ctkAbstractQObjectFactory<qSlicerAbstractModule>
{
public:

  typedef ctkAbstractQObjectFactory<qSlicerAbstractModule> Superclass;
  qSlicerCoreModuleFactory();
  virtual ~qSlicerCoreModuleFactory(){}

  virtual void registerItems();

  /// Extract module name given a core module classname
  /// For example: 
  ///  qSlicerCamerasModule -> cameras
  ///  qSlicerTransformsModule -> transforms
  static QString extractModuleName(const QString& className);

private:
  QCTK_DECLARE_PRIVATE(qSlicerCoreModuleFactory);
};

#endif
