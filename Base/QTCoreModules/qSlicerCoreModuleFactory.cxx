/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/

#include "qSlicerCoreModuleFactory.h"

// SlicerQT/CoreModules
#include "qSlicerTransformsModule.h"
#include "qSlicerCamerasModule.h"
// FIXME:Move the following to the Models module (when it will be ready in Qt.)
#include "qSlicerCoreApplication.h"
#include "qSlicerCoreIOManager.h"
#include "qSlicerModelsIO.h"
// endofFIXME
  
//-----------------------------------------------------------------------------
class qSlicerCoreModuleFactoryPrivate:public qCTKPrivate<qSlicerCoreModuleFactory>
{
public:
  QCTK_DECLARE_PUBLIC(qSlicerCoreModuleFactory);
  qSlicerCoreModuleFactoryPrivate(){}

  ///
  /// Add a module class to the core module factory
  template<typename ClassType>
  void registerCoreModule();
};

//-----------------------------------------------------------------------------
qSlicerCoreModuleFactory::qSlicerCoreModuleFactory():Superclass()
{
  QCTK_INIT_PRIVATE(qSlicerCoreModuleFactory);
}

//-----------------------------------------------------------------------------
void qSlicerCoreModuleFactory::registerItems()
{
  QCTK_D(qSlicerCoreModuleFactory);
  d->registerCoreModule<qSlicerTransformsModule>();
  d->registerCoreModule<qSlicerCamerasModule>();
  // FIXME:Move the following to the Models module (when it will be ready in Qt.)
  qSlicerCoreApplication::application()->coreIOManager()
    ->registerIO(new qSlicerModelsIO(0));
  qSlicerCoreApplication::application()->coreIOManager()
    ->registerIO(new qSlicerScalarOverlayIO(0));
  // endofFIXME
}

//-----------------------------------------------------------------------------
// qSlicerModuleFactoryPrivate methods

//-----------------------------------------------------------------------------
template<typename ClassType>
void qSlicerCoreModuleFactoryPrivate::registerCoreModule()
{
  QCTK_P(qSlicerCoreModuleFactory);
  
  QString _moduleName;
  if (!p->registerQObject<ClassType>(_moduleName))
    {
    qDebug() << "Failed to register module: " << _moduleName; 
    return;
    }
}
