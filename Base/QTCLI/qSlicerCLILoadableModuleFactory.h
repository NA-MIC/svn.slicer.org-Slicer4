/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/


#ifndef __qSlicerCLILoadableModuleFactory_h
#define __qSlicerCLILoadableModuleFactory_h

/// SlicerQT includes
#include "qSlicerAbstractModule.h"
#include "qSlicerBaseQTCLIExport.h"

/// CTK includes
#include <qCTKPimpl.h>
#include <ctkAbstractLibraryFactory.h>

//-----------------------------------------------------------------------------
class qSlicerCLILoadableModuleFactoryItem : public ctkFactoryLibraryItem<qSlicerAbstractModule>
{
public:
  // Convenient typedef
  typedef ctkFactoryLibraryItem<qSlicerAbstractModule> Superclass;
  
  explicit qSlicerCLILoadableModuleFactoryItem(const QString& itemKey, const QString& itemPath);
  virtual ~qSlicerCLILoadableModuleFactoryItem(){}

protected:
  /// Convenient typedef
  typedef qSlicerCLILoadableModuleFactoryItem Self;
  //typedef char * (*XMLModuleDescriptionFunction)();

  virtual qSlicerAbstractModule* instanciator();
};

//-----------------------------------------------------------------------------
class qSlicerCLILoadableModuleFactoryPrivate;

//-----------------------------------------------------------------------------
class Q_SLICER_BASE_QTCLI_EXPORT qSlicerCLILoadableModuleFactory :
  public ctkAbstractLibraryFactory<qSlicerAbstractModule,qSlicerCLILoadableModuleFactoryItem>
{
public:

  typedef ctkAbstractLibraryFactory<qSlicerAbstractModule,
                                     qSlicerCLILoadableModuleFactoryItem> Superclass;
  qSlicerCLILoadableModuleFactory();
  virtual ~qSlicerCLILoadableModuleFactory(){}

  virtual void registerItems();

  /// Extract module name given a library name
  /// See qSlicerUtils::extractModuleNameFromLibraryName
  static QString extractModuleName(const QString& libraryName);

private:
  QCTK_DECLARE_PRIVATE(qSlicerCLILoadableModuleFactory);
};

#endif
