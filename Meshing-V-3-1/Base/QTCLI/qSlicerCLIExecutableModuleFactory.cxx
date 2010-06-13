/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/

// Qt includes
#include <QStringList>

// SlicerQt includes
#include "qSlicerCLIExecutableModuleFactory.h"
#include "qSlicerCLIModule.h"
#include "qSlicerCLIModuleFactoryHelper.h"

//-----------------------------------------------------------------------------
qSlicerCLIExecutableModuleFactoryItem::qSlicerCLIExecutableModuleFactoryItem(const QString& itemKey,
  const QString& itemPath):Superclass(itemKey),Path(itemPath)
{
}

//-----------------------------------------------------------------------------
bool qSlicerCLIExecutableModuleFactoryItem::load()
{
  return false;
}

//-----------------------------------------------------------------------------
QString qSlicerCLIExecutableModuleFactoryItem::path()
{
  return this->Path;
}

//-----------------------------------------------------------------------------
qSlicerAbstractModule* qSlicerCLIExecutableModuleFactoryItem::instanciator()
{
  qDebug() << "CmdLineExecutableModuleItem::instantiate - name:" << this->path();
  return 0;
}

//-----------------------------------------------------------------------------
class qSlicerCLIExecutableModuleFactoryPrivate:public ctkPrivate<qSlicerCLIExecutableModuleFactory>
{
public:
  CTK_DECLARE_PUBLIC(qSlicerCLIExecutableModuleFactory);
  qSlicerCLIExecutableModuleFactoryPrivate()
    {
    }
};

//-----------------------------------------------------------------------------
qSlicerCLIExecutableModuleFactory::qSlicerCLIExecutableModuleFactory():Superclass()
{
  CTK_INIT_PRIVATE(qSlicerCLIExecutableModuleFactory);
}

//-----------------------------------------------------------------------------
void qSlicerCLIExecutableModuleFactory::registerItems()
{
  
}

//-----------------------------------------------------------------------------
// QString qSlicerCLIExecutableModuleFactory::objectNameToKey(const QString& objectName)
// {
//   return Self::extractModuleName(objectName);
// }

//-----------------------------------------------------------------------------
QString qSlicerCLIExecutableModuleFactory::extractModuleName(const QString& executableName)
{
  QString moduleName = executableName;

  // Remove extension if needed
  int index = moduleName.indexOf(".");
  if (index > 0)
    {
    moduleName.truncate(index);
    }

  return moduleName.toLower();
}
