/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// Qt includes
#include <QDirIterator>
#include <QSettings>
#include <QStringList>

// SlicerQt includes
#include "qSlicerLoadableModuleFactory.h"
#include "qSlicerCoreApplication.h"
#include "qSlicerUtils.h"

// For:
//  - Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR
#include "vtkSlicerConfigure.h"
  
//-----------------------------------------------------------------------------
class qSlicerLoadableModuleFactoryPrivate
{
public:
  ///
  /// Return a list of module paths
  QStringList modulePaths() const;
};

//-----------------------------------------------------------------------------
// qSlicerLoadableModuleFactoryPrivate Methods

//-----------------------------------------------------------------------------
QStringList qSlicerLoadableModuleFactoryPrivate::modulePaths() const
{
  qSlicerCoreApplication* app = qSlicerCoreApplication::application();
  Q_ASSERT(app);
  Q_ASSERT(!app->slicerHome().isEmpty());
  
  // On Win32, *both* paths have to be there, since scripts are installed
  // in the install location, and exec/libs are *automatically* installed
  // in intDir.
  QStringList defaultQTModulePaths;
  defaultQTModulePaths << app->slicerHome() + "/"
                                             + Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR;

   if (!app->intDir().isEmpty())
     {
     defaultQTModulePaths << app->slicerHome() + "/" + Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR + "/" + app->intDir();
     }

  QStringList userModulePaths = QSettings().value("Modules/Extensions").toStringList();
  QStringList qtModulePaths =  userModulePaths + defaultQTModulePaths;
  foreach(const QString& path, qtModulePaths)
    {
    app->addLibraryPath(path);
    }

  //qDebug() << "qtModulePaths:" << qtModulePaths;
  
  return qtModulePaths; 
}

//-----------------------------------------------------------------------------
// qSlicerLoadableModuleFactory Methods

//-----------------------------------------------------------------------------
qSlicerLoadableModuleFactory::qSlicerLoadableModuleFactory()
  : d_ptr(new qSlicerLoadableModuleFactoryPrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerLoadableModuleFactory::~qSlicerLoadableModuleFactory()
{
}

//-----------------------------------------------------------------------------
void qSlicerLoadableModuleFactory::registerItems()
{
  Q_D(qSlicerLoadableModuleFactory);

  this->registerAllFileItems(d->modulePaths());
}

//-----------------------------------------------------------------------------
QString qSlicerLoadableModuleFactory::fileNameToKey(const QString& fileName)const
{
  return qSlicerLoadableModuleFactory::extractModuleName(fileName);
}

//-----------------------------------------------------------------------------
QString qSlicerLoadableModuleFactory::extractModuleName(const QString& libraryName)
{
  return qSlicerUtils::extractModuleNameFromLibraryName(libraryName);
}
