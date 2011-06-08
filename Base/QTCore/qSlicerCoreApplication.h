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

#ifndef __qSlicerCoreApplication_h
#define __qSlicerCoreApplication_h

// Qt includes
#include <QApplication>

// CTK includes
#include <ctkVTKObject.h>

// SlicerCore includes
#include "vtkSlicerConfigure.h" // For Slicer_USE_PYTHONQT
#include "qSlicerBaseQTCoreExport.h"

class ctkErrorLogModel;
class QSettings;
class qSlicerCoreIOManager;
class qSlicerCoreCommandOptions;
class qSlicerCoreApplicationPrivate;
class qSlicerModuleManager;
#ifdef Slicer_USE_PYTHONQT
class qSlicerCorePythonManager;
#endif
class vtkSlicerApplicationLogic;
class vtkMRMLApplicationLogic;
class vtkMRMLScene;

class Q_SLICER_BASE_QTCORE_EXPORT qSlicerCoreApplication : public QApplication
{
  Q_OBJECT
  QVTK_OBJECT
public:

  typedef QApplication Superclass;
  qSlicerCoreApplication(int &argc, char **argv);
  virtual ~qSlicerCoreApplication();

  /// Return a reference to the application singleton
  static qSlicerCoreApplication* application();

  enum ApplicationAttribute
    {
    AA_DisablePython = 1000,
    AA_EnableTesting
    };

  /// \sa QCoreApplication::setAttribute
  static void setAttribute(qSlicerCoreApplication::ApplicationAttribute attribute, bool on = true);

  /// \sa QCoreApplication::testAttribute
  static bool testAttribute(qSlicerCoreApplication::ApplicationAttribute attribute);

  /// Parse arguments
  /// \note If exitWhenDone is True, it's your responsability to exit the application
  void parseArguments(bool& exitWhenDone);

  /// Get MRML Scene
  Q_INVOKABLE vtkMRMLScene* mrmlScene() const;

  /// Get application logic
  Q_INVOKABLE vtkSlicerApplicationLogic* appLogic() const;

  /// Get MRML ApplicationLogic
  Q_INVOKABLE vtkMRMLApplicationLogic* mrmlApplicationLogic() const;

  /// Get slicer home directory
  Q_INVOKABLE QString slicerHome() const;

  /// Get slicer temporary directory
  Q_INVOKABLE QString temporaryPath() const;

  /// Set slicer temporary directory
  Q_INVOKABLE void setTemporaryPath(const QString& path);
  
  /// Get slicer extension directory
  Q_INVOKABLE QString extensionsPath() const;

  /// Set slicer extension directory
  Q_INVOKABLE void setExtensionsPath(const QString& path);

  /// If any, this method return the build intermediate directory
  /// See $(IntDir) on http://msdn.microsoft.com/en-us/library/c02as0cs%28VS.71%29.aspx
  Q_INVOKABLE QString intDir()const;

  /// Return true is this instance of Slicer is running from an installed directory
  Q_INVOKABLE bool isInstalled()const;

#ifdef Slicer_USE_PYTHONQT
  /// Get python manager
  qSlicerCorePythonManager* corePythonManager()const;
  
  /// Set the IO manager
  /// \note qSlicerCoreApplication takes ownership of the object
  void setCorePythonManager(qSlicerCorePythonManager* pythonManager);
#endif

  /// Get the module manager
  Q_INVOKABLE qSlicerModuleManager* moduleManager()const;

  /// Get the IO manager
  Q_INVOKABLE qSlicerCoreIOManager* coreIOManager()const;

  /// Set the IO manager
  /// \note qSlicerCoreApplication takes ownership of the object
  void setCoreIOManager(qSlicerCoreIOManager* ioManager);

  /// Get coreCommandOptions
  qSlicerCoreCommandOptions* coreCommandOptions()const;

  /// Set coreCommandOptions
  /// \note qSlicerCoreApplication takes ownership of the object
  void setCoreCommandOptions(qSlicerCoreCommandOptions* options);

  /// Get errorLogModel
  Q_INVOKABLE ctkErrorLogModel* errorLogModel()const;

  /// Get application settings
  /// \note It will also instantiate a QSettings object if required.
  Q_INVOKABLE QSettings* settings()const;

  /// Disable application settings
  /// Instanciate a new empty ctkSettings object and associate it with this instance.
  /// \note The original settings won't deleted.
  void disableSettings();

  /// Clear application settings
  void clearSettings();

  /// Return the copyrights of Slicer
  virtual QString copyrights()const;

  /// Return the source repository URL associated to this build
  /// \sa discoverRepository
  Q_INVOKABLE QString repositoryUrl()const;

  /// Return the source repository Branch associated to this build
  /// \sa discoverRepository
  Q_INVOKABLE QString repositoryBranch()const;

  /// Return the source repository Revision associated to this build
  /// \sa discoverRepository
  Q_INVOKABLE QString repositoryRevision()const;

  /// Return the source repository Platform associated to this build
  /// \sa discoverRepository
  Q_INVOKABLE QString platform()const;

public slots:

  /// Restart the application with the arguments passed at startup time
  /// \sa QCoreApplication::arguments()
  static void restart();

protected:
  ///
  virtual void handlePreApplicationCommandLineArguments();
  /// If fileName is set (not empty), a custom fileName is used
  /// otherwise it uses the default QSettings constructor.
  /// \sa QSettings::QSettings(QObject* parent)
  virtual QSettings* newSettings(const QString& fileName = QString());

  /// Set MRML Scene
  virtual void setMRMLScene(vtkMRMLScene * scene);

protected slots:

  ///
  virtual void handleCommandLineArguments();
  void onSlicerApplicationLogicRequest(vtkObject*, void* , unsigned long);
  void processAppLogicModified();
  void processAppLogicReadData();
  void processAppLogicWriteData();

signals:
  void mrmlSceneChanged(vtkMRMLScene* mrmlScene);

protected:
  qSlicerCoreApplication(qSlicerCoreApplicationPrivate* pimpl, int &argc, char **argv);
  QScopedPointer<qSlicerCoreApplicationPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerCoreApplication);
  Q_DISABLE_COPY(qSlicerCoreApplication);
};

#endif
