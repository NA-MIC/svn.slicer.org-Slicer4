/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
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
#include <QList>
#include <QSettings>
#include <QSplashScreen>
#include <QString>
#include <QTimer>
#include <QTranslator>

// Slicer includes
#include "vtkSlicerConfigure.h" // For Slicer_USE_PYTHONQT Slicer_QM_OUTPUT_DIRS, Slicer_INSTALL_QM_DIR

// CTK includes
#include <ctkAbstractLibraryFactory.h>
#ifdef Slicer_USE_PYTHONQT
# include <ctkPythonConsole.h>
#endif

// MRMLWidgets includes
#include <qMRMLEventLoggerWidget.h>

// Slicer includes
#include "vtkSlicerVersionConfigure.h" // For Slicer_VERSION_FULL, Slicer_BUILD_CLI_SUPPORT

// SlicerApp includes
#include "qSlicerApplication.h"
#include "qSlicerApplicationHelper.h"
#ifdef Slicer_BUILD_CLI_SUPPORT
# include "qSlicerCLIExecutableModuleFactory.h"
# include "qSlicerCLILoadableModuleFactory.h"
#endif
#include "qSlicerAppMainWindow.h"
#include "qSlicerCommandOptions.h"
#include "qSlicerModuleFactoryManager.h"
#include "qSlicerModuleManager.h"
#include "qSlicerStyle.h"

// VTK includes
//#include <vtkObject.h>

#ifdef Slicer_USE_PYTHONQT
# include <PythonQtObjectPtr.h>
# include <PythonQtPythonInclude.h>
# include "qSlicerPythonManager.h"
# include "qSlicerSettingsPythonPanel.h"
#endif

#if defined (_WIN32) && !defined (Slicer_BUILD_WIN32_CONSOLE)
# include <windows.h>
# include <vtksys/SystemTools.hxx>
#endif

namespace
{

#ifdef Slicer_USE_QtTesting
//-----------------------------------------------------------------------------
void setEnableQtTesting()
{
  if (qSlicerApplication::application()->commandOptions()->enableQtTesting() ||
      qSlicerApplication::application()->settings()->value("QtTesting/Enabled").toBool())
    {
    QCoreApplication::setAttribute(Qt::AA_DontUseNativeMenuBar);
    }
}
#endif

#ifdef Slicer_USE_PYTHONQT

//----------------------------------------------------------------------------
void initializePythonConsole(ctkPythonConsole& pythonConsole)
{
  // Create python console
  Q_ASSERT(qSlicerApplication::application()->pythonManager());
  pythonConsole.initialize(qSlicerApplication::application()->pythonManager());

  QStringList autocompletePreferenceList;
  autocompletePreferenceList
      << "slicer" << "slicer.mrmlScene"
      << "qt.QPushButton";
  pythonConsole.completer()->setAutocompletePreferenceList(autocompletePreferenceList);

  //pythonConsole.setAttribute(Qt::WA_QuitOnClose, false);
  pythonConsole.resize(600, 280);

  qSlicerApplication::application()->settingsDialog()->addPanel(
    "Python", new qSlicerSettingsPythonPanel);

  // Show pythonConsole if required
  qSlicerCommandOptions * options = qSlicerApplication::application()->commandOptions();
  if(options->showPythonInteractor() && !options->runPythonAndExit())
    {
    pythonConsole.show();
    pythonConsole.activateWindow();
    pythonConsole.raise();
    }
}
#endif

//----------------------------------------------------------------------------
void showMRMLEventLoggerWidget()
{
  qMRMLEventLoggerWidget* logger = new qMRMLEventLoggerWidget(0);
  logger->setAttribute(Qt::WA_DeleteOnClose);
  logger->setConsoleOutputEnabled(false);
  logger->setMRMLScene(qSlicerApplication::application()->mrmlScene());

  QObject::connect(qSlicerApplication::application(),
                   SIGNAL(mrmlSceneChanged(vtkMRMLScene*)),
                   logger,
                   SLOT(setMRMLScene(vtkMRMLScene*)));

  logger->show();
}

//----------------------------------------------------------------------------
void splashMessage(QScopedPointer<QSplashScreen>& splashScreen, const QString& message)
{
  if (splashScreen.isNull())
    {
    return;
    }
  splashScreen->showMessage(message, Qt::AlignBottom | Qt::AlignHCenter);
  //splashScreen->repaint();
}

//----------------------------------------------------------------------------
void loadTranslations(const QString& dir)
{
  qSlicerApplication * app = qSlicerApplication::application();
  Q_ASSERT(app);

  QString localeFilter =
      QString( QString("*") + app->settings()->value("language").toString());
  localeFilter.resize(3);
  localeFilter += QString(".qm");

  QDir directory(dir);
  QStringList qmFiles = directory.entryList(QStringList(localeFilter));

  foreach(QString qmFile, qmFiles)
    {
    QTranslator* translator = new QTranslator();
    QString qmFilePath = QString(dir + QString("/") + qmFile);

    if(!translator->load(qmFilePath))
      {
      qDebug() << "The File " << qmFile << " hasn't been loaded in the translator";
      return;
      }
    app->installTranslator(translator);
    }
}

//----------------------------------------------------------------------------
void loadLanguage()
{
  qSlicerApplication * app = qSlicerApplication::application();
  Q_ASSERT(app);

  // we check if the application is installed or not.
  if (app->isInstalled())
    {
    QString qmDir = QString(Slicer_QM_DIR);
    loadTranslations(qmDir);
    }
  else
    {
    QStringList qmDirs = QString(Slicer_QM_OUTPUT_DIRS).split(";");
    foreach(QString qmDir, qmDirs)
      {
      loadTranslations(qmDir);
      }
    }
}

//----------------------------------------------------------------------------
int SlicerAppMain(int argc, char* argv[])
{
  QCoreApplication::setApplicationName("Slicer");
  QCoreApplication::setApplicationVersion(Slicer_VERSION_FULL);
  //vtkObject::SetGlobalWarningDisplay(false);
  QApplication::setDesktopSettingsAware(false);
  QApplication::setStyle(new qSlicerStyle);

  qSlicerApplication app(argc, argv);
  if (app.returnCode() != -1)
    {
    return app.returnCode();
    }

  // We load the language selected for the application
  loadLanguage();

#ifdef Slicer_USE_QtTesting
  setEnableQtTesting(); // disabled the native menu bar.
#endif

#ifdef Slicer_USE_PYTHONQT
  ctkPythonConsole pythonConsole;
  pythonConsole.setWindowTitle("Slicer Python Interactor");
  if (!qSlicerApplication::testAttribute(qSlicerApplication::AA_DisablePython))
    {
    initializePythonConsole(pythonConsole);
    }
#endif

  bool enableMainWindow = !app.commandOptions()->noMainWindow();
  enableMainWindow = enableMainWindow && !app.commandOptions()->runPythonAndExit();
  bool showSplashScreen = !app.commandOptions()->noSplash() && enableMainWindow;

  QScopedPointer<QSplashScreen> splashScreen;
  if (showSplashScreen)
    {
    QPixmap pixmap(":/SplashScreen.png");
    splashScreen.reset(new QSplashScreen(pixmap));
    splashMessage(splashScreen, "Initializing...");
    splashScreen->show();
    }

  qSlicerModuleManager * moduleManager = qSlicerApplication::application()->moduleManager();
  qSlicerModuleFactoryManager * moduleFactoryManager = moduleManager->factoryManager();
  moduleFactoryManager->addSearchPaths(app.commandOptions()->additonalModulePaths());
  qSlicerApplicationHelper::setupModuleFactoryManager(moduleFactoryManager);

  // Register and instantiate modules
  splashMessage(splashScreen, "Registering modules...");
  moduleFactoryManager->registerModules();
  qDebug() << "Number of registered modules:"
           << moduleFactoryManager->registeredModuleNames().count();
  splashMessage(splashScreen, "Instantiating modules...");
  moduleFactoryManager->instantiateModules();
  qDebug() << "Number of instantiated modules:"
           << moduleFactoryManager->instantiatedModuleNames().count();
  // Create main window
  splashMessage(splashScreen, "Initializing user interface...");
  QScopedPointer<qSlicerAppMainWindow> window;
  if (enableMainWindow)
    {
    window.reset(new qSlicerAppMainWindow);
    window->setWindowTitle(window->windowTitle()+ " " + Slicer_VERSION_FULL);
    }

  // Load all available modules
  foreach(const QString& name, moduleFactoryManager->instantiatedModuleNames())
    {
    Q_ASSERT(!name.isNull());
    splashMessage(splashScreen, "Loading module \"" + name + "\"...");
    moduleFactoryManager->loadModule(name);
    }
  qDebug() << "Number of loaded modules:" << moduleManager->modulesNames().count();

  splashMessage(splashScreen, QString());

  if (window)
    {
    window->setHomeModuleCurrent();
    window->show();
    }

  if (splashScreen && window)
    {
    splashScreen->finish(window.data());
    }

  // Process command line argument after the event loop is started
  QTimer::singleShot(0, &app, SLOT(handleCommandLineArguments()));

  // showMRMLEventLoggerWidget();

  // Look at QApplication::exec() documentation, it is recommended to connect
  // clean up code to the aboutToQuit() signal
  return app.exec();
}

} // end of anonymous namespace

#if defined (_WIN32) && !defined (Slicer_BUILD_WIN32_CONSOLE)
int __stdcall WinMain(HINSTANCE hInstance,
                      HINSTANCE hPrevInstance,
                      LPSTR lpCmdLine, int nShowCmd)
{
  Q_UNUSED(hInstance);
  Q_UNUSED(hPrevInstance);
  Q_UNUSED(nShowCmd);

  int argc;
  char **argv;
  vtksys::SystemTools::ConvertWindowsCommandLineToUnixArguments(
    lpCmdLine, &argc, &argv);

  int ret = SlicerAppMain(argc, argv);

  for (int i = 0; i < argc; i++)
    {
    delete [] argv[i];
    }
  delete [] argv;

  return ret;
}
#else
int main(int argc, char *argv[])
{
  return SlicerAppMain(argc, argv);
}
#endif
