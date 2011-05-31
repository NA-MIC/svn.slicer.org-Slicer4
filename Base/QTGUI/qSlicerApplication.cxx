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
#include <QCleanlooksStyle>
#include <QColor>
#include <QDebug>
#include <QFont>
#include <QFontDatabase>
#include <QFontInfo>
#include <QMainWindow>
#include <QMap>
#include <QPalette>
#include <QPluginLoader>
#include <QRect>
#include <QStyle>
#include <QWidget>

// CTK includes
#include <ctkColorDialog.h>
#include "ctkErrorLogModel.h"
#include <ctkIconEnginePlugin.h>
#include <ctkLogger.h>
#include <ctkSettings.h>
#include <ctkToolTipTrapper.h>

// QTGUI includes
#include "qSlicerApplication.h"
#include "qSlicerCoreApplication_p.h"
#include "qSlicerWidget.h"
#include "qSlicerIOManager.h"
#include "qSlicerCommandOptions.h"
#include "qSlicerLayoutManager.h"
#include "qSlicerStyle.h"
#ifdef Slicer_USE_PYTHONQT
# include "qSlicerPythonManager.h"
#endif

// qMRMLWidget includes
#include "qMRMLColorPickerWidget.h"

// MRMLLogic includes
#include <vtkMRMLApplicationLogic.h>

//--------------------------------------------------------------------------
static ctkLogger logger("org.slicer.base.qtgui.qSlicerApplication");
//--------------------------------------------------------------------------

//-----------------------------------------------------------------------------
class qSlicerApplicationPrivate : public qSlicerCoreApplicationPrivate
{
  Q_DECLARE_PUBLIC(qSlicerApplication);
protected:
  qSlicerApplication* const q_ptr;
public:
  typedef qSlicerCoreApplicationPrivate Superclass;

  qSlicerApplicationPrivate(qSlicerApplication& object, qSlicerCommandOptions * commandOptions, qSlicerIOManager * ioManager);
  virtual ~qSlicerApplicationPrivate();

  /// Convenient method regrouping all initialization code
  virtual void init();

  /// Initialize application style
  void initStyle();

  QMap<QWidget*,bool>                 TopLevelWidgetsSavedVisibilityState;
  Qt::WindowFlags                     DefaultWindowFlags;
  qSlicerLayoutManager*               LayoutManager;
  ctkToolTipTrapper*                  ToolTipTrapper;
  QSharedPointer<qMRMLColorPickerWidget> ColorDialogPickerWidget;
};


//-----------------------------------------------------------------------------
// qSlicerApplicationPrivate methods

//-----------------------------------------------------------------------------
qSlicerApplicationPrivate::qSlicerApplicationPrivate(
  qSlicerApplication& object, qSlicerCommandOptions * commandOptions, qSlicerIOManager * ioManager)
  : qSlicerCoreApplicationPrivate(object, commandOptions, ioManager), q_ptr(&object)
{
  this->LayoutManager = 0;
  this->ToolTipTrapper = 0;
}

//-----------------------------------------------------------------------------
qSlicerApplicationPrivate::~qSlicerApplicationPrivate()
{
}

//-----------------------------------------------------------------------------
void qSlicerApplicationPrivate::init()
{
  Q_Q(qSlicerApplication);

  this->Superclass::init();

  this->initStyle();

#ifdef Slicer_USE_PYTHONQT
  if (!qSlicerCoreApplication::testAttribute(qSlicerCoreApplication::AA_DisablePython))
    {
    // Note: qSlicerCoreApplication class takes ownership of the pythonManager and
    // will be responsible to delete it
    q->setCorePythonManager(new qSlicerPythonManager());
    }
#endif

  this->ToolTipTrapper = new ctkToolTipTrapper(q);
  this->ToolTipTrapper->setEnabled(false);

  this->ColorDialogPickerWidget =
    QSharedPointer<qMRMLColorPickerWidget>(new qMRMLColorPickerWidget(0));
  ctkColorDialog::addDefaultTab(this->ColorDialogPickerWidget.data(),
                                "Labels", SIGNAL(colorSelected(const QColor&)));
}
/*
#if !defined (QT_NO_LIBRARY) && !defined(QT_NO_SETTINGS)
Q_GLOBAL_STATIC_WITH_ARGS(QFactoryLoader, loaderV2,
    (QIconEngineFactoryInterfaceV2_iid, QLatin1String("/iconengines"), Qt::CaseInsensitive))
#endif
*/
//-----------------------------------------------------------------------------
void qSlicerApplicationPrivate::initStyle()
{
  Q_Q(qSlicerApplication);

  // Force showing the icons in the menus despite the native OS style
  // discourages it
  QCoreApplication::setAttribute(Qt::AA_DontShowIconsInMenus, false);

  // Init the style of the icons
  // The plugin qSlicerIconEnginePlugin is located in the iconengines
  // subdirectory of Slicer lib dir (typically lib/Slicer3).
  // By adding the path to the lib dir, Qt automatically loads the icon engine
  // plugin
  q->addLibraryPath(q->slicerHome() + "/" + Slicer_INSTALL_LIB_DIR);
}

//-----------------------------------------------------------------------------
// qSlicerApplication methods

//-----------------------------------------------------------------------------
qSlicerApplication::qSlicerApplication(int &_argc, char **_argv)
  : Superclass(new qSlicerApplicationPrivate(*this, new qSlicerCommandOptions, 0), _argc, _argv)
{
  Q_D(qSlicerApplication);
  // Note: Since QWidget/QDialog requires a QApplication to be successfully instantiated,
  //       qSlicerIOManager is not added to the constructor initialization list.
  //       Indeed, internally qSlicerIOManager registers qSlicerDataDialog, ...
  d->CoreIOManager = QSharedPointer<qSlicerIOManager>(new qSlicerIOManager);
}

//-----------------------------------------------------------------------------
qSlicerApplication::~qSlicerApplication()
{
}

//-----------------------------------------------------------------------------
qSlicerApplication* qSlicerApplication::application()
{
  qSlicerApplication* app = qobject_cast<qSlicerApplication*>(QApplication::instance());
  return app;
}

//-----------------------------------------------------------------------------
qSlicerCommandOptions* qSlicerApplication::commandOptions()
{
  qSlicerCommandOptions* _commandOptions =
    dynamic_cast<qSlicerCommandOptions*>(this->coreCommandOptions());
  Q_ASSERT(_commandOptions);
  return _commandOptions;
}

//-----------------------------------------------------------------------------
qSlicerIOManager* qSlicerApplication::ioManager()
{
  qSlicerIOManager* _ioManager = dynamic_cast<qSlicerIOManager*>(this->coreIOManager());
  Q_ASSERT(_ioManager);
  return _ioManager;
}

#ifdef Slicer_USE_PYTHONQT
//-----------------------------------------------------------------------------
qSlicerPythonManager* qSlicerApplication::pythonManager()
{
  qSlicerPythonManager* _pythonManager = 0;
  if (!qSlicerCoreApplication::testAttribute(qSlicerCoreApplication::AA_DisablePython))
    {
    _pythonManager = qobject_cast<qSlicerPythonManager*>(this->corePythonManager());
    Q_ASSERT(_pythonManager);
    }

  return _pythonManager;
}
#endif

//-----------------------------------------------------------------------------
void qSlicerApplication::setLayoutManager(qSlicerLayoutManager* layoutManager)
{
  Q_D(qSlicerApplication);
  d->LayoutManager = layoutManager;
  if (this->mrmlApplicationLogic())
    {
    this->mrmlApplicationLogic()->SetSliceLogics(
      layoutManager? layoutManager->mrmlSliceLogics() : 0);
    }
}

//-----------------------------------------------------------------------------
qSlicerLayoutManager* qSlicerApplication::layoutManager()const
{
  Q_D(const qSlicerApplication);
  return d->LayoutManager;
}

//-----------------------------------------------------------------------------
QMainWindow* qSlicerApplication::mainWindow()const
{
  foreach(QWidget * widget, this->topLevelWidgets())
    {
    QMainWindow* window = qobject_cast<QMainWindow*>(widget);
    if (window)
      {
      return window;
      }
    }
  return 0;
}

//-----------------------------------------------------------------------------
CTK_SET_CPP(qSlicerApplication, Qt::WindowFlags, setDefaultWindowFlags, DefaultWindowFlags);
CTK_GET_CPP(qSlicerApplication, Qt::WindowFlags, defaultWindowFlags, DefaultWindowFlags);

//-----------------------------------------------------------------------------
void qSlicerApplication::setTopLevelWidgetsVisible(bool visible)
{
  Q_D(qSlicerApplication);
  foreach(QWidget * widget, this->topLevelWidgets())
    {
    // Store current visibility state
    if (!visible)
      {
      if (!d->TopLevelWidgetsSavedVisibilityState.contains(widget))
        {
        d->TopLevelWidgetsSavedVisibilityState[widget] = widget->isVisible();
        }
      widget->hide();
      }
    else
      {
      QMap<QWidget*,bool>::const_iterator it = d->TopLevelWidgetsSavedVisibilityState.find(widget);

      // If widget state was saved, restore it. Otherwise skip.
      if (it != d->TopLevelWidgetsSavedVisibilityState.end())
        {
        widget->setVisible(it.value());
        }
      }
    }

  // Each time widget are set visible. Internal Map can be cleared.
  if (visible)
    {
    d->TopLevelWidgetsSavedVisibilityState.clear();
    }
}

//-----------------------------------------------------------------------------
void qSlicerApplication::setTopLevelWidgetVisible(qSlicerWidget* widget, bool visible)
{
  if (!widget) { return; }
  Q_ASSERT(!widget->parent());
  Q_D(qSlicerApplication);
  // When internal Map is empty, it means top widget are visible
  if (d->TopLevelWidgetsSavedVisibilityState.empty())
    {
    widget->setVisible(visible);
    }
  else
    {
    d->TopLevelWidgetsSavedVisibilityState[widget] = visible;
    }
}

//-----------------------------------------------------------------------------
void qSlicerApplication::handleCommandLineArguments()
{
  qSlicerCommandOptions* options = this->commandOptions();
  Q_ASSERT(options);

  if (options->noMainWindow() || options->disableMessageHandlers())
    {
    // If no UI is expected, it doesn't make sens to use registered handlers.
    // Let's disable them.
    this->errorLogModel()->disableAllMsgHandler();
    }

  this->Superclass::handleCommandLineArguments();

  this->setToolTipsEnabled(!options->disableToolTips());
}

//-----------------------------------------------------------------------------
QSettings* qSlicerApplication::newSettings(const QString& fileName)
{
  if (!fileName.isEmpty())
    {
    // Special case for tmp settings
    return new ctkSettings(fileName, QSettings::defaultFormat(), this);
    }
  return new ctkSettings(this);
}

//-----------------------------------------------------------------------------
void qSlicerApplication::setToolTipsEnabled(bool enable)
{
  Q_D(qSlicerApplication);
  d->ToolTipTrapper->setEnabled(!enable);
}

//-----------------------------------------------------------------------------
void qSlicerApplication::setMRMLScene(vtkMRMLScene* newMRMLScene)
{
  Q_D(qSlicerApplication);
  // we do it now because Superclass::setMRMLScene() emits the signal
  // mrmlSceneChanged and we don't want the set the scene after the signal
  // is sent.
  d->ColorDialogPickerWidget->setMRMLScene(newMRMLScene);
  this->Superclass::setMRMLScene(newMRMLScene);
}
