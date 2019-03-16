/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// Qt includes
#include <QDebug>

// Markups Logic includes
#include <vtkSlicerMarkupsLogic.h>

// MRMLDisplayableManager includes
#include <vtkMRMLSliceViewDisplayableManagerFactory.h>
#include <vtkMRMLThreeDViewDisplayableManagerFactory.h>

// QTGUI includes
#include <qSlicerApplication.h>
//#include <qSlicerCoreApplication.h>
#include <qSlicerIOManager.h>
#include <qSlicerNodeWriter.h>

// SubjectHierarchy Plugins includes
#include "qSlicerSubjectHierarchyPluginHandler.h"
#include "qSlicerSubjectHierarchyMarkupsPlugin.h"

// Markups includes
#include "qSlicerMarkupsModule.h"
#include "qSlicerMarkupsModuleWidget.h"
#include "qSlicerMarkupsReader.h"
//#include "qSlicerMarkupsSettingsPanel.h"
//#include "vtkSlicerMarkupsLogic.h"

// DisplayableManager initialization
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkSlicerMarkupsModuleMRMLDisplayableManager)

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_Markups
class qSlicerMarkupsModulePrivate
{
public:
  qSlicerMarkupsModulePrivate();
};

//-----------------------------------------------------------------------------
// qSlicerMarkupsModulePrivate methods

//-----------------------------------------------------------------------------
qSlicerMarkupsModulePrivate::qSlicerMarkupsModulePrivate()
= default;

//-----------------------------------------------------------------------------
// qSlicerMarkupsModule methods

//-----------------------------------------------------------------------------
qSlicerMarkupsModule::qSlicerMarkupsModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerMarkupsModulePrivate)
{
}

//-----------------------------------------------------------------------------
QStringList qSlicerMarkupsModule::categories()const
{
  return QStringList() << "" << "Informatics";
}

//-----------------------------------------------------------------------------
qSlicerMarkupsModule::~qSlicerMarkupsModule()
= default;

//-----------------------------------------------------------------------------
QString qSlicerMarkupsModule::helpText()const
{
  QString help =
    "A module to create and manage markups in 2D and 3D."
    " Replaces the Annotations module for fiducials.\n"
    "<a href=\"%1/Documentation/%2.%3/Modules/Markups\">"
    "%1/Documentation/%2.%3/Modules/Markups</a>\n";
  return help.arg(this->slicerWikiUrl()).arg(Slicer_VERSION_MAJOR).arg(Slicer_VERSION_MINOR);
}

//-----------------------------------------------------------------------------
QString qSlicerMarkupsModule::acknowledgementText()const
{
  return "This work was supported by NAMIC, NAC, and the Slicer Community.";
}

//-----------------------------------------------------------------------------
QStringList qSlicerMarkupsModule::contributors()const
{
  QStringList moduleContributors;
  moduleContributors << QString("Nicole Aucoin (BWH)");
  return moduleContributors;
}

//-----------------------------------------------------------------------------
QIcon qSlicerMarkupsModule::icon()const
{
  return QIcon(":/Icons/Markups.png");
}



//-----------------------------------------------------------------------------
void qSlicerMarkupsModule::setup()
{
  this->Superclass::setup();

  // Register displayable managers (same displayable manager handles both slice and 3D views)
  vtkMRMLSliceViewDisplayableManagerFactory::GetInstance()->RegisterDisplayableManager("vtkMRMLMarkupsDisplayableManager");
  vtkMRMLThreeDViewDisplayableManagerFactory::GetInstance()->RegisterDisplayableManager("vtkMRMLMarkupsDisplayableManager");

  // Register IO
  qSlicerIOManager* ioManager = qSlicerApplication::application()->ioManager();
  qSlicerMarkupsReader *markupsIO = new qSlicerMarkupsReader(vtkSlicerMarkupsLogic::SafeDownCast(this->logic()), this);
  ioManager->registerIO(markupsIO);
  ioManager->registerIO(new qSlicerNodeWriter(
                            "MarkupsFiducials", markupsIO->fileType(),
                            QStringList() << "vtkMRMLMarkupsNode", true, this));

  // settings
  /*
  if (qSlicerApplication::application())
    {
    qSlicerMarkupsSettingsPanel* panel =
      new qSlicerMarkupsSettingsPanel;
    qSlicerApplication::application()->settingsDialog()->addPanel(
      "Markups", panel);
    panel->setMarkupsLogic(vtkSlicerMarkupsLogic::SafeDownCast(this->logic()));
    }
  */
  // for now, don't use the settings panel as it's causing the logic values to
  // be reset on start up, just set things directly
  qSlicerMarkupsModuleWidget* moduleWidget = dynamic_cast<qSlicerMarkupsModuleWidget*>(this->widgetRepresentation());
  if (!moduleWidget)
    {
    qDebug() << "qSlicerMarkupsModule::setup: unable to get the markups version of the widget to set default display settings";
    }
  else
    {
    moduleWidget->updateLogicFromSettings();
    }

  // Register Subject Hierarchy core plugins
  qSlicerSubjectHierarchyPluginHandler::instance()->registerPlugin(new qSlicerSubjectHierarchyMarkupsPlugin());
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation * qSlicerMarkupsModule::createWidgetRepresentation()
{
  return new qSlicerMarkupsModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicerMarkupsModule::createLogic()
{
  return vtkSlicerMarkupsLogic::New();
}

//-----------------------------------------------------------------------------
QStringList qSlicerMarkupsModule::associatedNodeTypes() const
{
  return QStringList()
    << "vtkMRMLAnnotationFiducialNode"
    << "vtkMRMLMarkupsDisplayNode"
    << "vtkMRMLMarkupsFiducialNode"
    << "vtkMRMLMarkupsLineNode"
    << "vtkMRMLMarkupsAngleNode"
    << "vtkMRMLMarkupsCurveNode"
    << "vtkMRMLMarkupsClosedCurveNode"
    << "vtkMRMLMarkupsFiducialStorageNode";
}
