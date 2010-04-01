
// QT includes
#include <QtPlugin>

// qSlicer includes
#include <qSlicerCoreApplication.h>
#include <qSlicerCoreIOManager.h>

// Volumes Logic includes
#include <vtkSlicerVolumesLogic.h>

// Volumes QTModule includes
#include "qSlicerVolumesIO.h"
#include "qSlicerVolumesModule.h"
#include "qSlicerVolumesModuleWidget.h"

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(qSlicerVolumesModule, qSlicerVolumesModule);

//-----------------------------------------------------------------------------
class qSlicerVolumesModulePrivate: public qCTKPrivate<qSlicerVolumesModule>
{
public:
};

//-----------------------------------------------------------------------------
qSlicerVolumesModule::qSlicerVolumesModule(QObject* _parent)
  :Superclass(_parent)
{
  QCTK_INIT_PRIVATE(qSlicerVolumesModule);
}

//-----------------------------------------------------------------------------
void qSlicerVolumesModule::setup()
{
  this->Superclass::setup();
  qSlicerCoreApplication::application()->coreIOManager()->registerIO(
    new qSlicerVolumesIO(this));  
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleWidget * qSlicerVolumesModule::createWidgetRepresentation()
{
  return new qSlicerVolumesModuleWidget;
}

//-----------------------------------------------------------------------------
vtkSlicerLogic* qSlicerVolumesModule::createLogic()
{
  return vtkSlicerVolumesLogic::New();
}
