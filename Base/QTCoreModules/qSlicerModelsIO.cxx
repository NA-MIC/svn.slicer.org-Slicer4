/// Qt includes
#include <QDebug>
#include <QString>
#include <QVariant>

/// qSlicer includes
//#include "qSlicerAbstractModule.h"
//#include "qSlicerCoreApplication.h"
//#include "qSlicerModuleManager.h"
#include "qSlicerModelsIO.h"

/// Logic includes
#include "vtkSlicerModelsLogic.h"

/// MRML includes
#include <vtkMRMLModelNode.h>

//-----------------------------------------------------------------------------
qSlicerModelsIO::qSlicerModelsIO(QObject* _parent)
  :qSlicerIO(_parent)
{
}

//-----------------------------------------------------------------------------
QString qSlicerModelsIO::description()const
{
  return "Model";
}

//-----------------------------------------------------------------------------
qSlicerIO::IOFileType qSlicerModelsIO::fileType()const
{
  return qSlicerIO::VolumeFile;
}

//-----------------------------------------------------------------------------
QString qSlicerModelsIO::extensions()const
{
  return "*.vtk *.vtp *.g *.byu *.stl *.orig"
    "*.inflated *.sphere *.white *.smoothwm *.pial";
}

//-----------------------------------------------------------------------------
bool qSlicerModelsIO::load(const IOProperties& properties)
{
  Q_ASSERT(properties.contains("fileName"));
  QString fileName = properties["fileName"].toString();
  
  // vtkSlicerModelsLogic* modelsLogic = 
  //   vtkSlicerModelsLogic::SafeDownCast(
  //     qSlicerCoreApplication::application()->moduleManager()
  //     ->module("Models")->logic());
  vtkSlicerModelsLogic* modelsLogic = vtkSlicerModelsLogic::New();
  Q_ASSERT(modelsLogic);
  vtkMRMLModelNode* node = modelsLogic->AddModel(
    fileName.toLatin1().data());
  if (node)
    {
    this->setLoadedNodes(QStringList(QString(node->GetID())));
    }
  else
    {
    this->setLoadedNodes(QStringList());
    }
  modelsLogic->Delete();
  return node != 0;
}
