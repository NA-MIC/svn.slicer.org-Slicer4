// QT includes

// SlicerQT includes
#include "qSlicerDiffusionTensorVolumeDisplayWidget.h"
#include "qSlicerDiffusionWeightedVolumeDisplayWidget.h"
#include "qSlicerLabelMapVolumeDisplayWidget.h"
#include "qSlicerScalarVolumeDisplayWidget.h"
#include "qSlicerVolumeDisplayWidget.h"

// MRML includes
#include <vtkMRMLDiffusionTensorVolumeNode.h>
#include <vtkMRMLDiffusionWeightedVolumeNode.h>

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_Volumes
class qSlicerVolumeDisplayWidgetPrivate
{
  Q_DECLARE_PUBLIC(qSlicerVolumeDisplayWidget);

protected:
  qSlicerVolumeDisplayWidget* const q_ptr;

public:
  qSlicerVolumeDisplayWidgetPrivate(qSlicerVolumeDisplayWidget& object);
  void init();
  void setCurrentDisplayWidget(qSlicerWidget* displayWidget);

  qSlicerScalarVolumeDisplayWidget*            ScalarVolumeDisplayWidget;
  qSlicerLabelMapVolumeDisplayWidget*          LabelMapVolumeDisplayWidget;
  qSlicerDiffusionWeightedVolumeDisplayWidget* DWVolumeDisplayWidget;
  qSlicerDiffusionTensorVolumeDisplayWidget*   DTVolumeDisplayWidget;
};

// --------------------------------------------------------------------------
qSlicerVolumeDisplayWidgetPrivate::qSlicerVolumeDisplayWidgetPrivate(
  qSlicerVolumeDisplayWidget& object)
  : q_ptr(&object)
{
  this->ScalarVolumeDisplayWidget = 0;
  this->LabelMapVolumeDisplayWidget = 0;
  this->DWVolumeDisplayWidget = 0;
  this->DTVolumeDisplayWidget = 0;
}

// --------------------------------------------------------------------------
void qSlicerVolumeDisplayWidgetPrivate::init()
{
  Q_Q(qSlicerVolumeDisplayWidget);
  this->ScalarVolumeDisplayWidget = new qSlicerScalarVolumeDisplayWidget(q);
  q->addWidget(this->ScalarVolumeDisplayWidget);

  this->LabelMapVolumeDisplayWidget = new qSlicerLabelMapVolumeDisplayWidget(q);
  q->addWidget(this->LabelMapVolumeDisplayWidget);

  this->DWVolumeDisplayWidget = new qSlicerDiffusionWeightedVolumeDisplayWidget(q);
  q->addWidget(this->DWVolumeDisplayWidget);

  this->DTVolumeDisplayWidget = new qSlicerDiffusionTensorVolumeDisplayWidget(q);
  q->addWidget(this->DTVolumeDisplayWidget);
}

// --------------------------------------------------------------------------
void qSlicerVolumeDisplayWidgetPrivate::setCurrentDisplayWidget(
  qSlicerWidget* displayWidget)
{
  Q_Q(qSlicerVolumeDisplayWidget);
  qSlicerWidget* activeWidget = qobject_cast<qSlicerWidget*>(q->currentWidget());
  if (activeWidget == displayWidget)
    {
    return;
    }
  if (activeWidget)
    {
    // We must remove the node "before" the setting the scene to 0.
    // Because removing the scene could modify the observed node (e.g setting
    // the scene to 0 on a colortable combobox will set the color node of the
    // observed node to 0.
    vtkMRMLNode* emptyVolumeNode = 0;
    if (activeWidget == this->ScalarVolumeDisplayWidget)
      {
      this->ScalarVolumeDisplayWidget->setMRMLVolumeNode(emptyVolumeNode);
      }
    if (activeWidget == this->LabelMapVolumeDisplayWidget)
      {
      this->LabelMapVolumeDisplayWidget->setMRMLVolumeNode(emptyVolumeNode);
      }
    if (activeWidget == this->DWVolumeDisplayWidget)
      {
      this->DWVolumeDisplayWidget->setMRMLVolumeNode(emptyVolumeNode);
      }
    if (activeWidget == this->DTVolumeDisplayWidget)
      {
      this->DTVolumeDisplayWidget->setMRMLVolumeNode(emptyVolumeNode);
      }
    activeWidget->setMRMLScene(0);
    }
  // QStackWidget::setCurrentWidget(0) is not supported
  if (displayWidget)
    {
    q->setCurrentWidget(displayWidget);
    }
}

// --------------------------------------------------------------------------
// qSlicerVolumeDisplayWidget
// --------------------------------------------------------------------------
qSlicerVolumeDisplayWidget::qSlicerVolumeDisplayWidget(QWidget* parentWidget)
  : Superclass(parentWidget)
  , d_ptr(new qSlicerVolumeDisplayWidgetPrivate(*this))
{
  Q_D(qSlicerVolumeDisplayWidget);
  d->init();
}

// --------------------------------------------------------------------------
qSlicerVolumeDisplayWidget::~qSlicerVolumeDisplayWidget()
{
}

// --------------------------------------------------------------------------
void qSlicerVolumeDisplayWidget::setMRMLVolumeNode(vtkMRMLNode* volumeNode)
{
   Q_D(qSlicerVolumeDisplayWidget);
   qvtkDisconnect(0, vtkCommand::ModifiedEvent,
                  this, SLOT(updateFromMRML(vtkObject*)));

  if (volumeNode == 0)
    {
    d->setCurrentDisplayWidget(0);
    return;
    }

  vtkMRMLScene* scene = volumeNode->GetScene();
  vtkMRMLScalarVolumeNode* scalarVolumeNode =
    vtkMRMLScalarVolumeNode::SafeDownCast(volumeNode);
  vtkMRMLDiffusionWeightedVolumeNode* dwiVolumeNode =
    vtkMRMLDiffusionWeightedVolumeNode::SafeDownCast(volumeNode);
  vtkMRMLDiffusionTensorVolumeNode* dtiVolumeNode =
    vtkMRMLDiffusionTensorVolumeNode::SafeDownCast(volumeNode);
   if (dtiVolumeNode)
    {
    d->DTVolumeDisplayWidget->setMRMLScene(scene);
    d->DTVolumeDisplayWidget->setMRMLVolumeNode(volumeNode);
    d->setCurrentDisplayWidget(d->DTVolumeDisplayWidget);
    }
   else if (dwiVolumeNode)
    {
    d->DWVolumeDisplayWidget->setMRMLScene(scene);
    d->DWVolumeDisplayWidget->setMRMLVolumeNode(volumeNode);
    d->setCurrentDisplayWidget(d->DWVolumeDisplayWidget);
    }
   else if (scalarVolumeNode && !scalarVolumeNode->GetLabelMap())
    {
    qvtkConnect(volumeNode, vtkCommand::ModifiedEvent,
              this, SLOT(updateFromMRML(vtkObject*)));
    d->ScalarVolumeDisplayWidget->setMRMLScene(scene);
    d->ScalarVolumeDisplayWidget->setMRMLVolumeNode(volumeNode);
    d->setCurrentDisplayWidget(d->ScalarVolumeDisplayWidget);
    }
  else if (scalarVolumeNode && scalarVolumeNode->GetLabelMap())
    {
    qvtkConnect(volumeNode, vtkCommand::ModifiedEvent,
              this, SLOT(updateFromMRML(vtkObject*)));
    d->LabelMapVolumeDisplayWidget->setMRMLScene(scene);
    d->LabelMapVolumeDisplayWidget->setMRMLVolumeNode(volumeNode);
    d->setCurrentDisplayWidget(d->LabelMapVolumeDisplayWidget);
    }
}

// --------------------------------------------------------------------------
void qSlicerVolumeDisplayWidget::updateFromMRML(vtkObject* volume)
{
  vtkMRMLVolumeNode* volumeNode = vtkMRMLVolumeNode::SafeDownCast(volume);
  this->setMRMLVolumeNode(volumeNode);
}

