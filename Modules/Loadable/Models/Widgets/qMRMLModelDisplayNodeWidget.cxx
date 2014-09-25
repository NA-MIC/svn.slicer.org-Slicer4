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

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// QT includes
#include <QColor>

// qMRML includes
#include "qMRMLModelDisplayNodeWidget.h"
#include "ui_qMRMLModelDisplayNodeWidget.h"

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLColorTableNode.h>
#include <vtkMRMLModelDisplayNode.h>
#include <vtkMRMLModelNode.h>
#include <vtkMRMLModelHierarchyNode.h>
#include <vtkMRMLSelectionNode.h>

// VTK includes
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

//------------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_Models
class qMRMLModelDisplayNodeWidgetPrivate: public QWidget, public Ui_qMRMLModelDisplayNodeWidget
{
  Q_DECLARE_PUBLIC(qMRMLModelDisplayNodeWidget);

protected:
  qMRMLModelDisplayNodeWidget* const q_ptr;
  typedef QWidget Superclass;

public:
  qMRMLModelDisplayNodeWidgetPrivate(qMRMLModelDisplayNodeWidget& object);
  void init();

  virtual bool blockSignals(bool block);
  virtual void setRange(double min, double max);

  vtkSmartPointer<vtkMRMLModelDisplayNode> MRMLModelDisplayNode;
  vtkSmartPointer<vtkMRMLSelectionNode> MRMLSelectionNode;
};

//------------------------------------------------------------------------------
qMRMLModelDisplayNodeWidgetPrivate::qMRMLModelDisplayNodeWidgetPrivate(qMRMLModelDisplayNodeWidget& object)
  : q_ptr(&object)
{
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidgetPrivate::init()
{
  Q_Q(qMRMLModelDisplayNodeWidget);
  this->setupUi(q);

  QObject::connect(this->ScalarsVisibilityCheckBox, SIGNAL(toggled(bool)),
                   q, SLOT(setScalarsVisibility(bool)));
  QObject::connect(this->ActiveScalarComboBox, SIGNAL(currentArrayChanged(QString)),
                   q, SLOT(setActiveScalarName(QString)));
  QObject::connect(this->ScalarsColorNodeComboBox,
                   SIGNAL(currentNodeChanged(vtkMRMLNode*)),
                   q, SLOT(setScalarsColorNode(vtkMRMLNode*)));

  // make a radio button group for the scalar range flag options
  QButtonGroup* rangeFlagGroup = new QButtonGroup(q);
  rangeFlagGroup->addButton(this->ScalarsRangeFlagLUTRadioButton);
  rangeFlagGroup->addButton(this->ScalarsRangeFlagDisplayRadioButton);
  rangeFlagGroup->addButton(this->ScalarsRangeFlagDataRadioButton);
  rangeFlagGroup->addButton(this->ScalarsRangeFlagDataTypeRadioButton);
  // radio button group signals
  QObject::connect(this->ScalarsRangeFlagLUTRadioButton, SIGNAL(toggled(bool)),
    q, SLOT(setScalarsScalarRangeFlag()));
  QObject::connect(this->ScalarsRangeFlagDisplayRadioButton, SIGNAL(toggled(bool)),
    q, SLOT(setScalarsScalarRangeFlag()));
  QObject::connect(this->ScalarsRangeFlagDataRadioButton, SIGNAL(toggled(bool)),
    q, SLOT(setScalarsScalarRangeFlag()));
  QObject::connect(this->ScalarsRangeFlagDataTypeRadioButton, SIGNAL(toggled(bool)),
    q, SLOT(setScalarsScalarRangeFlag()));

  // auto/manual scalar range
  q->setAutoScalarRange(qMRMLModelDisplayNodeWidget::Auto);
  // scalar range
  QObject::connect(this->ScalarsDisplayScalarRangeDoubleRangeSlider,
//                   SIGNAL(valuesChanged(double,double)),
                   SIGNAL(positionsChanged(double,double)),
                   q, SLOT(setScalarsDisplayRange(double,double)));
  QObject::connect(this->ScalarsDisplayScalarRangeMinDoubleSpinBox,
                   SIGNAL(valueChanged(double)),
                   q, SLOT(setMinimumValue(double)));
  QObject::connect(this->ScalarsDisplayScalarRangeMaxDoubleSpinBox,
                   SIGNAL(valueChanged(double)),
                   q, SLOT(setMaximumValue(double)));

  QObject::connect(this->ScalarsDisplayScalarRangeAutoManualComboBox, SIGNAL(currentIndexChanged(int)),
                   q, SLOT(setAutoScalarRange(int)));

  //QObject::connect(this->ScalarsDisplayScalarRangeDoubleRangeSlider,
  //                 SIGNAL(positionsChanged(double,double)),
  //                 q, SLOT(setScalarsDisplayRange(double,double)));
  q->setEnabled(this->MRMLModelDisplayNode.GetPointer() != 0);

  this->MRMLDisplayNodeWidget->setSelectedVisible(false);
}

//------------------------------------------------------------------------------
bool qMRMLModelDisplayNodeWidgetPrivate::blockSignals(bool block)
{
  bool res = this->Superclass::blockSignals(block);
  this->ScalarsDisplayScalarRangeDoubleRangeSlider->blockSignals(block);
  this->ScalarsDisplayScalarRangeMinDoubleSpinBox->blockSignals(block);
  this->ScalarsDisplayScalarRangeMaxDoubleSpinBox->blockSignals(block);
  return res;
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidgetPrivate::setRange(double min, double max)
{
  this->ScalarsDisplayScalarRangeDoubleRangeSlider->setRange(min, max);
  this->ScalarsDisplayScalarRangeMinDoubleSpinBox->setRange(min, max);
  this->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setRange(min, max);
}

//------------------------------------------------------------------------------
qMRMLModelDisplayNodeWidget::qMRMLModelDisplayNodeWidget(QWidget *_parent)
  : QWidget(_parent)
  , d_ptr(new qMRMLModelDisplayNodeWidgetPrivate(*this))
{
  Q_D(qMRMLModelDisplayNodeWidget);
  d->init();
}

//------------------------------------------------------------------------------
qMRMLModelDisplayNodeWidget::~qMRMLModelDisplayNodeWidget()
{
}

//------------------------------------------------------------------------------
vtkMRMLModelDisplayNode* qMRMLModelDisplayNodeWidget::mrmlModelDisplayNode()const
{
  Q_D(const qMRMLModelDisplayNodeWidget);
  return d->MRMLModelDisplayNode;
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setMRMLModelOrHierarchyNode(vtkMRMLNode* node)
{
  // can be set from a model node or a model hierarchy node
  vtkMRMLModelNode* modelNode = vtkMRMLModelNode::SafeDownCast(node);
  vtkMRMLModelHierarchyNode *hierarchyNode = vtkMRMLModelHierarchyNode::SafeDownCast(node);
  vtkMRMLModelDisplayNode *modelDisplayNode = 0;
  if (modelNode)
    {
    vtkMRMLSelectionNode* selectionNode = this->getSelectionNode(modelNode->GetScene());
    std::string displayNodeName;
    if (selectionNode)
      {
      displayNodeName = selectionNode->GetModelHierarchyDisplayNodeClassName(
                        modelNode->GetClassName());
      }

    int nDisplayNodes = modelNode->GetNumberOfDisplayNodes();
    for (int i=0; i<nDisplayNodes; i++)
      {
        modelDisplayNode = vtkMRMLModelDisplayNode::SafeDownCast(modelNode->GetNthDisplayNode(i));
        if (displayNodeName.empty() || modelDisplayNode->IsA(displayNodeName.c_str()))
          {
          break;
          }
      }
    }
  else if (hierarchyNode)
    {
    modelDisplayNode = hierarchyNode->GetModelDisplayNode();
    }
  this->setMRMLModelDisplayNode(modelDisplayNode);
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setMRMLModelDisplayNode(vtkMRMLNode* node)
{
  this->setMRMLModelDisplayNode(vtkMRMLModelDisplayNode::SafeDownCast(node));
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setMRMLModelDisplayNode(vtkMRMLModelDisplayNode* ModelDisplayNode)
{
  Q_D(qMRMLModelDisplayNodeWidget);
  qvtkReconnect(d->MRMLModelDisplayNode, ModelDisplayNode, vtkCommand::ModifiedEvent,
                this, SLOT(updateWidgetFromMRML()));
  d->MRMLModelDisplayNode = ModelDisplayNode;
  d->MRMLDisplayNodeWidget->setMRMLDisplayNode(ModelDisplayNode);
  this->updateWidgetFromMRML();
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setScalarsVisibility(bool visible)
{
  Q_D(qMRMLModelDisplayNodeWidget);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  d->MRMLModelDisplayNode->SetScalarVisibility(visible);
}

//------------------------------------------------------------------------------
bool qMRMLModelDisplayNodeWidget::scalarsVisibility()const
{
  Q_D(const qMRMLModelDisplayNodeWidget);
  return d->ScalarsVisibilityCheckBox->isChecked();
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setActiveScalarName(const QString& arrayName)
{
  Q_D(qMRMLModelDisplayNodeWidget);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  d->MRMLModelDisplayNode->SetActiveScalarName(arrayName.toLatin1());

  // if there's no color node set for a non empty array name, use a default
  if (!arrayName.isEmpty() &&
      d->MRMLModelDisplayNode->GetColorNodeID() == NULL)
    {
    const char *colorNodeID = "vtkMRMLColorTableNodeRainbow";
    d->MRMLModelDisplayNode->SetAndObserveColorNodeID(colorNodeID);
    }
}

//------------------------------------------------------------------------------
QString qMRMLModelDisplayNodeWidget::activeScalarName()const
{
  Q_D(const qMRMLModelDisplayNodeWidget);
  // TODO: use currentArrayName()
  vtkAbstractArray* array = d->ActiveScalarComboBox->currentArray();
  return array ? array->GetName() : "";
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setScalarsColorNode(vtkMRMLNode* colorNode)
{
  this->setScalarsColorNode(vtkMRMLColorNode::SafeDownCast(colorNode));
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setScalarsColorNode(vtkMRMLColorNode* colorNode)
{
  Q_D(qMRMLModelDisplayNodeWidget);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  d->MRMLModelDisplayNode->SetAndObserveColorNodeID(colorNode ? colorNode->GetID() : NULL);
}

//------------------------------------------------------------------------------
vtkMRMLColorNode* qMRMLModelDisplayNodeWidget::scalarsColorNode()const
{
  Q_D(const qMRMLModelDisplayNodeWidget);
  return vtkMRMLColorNode::SafeDownCast(
    d->ScalarsColorNodeComboBox->currentNode());
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setScalarsScalarRangeFlag()
{
  Q_D(qMRMLModelDisplayNodeWidget);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  int flag = 0;
  if (d->ScalarsRangeFlagLUTRadioButton->isChecked())
    {
    flag = vtkMRMLModelDisplayNode::UseColorNodeScalarRange;
    }
  else if (d->ScalarsRangeFlagDisplayRadioButton->isChecked())
    {
    flag = vtkMRMLModelDisplayNode::UseDisplayNodeScalarRange;
    }
  else if (d->ScalarsRangeFlagDataRadioButton->isChecked())
    {
    flag = vtkMRMLModelDisplayNode::UseDataScalarRange;
    }
  else if (d->ScalarsRangeFlagDataTypeRadioButton->isChecked())
    {
    flag = vtkMRMLModelDisplayNode::UseDataTypeScalarRange;
    }
  d->MRMLModelDisplayNode->SetScalarRangeFlag(flag);
}

// --------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setAutoScalarRange(ControlMode autoScalarRange)
{
  Q_D(qMRMLModelDisplayNodeWidget);

  if (!d->MRMLModelDisplayNode)
    {
    return;
    }
  int oldAuto = d->MRMLModelDisplayNode->GetAutoScalarRange();

  //int disabledModify = this->MRMLModelDisplayNode->StartModify();
  d->MRMLModelDisplayNode->SetAutoScalarRange(
    autoScalarRange == qMRMLModelDisplayNodeWidget::Auto ? 1 : 0);
  //this->MRMLModelDisplayNode->EndModify(disabledModify);

  if (autoScalarRange == qMRMLModelDisplayNodeWidget::Auto)
    {
    // disable the range slider and min max spin boxes as the display
    // node scalar range will get updated from the data automatically
    d->ScalarsDisplayScalarRangeDoubleRangeSlider->setEnabled(false);
    d->ScalarsDisplayScalarRangeMinDoubleSpinBox->setEnabled(false);
    d->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setEnabled(false);
    }
  else
    {
    // make sure the slider and spin boxes are enabled
    d->ScalarsDisplayScalarRangeDoubleRangeSlider->setEnabled(true);
    d->ScalarsDisplayScalarRangeMinDoubleSpinBox->setEnabled(true);
    d->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setEnabled(true);
    }

  if (autoScalarRange != oldAuto)
    {
    emit this->autoScalarRangeValueChanged(
      autoScalarRange == qMRMLModelDisplayNodeWidget::Auto ?
        qMRMLModelDisplayNodeWidget::Auto : qMRMLModelDisplayNodeWidget::Manual);
    }
}

// --------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setAutoScalarRange(int autoScalarRange)
{
  switch(autoScalarRange)
    {
    case qMRMLModelDisplayNodeWidget::Auto:
      this->setAutoScalarRange(qMRMLModelDisplayNodeWidget::Auto);
      break;
    case qMRMLModelDisplayNodeWidget::Manual:
      this->setAutoScalarRange(qMRMLModelDisplayNodeWidget::Manual);
      break;
    default:
      break;
    }
}

// --------------------------------------------------------------------------
qMRMLModelDisplayNodeWidget::ControlMode qMRMLModelDisplayNodeWidget::autoScalarRange() const
{
  Q_D(const qMRMLModelDisplayNodeWidget);
  switch (d->ScalarsDisplayScalarRangeAutoManualComboBox->currentIndex())
    {
    case qMRMLModelDisplayNodeWidget::Auto:
      return qMRMLModelDisplayNodeWidget::Auto;
      break;
    case qMRMLModelDisplayNodeWidget::Manual:
      return qMRMLModelDisplayNodeWidget::Manual;
      break;
    }
  return qMRMLModelDisplayNodeWidget::Manual;
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setScalarsDisplayRange(double min, double max)
{
  Q_D(qMRMLModelDisplayNodeWidget);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  double *range = d->MRMLModelDisplayNode->GetScalarRange();
  if (range[0] != min ||
      range[1] != max)
    {
    d->MRMLModelDisplayNode->SetScalarRange(min, max);
    }
}

// --------------------------------------------------------------------------
double qMRMLModelDisplayNodeWidget::minimumValue() const
{
  Q_D(const qMRMLModelDisplayNodeWidget);

  double min = d->ScalarsDisplayScalarRangeDoubleRangeSlider->minimumValue();
  return min;
}

// --------------------------------------------------------------------------
double qMRMLModelDisplayNodeWidget::maximumValue() const
{
  Q_D(const qMRMLModelDisplayNodeWidget);

  double max = d->ScalarsDisplayScalarRangeDoubleRangeSlider->maximumValue();
  return max;
}

// --------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setMinimumValue(double min)
{
  this->setScalarsDisplayRange(min, this->maximumValue());
}

// --------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::setMaximumValue(double max)
{
  this->setScalarsDisplayRange(this->minimumValue(), max);
}

//------------------------------------------------------------------------------
void qMRMLModelDisplayNodeWidget::updateWidgetFromMRML()
{
  Q_D(qMRMLModelDisplayNodeWidget);
  this->setEnabled(d->MRMLModelDisplayNode.GetPointer() != 0);
  if (!d->MRMLModelDisplayNode.GetPointer())
    {
    return;
    }
  if (d->ScalarsVisibilityCheckBox->isChecked() !=
      (bool)d->MRMLModelDisplayNode->GetScalarVisibility())
    {
    d->ScalarsVisibilityCheckBox->setChecked(
      d->MRMLModelDisplayNode->GetScalarVisibility());
    }
  // update the scalar range flag radio buttons
  int scalarRangeFlag = d->MRMLModelDisplayNode->GetScalarRangeFlag();
  if (scalarRangeFlag == vtkMRMLDisplayNode::UseColorNodeScalarRange)
    {
    d->ScalarsRangeFlagLUTRadioButton->setChecked(true);
    }
  else if (scalarRangeFlag == vtkMRMLDisplayNode::UseDisplayNodeScalarRange)
    {
    d->ScalarsRangeFlagDisplayRadioButton->setChecked(true);
    }
  else if (scalarRangeFlag == vtkMRMLDisplayNode::UseDataScalarRange)
    {
    d->ScalarsRangeFlagDataRadioButton->setChecked(true);
    }
  else if (scalarRangeFlag == vtkMRMLDisplayNode::UseDataTypeScalarRange)
    {
    d->ScalarsRangeFlagDataTypeRadioButton->setChecked(true);
    }

  double widgetRangeMin = d->ScalarsDisplayScalarRangeDoubleRangeSlider->minimum();
  double widgetRangeMax = d->ScalarsDisplayScalarRangeDoubleRangeSlider->maximum();
  double low = d->ScalarsDisplayScalarRangeDoubleRangeSlider->minimumValue();
  double high = d->ScalarsDisplayScalarRangeDoubleRangeSlider->maximumValue();
  double *displayRange =  d->MRMLModelDisplayNode->GetScalarRange();
  // check if need to update the range of the widgets to accomodate the display node scalar range
  if (displayRange[0] < widgetRangeMin ||
      displayRange[1] > widgetRangeMax)
    {
    double newMin = widgetRangeMin;
    double newMax = widgetRangeMax;
    if (displayRange[0] < widgetRangeMin)
      {
      newMin = floor(displayRange[0] - 1.0);
      }
    if (displayRange[1] > widgetRangeMax)
      {
      newMax = ceil(displayRange[1] + 1.0);
      }
    d->setRange(newMin, newMax);
    }

  // We block here to prevent the widgets to call setScalarRange which could
  // change the AutoScalarRange from Auto into Manual.
  bool wasBlocking = d->blockSignals(true);
  if (low  != displayRange[0] ||
      high != displayRange[1])
    {
    d->ScalarsDisplayScalarRangeDoubleRangeSlider->setValues(displayRange[0],
                                                             displayRange[1]);
    }
  d->ScalarsDisplayScalarRangeMinDoubleSpinBox->setValue(displayRange[0]);
  d->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setValue(displayRange[1]);

  d->blockSignals(wasBlocking);

  switch (d->MRMLModelDisplayNode->GetAutoScalarRange())
    {
    case 1:
      d->ScalarsDisplayScalarRangeAutoManualComboBox->setCurrentIndex(qMRMLModelDisplayNodeWidget::Auto);
      if (scalarRangeFlag == vtkMRMLDisplayNode::UseDataScalarRange)
        {
        // disable user setting of the scalar range as it will be automatically over written
        d->ScalarsDisplayScalarRangeDoubleRangeSlider->setEnabled(false);
        d->ScalarsDisplayScalarRangeMinDoubleSpinBox->setEnabled(false);
        d->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setEnabled(false);
        }
      break;
    case 0:
      if (d->ScalarsDisplayScalarRangeAutoManualComboBox->currentIndex() ==
          qMRMLModelDisplayNodeWidget::Auto)
        {
        d->ScalarsDisplayScalarRangeAutoManualComboBox->setCurrentIndex(qMRMLModelDisplayNodeWidget::Manual);
        // enable user setting of scalar range
        d->ScalarsDisplayScalarRangeDoubleRangeSlider->setEnabled(true);
        d->ScalarsDisplayScalarRangeMinDoubleSpinBox->setEnabled(true);
        d->ScalarsDisplayScalarRangeMaxDoubleSpinBox->setEnabled(true);
        }
      break;
    }

  wasBlocking = d->ActiveScalarComboBox->blockSignals(true);
  d->ActiveScalarComboBox->setDataSet(
    d->MRMLModelDisplayNode->GetInputPolyData());
  d->ActiveScalarComboBox->blockSignals(wasBlocking);
  if (d->ActiveScalarComboBox->currentArrayName() !=
      d->MRMLModelDisplayNode->GetActiveScalarName())
    {
    d->ActiveScalarComboBox->setCurrentArray(
      d->MRMLModelDisplayNode->GetActiveScalarName());
    }
  // set the scalar range info
  QString scalarRangeString;
  if (!d->ActiveScalarComboBox->currentArrayName().isEmpty())
    {
    vtkPointData *pointData = NULL;
    if (d->MRMLModelDisplayNode->GetInputPolyData())
      {
      pointData = d->MRMLModelDisplayNode->GetInputPolyData()->GetPointData();
      }
    if (pointData &&
        pointData->GetArray(d->MRMLModelDisplayNode->GetActiveScalarName()))
      {
      double *range = pointData->GetArray(
        d->MRMLModelDisplayNode->GetActiveScalarName())->GetRange();
      if (range)
        {
        scalarRangeString = QString::number(range[0]) +
          QString(", ") +
          QString::number(range[1]);
        }
      }
    }
  d->ActiveScalarRangeLabel->setText(scalarRangeString);

  if (d->ScalarsColorNodeComboBox->mrmlScene() !=
      d->MRMLModelDisplayNode->GetScene())
    {
    d->ScalarsColorNodeComboBox->setMRMLScene(
      d->MRMLModelDisplayNode->GetScene());
    }
  if (d->ScalarsColorNodeComboBox->currentNodeID() !=
      d->MRMLModelDisplayNode->GetColorNodeID())
    {
    d->ScalarsColorNodeComboBox->setCurrentNodeID(
      d->MRMLModelDisplayNode->GetColorNodeID());
    }
}

vtkMRMLSelectionNode* qMRMLModelDisplayNodeWidget::getSelectionNode(vtkMRMLScene *mrmlScene)
{
  Q_D(qMRMLModelDisplayNodeWidget);

  if (d->MRMLSelectionNode.GetPointer() == 0)
    {
    std::vector<vtkMRMLNode *> selectionNodes;
    if (mrmlScene)
      {
      mrmlScene->GetNodesByClass("vtkMRMLSelectionNode", selectionNodes);
      }

    if (selectionNodes.size() > 0)
      {
      d->MRMLSelectionNode = vtkMRMLSelectionNode::SafeDownCast(selectionNodes[0]);
      }
    }
  return d->MRMLSelectionNode;
}

