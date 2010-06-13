
// Qt includes
#include <QDebug>
#include <QStack>

// qMRML includes
#include "qMRMLUtils.h"
#include "qMRMLWindowLevelWidget.h"
#include "ui_qMRMLWindowLevelWidget.h"

// MRML includes
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLScalarVolumeDisplayNode.h"

// VTK includes
#include "vtkSmartPointer.h"


//-----------------------------------------------------------------------------
class qMRMLWindowLevelWidgetPrivate: public ctkPrivate<qMRMLWindowLevelWidget>,
                                     public Ui_qMRMLWindowLevelWidget
{
public:
  qMRMLWindowLevelWidgetPrivate()
    {
    }
    
};

// --------------------------------------------------------------------------
qMRMLWindowLevelWidget::qMRMLWindowLevelWidget(QWidget* _parent) : Superclass(_parent)
{
  CTK_INIT_PRIVATE(qMRMLWindowLevelWidget);
  CTK_D(qMRMLWindowLevelWidget);
  
  d->setupUi(this);

  this->VolumeNode = NULL;
  this->VolumeDisplayNode = NULL;

  this->setAutoWindowLevel(1);

  // TODO replace with double window/level
  this->connect(d->WindowLevelRangeWidget, SIGNAL(minimumValueIsChanging(double)),
                SLOT(setMinimumValue(double)));
  this->connect(d->WindowLevelRangeWidget, SIGNAL(maximumValueIsChanging(double)),
                SLOT(setMaximumValue(double)));

  this->connect(d->AutoManualComboBox, SIGNAL(currentIndexChanged(int)),
                SLOT(setAutoWindowLevel(int)));

  // disable as there is not MRML Node associated with the widget
  this->setEnabled(false);
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setAutoWindowLevel(int autoWindowLevel)
{
  //CTK_D(qMRMLWindowLevelWidget);
  
  if (this->VolumeDisplayNode)
  {
    int oldAuto = this->VolumeDisplayNode->GetAutoWindowLevel();

    int disabledModify = this->VolumeDisplayNode->StartModify();
    this->VolumeDisplayNode->SetAutoWindowLevel(autoWindowLevel);
    this->VolumeDisplayNode->EndModify(disabledModify);

    if (autoWindowLevel != oldAuto)
    {
       emit this->autoWindowLevelValueChanged(autoWindowLevel);
    }

  }
}

// --------------------------------------------------------------------------
int qMRMLWindowLevelWidget::autoWindowLevel() const
{
  CTK_D(const qMRMLWindowLevelWidget);
  
  // Assumes settings of the sliders are all the same
  return d->AutoManualComboBox->currentIndex();
}


// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setWindowLevel(double window, double level)
{
  if (this->VolumeDisplayNode)
  {
    double oldWindow = this->VolumeDisplayNode->GetWindow();
    double oldLevel  = this->VolumeDisplayNode->GetLevel();

    int disabledModify = this->VolumeDisplayNode->StartModify();
    this->VolumeDisplayNode->SetWindow(window);
    this->VolumeDisplayNode->SetLevel(level);
    if (oldWindow != this->VolumeDisplayNode->GetWindow() ||
        oldLevel  != this->VolumeDisplayNode->GetLevel())
    {
      this->setAutoWindowLevel(0);
      emit this->windowLevelValuesChanged(window, level);
    }
    this->VolumeDisplayNode->EndModify(disabledModify);
  }
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMinMaxRangeValue(double min, double max)
{
  double window = max - min;
  double level = 0.5*(min+max);
  
  this->setWindowLevel(window, level);
}


// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setWindow(double window)
{
  if (this->VolumeDisplayNode)
  {
    double level  = this->VolumeDisplayNode->GetLevel();
    this->setWindowLevel(window, level);
  }
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setLevel(double level)
{
  if (this->VolumeDisplayNode)
  {
    double window = this->VolumeDisplayNode->GetWindow();
    this->setWindowLevel(window, level);
  }
}

// --------------------------------------------------------------------------
double qMRMLWindowLevelWidget::window() const
{
  CTK_D(const qMRMLWindowLevelWidget);

  double min = d->WindowLevelRangeWidget->minimumValue();
  double max = d->WindowLevelRangeWidget->maximumValue();

  return max - min;
}

// --------------------------------------------------------------------------
double qMRMLWindowLevelWidget::minimumValue() const
{
  CTK_D(const qMRMLWindowLevelWidget);

  double min = d->WindowLevelRangeWidget->minimumValue();
  return min;
}

// --------------------------------------------------------------------------
double qMRMLWindowLevelWidget::maximumValue() const
{
  CTK_D(const qMRMLWindowLevelWidget);

  double max = d->WindowLevelRangeWidget->maximumValue();
  return max;
}

// --------------------------------------------------------------------------
double qMRMLWindowLevelWidget::level() const
{
  CTK_D(const qMRMLWindowLevelWidget);

  double min = d->WindowLevelRangeWidget->minimumValue();
  double max = d->WindowLevelRangeWidget->maximumValue();

  return 0.5*(max + min);
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMRMLVolumeDisplayNode(vtkMRMLScalarVolumeDisplayNode* node)
{

  // each time the node is modified, the qt widgets are updated
  this->qvtkReconnect(this->VolumeDisplayNode, node, 
                       vtkCommand::ModifiedEvent, this, SLOT(updateWidgetFromMRML()));

  this->VolumeDisplayNode = node;

  this->setEnabled(node != 0);
  
  this->updateWidgetFromMRML();
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMRMLVolumeNode(vtkMRMLNode* node)
{
  this->setMRMLVolumeNode(vtkMRMLScalarVolumeNode::SafeDownCast(node));
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMRMLVolumeNode(vtkMRMLScalarVolumeNode* volumeNode)
{
  CTK_D(qMRMLWindowLevelWidget);
  
  if (volumeNode) 
  {
    double range[2];
    volumeNode->GetImageData()->GetScalarRange(range);
    d->WindowLevelRangeWidget->setMinimum(range[0]);
    d->WindowLevelRangeWidget->setMaximum(range[1]);
    this->setMRMLVolumeDisplayNode(vtkMRMLScalarVolumeDisplayNode::SafeDownCast(
              volumeNode->GetVolumeDisplayNode()));
  }
  this->VolumeNode = volumeNode;
}



// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMinimumValue(double min)
{
  this->setMinMaxRangeValue(min, this->maximumValue());
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMaximumValue(double max)
{
  this->setMinMaxRangeValue(this->minimumValue(), max);
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMinimum(double min)
{
  CTK_D(qMRMLWindowLevelWidget);
  d->WindowLevelRangeWidget->setMinimum(min);
}

// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::setMaximum(double max)
{
  CTK_D(qMRMLWindowLevelWidget);
  d->WindowLevelRangeWidget->setMaximum(max);
}



// --------------------------------------------------------------------------
void qMRMLWindowLevelWidget::updateWidgetFromMRML()
{
  CTK_D(qMRMLWindowLevelWidget);
  
  if (this->VolumeDisplayNode) 
  {
    d->AutoManualComboBox->setCurrentIndex(this->VolumeDisplayNode->GetAutoWindowLevel());

    double window = this->VolumeDisplayNode->GetWindow();
    double level = this->VolumeDisplayNode->GetLevel();
    double min = level - 0.5 * window;
    double max = level + 0.5 * window;

    if (this->VolumeNode) 
    {
      double range[2];
      this->VolumeNode->GetImageData()->GetScalarRange(range);
      d->WindowLevelRangeWidget->setMinimum(range[0]);
      d->WindowLevelRangeWidget->setMaximum(range[1]);
    }
    d->WindowLevelRangeWidget->setValues(min, max );
  }
}
