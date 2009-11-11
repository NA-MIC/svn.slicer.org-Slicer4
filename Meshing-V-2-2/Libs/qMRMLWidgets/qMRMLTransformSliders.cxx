
#include "qMRMLTransformSliders.h"
#include "qMRMLUtils.h"
#include "qMRMLLinearTransformSlider.h"

#include "ui_qMRMLTransformSliders.h" 

#include "vtkMRMLLinearTransformNode.h"

#include "vtkSmartPointer.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"

#include <QDebug>

//-----------------------------------------------------------------------------
class qMRMLTransformSliders::qInternal: public Ui::qMRMLTransformSliders
{
public:
  qInternal()
    {
    this->MRMLTransformNode = 0; 
    }
  TransformType                          TypeOfTransform;
  
  vtkMRMLLinearTransformNode*            MRMLTransformNode; 
};

// --------------------------------------------------------------------------
qMRMLTransformSliders::qMRMLTransformSliders(QWidget* parent) : Superclass(parent)
{
  this->Internal = new qInternal; 
  this->Internal->setupUi(this);
  
  this->setCoordinateReference(Self::GLOBAL); 
  this->setTypeOfTransform(Self::TRANSLATION);
  
  
  this->connect(this->Internal->LRSlider, SIGNAL(sliderMoved(double)), 
                SLOT(onLRSliderPositionChanged(double)));
  this->connect(this->Internal->PASlider, SIGNAL(sliderMoved(double)), 
                SLOT(onPASliderPositionChanged(double)));
  this->connect(this->Internal->ISSlider, SIGNAL(sliderMoved(double)), 
                SLOT(onISSliderPositionChanged(double)));

  this->connect(this->Internal->MinValueSpinBox, SIGNAL(valueChanged(double)), 
                SLOT(onMinimumChanged(double)));
  this->connect(this->Internal->MaxValueSpinBox, SIGNAL(valueChanged(double)), 
                SLOT(onMaximumChanged(double)));
  // the default values of min and max are set in the .ui file
  this->onMinimumChanged(this->Internal->MinValueSpinBox->value());
  this->onMaximumChanged(this->Internal->MaxValueSpinBox->value());

  // disable as there is not MRML Node associated with the widget
  this->setEnabled(false);
}

// --------------------------------------------------------------------------
qMRMLTransformSliders::~qMRMLTransformSliders()
{
  delete this->Internal; 
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setCoordinateReference(CoordinateReferenceType coordinateReference)
{
  qMRMLLinearTransformSlider::CoordinateReferenceType ref = qMRMLLinearTransformSlider::GLOBAL; 
  if (coordinateReference == LOCAL)
    {
    ref = qMRMLLinearTransformSlider::LOCAL; 
    }
  this->Internal->LRSlider->setCoordinateReference(ref);
  this->Internal->PASlider->setCoordinateReference(ref);
  this->Internal->ISSlider->setCoordinateReference(ref);
}

// --------------------------------------------------------------------------
qMRMLTransformSliders::CoordinateReferenceType qMRMLTransformSliders::coordinateReference() const
{
  // Assumes settings of the sliders are all the same
  qMRMLLinearTransformSlider::CoordinateReferenceType ref = 
    this->Internal->LRSlider->coordinateReference(); 
  return (ref == qMRMLLinearTransformSlider::GLOBAL) ? GLOBAL : LOCAL;
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setTypeOfTransform(TransformType typeOfTransform)
{
  if (this->Internal->TypeOfTransform == typeOfTransform) { return; }
  if (typeOfTransform == Self::TRANSLATION)
    {
    this->Internal->LRSlider->setTypeOfTransform(qMRMLLinearTransformSlider::TRANSLATION_LR); 
    this->Internal->PASlider->setTypeOfTransform(qMRMLLinearTransformSlider::TRANSLATION_PA);
    this->Internal->ISSlider->setTypeOfTransform(qMRMLLinearTransformSlider::TRANSLATION_IS);  
    }
  else if (typeOfTransform == Self::ROTATION)
    {
    this->Internal->LRSlider->setTypeOfTransform(qMRMLLinearTransformSlider::ROTATION_LR);
    this->Internal->PASlider->setTypeOfTransform(qMRMLLinearTransformSlider::ROTATION_PA);  
    this->Internal->ISSlider->setTypeOfTransform(qMRMLLinearTransformSlider::ROTATION_IS); 
    }
  this->Internal->TypeOfTransform = typeOfTransform; 
}

// --------------------------------------------------------------------------
qMRMLTransformSliders::TransformType qMRMLTransformSliders::typeOfTransform() const
{
  return this->Internal->TypeOfTransform;
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setMRMLTransformNode(vtkMRMLNode* node)
{
  this->setMRMLTransformNode(vtkMRMLLinearTransformNode::SafeDownCast(node));
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setMRMLTransformNode(vtkMRMLLinearTransformNode* transformNode)
{
  this->Internal->LRSlider->setMRMLTransformNode(transformNode); 
  this->Internal->PASlider->setMRMLTransformNode(transformNode); 
  this->Internal->ISSlider->setMRMLTransformNode(transformNode); 
  this->setEnabled(transformNode != 0); 
  this->Internal->MRMLTransformNode = transformNode; 
}

// --------------------------------------------------------------------------
vtkMRMLLinearTransformNode* qMRMLTransformSliders::mrmlTransformNode()const
{
  return this->Internal->MRMLTransformNode;
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setTitle(const QString& title)
{
  this->Internal->SlidersGroupBox->setTitle(title); 
}

// --------------------------------------------------------------------------
QString qMRMLTransformSliders::title()const
{
  return this->Internal->SlidersGroupBox->title(); 
}

// --------------------------------------------------------------------------
double qMRMLTransformSliders::minimum()const
{
  return this->Internal->MinValueSpinBox->value();
}

// --------------------------------------------------------------------------
double qMRMLTransformSliders::maximum()const
{
  return this->Internal->MaxValueSpinBox->value(); 
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setMinimum(double min)
{
  this->Internal->MinValueSpinBox->setValue(min); 
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setMaximum(double max)
{
  this->Internal->MaxValueSpinBox->setValue(max);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setRange(double min, double max)
{
  // Could be optimized here by blocking signals on spinboxes and manually
  // call the setRange method on the sliders. Does it really worth it ?
  this->Internal->MinValueSpinBox->setValue(min);
  this->Internal->MaxValueSpinBox->setValue(max);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::onMinimumChanged(double min)
{
  this->Internal->LRSlider->setMinimum(min);
  this->Internal->PASlider->setMinimum(min);
  this->Internal->ISSlider->setMinimum(min);
}
 
// --------------------------------------------------------------------------
void qMRMLTransformSliders::onMaximumChanged(double max)
{
  this->Internal->LRSlider->setMaximum(max);
  this->Internal->PASlider->setMaximum(max);
  this->Internal->ISSlider->setMaximum(max);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setMinMaxVisible(bool visible)
{
  this->Internal->MinMaxWidget->setVisible(visible);
}

// --------------------------------------------------------------------------
bool qMRMLTransformSliders::isMinMaxVisible()const
{
  return this->Internal->MinMaxWidget->isVisibleTo(
    const_cast<qMRMLTransformSliders*>(this));
}

// --------------------------------------------------------------------------
double qMRMLTransformSliders::singleStep()const
{
  // Assumes settings of the sliders are all the same
  return this->Internal->PASlider->singleStep(); 
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setSingleStep(double step)
{
  this->Internal->LRSlider->setSingleStep(step);
  this->Internal->PASlider->setSingleStep(step);
  this->Internal->ISSlider->setSingleStep(step);
}

// --------------------------------------------------------------------------
QString qMRMLTransformSliders::getLRLabel()const
{
  return this->Internal->LRLabel->text();
}

// --------------------------------------------------------------------------
QString qMRMLTransformSliders::getPALabel()const
{
  return this->Internal->PALabel->text();
}

// --------------------------------------------------------------------------
QString qMRMLTransformSliders::getISLabel()const
{
  return this->Internal->ISLabel->text();
}
  
// --------------------------------------------------------------------------
void qMRMLTransformSliders::setLRLabel(const QString& label)
{
  this->Internal->LRLabel->setText(label);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setPALabel(const QString& label)
{
  this->Internal->PALabel->setText(label);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::setISLabel(const QString& label)
{
  this->Internal->ISLabel->setText(label);
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::reset()
{
  this->Internal->LRSlider->reset();
  this->Internal->PASlider->reset();
  this->Internal->ISSlider->reset();
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::onLRSliderPositionChanged(double position)
{
  if (this->typeOfTransform() == Self::ROTATION)
    {
    // Reset other sliders
    this->Internal->PASlider->reset();
    this->Internal->ISSlider->reset();
    }
  else
    {
    emit this->sliderMoved(); 
    }
  // Update LR slider
  this->Internal->LRSlider->applyTransformation(position); 
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::onPASliderPositionChanged(double position)
{
  if (this->typeOfTransform() == Self::ROTATION)
    {
    // Reset other sliders
    this->Internal->LRSlider->reset();
    this->Internal->ISSlider->reset();
    }
  else
    {
    emit this->sliderMoved(); 
    }
  // Update PA slider
  this->Internal->PASlider->applyTransformation(position);  
}

// --------------------------------------------------------------------------
void qMRMLTransformSliders::onISSliderPositionChanged(double position)
{
  if (this->typeOfTransform() == Self::ROTATION)
    {
    // Reset other sliders
    this->Internal->LRSlider->reset();
    this->Internal->PASlider->reset();
    }
  else
    {
    emit this->sliderMoved(); 
    } 
  // Update IS slider
  this->Internal->ISSlider->applyTransformation(position); 
}
