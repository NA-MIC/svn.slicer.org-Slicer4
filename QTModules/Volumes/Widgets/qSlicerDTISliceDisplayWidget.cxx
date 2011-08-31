/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

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

#include "qSlicerDTISliceDisplayWidget.h"
#include "ui_qSlicerDTISliceDisplayWidget.h"

// Qt includes
#include <QDebug>

// MRML includes
#include "vtkMRMLDiffusionTensorDisplayPropertiesNode.h"
#include "vtkMRMLDiffusionTensorVolumeSliceDisplayNode.h"

// VTK includes
#include "vtkPolyData.h"

// STD includes

//-----------------------------------------------------------------------------
class qSlicerDTISliceDisplayWidgetPrivate
  : public Ui_qSlicerDTISliceDisplayWidget
{
  Q_DECLARE_PUBLIC(qSlicerDTISliceDisplayWidget);
protected:
  qSlicerDTISliceDisplayWidget* const q_ptr;
public:
  qSlicerDTISliceDisplayWidgetPrivate(qSlicerDTISliceDisplayWidget& object);
  ~qSlicerDTISliceDisplayWidgetPrivate();
  void init();
  void computeScalarBounds(double scalarBounds[2]);
  vtkMRMLDiffusionTensorVolumeSliceDisplayNode* DisplayNode;
};

//-----------------------------------------------------------------------------
qSlicerDTISliceDisplayWidgetPrivate
::qSlicerDTISliceDisplayWidgetPrivate(
  qSlicerDTISliceDisplayWidget& object)
  : q_ptr(&object)
{
  this->DisplayNode = 0;
}

//-----------------------------------------------------------------------------
qSlicerDTISliceDisplayWidgetPrivate
::~qSlicerDTISliceDisplayWidgetPrivate()
{
}

//-----------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidgetPrivate::init()
{
  Q_Q(qSlicerDTISliceDisplayWidget);

  this->setupUi(q);

  this->LineEigenVectorComboBox->setItemData(0, vtkMRMLDiffusionTensorDisplayPropertiesNode::Major);
  this->LineEigenVectorComboBox->setItemData(1, vtkMRMLDiffusionTensorDisplayPropertiesNode::Middle);
  this->LineEigenVectorComboBox->setItemData(2, vtkMRMLDiffusionTensorDisplayPropertiesNode::Minor);

  QObject::connect(this->GlyphVisibilityCheckBox, SIGNAL(toggled(bool)),
                   q, SLOT(setVisibility(bool)));
  QObject::connect(this->GlyphOpacitySliderWidget, SIGNAL(valueChanged(double)),
                   q, SLOT(setOpacity(double)));
  QObject::connect(this->GlyphScalarColorTableComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),
                   q, SLOT(setColorMap(vtkMRMLNode*)));
  QObject::connect(this->GlyphColorByScalarComboBox, SIGNAL(scalarInvariantChanged(int)),
                   q, SLOT(setColorGlyphBy(int)));
  QObject::connect(this->GlyphManualScalarRangeCheckBox, SIGNAL(toggled(bool)),
                   q, SLOT(setManualScalarRange(bool)));
  QObject::connect(this->GlyphScalarRangeWidget, SIGNAL(valuesChanged(double, double)),
                   q, SLOT(setScalarRange(double, double)));
  QObject::connect(this->GlyphGeometryComboBox, SIGNAL(currentIndexChanged(int)),
                   q, SLOT(setGlyphGeometry(int)));
  QObject::connect(this->GlyphScaleSliderWidget, SIGNAL(valueChanged(double)),
                   q, SLOT(setGlyphScaleFactor(double)));
  QObject::connect(this->GlyphSpacingSliderWidget, SIGNAL(valueChanged(double)),
                   q, SLOT(setGlyphSpacing(double)));
  QObject::connect(this->LineEigenVectorComboBox, SIGNAL(currentIndexChanged(int)),
                   q, SLOT(setGlyphEigenVector(int)));
  QObject::connect(this->TubeEigenVectorComboBox, SIGNAL(currentIndexChanged(int)),
                   q, SLOT(setGlyphEigenVector(int)));
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidgetPrivate::computeScalarBounds(double scalarBounds[2])
{
  Q_Q(qSlicerDTISliceDisplayWidget);
  const int ScalarInvariant = (q->displayPropertiesNode() ?
         q->displayPropertiesNode()->GetColorGlyphBy() : -1);

  if (vtkMRMLDiffusionTensorDisplayPropertiesNode::ScalarInvariantHasKnownScalarRange(ScalarInvariant))
  {
    vtkMRMLDiffusionTensorDisplayPropertiesNode::ScalarInvariantKnownScalarRange(ScalarInvariant, scalarBounds);
  } else {
      vtkPolyData* glyphs = this->DisplayNode->GetPolyData();
      if (glyphs)
        {
        glyphs->GetScalarRange(scalarBounds);
        }
  }
//  Commented this so the glyphs and bundles are colored consistently
//  this->DisplayNode->GetPolyData();
//  scalarBounds[0] = qMin (scalarBounds[0], q->displayNode()->GetScalarRange()[0]);
//  scalarBounds[1] = qMax (scalarBounds[1], q->displayNode()->GetScalarRange()[1]);
}

// --------------------------------------------------------------------------
qSlicerDTISliceDisplayWidget
::qSlicerDTISliceDisplayWidget(QWidget* parentWidget)
  : Superclass(parentWidget)
  , d_ptr(new qSlicerDTISliceDisplayWidgetPrivate(*this))
{
  Q_D(qSlicerDTISliceDisplayWidget);
  d->init();

  // disable as there is no MRML Node associated with the widget
  this->setEnabled(d->DisplayNode != 0);
}

// --------------------------------------------------------------------------
qSlicerDTISliceDisplayWidget
::~qSlicerDTISliceDisplayWidget()
{
}

// --------------------------------------------------------------------------
vtkMRMLDiffusionTensorDisplayPropertiesNode* qSlicerDTISliceDisplayWidget
::displayPropertiesNode()const
{
  Q_D(const qSlicerDTISliceDisplayWidget);
  return d->DisplayNode ?
    d->DisplayNode->GetDiffusionTensorDisplayPropertiesNode() : 0;
}

// --------------------------------------------------------------------------
vtkMRMLDiffusionTensorVolumeSliceDisplayNode* qSlicerDTISliceDisplayWidget
::displayNode()const
{
  Q_D(const qSlicerDTISliceDisplayWidget);
  return d->DisplayNode;
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setMRMLDTISliceDisplayNode(vtkMRMLNode* node)
{
  this->setMRMLDTISliceDisplayNode(
    vtkMRMLDiffusionTensorVolumeSliceDisplayNode::SafeDownCast(node));
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setMRMLDTISliceDisplayNode(
  vtkMRMLDiffusionTensorVolumeSliceDisplayNode* displayNode)
{
  Q_D(qSlicerDTISliceDisplayWidget);

  vtkMRMLDiffusionTensorVolumeSliceDisplayNode* oldDisplayNode =
    this->displayNode();
  vtkMRMLDiffusionTensorDisplayPropertiesNode* oldDisplayPropertiesNode =
    this->displayPropertiesNode();

  d->DisplayNode = displayNode;

  qvtkReconnect(oldDisplayNode, this->displayNode(),vtkCommand::ModifiedEvent,
                this, SLOT(updateWidgetFromMRML()));
  qvtkReconnect(oldDisplayPropertiesNode, this->displayPropertiesNode(),
                vtkCommand::ModifiedEvent, this, SLOT(updateWidgetFromMRML()));

  this->updateWidgetFromMRML();
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::updateWidgetFromMRML()
{
  Q_D(qSlicerDTISliceDisplayWidget);
  this->setEnabled(d->DisplayNode != 0);
  if (!d->DisplayNode)
    {
    return;
    }

  d->GlyphVisibilityCheckBox->setChecked(
    d->DisplayNode->GetVisibility());
  d->GlyphOpacitySliderWidget->setValue(
    d->DisplayNode->GetOpacity());
  d->GlyphScalarColorTableComboBox->setCurrentNode(
    d->DisplayNode->GetColorNode());
  d->GlyphManualScalarRangeCheckBox->setChecked(
    d->DisplayNode->GetAutoScalarRange() == 0);
  double scalarBounds[2];
  d->computeScalarBounds(scalarBounds);
  double singleStep = qAbs(scalarBounds[1] - scalarBounds[0]) / 100.;
  double i = 1.;
  int decimals = 0;
  while (i > singleStep)
    {
    ++decimals;
    i /= 10.;
    }
  // TBD: blockSignals are not very important, just reduce the noise resulting
  // from unnecessary updates.
  d->GlyphScalarRangeWidget->blockSignals(true);
  d->GlyphScalarRangeWidget->setDecimals(decimals);
  d->GlyphScalarRangeWidget->setSingleStep(i);
  d->GlyphScalarRangeWidget->setRange(scalarBounds[0], scalarBounds[1]);
  d->GlyphScalarRangeWidget->blockSignals(false);
  double scalarRange[2];
  d->DisplayNode->GetScalarRange(scalarRange);
  d->GlyphScalarRangeWidget->setValues(scalarRange[0], scalarRange[1]);

  vtkMRMLDiffusionTensorDisplayPropertiesNode* displayPropertiesNode =
    this->displayPropertiesNode();
  if (displayPropertiesNode)
    {
    d->GlyphColorByScalarComboBox->setScalarInvariant(displayPropertiesNode->GetColorGlyphBy());
    d->GlyphGeometryComboBox->setCurrentIndex(displayPropertiesNode->GetGlyphGeometry());
    d->GlyphScaleSliderWidget->setValue(displayPropertiesNode->GetGlyphScaleFactor());
    d->GlyphSpacingSliderWidget->setValue(
      displayPropertiesNode->GetLineGlyphResolution());
    int index = d->LineEigenVectorComboBox->findData(
      QVariant(displayPropertiesNode->GetGlyphEigenvector()));
    d->LineEigenVectorComboBox->setCurrentIndex(index);
    d->TubeEigenVectorComboBox->setCurrentIndex(index);
    }
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setColorGlyphBy(int scalarInvariant)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  
  if (!this->displayPropertiesNode())
    {
    return;
    }
  this->displayPropertiesNode()->SetColorGlyphBy(scalarInvariant);

  if (d->DisplayNode && (d->DisplayNode->GetAutoScalarRange()))
  {
    double scalarRange[2];
    d->DisplayNode->GetScalarRange(scalarRange);
    
    this->setScalarRange(scalarRange[0], scalarRange[1]);
  }


  if ( d->DisplayNode && (
    this->displayPropertiesNode()->GetScalarInvariant() == vtkMRMLDiffusionTensorDisplayPropertiesNode::ColorOrientation ||
    this->displayPropertiesNode()->GetScalarInvariant() == vtkMRMLDiffusionTensorDisplayPropertiesNode::ColorOrientationMiddleEigenvector ||
    this->displayPropertiesNode()->GetScalarInvariant() == vtkMRMLDiffusionTensorDisplayPropertiesNode::ColorOrientationMinEigenvector 
    ) )
  {
    d->DisplayNode->SetAndObserveColorNodeID("vtkMRMLColorTableNodeFullRainbow");
    d->GlyphScalarColorTableComboBox->setEnabled(false);
  } else {
    d->GlyphScalarColorTableComboBox->setEnabled(true);
  }

}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setColorMap(vtkMRMLNode* colorNode)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  if (!d->DisplayNode)
    {
    return;
    }
  d->DisplayNode->SetAndObserveColorNodeID(colorNode ? colorNode->GetID() : "");
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setOpacity(double opacity)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  Q_ASSERT(d->DisplayNode);
  d->DisplayNode->SetOpacity(opacity);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setVisibility(bool visible)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  Q_ASSERT(d->DisplayNode);
  d->DisplayNode->SetVisibility(visible);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setManualScalarRange(bool manual)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  Q_ASSERT(d->DisplayNode);
  d->DisplayNode->SetAutoScalarRange(manual ? 0 : 1);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setScalarRange(double min, double max)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  Q_ASSERT(d->DisplayNode);
  d->DisplayNode->SetScalarRange(min, max);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setGlyphGeometry(int index)
{
  Q_ASSERT(this->displayPropertiesNode());
  // 0 = Lines
  // 1 = Tubes
  // 2 = Ellipsoids
  // 3 = Superquadrics
  this->displayPropertiesNode()->SetGlyphGeometry(index);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setGlyphScaleFactor(double scaleFactor)
{
  Q_ASSERT(this->displayPropertiesNode());
  this->displayPropertiesNode()->SetGlyphScaleFactor(scaleFactor);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setGlyphSpacing(double spacing)
{
  Q_ASSERT(this->displayPropertiesNode());
  this->displayPropertiesNode()->SetLineGlyphResolution(spacing);
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setGlyphEigenVector(int index)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  Q_ASSERT(this->displayPropertiesNode());
  int eigenVector = d->LineEigenVectorComboBox->itemData(index).toInt();
  this->displayPropertiesNode()->SetGlyphEigenvector(eigenVector);
}

// --------------------------------------------------------------------------
bool qSlicerDTISliceDisplayWidget::isVisibilityHidden()const
{
  Q_D(const qSlicerDTISliceDisplayWidget);
  return d->GlyphVisibilityLabel->isVisibleTo(
    const_cast<qSlicerDTISliceDisplayWidget*>(this));
}

// --------------------------------------------------------------------------
void qSlicerDTISliceDisplayWidget::setVisibilityHidden(bool hide)
{
  Q_D(qSlicerDTISliceDisplayWidget);
  d->GlyphVisibilityLabel->setVisible(!hide);
  d->GlyphVisibilityCheckBox->setVisible(!hide);
}
