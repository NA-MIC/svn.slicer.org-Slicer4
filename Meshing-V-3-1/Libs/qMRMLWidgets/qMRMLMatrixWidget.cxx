
// Qt includes
#include <QDebug>

// qMRML includes
#include "qMRMLUtils.h"
#include "qMRMLMatrixWidget.h"

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLLinearTransformNode.h>

// VTK includes
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkWeakPointer.h>

//-----------------------------------------------------------------------------
class qMRMLMatrixWidgetPrivate: public ctkPrivate<qMRMLMatrixWidget>
{
public:
  qMRMLMatrixWidgetPrivate()
    {
    this->CoordinateReference = qMRMLMatrixWidget::GLOBAL; 
    this->MRMLTransformNode = 0; 
    }
  
  qMRMLMatrixWidget::CoordinateReferenceType   CoordinateReference;
  vtkWeakPointer<vtkMRMLLinearTransformNode>   MRMLTransformNode;
  // Warning, this is not the real "transform, the real can be retrieved
  // by qVTKAbstractMatrixWidget->transform();
  vtkSmartPointer<vtkTransform>                Transform;
};

// --------------------------------------------------------------------------
CTK_CONSTRUCTOR_1_ARG_CXX(qMRMLMatrixWidget, QWidget*);

// --------------------------------------------------------------------------
void qMRMLMatrixWidget::setCoordinateReference(CoordinateReferenceType _coordinateReference)
{
  CTK_D(qMRMLMatrixWidget);
  if (d->CoordinateReference == _coordinateReference)
    {
    return;
    }

  d->CoordinateReference = _coordinateReference;

  this->updateMatrix();
}

// --------------------------------------------------------------------------
qMRMLMatrixWidget::CoordinateReferenceType qMRMLMatrixWidget::coordinateReference() const
{
  return ctk_d()->CoordinateReference;
}

// --------------------------------------------------------------------------
void qMRMLMatrixWidget::setMRMLTransformNode(vtkMRMLNode* node)
{
  this->setMRMLTransformNode(vtkMRMLLinearTransformNode::SafeDownCast(node));
}

// --------------------------------------------------------------------------
void qMRMLMatrixWidget::setMRMLTransformNode(vtkMRMLLinearTransformNode* transformNode)
{
  CTK_D(qMRMLMatrixWidget);
  
  if (d->MRMLTransformNode == transformNode) 
    { 
    return; 
    }

  this->qvtkReconnect(d->MRMLTransformNode, transformNode,
                      vtkMRMLTransformableNode::TransformModifiedEvent, 
                      this, SLOT(updateMatrix())); 

  d->MRMLTransformNode = transformNode;
  
  this->updateMatrix();
}

// --------------------------------------------------------------------------
vtkMRMLLinearTransformNode* qMRMLMatrixWidget::mrmlTransformNode()const
{
  return ctk_d()->MRMLTransformNode;
}

// --------------------------------------------------------------------------
void qMRMLMatrixWidget::updateMatrix()
{
  CTK_D(qMRMLMatrixWidget);

  if (d->MRMLTransformNode == 0)
    {
    this->setMatrixInternal(0);
    d->Transform = 0;
    return;
    }
  
  vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  qMRMLUtils::getTransformInCoordinateSystem(
    d->MRMLTransformNode,
    d->CoordinateReference == Self::GLOBAL, 
    transform);
  // update the matrix with the new values.
  this->setMatrixInternal(transform->GetMatrix());
  // keep a ref on the transform otherwise, the matrix will be reset when transform
  // goes out of scope (because qVTKAbstractMatrixWidget has a weak ref on the matrix).
  d->Transform = transform;  
}
