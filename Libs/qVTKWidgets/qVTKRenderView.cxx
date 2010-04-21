#include "qVTKRenderView.h"
#include "qVTKRenderView_p.h"

// VTK includes
#include <vtkRendererCollection.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTextProperty.h>

// QT includes
#include <QTimer>
#include <QVBoxLayout>

// --------------------------------------------------------------------------
// qVTKRenderViewPrivate methods

// --------------------------------------------------------------------------
qVTKRenderViewPrivate::qVTKRenderViewPrivate()
{
  this->Renderer = vtkSmartPointer<vtkRenderer>::New();
  this->RenderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  this->Axes = vtkSmartPointer<vtkAxesActor>::New();
  this->Orientation = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
  this->CornerAnnotation = vtkSmartPointer<vtkCornerAnnotation>::New();
  this->RenderPending = false;
}

// --------------------------------------------------------------------------
void qVTKRenderViewPrivate::setupCornerAnnotation()
{
  if (!this->Renderer->HasViewProp(this->CornerAnnotation))
    {
    this->Renderer->AddViewProp(this->CornerAnnotation);
    this->CornerAnnotation->SetMaximumLineHeight(0.07);
    vtkTextProperty *tprop = this->CornerAnnotation->GetTextProperty();
    tprop->ShadowOn();
    }
  this->CornerAnnotation->ClearAllTexts();
}

//---------------------------------------------------------------------------
void qVTKRenderViewPrivate::setupRendering()
{
  Q_ASSERT(this->RenderWindow);
  this->RenderWindow->SetAlphaBitPlanes(1);
  this->RenderWindow->SetMultiSamples(0);
  this->RenderWindow->StereoCapableWindowOn();
  
  this->RenderWindow->GetRenderers()->RemoveAllItems();
  
  // Add renderer
  this->RenderWindow->AddRenderer(this->Renderer);
  
  // Setup the corner annotation
  this->setupCornerAnnotation();

  this->VTKWidget->SetRenderWindow(this->RenderWindow);
}

//---------------------------------------------------------------------------
void qVTKRenderViewPrivate::setupDefaultInteractor()
{
  CTK_P(qVTKRenderView);
  p->setInteractor(this->RenderWindow->GetInteractor());
}

//---------------------------------------------------------------------------
// qVTKRenderView methods

// --------------------------------------------------------------------------
qVTKRenderView::qVTKRenderView(QWidget* _parent) : Superclass(_parent)
{
  CTK_INIT_PRIVATE(qVTKRenderView);
  CTK_D(qVTKRenderView);
  
  d->VTKWidget = new QVTKWidget(this);
  this->setLayout(new QVBoxLayout);
  this->layout()->setMargin(0);
  this->layout()->setSpacing(0);
  this->layout()->addWidget(d->VTKWidget);

  d->setupRendering();
  d->setupDefaultInteractor();
}

// --------------------------------------------------------------------------
qVTKRenderView::~qVTKRenderView()
{
}

//----------------------------------------------------------------------------
CTK_GET_CXX(qVTKRenderView, vtkRenderWindowInteractor*, interactor, CurrentInteractor);

//----------------------------------------------------------------------------
void qVTKRenderView::scheduleRender()
{
  CTK_D(qVTKRenderView);
  if (!d->RenderPending)
    {
    d->RenderPending = true;
    QTimer::singleShot(0, this, SLOT(forceRender()));
    }
}

//----------------------------------------------------------------------------
void qVTKRenderView::forceRender()
{
  CTK_D(qVTKRenderView);
  d->RenderWindow->Render();
  d->RenderPending = false;
}

//----------------------------------------------------------------------------
void qVTKRenderView::setInteractor(vtkRenderWindowInteractor* newInteractor)
{
  Q_ASSERT(newInteractor);
  CTK_D(qVTKRenderView);
  d->RenderWindow->SetInteractor(newInteractor);
  d->Orientation->SetOrientationMarker(d->Axes);
  d->Orientation->SetInteractor(newInteractor);
  d->Orientation->SetEnabled(1);
  d->Orientation->InteractiveOff();
  d->CurrentInteractor = newInteractor; 
}

//----------------------------------------------------------------------------
void qVTKRenderView::setCornerAnnotationText(const QString& text)
{
  CTK_D(qVTKRenderView);
  d->CornerAnnotation->ClearAllTexts();
  d->CornerAnnotation->SetText(2, text.toLatin1());
}

// --------------------------------------------------------------------------
void qVTKRenderView::setBackgroundColor(double r, double g, double b)
{
  CTK_D(qVTKRenderView);
  double background_color[3] = {r, g, b};
  d->Renderer->SetBackground(background_color);
}

//----------------------------------------------------------------------------
void qVTKRenderView::resetCamera()
{
  CTK_D(qVTKRenderView);
  d->Renderer->ResetCamera();
}
