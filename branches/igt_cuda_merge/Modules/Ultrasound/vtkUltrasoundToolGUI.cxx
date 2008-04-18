#include "vtkUltrasoundToolGUI.h"
#include "vtkObjectFactory.h"

#include "vtkMatrix4x4.h"
#include "vtkRenderer.h"
#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkCylinderSource.h"
#include "vtkKWCheckButton.h"
#include "vtkCallbackCommand.h"
#include "vtkKWApplication.h"
#include "vtkKWMatrixWidget.h"

vtkCxxRevisionMacro(vtkUltrasoundToolGUI, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundToolGUI);

vtkUltrasoundToolGUI::vtkUltrasoundToolGUI()
{
this->Renderer = NULL;

    this->Rod = NULL;
    this->RodMapper = NULL;
    this->RodActor = NULL;

    this->cb_Enabled = NULL;

    this->Transform = vtkMatrix4x4::New();
    this->Transform->Identity();


    this->GUICallbackCommand = vtkCallbackCommand::New();
    this->GUICallbackCommand->SetCallback(&vtkUltrasoundToolGUI::ProcessGUIEventsStatic);
    this->GUICallbackCommand->SetClientData(this);

}
vtkUltrasoundToolGUI::~vtkUltrasoundToolGUI()
{
    this->Transform->Delete();
}

void vtkUltrasoundToolGUI::CreateWidget()
{
    this->Superclass::CreateWidget();
    

    this->cb_Enabled = vtkKWCheckButton::New();
    this->cb_Enabled->SetParent(this);
    this->cb_Enabled->Create();
    this->cb_Enabled->SetText("Enabled");
    this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->cb_Enabled->GetWidgetName());
    this->cb_Enabled->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);


    this->TransformWidget = vtkKWMatrixWidget::New();
    this->TransformWidget->SetParent(this);
    this->TransformWidget->Create();
    //this->TransformWidget->Re();
    this->TransformWidget->SetNumberOfColumns(3);
    this->TransformWidget->SetNumberOfRows(2);
    this->TransformWidget->SetElementValue(0,0,0);
    this->TransformWidget->SetElementValue(0,1,0);
    this->TransformWidget->SetElementValue(0,2,0);    
    this->TransformWidget->SetElementValue(1,0,0);
    this->TransformWidget->SetElementValue(1,1,0);
    this->TransformWidget->SetElementValue(1,2,0);
    this->GetApplication()->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->TransformWidget->GetWidgetName());
    this->TransformWidget->AddObserver(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);

}

void vtkUltrasoundToolGUI::AddGUIObservers()
{

}


void vtkUltrasoundToolGUI::SetTransformMatrix(vtkMatrix4x4* transform)
{
    this->Transform->DeepCopy(transform);
    
    if (this->Rod != NULL)
    {
        
    }
}


void vtkUltrasoundToolGUI::ProcessGUIEventsStatic(vtkObject *caller, unsigned long ev, void *callData, void* object)
{
    ((vtkUltrasoundToolGUI*)callData)->ProcessGUIEvents(caller, ev, callData);
}

void vtkUltrasoundToolGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long ev, void *callData )
{
    if(caller == this->cb_Enabled && this->Renderer != NULL)
    {
        if (this->cb_Enabled->GetSelectedState() == 1)
        {
            if (this->Rod == NULL)
            {
                this->Rod = vtkCylinderSource::New();
                this->RodMapper = vtkPolyDataMapper::New();

                this->Rod->SetHeight(500);
                this->Rod->SetRadius(5);
                this->Rod->SetCenter(0, 250, 0);
                this->Rod->SetResolution(20);

                this->RodMapper->SetInput(this->Rod->GetOutput());

                this->RodActor = vtkActor::New();
                this->RodActor->SetMapper(this->RodMapper);
            }
            this->Renderer->AddActor(this->RodActor);
        }
        else
        {
            this->Renderer->RemoveActor(this->RodActor);
        }
    }

    else if (caller == this->TransformWidget && this->Rod != NULL)
    {
        this->RodActor->SetPosition(
            this->TransformWidget->GetElementValueAsDouble(0,0),
            this->TransformWidget->GetElementValueAsDouble(0,1),
            this->TransformWidget->GetElementValueAsDouble(0,2));

        this->RodActor->SetOrientation(
            this->TransformWidget->GetElementValueAsDouble(1,0),
            this->TransformWidget->GetElementValueAsDouble(1,1),
            this->TransformWidget->GetElementValueAsDouble(1,2));

        if (this->Renderer != NULL)
            this->Renderer->Render();
    }
}
