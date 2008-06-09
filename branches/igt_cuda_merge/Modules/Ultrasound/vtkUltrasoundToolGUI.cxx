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

#include "vtkLinearTransform.h"

#include "vtkKWScaleWithEntry.h"
#include "vtkKWScale.h"
#include "vtkUltrasoundScanPlane.h"

#ifdef ASCENSION_MINIBIRD_SUPPORT
#include "vtkMiniBirdInstrumentTracker.h"
#endif /* ASCENSION_MINIBIRD_SUPPORT */


vtkCxxRevisionMacro(vtkUltrasoundToolGUI, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundToolGUI);

vtkUltrasoundToolGUI::vtkUltrasoundToolGUI()
{
    this->Renderer = NULL;

    this->Rod = NULL;
    this->RodMapper = NULL;
    this->RodActor = NULL;
    this->ScanPlane = NULL;

    this->cb_Enabled = NULL;

    this->Transform = vtkMatrix4x4::New();
    this->Transform->Identity();

    for (unsigned int i = 0; i < 3; i++)
    {
        this->ToolAdjustmentScales[i]  = NULL;
        this->ProbeAdjustmentScales[i] = NULL;
    }

    this->Tracker = NULL;
    this->GUICallbackCommand = vtkCallbackCommand::New();
    this->GUICallbackCommand->SetCallback(&vtkUltrasoundToolGUI::ProcessGUIEventsStatic);
    this->GUICallbackCommand->SetClientData(this);
}
vtkUltrasoundToolGUI::~vtkUltrasoundToolGUI()
{
    this->Transform->Delete();
#ifdef ASCENSION_MINIBIRD_SUPPORT
    if (this->Tracker != NULL)
        this->Tracker->Delete();
#endif /* ASCENSION_MINIBIRD_SUPPORT */
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

    // Rediculous stile to set stuff... KW is praised!
    for (unsigned int i = 0; i < 3; i++)
    {
        // TOOL
        this->ToolAdjustmentScales[i]  = vtkKWScaleWithEntry::New();
        this->ToolAdjustmentScales[i]->SetParent(this);
        this->ToolAdjustmentScales[i]->Create();
        this->ToolAdjustmentScales[i]->SetRange(-20, 20);
        this->ToolAdjustmentScales[i]->SetValue(0);
        this->ToolAdjustmentScales[i]->SetResolution(0.01);
        this->GetApplication()->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
            this->ToolAdjustmentScales[i]->GetWidgetName());
        this->ToolAdjustmentScales[i]->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)this->GUICallbackCommand);
    }
    this->ToolAdjustmentScales[0]->SetLabelText("Tool Adjust X:");
    this->ToolAdjustmentScales[1]->SetLabelText("Tool Adjust Y:");
    this->ToolAdjustmentScales[2]->SetLabelText("Tool Adjust Z:");

    for (unsigned i = 0; i < 3; i++)
    {
        // PROBE
        this->ProbeAdjustmentScales[i]  = vtkKWScaleWithEntry::New();
        this->ProbeAdjustmentScales[i]->SetParent(this);
        this->ProbeAdjustmentScales[i]->Create();
        this->ProbeAdjustmentScales[i]->SetRange(-20, 20);
        this->ProbeAdjustmentScales[i]->SetValue(0);
        this->ProbeAdjustmentScales[i]->SetResolution(0.01);
        this->GetApplication()->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
            this->ProbeAdjustmentScales[i]->GetWidgetName());
        this->ProbeAdjustmentScales[i]->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)this->GUICallbackCommand);
    }

    this->ProbeAdjustmentScales[0]->SetLabelText("Probe Adjust X:");
    this->ProbeAdjustmentScales[1]->SetLabelText("Probe Adjust Y:");
    this->ProbeAdjustmentScales[2]->SetLabelText("Probe Adjust Z:");

}

void vtkUltrasoundToolGUI::AddGUIObservers()
{

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
#ifdef ASCENSION_MINIBIRD_SUPPORT
            if (this->Tracker == NULL)
                this->Tracker = vtkMiniBirdInstrumentTracker::New();
#else

#endif
            if (this->RodActor == NULL)
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

                this->ScanPlane = vtkUltrasoundScanPlane::New();
                this->ScanPlane->SetScale(10,10);
                this->ScanPlane->SetRenderer(this->Renderer);
            }
            this->Renderer->AddActor(this->RodActor);

        }
        else
        {
            this->Renderer->RemoveActor(this->RodActor);

#ifdef ASCENSION_MINIBIRD_SUPPORT
            if (this->Tracker != NULL)
            {
                this->Tracker->Delete();
                this->Tracker = NULL; 
            }
#endif /* ASCENSION_MINIBIRD_SUPPORT */
        }
    }

    else if (caller == this->TransformWidget && this->Rod != NULL)
    {
        //this->RodActor->SetPosition(
        //    this->TransformWidget->GetElementValueAsDouble(0,0),
        //    this->TransformWidget->GetElementValueAsDouble(0,1),
        //    this->TransformWidget->GetElementValueAsDouble(0,2));

        //this->RodActor->SetOrientation(
        //    this->TransformWidget->GetElementValueAsDouble(1,0),
        //    this->TransformWidget->GetElementValueAsDouble(1,1),
        //    this->TransformWidget->GetElementValueAsDouble(1,2));


        if (this->Renderer != NULL)
            this->Renderer->Render();
    }

    if (ev == vtkKWScale::ScaleValueChangingEvent)
    {
#ifdef ASCENSION_MINIBIRD_SUPPORT
        if (this->Tracker != NULL)
        {
            this->Tracker->SetToolAdjustment(this->ToolAdjustmentScales[0]->GetWidget()->GetValue(), 
                this->ToolAdjustmentScales[1]->GetWidget()->GetValue(), 
                this->ToolAdjustmentScales[2]->GetWidget()->GetValue());
            this->Tracker->SetProbeAdjustment(this->ProbeAdjustmentScales[0]->GetValue(),
                this->ProbeAdjustmentScales[1]->GetValue(), this->ProbeAdjustmentScales[2]->GetValue());
        }
#endif /* ASCENSION_MINIBIRD_SUPPORT */
    }
}

void vtkUltrasoundToolGUI::UpdateTracker()
{
#ifdef ASCENSION_MINIBIRD_SUPPORT
    if (this->Tracker != NULL)
    {

        if (this->RodActor != NULL)
        {
            this->RodActor->SetPosition(0,0,0);
            this->RodActor->SetOrientation(0,0,0); 

            this->RodActor->RotateZ(90);
            this->Tracker->CalcInstrumentPos();

            this->RodActor->SetUserMatrix(NULL);
            this->RodActor->SetUserMatrix(this->Tracker->GetTransform());

            //this->ScanPlane->SetUserMatrix(NULL);
            this->ScanPlane->SetUserMatrix(this->Tracker->GetTransform());
        }
    }
#else
    if (this->RodActor != NULL)
    {
        this->RodActor->SetPosition(ProbeAdjustmentScales[0]->GetValue(), ProbeAdjustmentScales[1]->GetValue(), ProbeAdjustmentScales[2]->GetValue());
        this->RodActor->SetOrientation(ToolAdjustmentScales[0]->GetValue(), ToolAdjustmentScales[1]->GetValue(), ToolAdjustmentScales[2]->GetValue());
    }

#endif /* ASCENSION_MINIBIRD_SUPPORT */
}
