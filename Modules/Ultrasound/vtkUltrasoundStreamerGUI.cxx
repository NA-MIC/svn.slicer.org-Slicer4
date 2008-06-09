#include "vtkUltrasoundStreamerGUI.h"
#include "vtkObjectFactory.h"

#include "vtkKWLabel.h"
#include "vtkKWScale.h"
#include "vtkKWEntry.h"
#include "vtkKWCheckButton.h"
#include "vtkKWScaleWithLabel.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWApplication.h"
#include "vtkImageData.h"

#include "vtkCallbackCommand.h"

#include <sstream>

#include "vtkUltrasoundStreamSource.h"
#ifdef PHILLIPS_ULTRASOUND_SCANNER_SUPPORT
  #include "vtkPhilipsUltrasoundStreamSource.h"
#else 
  #include "vtkUltrasoundScannerReader.h"
#endif


vtkCxxRevisionMacro(vtkUltrasoundStreamerGUI, "$Revision 1.0 $");
vtkStandardNewMacro(vtkUltrasoundStreamerGUI);

vtkUltrasoundStreamerGUI::vtkUltrasoundStreamerGUI()
{
    this->cb_Enabled = NULL;
    this->sc_RefreshRate = NULL;
    this->el_Address = NULL;

    this->StreamSource = NULL;

    this->GUICallbackCommand = vtkCallbackCommand::New();
    this->GUICallbackCommand->SetCallback(&vtkUltrasoundStreamerGUI::ProcessGUIEventsStatic);
    this->GUICallbackCommand->SetClientData(this);
}

vtkUltrasoundStreamerGUI::~vtkUltrasoundStreamerGUI()
{
    if (this->cb_Enabled != NULL)
        this->cb_Enabled->Delete();
    if (this->sc_RefreshRate != NULL)
        this->sc_RefreshRate->Delete();
    if (this->el_Address != NULL)
        this->el_Address->Delete();

    if (this->StreamSource != NULL)
        this->StreamSource->Delete();
    //if (this->ImageData != NULL)
    //    this->ImageData->Delete();

    this->GUICallbackCommand->Delete();
}

bool vtkUltrasoundStreamerGUI::IsEnabled() const
{
    return (this->cb_Enabled != NULL && this->cb_Enabled->GetSelectedState() == 1);
}
int vtkUltrasoundStreamerGUI::GetRefreshRate() const
{
    return this->sc_RefreshRate->GetWidget()->GetValue();
}
const char* vtkUltrasoundStreamerGUI::GetAddress() const
{
    return this->el_Address->GetWidget()->GetValue();
}

void vtkUltrasoundStreamerGUI::CreateWidget()
{
    this->Superclass::CreateWidget();


    this->cb_Enabled = vtkKWCheckButton::New();
    this->cb_Enabled->SetParent(this);
    this->cb_Enabled->Create();
    this->cb_Enabled->SetText("Enabled");
    this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->cb_Enabled->GetWidgetName());

    this->el_Address = vtkKWEntryWithLabel::New();
    this->el_Address->SetParent(this);
    this->el_Address->Create();
    this->el_Address->SetLabelText("Source Address: ");
    this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->el_Address->GetWidgetName());


    this->sc_RefreshRate = vtkKWScaleWithLabel::New();
    this->sc_RefreshRate->SetParent(this);
    this->sc_RefreshRate->Create();
    this->sc_RefreshRate->GetWidget()->SetRange(1, 30);
    this->sc_RefreshRate->GetWidget()->SetValue(1);
    this->sc_RefreshRate->GetLabel()->SetText("Refresh Rate (fps):");
    this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->sc_RefreshRate->GetWidgetName());


    this->AddGUIObservers();
}

// Description:
// Add obsereves to GUI widgets
void vtkUltrasoundStreamerGUI::AddGUIObservers ( )
{
    this->cb_Enabled->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->el_Address->GetWidget()->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand*)this->GUICallbackCommand);
}

void vtkUltrasoundStreamerGUI::ProcessGUIEventsStatic(vtkObject *caller, unsigned long ev, void *callData, void* object)
{
    ((vtkUltrasoundStreamerGUI*)callData)->ProcessGUIEvents(caller, ev, callData);
}

void vtkUltrasoundStreamerGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long ev, void *callData )
{
    if (caller == this->cb_Enabled && ev == vtkKWCheckButton::SelectedStateChangedEvent)
    {
        // ENABLE
        if (cb_Enabled->GetSelectedState() == 1)
        {
            this->InvokeEvent(vtkUltrasoundStreamerGUI::EnablingEvent);
            if (this->StreamSource == NULL)
            {
#ifdef PHILLIPS_ULTRASOUND_SCANNER_SUPPORT
                this->StreamSource = vtkPhilipsUltrasoundStreamSource::New();
                if(this->el_Address->GetWidget()->GetValue()[0] == '\0')
                    this->el_Address->GetWidget()->SetValue("192.168.1.15");
#else /* TEST DATA SET */
                this->StreamSource = vtkUltrasoundScannerReader::New();
                if(this->el_Address->GetWidget()->GetValue()[0] == '\0')
                    this->el_Address->GetWidget()->SetValue("C:\\temp\\3DDCM002.raw");
//                    this->el_Address->GetWidget()->SetValue("E:\\Slicer-Demos\\UltrasoundCuda\\3DDCM002.raw");
#endif /*PHILLIPS_ULTRASOUND_SCANNER_SUPPORT */
            }
            // Register
            this->StreamSource->AddObserver(vtkUltrasoundStreamSource::ConnectionEstablished, (vtkCommand*)this->GUICallbackCommand);

            this->StreamSource->StartStreaming();
            this->StreamSource->FetchImageData(this->GetImageData());


            this->UpdateInput();
            this->InvokeEvent(vtkUltrasoundStreamerGUI::EnabledEvent);
        }
        //DISABLE 
        else // (cb_Enabled->GetSelectedState() == 0)
        {
            this->InvokeEvent(vtkUltrasoundStreamerGUI::DisablingEvent);
            if (this->StreamSource != NULL)
                this->StreamSource->StopStreaming();
            this->InvokeEvent(vtkUltrasoundStreamerGUI::DisabledEvent);
        }
    }
    else if (caller == this->el_Address->GetWidget() && ev == vtkKWEntry::EntryValueChangedEvent)
    {
        if (this->StreamSource != NULL)
        {
            this->StreamSource->SetSourceAddress(el_Address->GetWidget()->GetValue());

            if (this->cb_Enabled->GetSelectedState() == 1)
                this->StreamSource->Reconnect();
        }
    }

    else if (caller == this->StreamSource)
    {
        this->InvokeEvent(ev);
    }
}

void vtkUltrasoundStreamerGUI::UpdateInput()
{
    if (this->cb_Enabled->GetSelectedState() == 1)
    {
        this->StreamSource->FetchImageData(this->ImageData);

        std::stringstream str;
        this->InvokeEvent(vtkUltrasoundStreamerGUI::DataUpdatedEvent);

        // Replay with the framerate
        str << "after " << (int) (1000.0 / this->sc_RefreshRate->GetWidget()->GetValue()) << " %s UpdateInput";
        this->GetApplication()->Script(str.str().c_str(), this->GetTclName());
        //this->GetApplication()->Script("after idle %s UpdateInput", this->GetTclName());
    }
}
