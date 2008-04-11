#include "vtkUltrasoundModuleGUI.h"
#include "vtkUltrasoundModuleLogic.h"
#include "vtkObjectFactory.h"

#include "vtkKWEvent.h"
#include "vtkKWCheckButton.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenu.h"
#include "vtkKWScaleWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWScale.h"

#include "vtkSlicerApplication.h"
#include "vtkImageData.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLScalarVolumeDisplayNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkSlicerColorLogic.h"
#include "vtkMRMLScene.h"


#include "vtkUltrasoundStreamerGUI.h"

vtkCxxRevisionMacro(vtkUltrasoundModuleGUI, "$ Revision 1.0$");
vtkStandardNewMacro(vtkUltrasoundModuleGUI);


vtkUltrasoundModuleGUI::vtkUltrasoundModuleGUI()
{
    // Initialize Values
    this->Logic = NULL;
    this->VolumeNode = NULL;
    this->UltrasoundStreamerGUI = NULL;
}

vtkUltrasoundModuleGUI::~vtkUltrasoundModuleGUI()
{
    if (this->VolumeNode != NULL)
    {
        this->GetLogic()->GetMRMLScene()->RemoveNode(this->VolumeNode);
        this->VolumeNode->Delete();
        this->VolumeNode = NULL;
        this->UltrasoundStreamerGUI = NULL;
    }
}

// Description:
// Process events generated by Logic
void vtkUltrasoundModuleGUI::ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                                 void *callData )
{
}

/// GUI part
void vtkUltrasoundModuleGUI::BuildGUI ( )
{
    if (this->UltrasoundStreamerGUI != NULL)
        return;

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    this->GetMRMLScene();

    this->GetUIPanel()->AddPage("Ultrasound", "Ultrasound",NULL);

    // Define your help text and build the help frame here.
    const char *help = "Ultrasound. This is currently a prototype and will be under active development throughout 3DSlicer's Beta release.";
    const char *about = "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details.";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "Ultrasound" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    //
    //Ultrasound 
    //
    vtkSlicerModuleCollapsibleFrame *usFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    usFrame->SetParent(page);
    usFrame->Create();
    usFrame->ExpandFrame();
    usFrame->SetLabelText("Ultrasound");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        usFrame->GetWidgetName());

    this->UltrasoundStreamerGUI = vtkUltrasoundStreamerGUI::New();
    this->UltrasoundStreamerGUI->SetParent(usFrame->GetFrame());
    this->UltrasoundStreamerGUI->Create();
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->UltrasoundStreamerGUI->GetWidgetName());

    

}

// This method releases references and key-bindings,
// and optionally removes observers.
void vtkUltrasoundModuleGUI::TearDownGUI ( )
{
}

// Description:
// Methods for adding module-specific key bindings and
// removing them.
void vtkUltrasoundModuleGUI::CreateModuleEventBindings ( )
{
}

void vtkUltrasoundModuleGUI::ReleaseModuleEventBindings ( )
{
}

// Description:
// Add obsereves to GUI widgets
void vtkUltrasoundModuleGUI::AddGUIObservers ( )
{
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::EnablingEvent, (vtkCommand*)this->GUICallbackCommand);
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::DisabledEvent, (vtkCommand*)this->GUICallbackCommand);
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::DataUpdatedEvent, (vtkCommand*)this->GUICallbackCommand);
}


// Description:
// Remove obsereves to GUI widgets
void vtkUltrasoundModuleGUI::RemoveGUIObservers ( )
{
}

void vtkUltrasoundModuleGUI::RemoveMRMLNodeObservers ( )
{
}

void vtkUltrasoundModuleGUI::RemoveLogicObservers ( )
{
}



// Description:
// Process events generated by GUI widgets
void vtkUltrasoundModuleGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long event, void *callData )
{
    if (caller == this->UltrasoundStreamerGUI)
    {
        if (event == vtkUltrasoundStreamerGUI::DataUpdatedEvent)
        {
            this->VolumeNode->Modified();
        }
        else if (event == vtkUltrasoundStreamerGUI::EnablingEvent)
        {
            if (this->VolumeNode == NULL)
            {
                this->VolumeNode = AddVolumeNode("Ultrasound Streamer Node");
            }
            this->UltrasoundStreamerGUI->SetImageData(this->VolumeNode->GetImageData());
        }
    }
}


// Description:
// Process events generated by MRML
void vtkUltrasoundModuleGUI::ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData)
{
}


void vtkUltrasoundModuleGUI::CreateWidget()
{
}


// Description:
// Methods describe behavior at module enter and exit.
void vtkUltrasoundModuleGUI::Enter ( )
{
    vtkDebugMacro("Enter Ultrasound Module");

    if ( this->Built == false )
    {
        this->BuildGUI();
        this->AddGUIObservers();
    }
    this->CreateModuleEventBindings();
    //this->UpdateGUI();

    //this->NS_ImageData->SetMRMLScene(this->GetLogic()->GetMRMLScene());
    //this->NS_ImageData->UpdateMenu();
}

void vtkUltrasoundModuleGUI::Exit ( )
{
    vtkDebugMacro("Exit: removeObservers for VolumeRenderingModule");
    this->ReleaseModuleEventBindings();

}



void vtkUltrasoundModuleGUI::PrintSelf(ostream& os, vtkIndent indent)
{
}

vtkMRMLScalarVolumeNode* vtkUltrasoundModuleGUI::AddVolumeNode(const char* volumeNodeName)
{

    std::cerr << "AddVolumeNode(): called." << std::endl;

    vtkMRMLScalarVolumeNode *volumeNode = NULL;
    vtkImageData* imageData;
    vtkMRMLScene* scene = this->GetLogic()->GetMRMLScene();

    volumeNode = vtkMRMLScalarVolumeNode::New();

    volumeNode->SetName(volumeNodeName);
    volumeNode->SetDescription("Received by OpenIGTLink");

    imageData = vtkImageData::New();
    imageData->SetDimensions(80,80,160);///imgheader.size[0], imgheader.size[1], imgheader.size[2]);
    imageData->SetNumberOfScalarComponents(1);
    imageData->SetScalarTypeToUnsignedChar();
    imageData->AllocateScalars();

    volumeNode->SetAndObserveImageData(imageData);
    imageData->Delete();

    scene->AddNode(volumeNode);
    this->ApplicationLogic->GetSelectionNode()->SetReferenceActiveVolumeID(volumeNode->GetID());
    this->ApplicationLogic->PropagateVolumeSelection();

    return volumeNode;
}
