#include "vtkVolumeRenderingModuleGUI.h"

#include <ostream>
#include "vtkMRMLVolumeRenderingNode.h"
#include "vtkVolumeTextureMapper3D.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkIndent.h"
#include "vtkPointData.h"
#include "vtkPiecewiseFunction.h"
#include "vtkKWPiecewiseFunctionEditor.h"
#include "vtkKWColorTransferFunctionEditor.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkKWEntry.h"
#include "vtkVolume.h"
#include "vtkKWTreeWithScrollbars.h"
#include "vtkKWTree.h"
#include "vtkLabelMapPiecewiseFunction.h"
#include "vtkLabelMapColorTransferFunction.h"
#include "vtkKWProgressGauge.h"
#include "vtkKWHistogramSet.h"
#include "vtkKWTkUtilities.h"
#include "vtkTimerLog.h"
#include "vtkSlicerMRMLTreeWidget.h"
#include "vtkRendererCollection.h"
#include "vtkImageData.h"
#include "vtkImageMapper.h"
#include "vtkCylinderSource.h"
#include "vtkBMPWriter.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkPolyData.h"
#include "vtkTexture.h"
#include "vtkPlaneSource.h"
#include "vtkBMPReader.h"
#include "vtkFloatArray.h"
//#include "vtkSlicerFixedPointVolumeRayCastMapper.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include "vtkKWEvent.h"
#include "vtkSlicerVRHelper.h"
#include "vtkSlicerVRLabelmapHelper.h"
#include "vtkSlicerVRGrayscaleHelper.h"
#include "vtkMRMLVolumeRenderingNode.h"

vtkVolumeRenderingModuleGUI::vtkVolumeRenderingModuleGUI(void)
{
    //In Debug Mode
    this->DebugOff();
    this->presets=NULL;
    this->PreviousNS_ImageData="";
    this->PreviousNS_VolumeRenderingDataScene="";
    this->PreviousNS_VolumeRenderingSlicer="";
    this->PB_Testing=NULL;
    this->PB_CreateNewVolumeRenderingNode=NULL;
    this->NS_ImageData=NULL;
    this->NS_VolumeRenderingDataSlicer=NULL;
    this->NS_VolumeRenderingDataScene=NULL;
    this->EWL_CreateNewVolumeRenderingNode=NULL;

    //Frame Details
    this->detailsFrame=NULL;

    //Other members
    this->currentNode=NULL;
    this->presets=NULL;
    this->Helper=NULL;
}

vtkVolumeRenderingModuleGUI::~vtkVolumeRenderingModuleGUI(void)
{

    //Not Delete?!
    //vtkVolumeRenderingModuleLogic *Logic;
    //vtkSlicerViewerWidget *ViewerWidget;
    //vtkSlicerViewerInteractorStyle *InteractorStyle;
    //vtkMRMLVolumeRenderingNode  *currentNode;//really delete this

    if(this->PB_Testing)
    {
        this->PB_Testing->SetParent(NULL);
        this->PB_Testing->Delete();
        this->PB_Testing=NULL;
    }

    if (this->PB_CreateNewVolumeRenderingNode)
    {
        this->PB_CreateNewVolumeRenderingNode->SetParent(NULL);
        this->PB_CreateNewVolumeRenderingNode->Delete();
        this->PB_CreateNewVolumeRenderingNode=NULL;
    }

    if (this->NS_ImageData)
    {
        this->NS_ImageData->SetParent(NULL);
        this->NS_ImageData->Delete();
        this->NS_ImageData=NULL;
    }

    if(this->NS_VolumeRenderingDataScene)
    {
        this->NS_VolumeRenderingDataScene->SetParent(NULL);
        this->NS_VolumeRenderingDataScene->Delete();
        this->NS_VolumeRenderingDataScene=NULL;
    }

    if(this->NS_VolumeRenderingDataSlicer)
    {
        this->NS_VolumeRenderingDataSlicer->SetParent(NULL);
        this->NS_VolumeRenderingDataSlicer->Delete();
        this->NS_VolumeRenderingDataSlicer=NULL;
    }

    if(this->EWL_CreateNewVolumeRenderingNode)
    {
        this->EWL_CreateNewVolumeRenderingNode->SetParent(NULL);
        this->EWL_CreateNewVolumeRenderingNode->Delete();
        this->EWL_CreateNewVolumeRenderingNode=NULL;
    }


    if(this->presets)
    {
        this->presets->Delete();
        this->presets=NULL;
    }

    if(this->detailsFrame)
    {
        this->detailsFrame->Delete();
        this->detailsFrame=NULL;
    }
    if(this->Helper)
    {
        this->Helper->Delete();
        this->Helper=NULL;
    }
    if(this->currentNode)
    {
        this->currentNode->Delete();
        this->currentNode=NULL;
    }
    this->SetViewerWidget(NULL);
    this->SetInteractorStyle(NULL);
}
vtkVolumeRenderingModuleGUI* vtkVolumeRenderingModuleGUI::New() {
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVolumeRenderingModuleGUI");
    if(ret)
    {
        return (vtkVolumeRenderingModuleGUI*)ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkVolumeRenderingModuleGUI;


}
void vtkVolumeRenderingModuleGUI::PrintSelf(ostream& os, vtkIndent indent)
{
    os<<indent<<"vtkVolumeRenderingModuleGUI"<<endl;
    os<<indent<<"vtkVolumeRenderingModuleLogic"<<endl;
    if(this->GetLogic())
    {
        this->GetLogic()->PrintSelf(os,indent.GetNextIndent());
    }
}
void vtkVolumeRenderingModuleGUI::BuildGUI(void)
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    this->GetUIPanel()->AddPage("VolumeRendering","VolumeRendering",NULL);

    // Define your help text and build the help frame here.
    const char *help = "VolumeRendering. 3D Segmentation This module is currently a prototype and will be under active development throughout 3DSlicer's Beta release.";
    const char *about = "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details.";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "VolumeRendering" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    //
    //Load and save
    //
    vtkSlicerModuleCollapsibleFrame *loadSaveDataFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    loadSaveDataFrame->SetParent (this->UIPanel->GetPageWidget("VolumeRendering"));
    loadSaveDataFrame->Create();
    loadSaveDataFrame->ExpandFrame();
    loadSaveDataFrame->SetLabelText("Load and save");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        loadSaveDataFrame->GetWidgetName(), this->UIPanel->GetPageWidget("VolumeRendering")->GetWidgetName());

    //Testing Pushbutton
    this->PB_Testing= vtkKWPushButton::New();
    this->PB_Testing->SetParent(loadSaveDataFrame->GetFrame());
    this->PB_Testing->Create();
    this->PB_Testing->SetText("Testing");
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->PB_Testing->GetWidgetName());

    //NodeSelector  for Node from MRML Scene
    this->NS_ImageData=vtkSlicerNodeSelectorWidget::New();
    this->NS_ImageData->SetParent(loadSaveDataFrame->GetFrame());
    this->NS_ImageData->Create();
    this->NS_ImageData->NoneEnabledOn();
    this->NS_ImageData->SetLabelText("Source Volume");
    this->NS_ImageData->SetNodeClass("vtkMRMLScalarVolumeNode","","","");
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_ImageData->GetWidgetName());

    //NodeSelector for VolumeRenderingNode Preset
    this->NS_VolumeRenderingDataSlicer=vtkSlicerNodeSelectorVolumeRenderingWidget::New();
    this->NS_VolumeRenderingDataSlicer->SetParent(loadSaveDataFrame->GetFrame());
    this->NS_VolumeRenderingDataSlicer->Create();
    this->NS_VolumeRenderingDataSlicer->SetLabelText("Presets");
    this->NS_VolumeRenderingDataSlicer->EnabledOff();//By default off
    this->NS_VolumeRenderingDataSlicer->NoneEnabledOn();
    this->NS_VolumeRenderingDataSlicer->SetNodeClass("vtkMRMLVolumeRenderingNode","","","");
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_VolumeRenderingDataSlicer->GetWidgetName());

    //NodeSelector for VolumeRenderingNode Scene
    this->NS_VolumeRenderingDataScene=vtkSlicerNodeSelectorVolumeRenderingWidget::New();
    this->NS_VolumeRenderingDataScene->SetParent(loadSaveDataFrame->GetFrame());
    this->NS_VolumeRenderingDataScene->Create();
    this->NS_VolumeRenderingDataScene->NoneEnabledOn();
    this->NS_VolumeRenderingDataScene->SetLabelText("VolumeRenderingNode from Scene");
    this->NS_VolumeRenderingDataScene->EnabledOff();//By default off
    this->NS_VolumeRenderingDataScene->SetNodeClass("vtkMRMLVolumeRenderingNode","","","");
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_VolumeRenderingDataScene->GetWidgetName());
    //Missing: Load from file

    //Create New Volume Rendering Node
    //Entry With Label
    this->EWL_CreateNewVolumeRenderingNode=vtkKWEntryWithLabel::New();
    this->EWL_CreateNewVolumeRenderingNode->SetParent(loadSaveDataFrame->GetFrame());
    this->EWL_CreateNewVolumeRenderingNode->Create();
    this->EWL_CreateNewVolumeRenderingNode->SetLabelText("Name for new Volume Rendering Node");
    this->EWL_CreateNewVolumeRenderingNode->EnabledOff();
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2", this->EWL_CreateNewVolumeRenderingNode->GetWidgetName());


    this->PB_CreateNewVolumeRenderingNode=vtkKWPushButton::New();
    this->PB_CreateNewVolumeRenderingNode->SetParent(loadSaveDataFrame->GetFrame());
    this->PB_CreateNewVolumeRenderingNode->Create();
    this->PB_CreateNewVolumeRenderingNode->SetText("Create new VolumeRenderingNode");
    app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->PB_CreateNewVolumeRenderingNode->GetWidgetName());

    //Details frame
    this->detailsFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    this->detailsFrame->SetParent (this->UIPanel->GetPageWidget("VolumeRendering"));
    this->detailsFrame->Create();
    this->detailsFrame->ExpandFrame();
    this->detailsFrame->SetLabelText("Details");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->detailsFrame->GetWidgetName(), this->UIPanel->GetPageWidget("VolumeRendering")->GetWidgetName());


    //set subnodes
    //Delete frames
    if ( this->GetApplicationGUI() &&  this->GetApplicationGUI()->GetMRMLScene())
    {
        this->GetApplicationGUI()->GetMRMLScene()->AddObserver( vtkMRMLScene::SceneCloseEvent, this->MRMLCallbackCommand );
    }
    loadSaveDataFrame->Delete();
    this->Built=true;
}

void vtkVolumeRenderingModuleGUI::TearDownGUI(void)
{
    this->Exit();
    if ( this->Built )
    {
        this->RemoveGUIObservers();
    }
}

void vtkVolumeRenderingModuleGUI::CreateModuleEventBindings(void)
{
    vtkDebugMacro("VolumeRenderingModule: CreateModuleEventBindings: No ModuleEventBindings yet");
}

void vtkVolumeRenderingModuleGUI::ReleaseModuleEventBindings(void)
{
    vtkDebugMacro("VolumeRenderingModule: ReleaseModuleEventBindings: No ModuleEventBindings to remove yet");
}

void vtkVolumeRenderingModuleGUI::AddGUIObservers(void)
{

    this->NS_ImageData->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NS_VolumeRenderingDataScene->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NS_VolumeRenderingDataSlicer->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->PB_Testing->AddObserver(vtkKWPushButton::InvokedEvent,(vtkCommand *)this->GUICallbackCommand );
    this->PB_CreateNewVolumeRenderingNode->AddObserver(vtkKWPushButton::InvokedEvent,(vtkCommand*)this->GUICallbackCommand);


}
void vtkVolumeRenderingModuleGUI::RemoveGUIObservers(void)
{
    this->NS_ImageData->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->NS_VolumeRenderingDataScene->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->NS_VolumeRenderingDataSlicer->RemoveObservers(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->PB_Testing->RemoveObservers (vtkKWPushButton::InvokedEvent,(vtkCommand *)this->GUICallbackCommand);
    this->PB_CreateNewVolumeRenderingNode->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
}
void vtkVolumeRenderingModuleGUI::RemoveMRMLNodeObservers(void)
{

}
void vtkVolumeRenderingModuleGUI::RemoveLogicObservers(void)
{
}

void vtkVolumeRenderingModuleGUI::ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData)
{
    vtkDebugMacro("vtkVolumeRenderingModuleGUI::ProcessGUIEvents: event = " << event);


    //
    //Check PushButtons
    //
    vtkKWPushButton *callerObject=vtkKWPushButton::SafeDownCast(caller);
    // TODO Testing Button  Remove later
    if(callerObject==this->PB_Testing&&event==vtkKWPushButton::InvokedEvent)
    {
        int index=0;
        int count=this->GetLogic()->GetMRMLScene()->GetNumberOfNodesByClass("vtkMRMLModelNode");
        for(int i=0;i<count;i++)
        {
            (vtkMRMLModelNode::SafeDownCast(this->GetLogic()->GetMRMLScene()->GetNthNodeByClass(i,"vtkMRMLModelNode")))->GetModelDisplayNode()->VisibilityOff();
        }
    }
    //Create New VolumeRenderingNode
    else if (callerObject==this->PB_CreateNewVolumeRenderingNode&&event==vtkKWPushButton::InvokedEvent)
    {
        //Get a new auto currentNode
        this->InitializePipelineNewCurrentNode();
        this->currentNode->HideFromEditorsOff();
        //Set the right name
        const char *name=this->EWL_CreateNewVolumeRenderingNode->GetWidget()->GetValue();
        if(!name)
        {
            vtkErrorMacro("No Text for VolumeRenderingNode");
        }
        else 
        {
            this->currentNode->SetName(name);
        }
        //Remove Text from Entry
        this->EWL_CreateNewVolumeRenderingNode->GetWidget()->SetValue("");
        this->NS_VolumeRenderingDataScene->UpdateMenu();
    }
    //
    // End Check PushButtons
    // 

    //
    //Check Node Selectors
    //
    vtkSlicerNodeSelectorWidget *callerObjectNS=vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
    //Load Volume
    if(callerObjectNS==this->NS_ImageData&&event==vtkSlicerNodeSelectorWidget::NodeSelectedEvent)
    {
        if(this->NS_ImageData->GetSelected()==NULL)
        {
            //Unpack the details frame
            this->UnpackLabelMapGUI();
            this->UnpackSvpGUI();
            this->PreviousNS_ImageData="";
        }
        //Only proceed event,if new Node
        else if(strcmp(this->NS_ImageData->GetSelected()->GetID(),this->PreviousNS_ImageData.c_str())!=0)
        {
            vtkMRMLScalarVolumeNode *selectedImageData=vtkMRMLScalarVolumeNode::SafeDownCast(this->NS_ImageData->GetSelected());
            //This is a LabelMap
            if(selectedImageData->GetLabelMap()==1)
            {
                this->PackLabelMapGUI();

            }
            //This is NO LabelMap
            else
            {
                this->PackSvpGUI();


            }
            //Initialize the Pipeline
            this->InitializePipelineFromImageData();

            //update previous:
            this->PreviousNS_ImageData=this->NS_ImageData->GetSelected()->GetID();//only when not "None"


        }//else if
    }//if
    //Volume RenderingDataScene
    else if(callerObjectNS==this->NS_VolumeRenderingDataScene&&event==vtkSlicerNodeSelectorWidget::NodeSelectedEvent)
    {
        //Check for None selected //Just to be safe
        if(this->NS_VolumeRenderingDataScene->GetSelected()==NULL)
        {
            this->PreviousNS_VolumeRenderingDataScene="";
        }
        //Only proceed event,if new Node
        else if(strcmp(this->NS_VolumeRenderingDataScene->GetSelected()->GetID(),this->PreviousNS_VolumeRenderingDataScene.c_str())!=0)
        {
            this->InitializePipelineFromMRMLScene();
            this->PreviousNS_VolumeRenderingDataScene=this->NS_VolumeRenderingDataScene->GetSelected()->GetID();
        }
    }
    //VolumeRenderingDataSlicer
    else if(callerObjectNS==this->NS_VolumeRenderingDataSlicer&&event==vtkSlicerNodeSelectorWidget::NodeSelectedEvent)
    {
        //Check for None selected
        if(this->NS_VolumeRenderingDataSlicer->GetSelected()==NULL)
        {
            this->PreviousNS_VolumeRenderingSlicer="";
        }
        //Only proceed event,if other Node
        else if(strcmp(this->NS_VolumeRenderingDataSlicer->GetSelected()->GetID(),this->PreviousNS_VolumeRenderingSlicer.c_str())!=0)
        {
            //check if we have a preset or a "normal VolumeRenderingNode

            //We have a preset, we can find id in our presets
            if(this->presets->GetNodeByID(this->NS_VolumeRenderingDataSlicer->GetSelected()->GetID())!=NULL)
            {
                //Copy Preset Information in current Node
                this->currentNode->CopyParameterset(this->NS_VolumeRenderingDataSlicer->GetSelected());
                this->Helper->UpdateGUIElements();
            }
            //It's not a preset so just update references, and select the new Added Node
            else 
            {
                this->currentNode=vtkMRMLVolumeRenderingNode::SafeDownCast(this->NS_VolumeRenderingDataSlicer->GetSelected());
                this->currentNode->AddReference(this->NS_ImageData->GetSelected()->GetID());
                this->NS_VolumeRenderingDataScene->UpdateMenu();
                this->NS_VolumeRenderingDataScene->SetSelected(this->NS_VolumeRenderingDataSlicer->GetSelected());
            }
        }
    }
    //
    //End Check NodeSelectors
    //
    //Update GUI
    this->UpdateGUI();

}
void vtkVolumeRenderingModuleGUI::ProcessMRMLEvents(vtkObject *caller, unsigned long event, void *callData)
{
    if (event == vtkMRMLScene::SceneCloseEvent)
    {
        if(this->Helper!=NULL)
        {
            this->Helper->Delete();
            this->Helper=NULL;
        }
        //Reset every Node related stuff
        this->PreviousNS_ImageData="";
        this->PreviousNS_VolumeRenderingDataScene="";
        this->PreviousNS_VolumeRenderingSlicer="";
        this->currentNode=NULL;
        this->UpdateGUI();

    }
}

void vtkVolumeRenderingModuleGUI::Enter(void)
{

    vtkDebugMacro("Enter Volume Rendering Module");
    //Load Presets
    vtkMRMLVolumeRenderingNode *vrNode=vtkMRMLVolumeRenderingNode::New();

    if(!this->presets)
    {
        //Instance internal MRMLScene for Presets
        this->presets=vtkMRMLScene::New();
        //Register node class
        this->presets->RegisterNodeClass(vrNode);
        vrNode->Delete();


        //GetPath
        vtksys_stl::string slicerHome;
        if (vtksys::SystemTools::GetEnv("SLICER_HOME") == NULL)
        {
            if (vtksys::SystemTools::GetEnv("PWD") != NULL)
            {
                slicerHome =  vtksys_stl::string(vtksys::SystemTools::GetEnv("PWD"));
            }
            else
            {
                slicerHome =  vtksys_stl::string("");
            }
        }
        else
        {
            slicerHome = vtksys_stl::string(vtksys::SystemTools::GetEnv("SLICER_HOME"));
        }
        // check to see if slicer home was set
        vtksys_stl::vector<vtksys_stl::string> filesVector;
        filesVector.push_back(""); // for relative path
        filesVector.push_back(slicerHome);
        filesVector.push_back(vtksys_stl::string("Modules/VolumeRendering/presets.xml"));
        vtksys_stl::string presetFileName = vtksys::SystemTools::JoinPath(filesVector);
        this->presets->SetURL(presetFileName.c_str());
        this->presets->Connect();
        this->NS_VolumeRenderingDataSlicer->SetAdditionalMRMLScene(this->presets);
    }
    //End Load presets

    if ( this->Built == false )
    {
        this->BuildGUI();
        this->AddGUIObservers();
    }
    this->CreateModuleEventBindings();
    this->UpdateGUI();
}

void vtkVolumeRenderingModuleGUI::Exit(void)
{
    vtkDebugMacro("Exit: removeObservers for VolumeRenderingModule");
    this->ReleaseModuleEventBindings();
}

void vtkVolumeRenderingModuleGUI::UpdateGUI(void)
{

    //First of all check if we have a MRML Scene
    if (!this->GetLogic()->GetMRMLScene())
    {
        //if not return
        return;
    }
    if(this->NS_ImageData->GetMRMLScene()!=this->GetLogic()->GetMRMLScene())
    {
        //Update the NodeSelector for Volumes
        this->NS_ImageData->SetMRMLScene(this->GetLogic()->GetMRMLScene());
        this->NS_ImageData->UpdateMenu();
    }
    if(this->NS_VolumeRenderingDataScene->GetMRMLScene()!=this->GetLogic()->GetMRMLScene())
    {
        //Update NodeSelector for VolumeRendering Node
        this->NS_VolumeRenderingDataScene->SetMRMLScene(this->GetLogic()->GetMRMLScene());
        this->NS_VolumeRenderingDataScene->UpdateMenu();
    }



    //Set the new condition
    if(this->NS_ImageData->GetSelected()!=NULL&&(this->NS_VolumeRenderingDataScene->GetCondition()!=this->NS_ImageData->GetSelected()->GetID()))
    {
        this->NS_VolumeRenderingDataScene->SetCondition(this->NS_ImageData->GetSelected()->GetID(),vtkMRMLScalarVolumeNode::SafeDownCast(this->NS_ImageData->GetSelected())->GetLabelMap(),true);
        this->NS_VolumeRenderingDataScene->UpdateMenu();
    }


    //Take care about Presets...
    //We need None for this
    if(this->NS_VolumeRenderingDataSlicer->GetMRMLScene()!=this->GetLogic()->GetMRMLScene())
    {
        this->NS_VolumeRenderingDataSlicer->SetMRMLScene(this->GetLogic()->GetMRMLScene());
        this->NS_VolumeRenderingDataSlicer->UpdateMenu();
    }  

    if(this->NS_ImageData->GetSelected()!=NULL)
    {
        this->NS_VolumeRenderingDataSlicer->SetCondition(this->NS_ImageData->GetSelected()->GetID(),vtkMRMLScalarVolumeNode::SafeDownCast(this->NS_ImageData->GetSelected())->GetLabelMap(),false);
        this->NS_VolumeRenderingDataSlicer->UpdateMenu();
    }

    //Disable/Enable after Volume is selected
    if(this->NS_ImageData->GetSelected()!=NULL)
    {
        this->PB_CreateNewVolumeRenderingNode->EnabledOn();
        this->PB_Testing->EnabledOn();
        this->NS_VolumeRenderingDataScene->EnabledOn();
        this->NS_VolumeRenderingDataScene->NoneEnabledOff();
        this->EWL_CreateNewVolumeRenderingNode->EnabledOn();
        this->NS_VolumeRenderingDataSlicer->EnabledOn();
    }
    else
    {
        this->EWL_CreateNewVolumeRenderingNode->EnabledOff();
        this->PB_CreateNewVolumeRenderingNode->EnabledOff();
        this->PB_Testing->EnabledOff();
        this->NS_VolumeRenderingDataScene->NoneEnabledOn();
        this->NS_VolumeRenderingDataScene->EnabledOff();
        this->NS_VolumeRenderingDataSlicer->EnabledOff();
    }
    //In presets always "None" is selected
    this->NS_VolumeRenderingDataSlicer->SetSelected(NULL);
}
void vtkVolumeRenderingModuleGUI::SetViewerWidget(vtkSlicerViewerWidget *viewerWidget)
{
}
void vtkVolumeRenderingModuleGUI::SetInteractorStyle(vtkSlicerViewerInteractorStyle *interactorStyle)
{
}


void vtkVolumeRenderingModuleGUI::InitializePipelineFromMRMLScene()
{
    this->currentNode=vtkMRMLVolumeRenderingNode::SafeDownCast(this->NS_VolumeRenderingDataScene->GetSelected());
    vtkImageData* imageData=vtkMRMLScalarVolumeNode::SafeDownCast(this->NS_ImageData->GetSelected())->GetImageData();
    this->Helper->UpdateGUIElements();
    this->Helper->UpdateRendering();
}

void vtkVolumeRenderingModuleGUI::PackLabelMapGUI()
{
    this->UnpackSvpGUI();
    this->Helper=vtkSlicerVRLabelmapHelper::New();
    this->Helper->Init(this);

}

void vtkVolumeRenderingModuleGUI::UnpackLabelMapGUI()
{
    if(this->Helper!=NULL)
    {
        this->Helper->Delete();
        this->Helper=NULL;
    }
}

void vtkVolumeRenderingModuleGUI::PackSvpGUI()
{
    this->UnpackLabelMapGUI();
    this->Helper=vtkSlicerVRGrayscaleHelper::New();
    this->Helper->Init(this);
}
void vtkVolumeRenderingModuleGUI::UnpackSvpGUI()
{
    if(this->Helper!=NULL)
    {
        this->Helper->Delete();
        this->Helper=NULL;
    }
}

void vtkVolumeRenderingModuleGUI::InitializePipelineNewCurrentNode()
{
    //TODO move this part
    this->currentNode=vtkMRMLVolumeRenderingNode::New();
    this->currentNode->HideFromEditorsOff();
    //Add Node to Scene
    this->GetLogic()->GetMRMLScene()->AddNode(this->currentNode);
    this->currentNode->AddReference(this->NS_ImageData->GetSelected()->GetID());

    //Update the menu
    this->PreviousNS_VolumeRenderingDataScene=this->currentNode->GetID();
    this->NS_VolumeRenderingDataScene->SetSelected(this->currentNode);
    this->NS_VolumeRenderingDataScene->UpdateMenu(); 
    //this->PreviousNS_VolumeRenderingDataScene=this->Gui->GetcurrentNode()->GetID();
    //The Helper has to do something too
    this->Helper->InitializePipelineNewCurrentNode();
        this->Helper->UpdateGUIElements();
    this->Helper->UpdateRendering();


    //take care about references
}

void vtkVolumeRenderingModuleGUI::InitializePipelineFromImageData()
{


    //First check if we already have a reference
    const char *id=this->NS_ImageData->GetSelected()->GetID();
    //loop over existing Nodes in scene
    bool firstNodeFound=false;

    for( int i=0;i<this->GetLogic()->GetMRMLScene()->GetNumberOfNodesByClass("vtkMRMLVolumeRenderingNode");i++)
    {
        vtkMRMLVolumeRenderingNode *tmp=vtkMRMLVolumeRenderingNode::SafeDownCast(this->GetLogic()->GetMRMLScene()->GetNthNodeByClass(i,"vtkMRMLVolumeRenderingNode"));
        if(tmp->HasReference(id)&&!firstNodeFound)
        {
            //Select first found Node
            //So everyting will be treated when InitializeFromMRMLScene
            this->PreviousNS_VolumeRenderingDataScene=tmp->GetID();
            this->NS_VolumeRenderingDataScene->SetSelected(tmp);
            //We will call the initialize on our own.
            this->InitializePipelineFromMRMLScene();
            firstNodeFound=true;
        }//if
    }//for

    //If not initialize a new auto generated Volume Rendering Node
    if(!firstNodeFound)
    {
        this->InitializePipelineNewCurrentNode();
    }
    //Render it
    this->PipelineInitializedOn();
    this->Helper->UpdateRendering();
}

