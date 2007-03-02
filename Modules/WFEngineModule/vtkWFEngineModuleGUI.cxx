#include "vtkWFEngineModuleGUI.h"

#include "vtkMRMLWFEngineModuleNode.h"
//#include <WFDirectInterface.h>

#include <vtkObject.h>
#include <vtkObjectFactory.h>
#include <vtkCommand.h>
#include <vtkCallbackCommand.h>

#include <vtkKWWidget.h>
#include <vtkKWMenu.h>
#include <vtkKWMultiColumnList.h>
#include <vtkKWMultiColumnListWithScrollbars.h>
#include <vtkKWPushButtonSet.h>
#include <vtkKWPushButton.h>
#include <vtkKWLabel.h>

#include <vtkKWWidget.h>
#include <vtkKWFrame.h>
#include "vtkKWMyWizardWidget.h"
#include <vtkKWWizardStep.h>
#include "vtkKWMyWizardWorkflow.h"
#include <vtkKWStateMachineInput.h>
#include <vtkKWSpinBoxWithLabel.h>
#include <vtkKWSpinBox.h>
#include <vtkKWScaleWithEntry.h>
#include <vtkKWCheckButtonWithLabel.h>
#include <vtkKWEntryWithLabel.h>
#include <vtkKWEntry.h>
#include <vtkSlicerNodeSelectorWidget.h>
#include <vtkKWLoadSaveButtonWithLabel.h>
#include <vtkKWRadioButtonSetWithLabel.h>
#include <vtkKWRadioButton.h>
#include <vtkSlicerModelsGUI.h>
#include <vtkSlicerApplication.h>
#include <vtkSlicerModuleLogic.h>
#include <vtkSlicerVisibilityIcons.h>
#include <vtkSlicerModuleCollapsibleFrame.h>

#include <ModuleDescription.h>
#include <ModuleDescriptionParser.h>
#include <vtkSlicerParameterWidget.h>
#include <vtkSlicerModuleLogic.h>

#include "WFStateConverter.h"
#include "vtkWFStepHandler.h"

#include <map>
#include <iostream>

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkWFEngineModuleGUI );
vtkCxxRevisionMacro ( vtkWFEngineModuleGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkWFEngineModuleGUI::vtkWFEngineModuleGUI ( )
{
    this->Logic = NULL;
    this->m_curStepID = -1;
    this->m_curWFStep = NULL;
    this->m_curWizWidg = NULL;
    this->m_selectedWF = -1;
    this->m_mclDW = NULL;
    this->m_pbtnSet = NULL;
//    this->m_wfDI = NULL;
    this->m_wizFrame = NULL;
    this->m_curNameToValueMap = NULL;
    this->m_wfStepHandler = NULL;
    this->m_inTransition = false;
    //this->DebugOn();
}


//---------------------------------------------------------------------------
vtkWFEngineModuleGUI::~vtkWFEngineModuleGUI ( )
{
    this->SetModuleLogic ( NULL );
}


//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "WFEngineModuleGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";

    // print widgets?
//    os << indent << "LoadSceneButton" << this->GetLoadSceneButton ( ) << "\n";
}



//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::RemoveGUIObservers ( )
{
  vtkDebugMacro("vtkWFEngineModuleGUI: RemoveGUIObservers\n");
}


//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::AddGUIObservers ( )
{
  vtkDebugMacro("vtkWFEngineModuleGUI: AddGUIObservers\n");      
}



//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::ProcessGUIEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{
    // nothing to do here yet...
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );
  vtkKWMenu *m = vtkKWMenu::SafeDownCast ( caller );
  
  return;
}

//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::ProcessLogicEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::ProcessMRMLEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{    
}


//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::Enter ( )
{
    vtkDebugMacro("vtkWFEngineModuleGUI: Enter\n");
    std::cout<<"ENTER"<<std::endl;
}

//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::Exit ( )
{
    vtkDebugMacro("vtkWFEngineModuleGUI: Exit\n");
}


//---------------------------------------------------------------------------
void vtkWFEngineModuleGUI::BuildGUI ( )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    // Define your help text here.


    // ---
    // MODULE GUI FRAME 
    // configure a page for a model loading UI for now.
    // later, switch on the modulesButton in the SlicerControlGUI
    // ---
    // create a page
    this->UIPanel->AddPage ( "WFEngineModule", "WFEngineModule", NULL );

    const char *help = "The WFEngine Module helps users with an easy implementation of a workflow driven user interface.";
    const char *about = "This work was supported by BIRN, NA-MIC, NAC, NCIGT, and the Slicer Community. See http://www.slicer.org for details. ";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "WFEngineModule" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    
    //Insert here the WFEngine GUI
    // ---
    // LOAD FRAME            
    vtkSlicerModuleCollapsibleFrame *loadFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    loadFrame->SetParent ( page );
    loadFrame->Create ( );
    loadFrame->SetLabelText ("Loadable Workflows");
//    loadFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  loadFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("WFEngineModule")->GetWidgetName());
    
    m_mclDW = vtkKWMultiColumnList::New();    
    m_mclDW->SetParent(loadFrame->GetFrame());
    m_mclDW->Create();
    
    m_mclDW->AddColumn("Workflow-Name");
    m_mclDW->AddColumn("File-Name");
    m_mclDW->AddColumn("Created");
    
    vtkCallbackCommand *mclDWSelectionChangedCmd = vtkCallbackCommand::New();
    mclDWSelectionChangedCmd->SetClientData(this);
    mclDWSelectionChangedCmd->SetCallback(&vtkWFEngineModuleGUI::mclDWSelectionChangedCallback);
    
    m_mclDW->AddObserver(vtkKWMultiColumnList::SelectionChangedEvent, mclDWSelectionChangedCmd);
    
    app->Script("pack %s -side top -anchor nw -expand y -fill both -padx 2 -pady 2", 
            m_mclDW->GetWidgetName());
    
    vtkKWFrame *buttonFrame = vtkKWFrame::New();
    buttonFrame->SetParent(page);
    buttonFrame->Create();
    buttonFrame->SetBorderWidth(2);
    buttonFrame->SetReliefToFlat();

    app->Script("pack %s -side top -anchor se -expand n -fill x -padx 2 -pady 2", 
            buttonFrame->GetWidgetName());

    m_pbtnSet = vtkKWPushButtonSet::New();
    m_pbtnSet->SetParent(buttonFrame);
    m_pbtnSet->PackHorizontallyOn();
    
    m_pbtnSet->Create();
    
    vtkKWPushButton *pbtn = m_pbtnSet->AddWidget(0);
    pbtn->SetText("Load");
    pbtn->SetEnabled(0);
    
    vtkCallbackCommand *loadBtnPushCmd = vtkCallbackCommand::New();
    loadBtnPushCmd->SetCallback(&vtkWFEngineModuleGUI::loadBtnPushCmdCallback);
    loadBtnPushCmd->SetClientData(this);
    
    pbtn->AddObserver(vtkKWPushButton::InvokedEvent, loadBtnPushCmd);
    
//    pbtn = m_pbtnSet->AddWidget(1);
//    pbtn->SetText("Close");
//    pbtn->SetEnabled(1);
    
       
    app->Script("pack %s -side top -anchor se -expand n -fill none -padx 2 -pady 2", 
            m_pbtnSet->GetWidgetName());
    
//    // add button to load a scene. this is wrong widget, but for now let it sit.
//    this->LoadSceneButton = vtkKWPushButton::New ( );
//    this->LoadSceneButton->SetParent ( loadFrame->GetFrame() );
//    this->LoadSceneButton->Create();
//    this->LoadSceneButton->SetText ( "LoadScene" );
//    this->LoadSceneButton->SetBalloonHelpString ( "Select all search terms for use");
//    app->Script ( "pack %s -side top -padx 3 -pady 3", this->LoadSceneButton->GetWidgetName() );
    this->ConnectToWFEngine();
}

int vtkWFEngineModuleGUI::ConnectToWFEngine()
{
    this->m_wfStepHandler = vtkWFStepHandler::New();
    if(this->m_wfStepHandler->InitializeWFEngine() == vtkWFStepHandler::SUCC)
    {
        this->m_wfStepHandler->SetApplication(this->GetApplication());
        this->UpdateWorkflowList();
    }
}

void vtkWFEngineModuleGUI::addWorkflowToList(const char* workflowName, const char* fileName, int date)
{
    int id = this->m_mclDW->GetNumberOfRows();
    this->m_mclDW->InsertCellText(id,0, workflowName);
    this->m_mclDW->InsertCellText(id,1, fileName);
    
    time_t tim = date; 
    this->m_mclDW->InsertCellText(id,2, ctime(&tim));    
//    this->listToWFMap.insert(std::make_pair(id, fileName));
}

void vtkWFEngineModuleGUI::SaveState()
{
    
}

void vtkWFEngineModuleGUI::mclDWSelectionChangedCallback(vtkObject* obj, unsigned long,void* param, void*)
{
    vtkWFEngineModuleGUI *myDW = (vtkWFEngineModuleGUI*)param;
    
    if(myDW->m_mclDW->GetIndexOfFirstSelectedRow() != -1)
    {
        myDW->m_pbtnSet->GetWidget(0)->SetEnabled(1);
    }
    
    if(myDW->m_mclDW->GetIndexOfFirstSelectedRow() == myDW->m_selectedWF)
    {
        myDW->m_pbtnSet->GetWidget(0)->SetText("Unload");
    }
    else
        myDW->m_pbtnSet->GetWidget(0)->SetText("Load");
}

void vtkWFEngineModuleGUI::createWizard()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "WFEngineModule" );
    //Insert here the WFEngine GUI
    // ---
    // LOAD FRAME
    
    // get Name for selected work-flow
    std::string name = this->m_mclDW->GetCellText(this->m_mclDW->GetIndexOfFirstSelectedRow(),0);
    
    this->m_wizFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    m_wizFrame->SetParent ( page );
    m_wizFrame->Create ( );
    m_wizFrame->SetLabelText (name.c_str());
//    loadFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
            m_wizFrame->GetWidgetName(),
                  this->UIPanel->GetPageWidget("WFEngineModule")->GetWidgetName());
    
    m_curWizWidg = vtkKWMyWizardWidget::New();
    
    m_curWizWidg->SetParent(m_wizFrame->GetFrame());
    m_curWizWidg->Create();
    m_curWizWidg->SetClientAreaMinimumHeight(300);
    
    this->GetApplication()->Script("pack %s -side top -anchor ne -expand y -fill both -padx 2 -pady 2", 
            m_curWizWidg->GetWidgetName());
    
    vtkKWMyWizardWorkflow *wizWorkflow = m_curWizWidg->GetMyWizardWorkflow();
                  
//    create a virtual first and last step to take cover for the input and out vars in that both steps
    
    vtkKWWizardStep *virtFirstStep = vtkKWWizardStep::New();
//    vtkKWStateMachineState *virtFirstState = vtkKWStateMachineState::New();
    virtFirstStep->SetName("Welcome");
    virtFirstStep->SetDescription("");
    
    wizWorkflow->AddStep(virtFirstStep);
    
    vtkKWWizardStep *virtLastStep = vtkKWWizardStep::New();
    virtLastStep->SetName("Good Bye");
    virtLastStep->SetDescription("");
    
    wizWorkflow->AddNextStep(virtLastStep);
    
    wizWorkflow->SetInitialStep(virtFirstStep);
    wizWorkflow->SetFinishStep(virtLastStep);
    wizWorkflow->CreateGoToTransitionsToFinishStep();
    // Comment:
    // listen to the workflow next and back Events to handle the workflow dynamically
    // 
    
    vtkCallbackCommand *wizCB = vtkCallbackCommand::New();
    
    wizCB->SetCallback(&vtkWFEngineModuleGUI::nextTransitionCallback);
    wizCB->SetClientData(this);
    this->m_curWizWidg->AddObserver(vtkKWMyWizardWidget::nextButtonClicked, wizCB);
    
    wizCB = vtkCallbackCommand::New();
    
    wizCB->SetCallback(&vtkWFEngineModuleGUI::backTransitionCallback);
    wizCB->SetClientData(this);
    this->m_curWizWidg->AddObserver(vtkKWMyWizardWidget::backButtonClicked, wizCB);
    
    // create a new MRMLNode to get track of all workflow variables
    
    vtkMRMLWFEngineModuleNode *myWFENode = vtkMRMLWFEngineModuleNode::New();
    myWFENode->SetName(name.c_str());
    myWFENode->SetDescription("WFEngineModule: Current selected work-flow node");
    myWFENode->SetScene(this->Logic->GetMRMLScene());
    this->Logic->GetMRMLScene()->AddNode(myWFENode);
    this->SetWFEngineModuleNode(myWFENode);
}

void vtkWFEngineModuleGUI::closeBtnPushCmdCallback(vtkObject* obj, unsigned long, void* param, void*)
{
    vtkWFEngineModuleGUI *myDW = (vtkWFEngineModuleGUI*)param;
    myDW->SaveState();
    myDW->closeWorkflow();
//    myDW->OK();
}

void vtkWFEngineModuleGUI::loadBtnPushCmdCallback(vtkObject* obj, unsigned long, void* param, void*)
{
    vtkWFEngineModuleGUI *myDW = (vtkWFEngineModuleGUI*)param;
    std::string selectedWF = myDW->m_mclDW->GetCellText(myDW->m_mclDW->GetIndexOfFirstSelectedRow(), 1);
    std::cout<<selectedWF<<std::endl;
    vtkKWPushButton *loadBtn = vtkKWPushButton::SafeDownCast(obj);
    if(loadBtn != NULL && myDW->m_wizFrame != NULL)
    {
        myDW->m_curWizWidg->Delete();
        myDW->m_curWizWidg = NULL;
        myDW->deleteWizardWidgetContainer();
        
        vtkKWWidget *page = myDW->UIPanel->GetPageWidget ( "WFEngineModule" );
        for(int i = 0; i < page->GetNumberOfChildren(); i++)
        {
            vtkSlicerModuleCollapsibleFrame *curFrame = vtkSlicerModuleCollapsibleFrame::SafeDownCast(page->GetNthChild(i));
            if(curFrame)
            {
                curFrame->ExpandFrame();
            }
        }
        
        loadBtn->SetText("Load");
//        myDW->m_nbDW->SetPageEnabled(myDW->m_selWF,0);
        myDW->m_wfStepHandler->CloseWorkflowSession();
        myDW->m_wizFrame = NULL;
        myDW->m_selectedWF = -1;
    }
    else if(loadBtn != NULL && myDW->m_wizFrame == NULL)
    {
        myDW->m_wfStepHandler->LoadNewWorkflowSession(selectedWF);
        
        vtkKWWidget *page = myDW->UIPanel->GetPageWidget ( "WFEngineModule" );
        for(int i = 0; i < page->GetNumberOfChildren(); i++)
        {
            vtkSlicerModuleCollapsibleFrame *curFrame = vtkSlicerModuleCollapsibleFrame::SafeDownCast(page->GetNthChild(i));
            if(curFrame)
            {
                curFrame->CollapseFrame ( );
            }
        }
        myDW->createWizard();
//        myDW->m_nbDW->RaisePage(myDW->m_selWF);
        myDW->m_mclDW->SetRowBackgroundColor(myDW->m_mclDW->GetIndexOfFirstSelectedRow(),128,255,128);
        myDW->m_selectedWF = myDW->m_mclDW->GetIndexOfFirstSelectedRow();
        myDW->m_mclDW->InvokeEvent(vtkKWMultiColumnList::SelectionChangedEvent, myDW);        
    }    
}

void vtkWFEngineModuleGUI::closeWorkflow()
{
    this->m_wfStepHandler->CloseWorkflowSession();
}

void vtkWFEngineModuleGUI::workStepValidationCallBack(WFEngine::nmWFStepObject::WFStepObject *nextWS)
{
    vtkKWMyWizardWorkflow *curWF = this->m_curWizWidg->GetMyWizardWorkflow();    
    std::cout<<"Steps in navigation stack: "<<curWF->GetNumberOfStepsInNavigationStack()<<std::endl;
    
    if(nextWS)
    {
        //initialize TCL Conditions
//        this->initializeTCLConditions(nextWS);
        
        WFStateConverter *wfSC = new WFStateConverter(nextWS);
        vtkKWWizardStep *nextStep = wfSC->GetKWWizardStep();
                
//        nextStep->SetValidateCommand(this, "workStepValidationCallBack");
        nextStep->SetShowUserInterfaceCommand(this, "workStepGUICallBack");
        curWF->AddStep(nextStep);
        
        vtkKWStateMachineInput *validStepInput = vtkKWStateMachineInput::New();
        validStepInput->SetName("nextStepValidationInput");
        
        curWF->AddInput(validStepInput);
        
        curWF->CreateNextTransition(curWF->GetCurrentStep(), validStepInput, nextStep);
        curWF->CreateBackTransition(curWF->GetCurrentStep(), nextStep);
        
        curWF->CreateNextTransition(
                nextStep,
                vtkKWWizardStep::GetValidationSucceededInput(),
                curWF->GetFinishStep());
        curWF->CreateBackTransition(nextStep, curWF->GetFinishStep());
        
        curWF->CreateGoToTransitionsToFinishStep();
        
        curWF->PushInput(validStepInput);
        
        this->m_curWFStep = nextWS;        
    }
    else
    {
        curWF->CreateNextTransition(
                curWF->GetCurrentStep(),
                vtkKWWizardStep::GetValidationSucceededInput(),
                curWF->GetFinishStep());
        curWF->CreateBackTransition(curWF->GetFinishStep(),
                curWF->GetCurrentStep());
        curWF->PushInput(vtkKWWizardStep::GetValidationSucceededInput());
        this->m_curWFStep = NULL;
    }
    curWF->ProcessInputs();
    this->m_curStepID = curWF->GetCurrentStep()->GetId();
}

void vtkWFEngineModuleGUI::nextTransitionCallback(vtkObject* obj, unsigned long id,void* callBackData, void*)
{
    std::cout<<"nextTransitionCallback: "<<id<<std::endl;
    vtkWFEngineModuleGUI *wfEngineModule = (vtkWFEngineModuleGUI*)callBackData;
    if(wfEngineModule)
    {
        wfEngineModule->UpdateMRML();
        wfEngineModule->UpdateParameter();
        //check step validation

        if(wfEngineModule->m_wfStepHandler->LoadNextWorkStep() == vtkWFStepHandler::SUCC)
        {
            wfEngineModule->workStepValidationCallBack(wfEngineModule->m_wfStepHandler->GetLoadedWFStep());            
        }
        else
        {
            wfEngineModule->m_curWizWidg->SetErrorText(wfEngineModule->m_wfStepHandler->GetLastError().c_str());
            wfEngineModule->UpdateGUI();
            
            wfEngineModule->m_curWizWidg->GetWizardWorkflow()->PushInput(vtkKWWizardStep::GetValidationFailedInput());
        }
    }
}

void vtkWFEngineModuleGUI::backTransitionCallback(vtkObject* obj, unsigned long id,void* callBackData, void*)
{
    std::cout<<"backTransitionCallback: "<<id<<std::endl;
    vtkWFEngineModuleGUI *wfEngineModule = (vtkWFEngineModuleGUI*)callBackData;
    if(wfEngineModule)
    {
        vtkKWMyWizardWorkflow *wizWF = wfEngineModule->m_curWizWidg->GetMyWizardWorkflow();
        if(wfEngineModule->m_wfStepHandler->LoadBackWorkStep() == vtkWFStepHandler::SUCC)
        {
            wfEngineModule->workStepValidationCallBack(wfEngineModule->m_wfStepHandler->GetLoadedWFStep());   
        }
        else
        {
            wfEngineModule->m_curWFStep = NULL;
            wfEngineModule->m_curWizWidg->GetWizardWorkflow()->AttemptToGoToPreviousStep();            
        }
        
        wfEngineModule->workStepGUICallBack();
        //because there is no GUI callback in a backtransition we call this manually
        
        wfEngineModule->m_curWizWidg->Update();
    }
}

void vtkWFEngineModuleGUI::workStepGUICallBack()
{
    this->UpdateGUI();
}

void vtkWFEngineModuleGUI::deleteWizardWidgetContainer()
{
    std::cout<<"deleteWizardWidgetContainer"<<std::endl;
    if(this->m_wizFrame)
    {
//        vtkKWWidget *page = this->UIPanel->GetPageWidget ( "WFEngineModule" );
        this->m_wizFrame->Unpack();
        this->m_wizFrame->Delete();
        this->m_wizFrame = NULL;
    }
       
}

void vtkWFEngineModuleGUI::UpdateMRML()
{
    //check if information is available
    if(!this->m_curNameToValueMap)
    {
        return;
    }
    
    vtkMRMLWFEngineModuleNode* n = this->GetWFEngineModuleNode();
    bool createdNode = false;
    if (n == NULL)
    {
        n = vtkMRMLWFEngineModuleNode::New();
        
        if (n == NULL)
        {
//        this->InUpdateMRML = false;
        vtkDebugMacro("No CommandLineModuleNode available");
        return;
        }
      
      // set the a module description for this node
//      n->SetModuleDescription( this->ModuleDescriptionObject );
      
      // set an observe new node in Logic
//      this->Logic->SetCommandLineModuleNode(n);
      vtkSetAndObserveMRMLNodeMacro(this->WFEngineModuleNode,n);

      createdNode = true;
    }
    
    std::map<std::string, std::string>::iterator iter;        
    
    for(iter = this->m_curNameToValueMap->begin(); iter != this->m_curNameToValueMap->end(); iter++)
    {
        
    }//for
    
    //set the map back to null
//    this->m_curNameToWidgetMap = NULL;
}

void vtkWFEngineModuleGUI::SetWFEngineModuleNode(vtkMRMLWFEngineModuleNode *node)
{
    this->WFEngineModuleNode = node;
}

void vtkWFEngineModuleGUI::widgetChangedCallback(vtkObject* obj, unsigned long eid, void* clientData, void* callData)
{
    std::cout<<"something Changed Called Back;-)"<<std::endl;
}

const char *vtkWFEngineModuleGUI::getStepInputValueByName(std::string name)
{
    //check if information is available
    if(!this->WFEngineModuleNode)
    {
        return "";
    }
    
    if(!this->m_curWFStep)
    {
        return "";
    }
    
    std::string curAttributeName = this->m_curWFStep->GetID();
    curAttributeName.append("." + name);
    
    std::cout<<curAttributeName<<std::endl;
    return this->WFEngineModuleNode->GetAttribute(curAttributeName.c_str());
                   
}

void vtkWFEngineModuleGUI::UpdateWorkflowList()
{
    std::vector<vtkWFStepHandler::workflowDesc*> *tmpWFDescList = this->m_wfStepHandler->GetKnownWorkflowDescriptions();
    std::vector<vtkWFStepHandler::workflowDesc*>::iterator iter;
    
    if(this->m_mclDW)
    {
        this->m_mclDW->DeleteAllRows();
        for(iter = tmpWFDescList->begin(); iter != tmpWFDescList->end(); iter++)
        {
            vtkWFStepHandler::workflowDesc *tmpWFDesc = *iter;
            this->addWorkflowToList(tmpWFDesc->workflowName.c_str(), tmpWFDesc->fileName.c_str(), tmpWFDesc->createTime);
        }
    }        
}

void vtkWFEngineModuleGUI::UpdateParameter()
{
    if(this->m_wfStepHandler && this->m_curNameToValueMap)
    {
        std::map<std::string, std::string>::iterator iter;
        for(iter = this->m_curNameToValueMap->begin(); iter != this->m_curNameToValueMap->end(); iter++)
        {
            const char* value = this->getStepInputValueByName(iter->first);
            std::string strValue = value;
            std::cout<<strValue<<std::endl;
            std::cout<<"name: "<<iter->first<<" value: "<<value<<std::endl;
            this->m_wfStepHandler->AddParameter(iter->first.c_str(), value);
        }        
    }
}

void vtkWFEngineModuleGUI::UpdateGUI()
{
    if(this->m_curWizWidg)
    {
        // Destroy all ClientAreaChildren!
        this->m_curWizWidg->GetClientArea()->RemoveAllChildren();        
    }
    
    if(!this->m_wfStepHandler)
    {
        return;
    }
    
    if(!this->m_curWFStep)
    {
        return;
    }
    
//    ModuleDescription curModuleDesc;
//    
//    ModuleDescriptionParser curMDParser;
//    
//    std::string guiDesc = this->m_curWFStep->GetGUIDescription();
//    std::cout<<"guiParser: ";
//    std::cout<<curMDParser.Parse(guiDesc, curModuleDesc)<<std::endl;
    
    this->m_curParameterWidgets = vtkSlicerParameterWidget::New();
    
    vtkCallbackCommand *widgetChangedCBC = vtkCallbackCommand::New();
    m_curParameterWidgets->SetApplication(this->GetApplication());
    vtkSlicerModuleLogic *myModuleLogic = vtkSlicerModuleLogic::New();
    
    m_curParameterWidgets->SetParent(this->m_curWizWidg->GetClientArea());
    m_curParameterWidgets->SetSlicerModuleLogic(myModuleLogic);
    
    // set ModuleNode to the new step
    m_curParameterWidgets->SetWidgetID(this->m_curWFStep->GetID());
    m_curParameterWidgets->SetMRMLNode(this->GetWFEngineModuleNode());
    
    m_curParameterWidgets->SetModuleDescription(this->m_wfStepHandler->GetCurrentModuleDescription());
    m_curParameterWidgets->SetErrorMap(this->m_wfStepHandler->GetValidationErrorMap());
    m_curParameterWidgets->CreateWidgets();
    
    widgetChangedCBC->SetClientData(this);
    widgetChangedCBC->SetCallback(vtkWFEngineModuleGUI::widgetChangedCallback);
    m_curParameterWidgets->AddObserver(vtkSlicerParameterWidget::ParameterWidgetChangedEvent, widgetChangedCBC);
    
    widgetChangedCBC->Delete();
    std::cout<<"vector size "<<m_curParameterWidgets->size()<<std::endl;
    
    if(m_curParameterWidgets->size() == 0)
    {
        return;
    }
    
    // Build GUI out of the widget list
    vtkKWWidget *curGroupWidget = m_curParameterWidgets->GetNextWidget();
    this->m_curNameToValueMap = new std::map<std::string, std::string>;
    while(curGroupWidget)
    {
        this->GetApplication()->Script("pack %s -side top -anchor ne -expand y -fill both -padx 2 -pady 2",
                curGroupWidget->GetWidgetName());
        std::vector<ModuleParameter> *curParameterList = m_curParameterWidgets->GetCurrentParameters();
        
        std::vector<ModuleParameter>::iterator iter;
        for(iter = curParameterList->begin(); iter != curParameterList->end(); iter++)
        {
            this->m_curNameToValueMap->insert(std::make_pair(iter->GetName(),iter->GetDefault()));
        }
        curGroupWidget = m_curParameterWidgets->GetNextWidget();  
    }
}

//void vtkWFEngineModuleGUI::initializeTCLConditions(WFEngine::nmWFStepObject::WFStepObject* curWS)
//{
//    this->m_condChecker = vtkWFStepConditionChecker::New();
//    this->m_condChecker->SetApplication(this->GetApplication());
//    this->m_condChecker->LoadStepValidationFunction(curWS->GetTCLValidationFunction().c_str());
//    this->m_condChecker->LoadNextStepFunction(curWS->GetTCLNextWorkstepFunction().c_str());
////    condChecker->SetTCLInterp(this->GetApplication()->)    
//}

//void vtkWFEngineModuleGUI::validateStep(WFEngine::nmWFStepObject::WFStepObject* curWS)
//{
//    if(this->m_condChecker)
//    {
//        std::map<std::string, std::string>::iterator iter;
//        for(iter = this->m_curNameToValueMap->begin(); iter != this->m_curNameToValueMap->end(); iter++)
//        {
//            std::cout<<"name: "<<iter->first<<" value: "<<iter->second<<std::endl;
//            this->m_condChecker->AddParameter(iter->first.c_str(), iter->second.c_str());
//        }
//        int result = this->m_condChecker->ValidateStep();
//    }
//}
