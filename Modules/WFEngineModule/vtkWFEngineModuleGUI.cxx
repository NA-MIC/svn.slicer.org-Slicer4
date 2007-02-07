#include "vtkWFEngineModuleGUI.h"
#include <WFDirectInterface.h>

#include <vtkObject.h>
#include "vtkObjectFactory.h"
#include "vtkCommand.h"

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

#include <string>

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
    this->m_wfDI = NULL;
    this->m_wizFrame = NULL;
    
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
    m_wfDI = WFDirectInterface::New();
    
    if(m_wfDI->InitializeWFEngine())
    {
        std::vector<WFDirectInterface::workflowDesc*> knownWFs;
        knownWFs = m_wfDI->GetAllKnownWorkflows();
        
        std::cout<<"knownWFs.size() "<<knownWFs.size()<<std::endl;
        
        for(std::vector<WFDirectInterface::workflowDesc*>::iterator iter = knownWFs.begin(); iter != knownWFs.end(); iter++)
        {
            WFDirectInterface::workflowDesc *curWFDesc = *iter;
            this->addWorkflowToList(curWFDesc->workflowName.c_str(), curWFDesc->fileName.c_str(), curWFDesc->createTime);
        }
        
        return 1;
    }
    else return 0;
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
    this->m_wizFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    m_wizFrame->SetParent ( page );
    m_wizFrame->Create ( );
    m_wizFrame->SetLabelText (this->m_mclDW->GetCellText(this->m_mclDW->GetIndexOfFirstSelectedRow(),0));
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
    
//    vtkCallbackCommand *wizCB = vtkCallbackCommand::New();
//    wizCB->SetCallback(&vtkWFEngineModuleGUI::nextTransitionCallback);
//    wizCB->SetClientData(this);
//    wizWorkflow->AddObserver(vtkKWMyWizardWorkflow::nextTransitionStartEvent, wizCB);
//    
//    wizCB = vtkCallbackCommand::New();
//    wizCB->SetCallback(&vtkWFEngineModuleGUI::nextTransitionCallback);
//    wizCB->SetClientData(this);
//    wizWorkflow->AddObserver(vtkKWMyWizardWorkflow::backTransitionEndEvent, wizCB);
//    
//    wizCB = vtkCallbackCommand::New();
//    wizCB->SetCallback(&vtkWFEngineModuleGUI::backTransitionCallback);
//    wizCB->SetClientData(this);
//    wizWorkflow->AddObserver(vtkKWMyWizardWorkflow::backTransitionStartEvent, wizCB);
//    wizCB = vtkCallbackCommand::New();
//    wizCB->SetCallback(&vtkWFEngineModuleGUI::backTransitionCallback);
//    wizCB->SetClientData(this);
//    wizWorkflow->AddObserver(vtkKWMyWizardWorkflow::backTransitionEndEvent, wizCB);
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
        myDW->m_wfDI->CloseWorkflowManager();
        myDW->m_wizFrame = NULL;
        myDW->m_selectedWF = -1;
    }
    else if(loadBtn != NULL && myDW->m_wizFrame == NULL)
    {
        myDW->m_wfDI->loadWorkflowFromFile(selectedWF);
        
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
    this->m_wfDI->CloseWorkflowManager();
}

void vtkWFEngineModuleGUI::workStepValidationCallBack(WFStepObject *nextWS)
{
//    vtkWFDynamicWizard *wfDW = (vtkWFDynamicWizard*)param;
        
//    std::cout<<obj->GetClassName()<<std::endl;
//    std::cout<<this->m_curStepID<<" == "<<curWF->GetCurrentStep()->GetId() + 1<<std::endl;
//    if(this->m_curStepID != (curWF->GetCurrentStep()->GetId() + 1))
//    {
//        std::cout<<"***workStepCallBack***"<<std::endl;
//        
//        m_curWFStep = NULL;
////        std::cout<<"***Current StateID: "<<curWF->GetCurrentStep()->GetId()<<" stateMapSize: "<<this->m_kwwID2StepIDMap.size()<<std::endl;
//        std::cout<<"***Stack-Width: "<<curWF->GetNumberOfStepsInNavigationStack()<<" Number of known steps: "<<curWF->GetNumberOfSteps()<<std::endl;
//        if(curWF->GetCurrentStep()->GetId() < curWF->GetNumberOfStepsInNavigationStack())
//        {
//            std::cout<<"Back pressed at position: "<<curWF->GetCurrentStep()->GetId()<<std::endl;
//            m_curWFStep = this->m_wfDI->getBackWorkStep();
//    //        this->m_kwwID2StepIDMap.erase(iter->first);
//        }
//        else
//        {
//            std::cout<<"Next pressed at position: "<<curWF->GetCurrentStep()->GetId()<<std::endl;
//            m_curWFStep = this->m_wfDI->getNextWorkStep();
//        }
    vtkKWMyWizardWorkflow *curWF = this->m_curWizWidg->GetMyWizardWorkflow();    
    int nb_steps_in_stack = curWF->GetNumberOfStepsInNavigationStack();
    std::cout<<"Steps in navigation stack: "<<nb_steps_in_stack<<std::endl;
    if(nextWS)
    {
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
        wfEngineModule->workStepValidationCallBack(wfEngineModule->m_wfDI->getNextWorkStep());
    }
}

void vtkWFEngineModuleGUI::backTransitionCallback(vtkObject* obj, unsigned long id,void* callBackData, void*)
{
    std::cout<<"backTransitionCallback: "<<id<<std::endl;
    vtkWFEngineModuleGUI *wfEngineModule = (vtkWFEngineModuleGUI*)callBackData;
    if(wfEngineModule)
    {
        vtkKWMyWizardWorkflow *wizWF = wfEngineModule->m_curWizWidg->GetMyWizardWorkflow();
        wfEngineModule->workStepValidationCallBack(wfEngineModule->m_wfDI->getBackWorkStep());
        wfEngineModule->workStepGUICallBack();
        //because there is no GUI callback in a backtransition we call this manually
        
        wfEngineModule->m_curWizWidg->Update();
    }
}

void vtkWFEngineModuleGUI::workStepGUICallBack()
{
    if(!this->m_curWFStep)
    {
        return;
    }
    
    // Destroy all ClientAreaChildren!
    this->m_curWizWidg->GetClientArea()->RemoveAllChildren();
    
    ModuleDescription curModuleDesc;
    
    ModuleDescriptionParser curMDParser;
    
    std::string guiDesc = this->m_curWFStep->GetGUIDescription();
    std::cout<<"guiParser: ";
    std::cout<<curMDParser.Parse(guiDesc, curModuleDesc)<<std::endl;
    
    vtkSlicerParameterWidget *myParameterWidgets = vtkSlicerParameterWidget::New();
    myParameterWidgets->SetApplication(this->GetApplication());
    vtkSlicerModuleLogic *myModuleLogic = vtkSlicerModuleLogic::New();
    
    myParameterWidgets->SetParent(this->m_curWizWidg->GetClientArea());
    myParameterWidgets->SetSlicerModuleLogic(myModuleLogic);
    myParameterWidgets->SetModuleDescription(&curModuleDesc);
    myParameterWidgets->CreateWidgets();
    std::cout<<"vector size "<<myParameterWidgets->size()<<std::endl;
    
    if(myParameterWidgets->size() == 0)
    {
        return;
    }
    
    // Build GUI out of the widget list
    vtkKWWidget *curParameterWidget = myParameterWidgets->GetNextWidget();    
    while(curParameterWidget)
    {
//        std::cout<<curParameterWidget->GetWidgetName()<<std::endl;
//        -side top -anchor ne -expand y -fill both -padx 2 -pady 2
//        this->GetApplication()->Script("pack %s -in %s",
//                curParameterWidget->GetWidgetName(), this->m_curWizWidg->GetClientArea()->GetWidgetName());
        this->GetApplication()->Script("pack %s -side top -anchor ne -expand y -fill both -padx 2 -pady 2",
                curParameterWidget->GetWidgetName());
        curParameterWidget = myParameterWidgets->GetNextWidget();
    }    
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
