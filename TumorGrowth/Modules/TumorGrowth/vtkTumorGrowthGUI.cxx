#include "vtkTumorGrowthGUI.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkMRMLTumorGrowthNode.h"
#include "vtkMRMLScene.h"

#include "vtkSlicerApplication.h"

#include "vtkKWMessageDialog.h"
#include "vtkKWProgressGauge.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"

// For PopulateTestingData()
#include "vtkSlicerVolumesGUI.h"
#include "vtkSlicerVolumesLogic.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkDirectory.h"
#include "vtkIntArray.h"
#include "vtkTumorGrowthFirstScanStep.h"
#include "vtkTumorGrowthROIStep.h"
#include "vtkTumorGrowthSegmentationStep.h"
#include "vtkTumorGrowthSecondScanStep.h"
#include "vtkTumorGrowthAnalysisStep.h"

// #include "CSAILLogo.h"
#include "vtkKWIcon.h"

vtkCxxSetObjectMacro(vtkTumorGrowthGUI,Node,vtkMRMLTumorGrowthNode);
vtkCxxSetObjectMacro(vtkTumorGrowthGUI,Logic,vtkTumorGrowthLogic);

//----------------------------------------------------------------------------
vtkTumorGrowthGUI* vtkTumorGrowthGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = 
    vtkObjectFactory::CreateInstance("vtkTumorGrowthGUI");
  if (ret)
    {
    return (vtkTumorGrowthGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTumorGrowthGUI;
}

//----------------------------------------------------------------------------
vtkTumorGrowthGUI::vtkTumorGrowthGUI()
{
  this->Logic        = NULL;
  this->Node         = NULL;
  this->ModuleName   = NULL;

  this->WizardWidget     = vtkKWWizardWidget::New();
  this->FirstScanStep    = NULL;
  this->ROIStep    = NULL;
  this->SegmentationStep = NULL;
  this->SecondScanStep   = NULL;
  this->AnalysisStep     = NULL;

//  vtkKWIcon* logo = vtkKWIcon::New();
//   logo->SetImage(image_CSAILLogo,
//                 image_CSAILLogo_width, image_CSAILLogo_height,
//                 image_CSAILLogo_pixel_size, image_CSAILLogo_length,
//                 0);
//  this->Logo = logo;
//  logo->Delete();
}

//----------------------------------------------------------------------------
vtkTumorGrowthGUI::~vtkTumorGrowthGUI()
{
  this->RemoveMRMLNodeObservers();
  this->RemoveLogicObservers();

  this->SetLogic(NULL);
  this->SetNode(NULL);

  if (this->WizardWidget)
    {
    this->WizardWidget->Delete();
    this->WizardWidget = NULL;
    }

  if (this->FirstScanStep)
    {
    this->FirstScanStep->Delete();
    this->FirstScanStep = NULL;
  }

  if (this->ROIStep)
    {
    this->ROIStep->Delete();
    this->ROIStep = NULL;
  }

  if (this->SegmentationStep)
    {
    this->SegmentationStep->Delete();
    this->SegmentationStep = NULL;
  }

  if (this->SecondScanStep)
    {
    this->SecondScanStep->Delete();
    this->SecondScanStep = NULL;
  }
  if (this->AnalysisStep)
    {
    this->AnalysisStep->Delete();
    this->AnalysisStep = NULL;
  }

}

//----------------------------------------------------------------------------
void vtkTumorGrowthGUI::RemoveMRMLNodeObservers()
{
}

//----------------------------------------------------------------------------
void vtkTumorGrowthGUI::RemoveLogicObservers()
{
}

//----------------------------------------------------------------------------
void vtkTumorGrowthGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::AddGUIObservers() 
{
  // observe when nodes are added or removed from the scene
  vtkIntArray* events = vtkIntArray::New();
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  if (this->GetMRMLScene() != NULL)
    {
    this->SetAndObserveMRMLSceneEvents(this->GetMRMLScene(), events);
    }
  
  // Here nothing happens normally other bc the individual pannels are not created yet and the one for the first step is already created 
  // - add if clause so that same event is not added twice 
  // The wizrad creates them once they are shown on the gui for the first time - and does not delete them afterwards - strange 
  // Have to list them here so if they are all deleted and this function is called afterwards the missing ones are created again 
  if (this->FirstScanStep) this->FirstScanStep->AddGUIObservers();
  if (this->ROIStep) this->ROIStep->AddGUIObservers();
  if (this->SecondScanStep) this->SecondScanStep->AddGUIObservers();

  events->Delete();
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::RemoveGUIObservers()
{
  if (this->FirstScanStep)    this->FirstScanStep->RemoveGUIObservers();
  if (this->ROIStep)          this->ROIStep->RemoveGUIObservers();
  if (this->SegmentationStep) this->SegmentationStep->RemoveGUIObservers();
  if (this->SecondScanStep)   this->SecondScanStep->RemoveGUIObservers();
  if (this->AnalysisStep)     this->AnalysisStep->RemoveGUIObservers();
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::ProcessGUIEvents(vtkObject *caller,
                                                      unsigned long event,
                                                      void *callData) 
{
  if (this->FirstScanStep)    this->FirstScanStep->ProcessGUIEvents(caller, event, callData); 
  if (this->ROIStep)          this->ROIStep->ProcessGUIEvents(caller, event, callData); 
  if (this->SegmentationStep) this->SegmentationStep->ProcessGUIEvents(caller, event, callData); 
  if (this->SecondScanStep)   this->SecondScanStep->ProcessGUIEvents(caller, event, callData); 
  if (this->AnalysisStep)     this->AnalysisStep->ProcessGUIEvents(caller, event, callData); 
}


//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::ProcessLogicEvents (
  vtkObject *caller, unsigned long event, void *callData )
{
  if ( !caller || !this->WizardWidget)
    {
    return;
    }

  // process Logic changes
  vtkTumorGrowthLogic *callbackLogic = 
    vtkTumorGrowthLogic::SafeDownCast(caller);
  
  if ( callbackLogic == this->GetLogic ( ) && 
    event == vtkCommand::ProgressEvent) 
    {
    this->UpdateRegistrationProgress();
    }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthGUI::UpdateRegistrationProgress()
{
  // Kilian: Do this later for analysis 
  double progress = this->Logic->GetProgressGlobalFractionCompleted();
  if(progress>=0 && progress <=1)
    {
    this->GetApplicationGUI()->GetMainSlicerWindow()->GetProgressGauge()->
      SetValue(progress*100);
    }
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::UpdateMRML()
{
  // std::cout <<"UpdateMRML gets called!" << "\n";
  vtkMRMLTumorGrowthNode* n = this->GetNode();

  if (n == NULL) {
    std::cout <<"UpdateMRML: n is null, create new one?!" << "\n";

    //    no parameter node selected yet, create new
    vtkMRMLTumorGrowthNode* TumorGrowthNode = vtkMRMLTumorGrowthNode::New();
    n = TumorGrowthNode;
    this->GetMRMLScene()->AddNode(n);
    this->Logic->SetAndObserveTumorGrowthNode(n);
    vtkSetAndObserveMRMLNodeMacro(this->Node, n);
  }

  // save node parameters for Undo
  this->GetLogic()->GetMRMLScene()->SaveStateForUndo(n);

  // Update individual entries 
  if (this->FirstScanStep)    this->FirstScanStep->UpdateMRML(); 
  if (this->ROIStep)          this->ROIStep->UpdateMRML(); 
  if (this->SegmentationStep) this->SegmentationStep->UpdateMRML(); 
  if (this->SecondScanStep)   this->SecondScanStep->UpdateMRML(); 
  if (this->AnalysisStep)     this->AnalysisStep->UpdateMRML(); 
}

// according to vtkGradnientAnisotrpoicDiffusionoFilterGUI
//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::UpdateGUI()
{
  vtkMRMLTumorGrowthNode* n = this->GetNode();
  if (n != NULL)
    {
      // This might have to be changed bc instances might not yet be created 
      if (this->FirstScanStep)    this->FirstScanStep->UpdateGUI(); 
      if (this->ROIStep)          this->ROIStep->UpdateGUI(); 
      if (this->SegmentationStep) this->SegmentationStep->UpdateGUI(); 
      if (this->SecondScanStep)   this->SecondScanStep->UpdateGUI(); 
      if (this->AnalysisStep)     this->AnalysisStep->UpdateGUI(); 
    }
}


//  according to vtkGradnientAnisotrpoicDiffusionoFilterGUI

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::ProcessMRMLEvents(vtkObject *caller,
                                       unsigned long event,
                                       void *callData) 
{

  // cout << "============ vtkTumorGrowthGUI::ProcessMRMLEvents Start ==========" << endl;

  // TODO: map the object and event to strings for tcl
  
  //vtksys_stl::cout << "ProcessMRMLEvents()" << vtksys_stl::endl;
  // if parameter node has been changed externally, update GUI widgets
  // with new values 
  vtkMRMLTumorGrowthNode* node = vtkMRMLTumorGrowthNode::SafeDownCast(caller);
  if (node != NULL && this->GetNode() == node) 
  {
     this->UpdateGUI();
  }

  // cout << "============ vtkTumorGrowthGUI::ProcessMRMLEvents End ==========" << endl;
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::BuildGUI() 
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

  const char *help = "**TumorGrowth Module:** **Under Construction** ";
  
  this->Logic->RegisterMRMLNodesWithScene();

  this->UIPanel->AddPage("TumorGrowth", "TumorGrowth", NULL);
  vtkKWWidget *module_page = 
    this->UIPanel->GetPageWidget("TumorGrowth");

  //this->PopulateTestingData();

  // -----------------------------------------------------------------------
  // Help

  vtkSlicerModuleCollapsibleFrame *help_frame = 
    vtkSlicerModuleCollapsibleFrame::New();
  help_frame->SetParent(module_page);
  help_frame->Create();
  help_frame->CollapseFrame();
  help_frame->SetLabelText("Help");
  help_frame->Delete();

  app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
              help_frame->GetWidgetName(), 
              module_page->GetWidgetName());
  
  // configure the parent classes help text widget

  this->HelpText->SetParent(help_frame->GetFrame());
  this->HelpText->Create();
  this->HelpText->SetHorizontalScrollbarVisibility(0);
  this->HelpText->SetVerticalScrollbarVisibility(1);
  this->HelpText->GetWidget()->SetText(help);
  this->HelpText->GetWidget()->SetReliefToFlat();
  this->HelpText->GetWidget()->SetWrapToWord();
  this->HelpText->GetWidget()->ReadOnlyOn();
  this->HelpText->GetWidget()->QuickFormattingOn();

  app->Script("pack %s -side top -fill x -expand y -anchor w -padx 2 -pady 4",
              this->HelpText->GetWidgetName());

  // -----------------------------------------------------------------------
  // Define Wizard with the order of the steps

  vtkSlicerModuleCollapsibleFrame *wizard_frame = 
    vtkSlicerModuleCollapsibleFrame::New();
  wizard_frame->SetParent(module_page);
  wizard_frame->Create();
  wizard_frame->SetLabelText("Wizard");
  wizard_frame->ExpandFrame();

  app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
              wizard_frame->GetWidgetName(), 
              module_page->GetWidgetName());
   
  this->WizardWidget->SetParent(wizard_frame->GetFrame());
  this->WizardWidget->Create();
  this->WizardWidget->GetSubTitleLabel()->SetHeight(1);
  this->WizardWidget->SetClientAreaMinimumHeight(150);
  //this->WizardWidget->SetButtonsPositionToTop();

  this->WizardWidget->HelpButtonVisibilityOff();
  this->WizardWidget->CancelButtonVisibilityOff();
  this->WizardWidget->FinishButtonVisibilityOff();

  app->Script("pack %s -side top -anchor nw -fill both -expand y",
              this->WizardWidget->GetWidgetName());
  wizard_frame->Delete();
 
  vtkKWWizardWorkflow *wizard_workflow = 
    this->WizardWidget->GetWizardWorkflow();
  vtkNotUsed(vtkKWWizardWidget *wizard_widget = this->WizardWidget;);

  // -----------------------------------------------------------------
  // Parameter Set step
   
  // To add a step to the wizard
  // - Create files in Wizard directory
  // - Inlude them into CMakeLists.txt
  // add variable to vtkTumorGrowthGUI.h
  // add to vtkTumorGrowthGUI.cxx : 
  //    - header file
  //    - vtkTumorGrowthGUI::vtkTumorGrowthGUI(), 
  //    - vtkTumorGrowthGUI::~vtkTumorGrowthGUI(
  //    - here
  //    - Tear Down GUI


  if (!this->FirstScanStep)
  {
    this->FirstScanStep = vtkTumorGrowthFirstScanStep::New();
    this->FirstScanStep->SetGUI(this);
  }
  wizard_workflow->AddStep(this->FirstScanStep);

  if (!this->ROIStep)
    {
    this->ROIStep = vtkTumorGrowthROIStep::New();
    this->ROIStep->SetGUI(this);
    
    }
  wizard_workflow->AddNextStep(this->ROIStep);

  if (!this->SegmentationStep)
    {
    this->SegmentationStep = vtkTumorGrowthSegmentationStep::New();
    this->SegmentationStep->SetGUI(this);
    }
  wizard_workflow->AddNextStep(this->SegmentationStep);

  if (!this->SecondScanStep)
    {
    this->SecondScanStep = vtkTumorGrowthSecondScanStep::New();
    this->SecondScanStep->SetGUI(this);
    }
  wizard_workflow->AddNextStep(this->SecondScanStep);

  if (!this->AnalysisStep)
    {
    this->AnalysisStep = vtkTumorGrowthAnalysisStep::New();
    this->AnalysisStep->SetGUI(this);
    }
  wizard_workflow->AddNextStep(this->AnalysisStep);

  // -----------------------------------------------------------------
  // Initial and finish step
  wizard_workflow->SetFinishStep(this->AnalysisStep);
  wizard_workflow->CreateGoToTransitionsToFinishStep();
  wizard_workflow->SetInitialStep(this->FirstScanStep);
  this->ROIStep->GetInteractionState();
  // This way we can restart the machine - did not work 
  // wizard_workflow->CreateGoToTransitions(wizard_workflow->GetInitialStep());
}

//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::TearDownGUI() 
{
   if (this->FirstScanStep)
   {
     this->FirstScanStep->SetGUI(NULL);
   }

   if (this->ROIStep)
   {
     this->ROIStep->SetGUI(NULL);
   }

   if (this->SegmentationStep)
   {
     this->SegmentationStep->SetGUI(NULL);
   }

   if (this->SecondScanStep)
   {
     this->SecondScanStep->SetGUI(NULL);
   }

   if (this->AnalysisStep)
   {
     this->AnalysisStep->SetGUI(NULL);
   }

}

//---------------------------------------------------------------------------
unsigned long vtkTumorGrowthGUI::
AddObserverByNumber(vtkObject *observee, unsigned long event) 
{
  return (observee->AddObserver(event, 
                                (vtkCommand *)this->GUICallbackCommand));
} 


//---------------------------------------------------------------------------
void vtkTumorGrowthGUI::PopulateTestingData() 
{
  this->Logic->PopulateTestingData();

  vtkSlicerModuleGUI *m = vtkSlicerApplication::SafeDownCast(
    this->GetApplication())->GetModuleGUIByName("Volumes"); 

  if ( m != NULL ) 
    {
    vtkSlicerVolumesLogic* volume_logic = 
      vtkSlicerVolumesGUI::SafeDownCast(m)->GetLogic();
    vtksys_stl::string file_path = vtksys::SystemTools::GetEnv("HOME");
#ifdef _WIN32
    file_path.append("\\tmp\\TumorGrowthTestImages");
    if (!vtksys::SystemTools::FileIsDirectory(file_path.c_str()))
      {
      file_path = vtksys::SystemTools::GetEnv("HOME");
      file_path.append("\\temp\\TumorGrowthTestImages");
      }
    file_path.append("\\");
#else
    file_path.append("/tmp/TumorGrowthTestImages/");
#endif
    
    vtkDirectory *dir = vtkDirectory::New();
    if (!dir->Open(file_path.c_str()))
      {
      dir->Delete();
      return;
      }
    
    for (int i = 0; i < dir->GetNumberOfFiles(); i++)
      {
      vtksys_stl::string filename = dir->GetFile(i);
      //skip . and ..
      if (strcmp(filename.c_str(), ".") == 0)
        {
        continue;
        }
      else if (strcmp(filename.c_str(), "..") == 0)
        {
        continue;
        }

      vtksys_stl::string fullName = file_path;
      fullName.append(filename.c_str());
      if (strcmp(vtksys::SystemTools::
                 GetFilenameExtension(fullName.c_str()).c_str(), ".mhd") != 0)
        {
        continue;
        }

      if (vtksys::SystemTools::FileExists(fullName.c_str()) &&
          !vtksys::SystemTools::FileIsDirectory(fullName.c_str()))
        {
        volume_logic->AddArchetypeVolume((char*)(fullName.c_str()), 1, 0, 
                                         filename.c_str()); 
        }
      }
    dir->Delete();
       
//     this->MRMLManager->SetTreeNodeSpatialPriorVolumeID(
//       this->MRMLManager->GetTreeRootNodeID(), 
//       this->MRMLManager->GetVolumeNthID(0));
// 
//     this->MRMLManager->SetRegistrationAtlasVolumeID(
//       this->MRMLManager->GetVolumeNthID(0));
//     this->MRMLManager->AddTargetSelectedVolume(
//       this->MRMLManager->GetVolumeNthID(1));
//     this->MRMLManager->SetRegistrationTargetVolumeID(
//       this->MRMLManager->GetVolumeNthID(1));

    // this->MRMLManager->SetSaveWorkingDirectory(file_path.c_str());
    // this->MRMLManager->SetSaveTemplateFilename(file_path.append("TumorGrowthSTemplate.mrml").c_str());
    }
} 
