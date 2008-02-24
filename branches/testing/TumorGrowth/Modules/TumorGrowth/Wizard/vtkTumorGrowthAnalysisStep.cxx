#include "vtkTumorGrowthAnalysisStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWThumbWheel.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWEntry.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkKWScale.h"
#include "vtkSlicerApplication.h"
#include "vtkKWPushButton.h"
#include "vtkKWMessageDialog.h"
#include "vtkWindowToImageFilter.h"
#include "vtkPNGWriter.h"
#include "vtkImageAppend.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthAnalysisStep);
vtkCxxRevisionMacro(vtkTumorGrowthAnalysisStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::vtkTumorGrowthAnalysisStep()
{
  this->SetName("Analysis"); 
  this->SetDescription("Analysis of Tumor Growth"); 
  this->WizardGUICallbackCommand->SetCallback(vtkTumorGrowthAnalysisStep::WizardGUICallback);

  this->SensitivityScale = NULL;
  this->GrowthLabel = NULL;

  this->ButtonsSave = NULL;
  this->ButtonsSnapshot = NULL;
  this->FrameButtons = NULL;
  this->SnapshotCount = 0;

  this->FrameDeformable     = NULL;
  this->FrameDeformableCol1 = NULL;
  this->FrameDeformableCol2 = NULL;
  this->DeformableTextLabel = NULL;
  this->DeformableMeassureLabel = NULL;

}

//----------------------------------------------------------------------------
vtkTumorGrowthAnalysisStep::~vtkTumorGrowthAnalysisStep()
{

  if (this->ButtonsSave)
    {
    this->ButtonsSave->Delete();
    this->ButtonsSave  = NULL;
    }

  if (this->ButtonsSnapshot)
    {
    this->ButtonsSnapshot->Delete();
    this->ButtonsSnapshot  = NULL;
    }

  if (this->FrameButtons)
    {
    this->FrameButtons->Delete();
    this->FrameButtons  = NULL;
    }
  
  if (this->SensitivityScale)
    {
    this->SensitivityScale->Delete();
    this->SensitivityScale = NULL;
    }
  if (this->GrowthLabel) 
    {
      this->GrowthLabel->Delete();
      this->GrowthLabel = NULL;
    }

 if (this->FrameDeformable) 
    {
      this->FrameDeformable->Delete();
      this->FrameDeformable = NULL;
    }

 if (this->FrameDeformableCol1) 
    {
      this->FrameDeformableCol1->Delete();
      this->FrameDeformableCol1 = NULL;
    }

 if (this->FrameDeformableCol2) 
    {
      this->FrameDeformableCol2->Delete();
      this->FrameDeformableCol2 = NULL;
    }

 if (this->DeformableTextLabel) 
    {
      this->DeformableTextLabel->Delete();
      this->DeformableTextLabel = NULL;
    }

 if (this->DeformableMeassureLabel) 
    {
      this->DeformableMeassureLabel->Delete();
      this->DeformableMeassureLabel = NULL;
    }

}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::AddGUIObservers() 
{
  // cout << "vtkTumorGrowthROIStep::AddGUIObservers()" << endl; 
  // Make sure you do not add the same event twice - need to do it bc of wizrd structure
  if (this->ButtonsSnapshot && (!this->ButtonsSnapshot->HasObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand))) 
    {
      this->ButtonsSnapshot->AddObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand );  
    } 

  if (this->ButtonsSave && (!this->ButtonsSave->HasObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand))) 
    {
      this->ButtonsSave->AddObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand );  
    } 
}

void vtkTumorGrowthAnalysisStep::RemoveGUIObservers() 
{
  // cout << "vtkTumorGrowthAnalysisStep::RemoveGUIObservers" << endl;
  if (this->ButtonsSnapshot) 
    {
      this->ButtonsSnapshot->RemoveObservers(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand);  
    }

  if (this->ButtonsSave) 
  {
      this->ButtonsSave->RemoveObservers(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand);  
  }
}

void vtkTumorGrowthAnalysisStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
  // cout << "void vtkTumorGrowthAnalysisStep::WizardGUICallback" << endl;
    vtkTumorGrowthAnalysisStep *self = reinterpret_cast<vtkTumorGrowthAnalysisStep *>(clientData);
    if (self) { self->ProcessGUIEvents(caller, event, callData); }


}


void vtkTumorGrowthAnalysisStep::ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) {

  // cout << "vtkTumorGrowthAnalysisStep::ProcessGUIEvents" << endl;

  if (event == vtkKWPushButton::InvokedEvent) {
    vtkKWPushButton *button = vtkKWPushButton::SafeDownCast(caller);
    if (this->ButtonsSnapshot && (button == this->ButtonsSnapshot)) 
    { 
      this->TakeScreenshot(); 
    }
    else if (this->ButtonsSave && (button == this->ButtonsSave)) 
    { 
      vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
      if (node) {
        // Depends on analysis
        // Save Data 
    vtkMRMLVolumeNode *volumeAnalysisNode = NULL;
        if (node->GetAnalysis_Intensity_Flag()) { 
          volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetAnalysis_Intensity_Ref()));
    } else if (node->GetAnalysis_Deformable_Flag()) { 
      volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetAnalysis_Deformable_Ref()));
    }
    if (volumeAnalysisNode) {
      vtkTumorGrowthLogic *Logic = this->GetGUI()->GetLogic();
      Logic->SaveVolumeForce(vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()),volumeAnalysisNode);
    } 

        // Save MRML 
        node->GetScene()->SetRootDirectory(node->GetWorkingDir());

        std::string fileName(node->GetWorkingDir());
        fileName.append("/Data.mrml");
        node->GetScene()->SetURL(fileName.c_str());

        // Saves file  
        node->GetScene()->Commit();

        std::string infoMsg("Saved Data to ");
        infoMsg.append(node->GetWorkingDir());

        vtkKWMessageDialog::PopupMessage(this->GetGUI()->GetApplication(), this->GetGUI()->GetApplicationGUI()->GetMainSlicerWindow(),
                                         "Tumor Growth",infoMsg.c_str(), vtkKWMessageDialog::OkDefault);

      } else {
        this->GetGUI()->GetApplicationGUI()->ProcessSaveSceneAsCommand();
        node = this->GetGUI()->GetNode();
      }
   
      // Save Results to file 
      if (node) { 
        std::string fileName(node->GetWorkingDir());
        fileName.append("/AnalysisOutcome.log");
        std::ofstream outfile(fileName.c_str());
        if (outfile.fail()) {
      cout << "Error: vtkTumorGrowthAnalysisStep::ProcessGUIEvents: Cannot write to file " << fileName.c_str() << endl;
    } else {     
       this->GetGUI()->GetLogic()->PrintResult(outfile, vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()));
      cout << "Wrote outcome of analysis to " << fileName.c_str() << endl;
    }
      }

    }
  }
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::ShowUserInterface()
{
  // ----------------------------------------
  // Display Analysis Volume 
  // ----------------------------------------  
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
  if (node) { 
    vtkMRMLVolumeNode *volumeSampleNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_SuperSampleRef()));
    vtkMRMLVolumeNode *volumeAnalysisNode = NULL;
    if (node->GetAnalysis_Intensity_Flag()) {
      volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetAnalysis_Intensity_Ref()));
    } else if (node->GetAnalysis_Deformable_Flag()) {
      volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetAnalysis_Deformable_Ref()));
    } else {
      volumeAnalysisNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan2_LocalRef()));
    }

    if (volumeSampleNode && volumeAnalysisNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeSampleNode->GetID());

      vtkSlicerApplicationGUI *applicationGUI     = this->GetGUI()->GetApplicationGUI();

      double oldSliceSetting[3];
      oldSliceSetting[0] = double(applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetOffsetScale()->GetValue());
      oldSliceSetting[1] = double(applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetOffsetScale()->GetValue());
      oldSliceSetting[2] = double(applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetOffsetScale()->GetValue());

      applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeAnalysisNode);
      applicationGUI->GetSlicesControlGUI()->GetSliceFadeScale()->SetValue(0.6);

      applicationLogic->PropagateVolumeSelection();

      // Return to original slice position 
      applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[0]);
      applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[1]);
      applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[2]);
    } 
  }

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------

  this->vtkTumorGrowthStep::ShowUserInterface();

  this->Frame->SetLabelText("Intensity Pattern");
  if (node->GetAnalysis_Intensity_Flag()) {
    this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());
  }

  if (!this->SensitivityScale)
    {
    this->SensitivityScale = vtkKWThumbWheel::New();
    }
  if (!this->SensitivityScale->IsCreated())
  {
    this->SensitivityScale->SetParent(this->Frame->GetFrame());
    this->SensitivityScale->Create();
    this->SensitivityScale->SetRange(0.0,1.0);
    this->SensitivityScale->SetMinimumValue(0.0);
    this->SensitivityScale->ClampMinimumValueOn(); 
    this->SensitivityScale->SetMaximumValue(1.0);
    this->SensitivityScale->ClampMaximumValueOn(); 
    this->SensitivityScale->SetResolution(0.75);
    this->SensitivityScale->SetLinearThreshold(1);
    this->SensitivityScale->SetThumbWheelSize (TUMORGROWTH_WIDGETS_SLIDER_WIDTH,TUMORGROWTH_WIDGETS_SLIDER_HEIGHT);
    this->SensitivityScale->DisplayEntryOn();
    this->SensitivityScale->DisplayLabelOn();
    this->SensitivityScale->GetLabel()->SetText("Sensitivity");
    this->SensitivityScale->SetCommand(this,"SensitivityChangedCallback");
    this->SensitivityScale->DisplayEntryAndLabelOnTopOff(); 
    this->SensitivityScale->SetBalloonHelpString("The further the wheel is turned to the right the more robust the result");

    // this->SensitivityScale->GetEntry()->SetCommandTriggerToAnyChange();
  }

  // Initial value 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (mrmlNode) this->SensitivityScale->SetValue(mrmlNode->GetAnalysis_Intensity_Sensitivity());
  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->SensitivityScale->GetWidgetName());

  if (!this->GrowthLabel)
    {
    this->GrowthLabel = vtkKWLabel::New();
    }
  if (!this->GrowthLabel->IsCreated())
  {
    this->GrowthLabel->SetParent(this->Frame->GetFrame());
    this->GrowthLabel->Create();
  }
  this->Script( "pack %s -side top -anchor nw -padx 2 -pady 2", this->GrowthLabel->GetWidgetName());


  if (!this->FrameDeformable)
  {
    this->FrameDeformable = vtkKWFrameWithLabel::New();
  }
  if (!this->FrameDeformable->IsCreated())
  {
      vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
      this->FrameDeformable->SetParent(wizard_widget->GetClientArea());
      this->FrameDeformable->Create();
      this->FrameDeformable->SetLabelText("Deformable Map");
      this->FrameDeformable->AllowFrameToCollapseOff();
  }
  if (node->GetAnalysis_Deformable_Flag()) {
     this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->FrameDeformable->GetWidgetName());
  }

  if (!this->FrameDeformableCol1)
  {
    this->FrameDeformableCol1 = vtkKWFrame::New();
  }
  if (!this->FrameDeformableCol1->IsCreated())
  {
      this->FrameDeformableCol1->SetParent(this->FrameDeformable->GetFrame());
      this->FrameDeformableCol1->Create();
  }

  if (!this->FrameDeformableCol2)
  {
    this->FrameDeformableCol2 = vtkKWFrame::New();
  }
  if (!this->FrameDeformableCol2->IsCreated())
  {
      this->FrameDeformableCol2->SetParent(this->FrameDeformable->GetFrame());
      this->FrameDeformableCol2->Create();
  }

  this->Script("pack %s %s -side left -anchor nw -fill x -padx 0 -pady 0", this->FrameDeformableCol1->GetWidgetName(),this->FrameDeformableCol2->GetWidgetName());


  if (!this->DeformableTextLabel)
    {
    this->DeformableTextLabel = vtkKWLabel::New();
    }
  if (!this->DeformableTextLabel->IsCreated())
  {
    this->DeformableTextLabel->SetParent(this->FrameDeformableCol1);
    this->DeformableTextLabel->Create();
  }

  this->DeformableTextLabel->SetText("Segmentation Metric: \nJacobian Metric:");
  this->Script( "pack %s -side left -anchor nw -padx 2 -pady 2", this->DeformableTextLabel->GetWidgetName());


  if (!this->DeformableMeassureLabel)
    {
    this->DeformableMeassureLabel = vtkKWLabel::New();
    }
  if (!this->DeformableMeassureLabel->IsCreated())
  {
    this->DeformableMeassureLabel->SetParent(this->FrameDeformableCol2);
    this->DeformableMeassureLabel->Create();
  }

  {
    char TEXT[1024];
    sprintf(TEXT,"%.3f mm%c (%d Voxels)\n%.3f mm%c (%d Voxels)", 
        mrmlNode->GetAnalysis_Deformable_SegmentationGrowth(),179,int(mrmlNode->GetAnalysis_Deformable_SegmentationGrowth()/mrmlNode->GetScan1_VoxelVolume()), 
        mrmlNode->GetAnalysis_Deformable_JacobianGrowth(),179,int(mrmlNode->GetAnalysis_Deformable_JacobianGrowth()/mrmlNode->GetScan1_VoxelVolume()));
    this->DeformableMeassureLabel->SetText(TEXT);

  }
  this->Script( "pack %s -side left -anchor nw -padx 2 -pady 0", this->DeformableMeassureLabel->GetWidgetName());


  // Define buttons 
  if (!this->FrameButtons)
  {
    this->FrameButtons = vtkKWFrameWithLabel::New();
  }
  if (!this->FrameButtons->IsCreated())
  {
      vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
      this->FrameButtons->SetParent(wizard_widget->GetClientArea());
      this->FrameButtons->Create();
      this->FrameButtons->SetLabelText("Save Analysis");
      this->FrameButtons->AllowFrameToCollapseOff();
  }
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->FrameButtons->GetWidgetName());

 if (!this->ButtonsSnapshot) {
    this->ButtonsSnapshot = vtkKWPushButton::New();
  }

  if (!this->ButtonsSnapshot->IsCreated()) {
    this->ButtonsSnapshot->SetParent(this->FrameButtons->GetFrame());
    this->ButtonsSnapshot->Create();
    this->ButtonsSnapshot->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsSnapshot->SetText("Snapshot");
    // this->ButtonsSnapshot->EnabledOff();
  }

  if (!this->ButtonsSave) {
    this->ButtonsSave = vtkKWPushButton::New();
  }
  if (!this->ButtonsSave->IsCreated()) {
    this->ButtonsSave->SetParent(this->FrameButtons->GetFrame());
    this->ButtonsSave->Create();
    this->ButtonsSave->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsSave->SetText("Data");
  }

  this->Script("pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2", 
                this->ButtonsSnapshot->GetWidgetName(),this->ButtonsSave->GetWidgetName());

  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    // wizard_widget->GetOKButton()->SetText("Run");
    wizard_widget->GetCancelButton()->SetText("OK"); 
    wizard_widget->GetCancelButton()->SetCommand(this, "ResetPipelineCallback");
    wizard_widget->GetCancelButton()->EnabledOn();
    wizard_widget->OKButtonVisibilityOff();

  }

  this->CreateGridButton();

  // Show results 
  this->SensitivityChangedCallback(0.0);
  this->AddGUIObservers();
}


//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::SensitivityChangedCallback(double value)
{
  // Sensitivity has changed because of user interaction
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (!this->SensitivityScale || !mrmlNode || !this->GrowthLabel ) return;
  mrmlNode->SetAnalysis_Intensity_Sensitivity(this->SensitivityScale->GetValue());
  double Growth = this->GetGUI()->GetLogic()->MeassureGrowth();
  // show here 
  char TEXT[1024];
  // cout << "---------- " << Growth << " " << mrmlNode->GetSuperSampled_VoxelVolume() << " " << mrmlNode->GetSuperSampled_RatioNewOldSpacing() << endl;;
  sprintf(TEXT,"Intensity Metric: %.3f mm%c (%d Voxels)", Growth*mrmlNode->GetSuperSampled_VoxelVolume(),179,int(Growth*mrmlNode->GetSuperSampled_RatioNewOldSpacing()));

  this->GrowthLabel->SetText(TEXT);
  // Show updated results 
  vtkMRMLVolumeNode *analysisNode = vtkMRMLVolumeNode::SafeDownCast(mrmlNode->GetScene()->GetNodeByID(mrmlNode->GetAnalysis_Intensity_Ref()));
  if (analysisNode) analysisNode->Modified();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::TakeScreenshot() {
  vtkImageAppend *screen = vtkImageAppend::New();
    screen->SetAppendAxis(0);

  vtkWindowToImageFilter *window0 = vtkWindowToImageFilter::New();
     window0->SetInput(this->GetGUI()->GetApplicationGUI()->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetRenderWindow());
     screen->AddInput(window0->GetOutput());

  vtkWindowToImageFilter *window1 = vtkWindowToImageFilter::New();
     window1->SetInput(this->GetGUI()->GetApplicationGUI()->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetRenderWindow());
     screen->AddInput(window1->GetOutput());

  vtkWindowToImageFilter *window2 = vtkWindowToImageFilter::New();
     window2->SetInput(this->GetGUI()->GetApplicationGUI()->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetRenderWindow());
     screen->AddInput(window2->GetOutput());
     
  screen->Update();

  vtkPNGWriter *saveWriter = vtkPNGWriter::New();
     saveWriter->SetInput(screen->GetOutput());
     this->SnapshotCount ++;
     std::stringstream ss;
     
     char fileName[1024];
     sprintf(fileName,"%s/TG_Screenshot_%03d.png",this->GetGUI()->GetNode()->GetWorkingDir(),this->SnapshotCount);
     saveWriter->SetFileName(fileName);
     cout << "Snapshot is saved to " << fileName << endl;
  saveWriter->Write();

  saveWriter->Delete();
 
  window0->Delete();
  window1->Delete();
  window2->Delete();

  screen->Delete();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::ResetPipelineCallback() 
{
  // Sensitivity has changed because of user interaction 
  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
  vtkKWWizardWorkflow *wizard_workflow = wizard_widget->GetWizardWorkflow();
  // Go Back to the beginning - you can also make this more generale by first getting the number of states 
  // and then doing a loop 
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
  wizard_workflow->AttemptToGoToPreviousStep();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthAnalysisStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

void  vtkTumorGrowthAnalysisStep::RemoveResults()  { 
    vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
    if (!Node) return;
    {
       vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetAnalysis_Intensity_Ref()));
       if (currentNode) { this->GetGUI()->GetMRMLScene()->RemoveNode(currentNode); }
    }
}
