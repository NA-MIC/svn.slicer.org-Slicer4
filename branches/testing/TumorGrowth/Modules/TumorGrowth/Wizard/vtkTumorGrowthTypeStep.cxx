#include "vtkTumorGrowthTypeStep.h"
#include "vtkTumorGrowthGUI.h"

#include "vtkMRMLTumorGrowthNode.h"

#include "vtkTumorGrowthLogic.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerModelsLogic.h"
#include "vtkKWCheckButton.h"
#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkSlicerApplication.h"
#include "vtkKWScale.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWMessageDialog.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthTypeStep);
vtkCxxRevisionMacro(vtkTumorGrowthTypeStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthTypeStep::vtkTumorGrowthTypeStep()
{
  this->SetName("3/4. Define Metric"); 
  this->SetDescription("We provide several metrics to meassure growth."); 
  this->WizardGUICallbackCommand->SetCallback(vtkTumorGrowthTypeStep::WizardGUICallback);

  this->FrameTypeIntensity = NULL;
  this->FrameTypeSegmentation = NULL;
  this->FrameTypeJacobian = NULL;

  this->TypeIntensityCheckButton  = NULL;
  this->TypeSegmentationCheckButton  = NULL;
  this->TypeJacobianCheckButton  = NULL;
  
}

//----------------------------------------------------------------------------
vtkTumorGrowthTypeStep::~vtkTumorGrowthTypeStep()
{

  if (this->FrameTypeIntensity) 
    {
    this->FrameTypeIntensity->Delete();
    this->FrameTypeIntensity = NULL;
    }


  if (this->FrameTypeSegmentation) 
    {
    this->FrameTypeSegmentation->Delete();
    this->FrameTypeSegmentation = NULL;
    }

  if (this->FrameTypeJacobian) 
    {
    this->FrameTypeJacobian->Delete();
    this->FrameTypeJacobian = NULL;
    }

  if (this->TypeIntensityCheckButton)
    {
    this->TypeIntensityCheckButton->Delete();
    this->TypeIntensityCheckButton = NULL;
    }

  if (this->TypeSegmentationCheckButton)
    {
    this->TypeSegmentationCheckButton->Delete();
    this->TypeSegmentationCheckButton = NULL;
    }

  if (this->TypeJacobianCheckButton)
    {
    this->TypeJacobianCheckButton->Delete();
    this->TypeJacobianCheckButton = NULL;
    }
}


//----------------------------------------------------------------------------
void vtkTumorGrowthTypeStep::AddGUIObservers() 
{
  // cout << "vtkTumorGrowthTypeStep::AddGUIObservers Start" << endl;
  if (this->TypeIntensityCheckButton && (!this->TypeIntensityCheckButton->HasObserver(vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand)))  {
    this->TypeIntensityCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand );
  }

  if (this->TypeSegmentationCheckButton && (!this->TypeSegmentationCheckButton->HasObserver(vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand)))  {
    this->TypeSegmentationCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand );
  }

  if (this->TypeJacobianCheckButton && (!this->TypeJacobianCheckButton->HasObserver(vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand)))  {
    this->TypeJacobianCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand );
  }


}

void vtkTumorGrowthTypeStep::RemoveGUIObservers() 
{
  if (this->TypeIntensityCheckButton) 
    {
      this->TypeIntensityCheckButton->RemoveObservers( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand);  
    }


  if (this->TypeSegmentationCheckButton) 
    {
      this->TypeSegmentationCheckButton->RemoveObservers( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand);  
    }

  if (this->TypeJacobianCheckButton) 
    {
      this->TypeJacobianCheckButton->RemoveObservers( vtkKWCheckButton::SelectedStateChangedEvent, this->WizardGUICallbackCommand);  
    }
}

void vtkTumorGrowthTypeStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
    vtkTumorGrowthTypeStep *self = reinterpret_cast<vtkTumorGrowthTypeStep *>(clientData);
    if (self) { self->ProcessGUIEvents(caller, event, callData); }


}

void vtkTumorGrowthTypeStep::ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) {
  if (event == vtkKWCheckButton::SelectedStateChangedEvent ) {
    if (this->TypeIntensityCheckButton == vtkKWCheckButton::SafeDownCast(caller)) {  
      if (this->TypeIntensityCheckButton->GetSelectedState()) {
    // turn off the other ones 
    this->TypeSegmentationCheckButton->SelectedStateOff();
    this->TypeJacobianCheckButton->SelectedStateOff();
      } else {
    // Turn it on again bc you cannot check it off 
        if (!this->TypeSegmentationCheckButton->GetSelectedState() && !this->TypeJacobianCheckButton->GetSelectedState()) {
      this->TypeIntensityCheckButton->SelectedStateOn();
    }
      }
      vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
      wizard_widget->GetCancelButton()->EnabledOn();

      return; 
    }
    if (this->TypeSegmentationCheckButton == vtkKWCheckButton::SafeDownCast(caller)) {  
      if (this->TypeSegmentationCheckButton->GetSelectedState()) {
    // turn off the other ones 
    this->TypeIntensityCheckButton->SelectedStateOff();
    this->TypeJacobianCheckButton->SelectedStateOff();
      } else {
    // Turn it on again bc you cannot check it off 
        if (!this->TypeIntensityCheckButton->GetSelectedState() && !this->TypeJacobianCheckButton->GetSelectedState()) {
      this->TypeSegmentationCheckButton->SelectedStateOn();
    }
      }
      vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
      wizard_widget->GetCancelButton()->EnabledOn();
      return;
    }

    if (this->TypeJacobianCheckButton == vtkKWCheckButton::SafeDownCast(caller)) {  
      if (this->TypeJacobianCheckButton->GetSelectedState()) {
    // turn off the other ones 
    this->TypeSegmentationCheckButton->SelectedStateOff();
    this->TypeIntensityCheckButton->SelectedStateOff();
      } else {
    // Turn it on again bc you cannot check it off 
        if (!this->TypeSegmentationCheckButton->GetSelectedState() && !this->TypeIntensityCheckButton->GetSelectedState()) {
      this->TypeJacobianCheckButton->SelectedStateOn();
    }
      }
      vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
      wizard_widget->GetCancelButton()->EnabledOn();
    }
  }
}

void vtkTumorGrowthTypeStep::RemoveAnalysisOutput() {
  this->GetGUI()->GetLogic()->DeleteAnalyzeOutput(vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()));      
}



//----------------------------------------------------------------------------
void vtkTumorGrowthTypeStep::ShowUserInterface()
{
  // cout << "vtkTumorGrowthTypeStep::ShowUserInterface Start" << endl;
  // ----------------------------------------
  // Display Super Sampled Volume 
  // ---------------------------------------- 
  this->RemoveAnalysisOutput();
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
  if (node) { 
    vtkMRMLVolumeNode *volumeSampleNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_SuperSampleRef()));
    vtkMRMLVolumeNode *volumeSegmentNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_SegmentRef()));
    if (volumeSampleNode && volumeSegmentNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeSampleNode->GetID());

      vtkSlicerApplicationGUI *applicationGUI     = this->GetGUI()->GetApplicationGUI();
      double oldSliceSetting[3];
      oldSliceSetting[0] = double(applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetOffsetScale()->GetValue());
      oldSliceSetting[1] = double(applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetOffsetScale()->GetValue());
      oldSliceSetting[2] = double(applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetOffsetScale()->GetValue());

      applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeSegmentNode);
      applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeSegmentNode);
      applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetForegroundSelector()->SetSelected(volumeSegmentNode);
      applicationGUI->GetSlicesControlGUI()->GetSliceFadeScale()->SetValue(0.6);
      applicationLogic->PropagateVolumeSelection();

      applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[0]);
      applicationGUI->GetMainSliceGUI1()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[1]);
      applicationGUI->GetMainSliceGUI2()->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[2]);
    } 
  }

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------

  this->vtkTumorGrowthStep::ShowUserInterface();

  // Create the frame
  // Needs to be check bc otherwise with wizrd can be created over again

  this->Frame->SetLabelText("Select Growth Metric");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());


  if (!this->FrameTypeIntensity)
    {
    this->FrameTypeIntensity = vtkKWFrame::New();
    }
  if (!this->FrameTypeIntensity->IsCreated())
    {
      this->FrameTypeIntensity->SetParent(this->Frame->GetFrame());
      this->FrameTypeIntensity->Create();
  }

  if (!this->FrameTypeSegmentation)
    {
    this->FrameTypeSegmentation = vtkKWFrame::New();
    }
  if (!this->FrameTypeSegmentation->IsCreated())
    {
      this->FrameTypeSegmentation->SetParent(this->Frame->GetFrame());
      this->FrameTypeSegmentation->Create();
  }

  if (!this->FrameTypeJacobian)
    {
    this->FrameTypeJacobian = vtkKWFrame::New();
    }
  if (!this->FrameTypeJacobian->IsCreated())
    {
      this->FrameTypeJacobian->SetParent(this->Frame->GetFrame());
      this->FrameTypeJacobian->Create();
  }

  this->Script("pack %s %s %s -side top -anchor nw -fill x -padx 0 -pady 0", 
                this->FrameTypeIntensity->GetWidgetName(),
            this->FrameTypeSegmentation->GetWidgetName(),
                this->FrameTypeJacobian->GetWidgetName());


  if (!this->TypeIntensityCheckButton) {
    this->TypeIntensityCheckButton = vtkKWCheckButton::New();
  }

  if (!this->TypeIntensityCheckButton->IsCreated()) {
    this->TypeIntensityCheckButton->SetParent(this->FrameTypeIntensity);
    this->TypeIntensityCheckButton->Create();
    this->TypeIntensityCheckButton->SelectedStateOn();
    this->TypeIntensityCheckButton->SetText("Detect Growth Patterns (conservative)");
  }

  if (!this->TypeSegmentationCheckButton) {
    this->TypeSegmentationCheckButton = vtkKWCheckButton::New();
  }


  if (!this->TypeSegmentationCheckButton->IsCreated()) {
    this->TypeSegmentationCheckButton->SetParent(this->FrameTypeSegmentation);
    this->TypeSegmentationCheckButton->Create();
    this->TypeSegmentationCheckButton->SelectedStateOff();
    this->TypeSegmentationCheckButton->SetText("Map Segmentation to Scan 2 (moderate)");
    this->TypeSegmentationCheckButton->EnabledOff();
  }

  if (!this->TypeJacobianCheckButton) {
    this->TypeJacobianCheckButton = vtkKWCheckButton::New();
  }

  if (!this->TypeJacobianCheckButton->IsCreated()) {
    this->TypeJacobianCheckButton->SetParent(this->FrameTypeJacobian);
    this->TypeJacobianCheckButton->Create();
    this->TypeJacobianCheckButton->SelectedStateOff();
    this->TypeJacobianCheckButton->SetText("Analyze Deformation Map (progressive) ");
    // Currently not yet implemented
    this->TypeJacobianCheckButton->EnabledOff();
  }
  this->Script("pack %s %s %s -side left -anchor nw -fill x -padx 2 -pady 2", 
                this->TypeIntensityCheckButton->GetWidgetName(),
            this->TypeSegmentationCheckButton->GetWidgetName(),
                this->TypeJacobianCheckButton->GetWidgetName());


  {
    vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
    wizard_widget->GetCancelButton()->SetText("Analyze");
    // wizard_widget->GetCancelButton()->EnabledOff();
  }

  this->AddGUIObservers(); 

}

//----------------------------------------------------------------------------
void vtkTumorGrowthTypeStep::TransitionCallback( ) 
{
  if (!this->TypeIntensityCheckButton ||!this->TypeSegmentationCheckButton || !this->TypeJacobianCheckButton) return;

  vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget();
  vtkKWWizardWorkflow *wizard_workflow = wizard_widget->GetWizardWorkflow();

  wizard_widget->GetCancelButton()->EnabledOn();
  // ============================
  // make sure that analyze related nodes are empty 
  // Delete old attached node first 
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return;
  {
    vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetAnalysis_Ref()));
    if (currentNode) { this->GetGUI()->GetMRMLScene()->RemoveNode(currentNode); }
  }
  
  vtkTumorGrowthLogic* Logic = this->GetGUI()->GetLogic();
  
  //  Process images 
  if (this->TypeIntensityCheckButton->GetSelectedState()) {
    if (!Logic->AnalyzeGrowth(vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()))) return;
  }
  
  if (this->TypeSegmentationCheckButton->GetSelectedState()) {
    cout << "Currently not implemented" << endl;
    return;
  }
  
  if (this->TypeJacobianCheckButton->GetSelectedState()) {
    cout << "Currently not implemented" << endl;
    return;
  }
  wizard_workflow->AttemptToGoToNextStep();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthTypeStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
