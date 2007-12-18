#include "vtkTumorGrowthROIStep.h"

#include "vtkTumorGrowthGUI.h"
#include "vtkTumorGrowthLogic.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWLabel.h"
#include "vtkKWMatrixWidgetWithLabel.h"
#include "vtkKWMatrixWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkKWMessageDialog.h"
#include "vtkImageRectangularSource.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerVolumesGUI.h" 
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkKWScale.h"
//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkTumorGrowthROIStep);
vtkCxxRevisionMacro(vtkTumorGrowthROIStep, "$Revision: 1.2 $");

//----------------------------------------------------------------------------
vtkTumorGrowthROIStep::vtkTumorGrowthROIStep()
{
  this->SetName("2/4. Define Region of Interest"); 
  this->SetDescription("Define ROI by clicking with <ctrl>-left mouse button around the tumor"); 
  this->WizardGUICallbackCommand->SetCallback(vtkTumorGrowthROIStep::WizardGUICallback);

  this->FrameButtons    = NULL;
  this->FrameBlank      = NULL;
  this->FrameROI        = NULL;
  this->ButtonsShow     = NULL;
  this->ButtonsReset    = NULL;
  this->ROIMinVector    = NULL;
  this->ROIMaxVector    = NULL;
  this->ROILabelMapNode = NULL;
}

//----------------------------------------------------------------------------
vtkTumorGrowthROIStep::~vtkTumorGrowthROIStep()
{
  if (this->FrameButtons)
  {
    this->FrameButtons->Delete();
    this->FrameButtons = NULL;
  }

  if (this->FrameBlank)
  {
    this->FrameBlank->Delete();
    this->FrameBlank = NULL;
  }

  if (this->FrameROI)
  {
    this->FrameROI->Delete();
    this->FrameROI = NULL;
  }

  if (this->ROIMinVector)
  {
    this->ROIMinVector->Delete();
    this->ROIMinVector = NULL;
  }

  if (this->ROIMaxVector )
  {
    this->ROIMaxVector->Delete();
    this->ROIMaxVector  = NULL;
  }

  if (this->ROILabelMapNode) this->ROIMapRemove();
}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ShowUserInterface()
{

  // ----------------------------------------
  // Display Super Sampled Volume 
  // ----------------------------------------
  vtkMRMLTumorGrowthNode* node = this->GetGUI()->GetNode();
  if (node) {
    vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_Ref()));
    if (volumeNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeNode->GetID());
      applicationLogic->PropagateVolumeSelection();
    } 
  } else {
    cout << "no node "  << endl;
  }

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------

  // cout << "vtkTumorGrowthROIStep::ShowUserInterface()" << endl;
  this->vtkTumorGrowthStep::ShowUserInterface();

  // Create the frame
  // Needs to be check bc otherwise with wizrd can be created over again

  this->Frame->SetLabelText("Define ROI");
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 2", this->Frame->GetWidgetName());

  if (!this->FrameButtons)
    {
    this->FrameButtons = vtkKWFrame::New();
    }
  if (!this->FrameButtons->IsCreated())
    {
      this->FrameButtons->SetParent(this->Frame->GetFrame());
    this->FrameButtons->Create();
    // this->FrameButtons->SetLabelText("");
    // define buttons 
  }
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 0", this->FrameButtons->GetWidgetName());

  if (!this->FrameBlank)
    {
    this->FrameBlank = vtkKWFrame::New();
    }
  if (!this->FrameBlank->IsCreated())
    {
      this->FrameBlank->SetParent(this->Frame->GetFrame());
    this->FrameBlank->Create();
    // this->FrameButtons->SetLabelText("");
    // define buttons 
  }
  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 4", this->FrameBlank->GetWidgetName());

  if (!this->FrameROI)
    {
    this->FrameROI = vtkSlicerModuleCollapsibleFrame::New();
    }
  if (!this->FrameROI->IsCreated())
    {
      this->FrameROI->SetParent(this->Frame->GetFrame());
    this->FrameROI->Create();
    this->FrameROI->SetLabelText("Manual");
    this->FrameROI->CollapseFrame();
  }

  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 0", this->FrameROI->GetWidgetName());

  if (!this->ButtonsShow) {
    this->ButtonsShow = vtkKWPushButton::New();
  }

  if (!this->ButtonsShow->IsCreated()) {
    this->ButtonsShow->SetParent(this->FrameButtons);
    this->ButtonsShow->Create();
    this->ButtonsShow->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsShow->SetText("Show ROI");
  }

  if (!this->ButtonsReset) {
    this->ButtonsReset = vtkKWPushButton::New();
  }
  if (!this->ButtonsReset->IsCreated()) {
    this->ButtonsReset->SetParent(this->FrameButtons);
    this->ButtonsReset->Create();
    this->ButtonsReset->SetWidth(TUMORGROWTH_MENU_BUTTON_WIDTH);
    this->ButtonsReset->SetText("Reset");
  }

  this->Script("pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2", 
                this->ButtonsShow->GetWidgetName(),this->ButtonsReset->GetWidgetName());

  if (!this->ROIMaxVector)
    {
    this->ROIMaxVector = vtkKWMatrixWidgetWithLabel::New();
    }
  if (!this->ROIMaxVector->IsCreated())
    {
      this->ROIMaxVector->SetParent(this->FrameROI->GetFrame());
    this->ROIMaxVector->Create();
    this->ROIMaxVector->SetLabelText("Max :");
    this->ROIMaxVector->SetLabelPositionToLeft();
    this->ROIMaxVector->ExpandWidgetOff();
    this->ROIMaxVector->GetLabel()->SetWidth(TUMORGROWTH_WIDGETS_LABEL_WIDTH - 16);
    this->ROIMaxVector->SetBalloonHelpString("Set the upper right hand corner of region of interest.");
    
    vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
      matrix->SetNumberOfColumns(3);
      matrix->SetNumberOfRows(1);
      matrix->SetElementWidth(4);
      matrix->SetRestrictElementValueToInteger();
      matrix->SetElementChangedCommand(this, "ROIMaxChangedCallback");
      matrix->SetElementChangedCommandTriggerToAnyChange();
    }
    // Set it up so it has default value from MRML file 

  if (!this->ROIMinVector)
    {
    this->ROIMinVector = vtkKWMatrixWidgetWithLabel::New();
    }
  if (!this->ROIMinVector->IsCreated())
    {
    this->ROIMinVector->SetParent(this->FrameROI->GetFrame());
    this->ROIMinVector->Create();
    this->ROIMinVector->SetLabelText("Min :");
    this->ROIMinVector->SetLabelPositionToLeft();
    this->ROIMinVector->ExpandWidgetOff();
    this->ROIMinVector->GetLabel()->SetWidth(TUMORGROWTH_WIDGETS_LABEL_WIDTH - 16);
    this->ROIMinVector->SetBalloonHelpString("Set the upper right hand corner of region of interest.");
    
    vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
      matrix->SetNumberOfColumns(3);
      matrix->SetNumberOfRows(1);
      matrix->SetElementWidth(4);
      matrix->SetRestrictElementValueToInteger();
      matrix->SetElementChangedCommand(this, "ROIMinChangedCallback");
      matrix->SetElementChangedCommandTriggerToAnyChange();
      // You do not need that bc youo only have one node 
      // matrix->SetReadOnly(mrmlManager->GetTreeNodeDistributionSpecificationMethod(sel_vol_id) != vtkEMSegmentMRMLManager::DistributionSpecificationManual);
      // char buffer[256];
      // sprintf(buffer, "IntensityDistributionMeanChangedCallback %d", sel_vol_id);
    }
  this->Script("pack %s %s -side top -anchor nw -padx 2 -pady 2",this->ROIMinVector->GetWidgetName(),this->ROIMaxVector->GetWidgetName());
 
  // Set it up so it has default value from MRML file 
  this->ROIUpdateWithNode();
  {
   vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget(); 
   wizard_widget->BackButtonVisibilityOn();
   wizard_widget->GetCancelButton()->EnabledOn();
  }
  // Very Important 
  this->AddGUIObservers();
  // Keep seperate bc GUIObserver is also called from vtkTumorGrowthGUI ! 
  // You only want to add the observers below when the step is active 
  this->AddROISamplingGUIObservers();

  // this->TransitionCallback();
}


//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ROIMaxChangedCallback(int row, int col, const char *value) 
{
  // cout << "vtkTumorGrowthROIStep::ROIMaxChangedCallback" << endl;
  // Threshold has changed because of user interaction 
  // vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (!this->ROIMaxVector || !mrmlNode) return;

  vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
  mrmlNode->SetROIMax(col,atoi(matrix->GetElementValue(0,col)));  
  // cout << "vtkTumorGrowthROIStep::ROIMaxChangedCallback End" << endl;
}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::ROIMinChangedCallback(int row, int col, const char *value)  
{
  // cout << "vtkTumorGrowthROIStep::ROIMinChangedCallback" << endl;

  // Threshold has changed because of user interaction  - this is so it is written right away to the node 
  vtkMRMLTumorGrowthNode *mrmlNode = this->GetGUI()->GetNode();
  if (!this->ROIMinVector || !mrmlNode) return;

  vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
  mrmlNode->SetROIMin(col,atoi(matrix->GetElementValue(0,col)));  

  //char buffer[256];   
  // sprintf(buffer,  "%s %s %s",matrix->GetElementValue(0,0),matrix->GetElementValue(0,1),matrix->GetElementValue(0,2));
  // mrmlNode->SetROIMin(buffer);  
}

//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::AddGUIObservers() 
{
  // cout << "vtkTumorGrowthROIStep::AddGUIObservers()" << endl; 
  // Make sure you do not add the same event twice - need to do it bc of wizrd structure
  if (this->ButtonsShow && (!this->ButtonsShow->HasObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand))) 
    {
      this->ButtonsShow->AddObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand );  
    } 

  if (this->ButtonsReset && (!this->ButtonsReset->HasObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand))) 
    {
      this->ButtonsReset->AddObserver(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand );  
    } 
}

void vtkTumorGrowthROIStep::AddROISamplingGUIObservers() {
  vtkRenderWindowInteractor *rwi0 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI0()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi0->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);

  // Slice GUI 1

  vtkRenderWindowInteractor *rwi1 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI1()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi1->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent,this->WizardGUICallbackCommand);

  // Slice GUI 2

  vtkRenderWindowInteractor *rwi2 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI2()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi2->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);
} 


void vtkTumorGrowthROIStep::RemoveGUIObservers() 
{
  if (this->ButtonsShow) 
    {
      this->ButtonsShow->RemoveObservers(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand);  
    }

  if (this->ButtonsReset) 
  {
      this->ButtonsReset->RemoveObservers(vtkKWPushButton::InvokedEvent, this->WizardGUICallbackCommand);  
  }
  this->RemoveROISamplingGUIObservers();
}


void vtkTumorGrowthROIStep::RemoveROISamplingGUIObservers() {
  // Slice GUI 0
  vtkRenderWindowInteractor *rwi0 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI0()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi0->GetInteractorStyle()->RemoveObservers(
    vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);

  // Slice GUI 1

  vtkRenderWindowInteractor *rwi1 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI1()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi1->GetInteractorStyle()->RemoveObservers(
    vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);

  // Slice GUI 2

  vtkRenderWindowInteractor *rwi2 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI2()->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi2->GetInteractorStyle()->RemoveObservers(
    vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);

}

void vtkTumorGrowthROIStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
    vtkTumorGrowthROIStep *self = reinterpret_cast<vtkTumorGrowthROIStep *>(clientData);
    if (self) { self->ProcessGUIEvents(caller, event, callData); }


}

void vtkTumorGrowthROIStep::ROIReset() {
  if (this->ROIMinVector) 
  {
    vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
    matrix->SetElementValue(0,0,"-1");
    matrix->SetElementValue(0,1,"-1");
    matrix->SetElementValue(0,2,"-1");
  }
  if (this->ROIMaxVector) {
    vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
    matrix->SetElementValue(0,0,"-1");
    matrix->SetElementValue(0,1,"-1");
    matrix->SetElementValue(0,2,"-1");
  }
}


void vtkTumorGrowthROIStep::ROIUpdateWithNewSample(int ijkSample[3]) {
  if (this->ROIMinVector) 
  {
    vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
    for (int i = 0; i < 3 ; i++) {
      int val = atoi(matrix->GetElementValue(0,i));
      if ((val < 0) || val > ijkSample[i] ) {
        char buffer [256];
        sprintf(buffer,"%d",ijkSample[i]);
        matrix->SetElementValue(0,i,buffer);
      }
    }
  }
  if (this->ROIMaxVector) {
    vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
    for (int i = 0; i < 3 ; i++) {
      int val = atoi(matrix->GetElementValue(0,i));
      if ((val < 0) || val < ijkSample[i] ) {
        char buffer [256];
        sprintf(buffer,"%d",ijkSample[i]);
        matrix->SetElementValue(0,i,buffer);
      }
    }
  }
}


void vtkTumorGrowthROIStep::ROIUpdateWithNode() {
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return;
  
 
  if (this->ROIMinVector) 
  {
    vtkKWMatrixWidget *matrix = this->ROIMinVector->GetWidget();
    for (int i = 0; i < 3 ; i++) {
      char buffer [256];
      sprintf(buffer,"%d",Node->GetROIMin(i));
      matrix->SetElementValue(0,i,buffer);
    }
    // For Debugging
    //matrix->SetElementValue(0,0,"72");
    //matrix->SetElementValue(0,1,"128");
    //matrix->SetElementValue(0,2,"95");
  }
  if (this->ROIMaxVector) {
    vtkKWMatrixWidget *matrix = this->ROIMaxVector->GetWidget();
    for (int i = 0; i < 3 ; i++) {
      char buffer [256];
      sprintf(buffer,"%d",Node->GetROIMax(i));
      matrix->SetElementValue(0,i,buffer);
    }
    // For Debugging
    //matrix->SetElementValue(0,0,"92");
    //matrix->SetElementValue(0,1,"161");
    //matrix->SetElementValue(0,2,"104");
  }
}



// Return 1 if it is a valid ROI and zero otherwise
int vtkTumorGrowthROIStep::ROICheck() {
  // Define Variables
  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) return 0;

  vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
  if (!volumeNode) return 0;
  return this->GetGUI()->GetLogic()->CheckROI(volumeNode);
}


int vtkTumorGrowthROIStep::ROIMapShow() {
  // Initialize
  if (!this->ROICheck()) {
    vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Please define ROI correctly before pressing button", vtkKWMessageDialog::ErrorIcon);
    return 0;
  }

  vtkMRMLTumorGrowthNode* Node      =  this->GetGUI()->GetNode();
  if (!Node) return 0;

  vtkMRMLScene* mrmlScene           =  Node->GetScene();
  vtkMRMLNode* mrmlFristScanRefNode =  mrmlScene->GetNodeByID(Node->GetScan1_Ref());
  vtkMRMLVolumeNode* volumeNode     =  vtkMRMLVolumeNode::SafeDownCast(mrmlFristScanRefNode);
  if (!volumeNode) return 0;

  int size[3]   = {Node->GetROIMax(0) - Node->GetROIMin(0) + 1, Node->GetROIMax(1) - Node->GetROIMin(1) + 1, Node->GetROIMax(2) - Node->GetROIMin(2) + 1};
  int center[3] = {(Node->GetROIMax(0) + Node->GetROIMin(0))/2 ,(Node->GetROIMax(1) + Node->GetROIMin(1))/2, (Node->GetROIMax(2) + Node->GetROIMin(2))/2};
  int* dimensions = volumeNode->GetImageData()->GetDimensions();

  // Define LabelMap 
  vtkImageRectangularSource* ROILabelMap =  vtkImageRectangularSource::New();
  ROILabelMap->SetCenter(center);
  ROILabelMap->SetSize(size);
  ROILabelMap->SetWholeExtent(0,dimensions[0] -1,0,dimensions[1] -1, 0,dimensions[2] -1); 
  ROILabelMap->SetOutputScalarTypeToShort();
  ROILabelMap->SetInsideGraySlopeFlag(0); 
  ROILabelMap->SetInValue(17);
  ROILabelMap->SetOutValue(0);
  ROILabelMap->Update();

  // Show map in Slicer 3 
  //  set scene [[$this GetLogic] GetMRMLScene]
  //  set volumesLogic [$::slicer3::VolumesGUI GetLogic]
  vtkSlicerApplication *application   = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkSlicerApplicationGUI *applicationGUI = this->GetGUI()->GetApplicationGUI();
  vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();

  vtkSlicerVolumesGUI  *volumesGUI    = vtkSlicerVolumesGUI::SafeDownCast(application->GetModuleGUIByName("Volumes")); 
  vtkSlicerVolumesLogic *volumesLogic = volumesGUI->GetLogic();
  // set labelNode [$volumesLogic CreateLabelVolume $scene $volumeNode $name]
  if (this->ROILabelMapNode) this->ROIMapRemove(); 
  this->ROILabelMapNode = volumesLogic->CreateLabelVolume(mrmlScene,volumeNode, "TG_ROI");
  this->ROILabelMapNode->SetAndObserveImageData(ROILabelMap->GetOutput());

  // Now show in foreground 
  //  make the source node the active background, and the label node the active label
  // set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
  //$selectionNode SetReferenceActiveVolumeID [$volumeNode GetID]
  //$selectionNode SetReferenceActiveLabelVolumeID [$labelNode GetID]

  applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetBackgroundSelector()->SetSelected(volumeNode);
  applicationGUI->GetMainSliceGUI0()->GetSliceController()->GetForegroundSelector()->SetSelected(this->ROILabelMapNode);
  applicationGUI->GetSlicesControlGUI()->GetSliceFadeScale()->SetValue(0.6);

  //[[$this GetLogic] GetApplicationLogic]  PropagateVolumeSelection
  applicationLogic->PropagateVolumeSelection();
  ROILabelMap->Delete();

  return 1;
}

void vtkTumorGrowthROIStep::ROIMapRemove() {
  this->GetGUI()->GetMRMLScene()->RemoveNode(this->ROILabelMapNode);
  this->ROILabelMapNode = NULL;
}




void vtkTumorGrowthROIStep::RetrieveInteractorIJKCoordinates(vtkSlicerSliceGUI *sliceGUI, vtkRenderWindowInteractor *rwi,int coords[3]) {

  vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
  if (!Node) {
    cout << "ERROR: vtkTumorGrowthROIStep::RetrieveInteractorIJKCoordinates: No Node" << endl;
    return;
  } 

  vtkMRMLNode* mrmlNode =   Node->GetScene()->GetNodeByID(Node->GetScan1_Ref());
  vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(mrmlNode);

  if (!volumeNode)
    {
      cout << "ERROR: vtkTumorGrowthROIStep::RetrieveInteractorIJKCoordinates: No Scan1_Ref" << endl;
      return;
    }

  // --------------------------------------------------------------
  // Compute RAS coordinates
   int point[2];
   rwi->GetLastEventPosition(point);
   double inPt[4] = {point[0], point[1], 0, 1};
   double rasPt[4];
   vtkMatrix4x4 *matrix = sliceGUI->GetLogic()->GetSliceNode()->GetXYToRAS();
   matrix->MultiplyPoint(inPt, rasPt); 

  // --------------------------------------------------------------
  // Compute IJK coordinates
  double ijkPt[4];
  vtkMatrix4x4* rasToijk = vtkMatrix4x4::New();
  volumeNode->GetRASToIJKMatrix(rasToijk);
  rasToijk->MultiplyPoint(rasPt, ijkPt);
  rasToijk->Delete();

  // --------------------------------------------------------------
  // Check validity of coordinates
  int* dimensions = volumeNode->GetImageData()->GetDimensions();
  for (int i = 0 ; i < 3 ; i++) {
    if (ijkPt[i] < 0 ) ijkPt[i] = 0;
    else if (ijkPt[i] >=  dimensions[i] ) ijkPt[i] = dimensions[i] -1;    
  }
  coords[0] = int(round(ijkPt[0]));  coords[1] = int(round(ijkPt[1])); coords[2] = int(round(ijkPt[2])); 

  // cout << "Sample:  " << rasPt[0] << " " <<  rasPt[1] << " " << rasPt[2] << " " << rasPt[3] << endl;
  // cout << "Coord: " << coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << endl;

}
void vtkTumorGrowthROIStep::ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) {

  if (event == vtkKWPushButton::InvokedEvent) {
    vtkKWPushButton *button = vtkKWPushButton::SafeDownCast(caller);
    if (this->ButtonsShow && (button == this->ButtonsShow)) 
    { 
      if (this->ROILabelMapNode) {
        this->ButtonsShow->SetText("Show ROI");
        this->ROIMapRemove();
      } else { 
        if (this->ROIMapShow()) { 
          this->ButtonsShow->SetText("Hide ROI");
        }
      }
    }
    if (this->ButtonsReset && (button == this->ButtonsReset)) 
    { 
      this->ROIReset();
    }
    return;
  }

  vtkSlicerInteractorStyle *s = vtkSlicerInteractorStyle::SafeDownCast(caller);
  if (s && event == vtkCommand::LeftButtonPressEvent)
  {
    // Retrieve Coordinates and update ROI
    int index = 0; 
    vtkSlicerSliceGUI *sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(
      this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI0();

    vtkRenderWindowInteractor *rwi = sliceGUI->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();
    while (index < 2 && (s != rwi->GetInteractorStyle())) {
        index ++;
        if (index == 1) {
          sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI1();
        } else {
          sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI2();
        }
        rwi = sliceGUI->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();
    }
    int ijkCoords[3];
    this->RetrieveInteractorIJKCoordinates(sliceGUI, rwi, ijkCoords);
    this->ROIUpdateWithNewSample(ijkCoords);

  }    
  // Define SHOW Button 
}



//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::TransitionCallback() 
{
  // cout << "vtkTumorGrowthROIStep::TransitionCallback() Start" << endl; 
  if (this->ROICheck()) { 
     // ----------------------------
     // Create SuperSampledVolume 
    vtkSlicerApplication *application   = vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication());
    vtkMRMLScalarVolumeNode *outputNode = this->GetGUI()->GetLogic()->CreateSuperSample(1,application);
     if (outputNode) {
       // Prepare to update mrml node with results 
       vtkMRMLTumorGrowthNode* Node = this->GetGUI()->GetNode();
       if (!Node) return;
              
       // Delete old attached node first 
       vtkMRMLVolumeNode* currentNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_SuperSampleRef()));
       if (currentNode) { this->GetGUI()->GetMRMLScene()->RemoveNode(currentNode); }

       // Update node 
       Node->SetScan1_SuperSampleRef(outputNode->GetID());

       // Proceed to next step 
       this->GUI->GetWizardWidget()->GetWizardWorkflow()->AttemptToGoToNextStep();

     } else {
       vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Could not proceed to next step - scan1 might have disappeared", vtkKWMessageDialog::ErrorIcon); 
     }
     // ---------------------------------
   } else {     
     vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),"Tumor Growth", "Please define ROI correctly before proceeding", vtkKWMessageDialog::ErrorIcon);
   }
}

//----------------------------------------------------------------------------
void  vtkTumorGrowthROIStep::HideUserInterface()
{
  // cout << "vtkTumorGrowthROIStep::HideUserInterface() Start" << endl;
  this->Superclass::HideUserInterface();
  this->RemoveROISamplingGUIObservers();
  // cout << "vtkTumorGrowthROIStep::HideUserInterface() End" << endl;
}


//----------------------------------------------------------------------------
void vtkTumorGrowthROIStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
