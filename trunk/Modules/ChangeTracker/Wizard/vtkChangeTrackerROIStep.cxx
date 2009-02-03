#include "vtkChangeTrackerROIStep.h"

#include "vtkChangeTrackerGUI.h"
#include "vtkChangeTrackerLogic.h"
#include "vtkMRMLChangeTrackerNode.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWLabel.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerSlicesControlGUI.h"
#include "vtkKWMessageDialog.h"
#include "vtkImageRectangularSource.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerVolumesGUI.h" 
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkKWScale.h"

//#include "vtkKWMatrixWidget.h"
#include "vtkMRMLROINode.h"
#include <sstream>
#include "vtkObserverManager.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include "vtkVolumeTextureMapper3D.h"
#include "vtkPiecewiseFunction.h"
#include "vtkSlicerROIDisplayWidget.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkChangeTrackerROIStep);
vtkCxxRevisionMacro(vtkChangeTrackerROIStep, "$Revision: 1.2 $");

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

//----------------------------------------------------------------------------
vtkChangeTrackerROIStep::vtkChangeTrackerROIStep()
{
  this->SetName("2/4. Define Volume of Interest"); 
  this->SetDescription("Define VOI by clicking left mouse button around the tumor or moving sliders"); 
  this->WizardGUICallbackCommand->SetCallback(vtkChangeTrackerROIStep::WizardGUICallback);

  this->FrameButtons    = NULL;
  this->FrameBlank      = NULL;
  this->FrameROI        = NULL;
  this->FrameROIX       = NULL;
  this->FrameROIY       = NULL;
  this->FrameROIZ       = NULL;
  this->ButtonsShow     = NULL;
  this->ButtonsReset    = NULL;
  this->ROIX            = NULL;
  this->ROIY            = NULL;
  this->ROIZ            = NULL;
  this->LabelROIX       = NULL;
  this->LabelROIY       = NULL;
  this->LabelROIZ       = NULL;
  this->ROILabelMapNode = NULL;
  this->ROILabelMap     = NULL;
  this->ROIHideFlag     = 0; 

  this->roiNode = NULL;
  this->roiWidget = NULL;
  this->roiUpdateGuard = false;
}

//----------------------------------------------------------------------------
vtkChangeTrackerROIStep::~vtkChangeTrackerROIStep()
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

  if (this->FrameROIX)
  {
    this->FrameROIX->Delete();
    this->FrameROIX = NULL;
  }

  if (this->FrameROIY)
  {
    this->FrameROIY->Delete();
    this->FrameROIY = NULL;
  }

  if (this->FrameROIZ)
  {
    this->FrameROIZ->Delete();
    this->FrameROIZ = NULL;
  }
  if (this->ROIX)
  {
    this->ROIX->Delete();
    this->ROIX = NULL;
  }

  if (this->ROIY)
  {
    this->ROIY->Delete();
    this->ROIY = NULL;
  }

  if (this->ROIZ)
  {
    this->ROIZ->Delete();
    this->ROIZ = NULL;
  }

  if (this->LabelROIX)
  {
    this->LabelROIX->Delete();
    this->LabelROIX = NULL;
  }

  if (this->LabelROIY)
  {
    this->LabelROIY->Delete();
    this->LabelROIY = NULL;
  }

  if (this->LabelROIZ)
  {
    this->LabelROIZ->Delete();
    this->LabelROIZ = NULL;
  }

  if (this->ButtonsShow) {
    this->ButtonsShow->Delete();
    this->ButtonsShow= NULL;
  }

  if (this->ButtonsReset) {
    this->ButtonsReset->Delete();
    this->ButtonsReset= NULL;
  }

  if (this->ROILabelMapNode || this->ROILabelMap) this->ROIMapRemove();

  if(this->roiNode)
  {
    this->roiNode->Delete();
    this->roiNode = NULL;
  }

  if(this->roiWidget)
    {
    this->roiWidget->Delete();
    this->roiWidget = NULL;
    }
}

void vtkChangeTrackerROIStep::DeleteSuperSampleNode() 
{
  this->GetGUI()->GetLogic()->DeleteSuperSample(1);
} 

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::ShowUserInterface()
{
  // cout << "vtkChangeTrackerROIStep::ShowUserInterface() Start " << endl;
  // ----------------------------------------
  // Display Scan1, Delete Super Sampled and Grid  
  // ----------------------------------------
  this->DeleteSuperSampleNode();

  // debugging compareview
  if(false){
    vtkMRMLChangeTrackerNode* node = this->GetGUI()->GetNode();
//    vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
//    applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeSampleNode->GetID());

    vtkSlicerApplicationGUI *applicationGUI     = this->GetGUI()->GetApplicationGUI();


    double oldSliceSetting[3];
    oldSliceSetting[0] = double(applicationGUI->GetMainSliceGUI("Red")->GetSliceController()->GetOffsetScale()->GetValue());
    oldSliceSetting[1] = double(applicationGUI->GetMainSliceGUI("Yellow")->GetSliceController()->GetOffsetScale()->GetValue());
    oldSliceSetting[2] = double(applicationGUI->GetMainSliceGUI("Green")->GetSliceController()->GetOffsetScale()->GetValue());
//SetBackgroundVolumeID( node->GetScan1_Ref());
    // set the layout to CompareView
    applicationGUI->GetGUILayoutNode()->SetNumberOfCompareViewRows(2);
    applicationGUI->GetGUILayoutNode()->SetNumberOfCompareViewColumns(1);
    applicationGUI->GetGUILayoutNode()->SetNumberOfCompareViewLightboxRows(1);
    applicationGUI->GetGUILayoutNode()->SetNumberOfCompareViewLightboxColumns(4);
    applicationGUI->GetGUILayoutNode()->SetViewArrangement(vtkMRMLLayoutNode::SlicerLayoutCompareView);

    cout << "Compare0 node: " << applicationGUI->GetMainSliceGUI("Compare0")->GetLogic()->GetSliceCompositeNode() << endl;
    cout << "Compare1 node: " << applicationGUI->GetMainSliceGUI("Compare1")->GetLogic()->GetSliceCompositeNode() << endl;

    applicationGUI->GetMainSliceGUI("Compare0")->GetLogic()->GetSliceCompositeNode()->SetBackgroundVolumeID( node->GetScan1_Ref());
    cerr << "Setting Compare0 background volume id to: " << node->GetScan1_Ref() << endl;
    cerr << "Current Compare0 background volume id: " << 
      applicationGUI->GetMainSliceGUI("Compare0")->GetLogic()->GetSliceCompositeNode()->GetBackgroundVolumeID() << endl;

    applicationGUI->GetMainSliceGUI("Compare1")->GetLogic()->GetSliceCompositeNode()->SetBackgroundVolumeID( node->GetScan2_Ref());
    cerr << "Setting Compare1 background volume id to: " << node->GetScan2_Ref() << endl;
    cerr << "Current Compare1 background volume id: " << 
      applicationGUI->GetMainSliceGUI("Compare1")->GetLogic()->GetSliceCompositeNode()->GetBackgroundVolumeID() << endl;

    applicationGUI->GetMainSliceGUI("Compare0")->GetLogic()->GetSliceCompositeNode()->SetForegroundVolumeID( "" );
    applicationGUI->GetMainSliceGUI("Compare1")->GetLogic()->GetSliceCompositeNode()->SetForegroundVolumeID( "" );
    this->Script("proc CTsetid {cnodeid vnodeid} { set cnode [$::slicer3::MRMLScene GetNodeByID $cnodeid]; $cnode SetReferenceBackgroundVolumeID $vnodeid}");
    this->Script("after idle CTsetid %s %s",  applicationGUI->GetMainSliceGUI("Compare0")->GetLogic()->GetSliceCompositeNode()->GetID(), node->GetScan1_Ref());
    this->Script("after idle CTsetid %s %s",  applicationGUI->GetMainSliceGUI("Compare1")->GetLogic()->GetSliceCompositeNode()->GetID(), node->GetScan2_Ref());
  }

  vtkMRMLChangeTrackerNode* node = this->GetGUI()->GetNode();
  int dimensions[3]={1,1,1};
  if (node) {
    vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_Ref()));
    if (volumeNode) {
      vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();
      applicationLogic->GetSelectionNode()->SetActiveVolumeID(volumeNode->GetID());
      applicationLogic->PropagateVolumeSelection(); 
      if (!volumeNode->GetImageData()) {
     vtkKWMessageDialog::PopupMessage(this->GetGUI()->GetApplication(), 
                                      this->GetGUI()->GetApplicationGUI()->GetMainSlicerWindow(),
                                      "Change Tracker", 
                                      "No image data associated with Scan 1", 
                                      vtkKWMessageDialog::ErrorIcon);
     return;
      }
      memcpy(dimensions,volumeNode->GetImageData()->GetDimensions(),sizeof(int)*3);
      // Load File 

      char fileName[1024];
      sprintf(fileName,"%s/TG_Analysis_Intensity.nhdr",node->GetWorkingDir());
       // vtkMRMLVolumeNode* tmp =  this->GetGUI()->GetLogic()->LoadVolume(vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication()),fileName,1,"TG_analysis");
    } 
  } else {
    cout << "no node "  << endl;
  }

  this->GridRemove();

  // ----------------------------------------
  // Build GUI 
  // ----------------------------------------

  this->vtkChangeTrackerStep::ShowUserInterface();
  // Create the frame
  // Needs to be check bc otherwise with wizrd can be created over again

  this->Frame->SetLabelText("Define VOI");
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
    this->FrameROI->SetLabelText("ROI Widget controls");
    // this->FrameROI->CollapseFrame();
  }

  this->Script("pack %s -side top -anchor nw -fill x -padx 0 -pady 0", this->FrameROI->GetWidgetName());

  if (!this->ButtonsShow) {
    this->ButtonsShow = vtkKWPushButton::New();
  }

  if (!this->ButtonsShow->IsCreated()) {
    this->ButtonsShow->SetParent(this->FrameButtons);
    this->ButtonsShow->Create();
    this->ButtonsShow->SetWidth(CHANGETRACKER_MENU_BUTTON_WIDTH);
    this->ButtonsShow->SetText("Show render");
    this->ButtonsShow->SetBalloonHelpString("Show/hide VOI in image viewer"); 
  }

  if (!this->ButtonsReset) {
    this->ButtonsReset = vtkKWPushButton::New();
  }
  if (!this->ButtonsReset->IsCreated()) {
    this->ButtonsReset->SetParent(this->FrameButtons);
    this->ButtonsReset->Create();
    this->ButtonsReset->SetWidth(CHANGETRACKER_MENU_BUTTON_WIDTH);
    this->ButtonsReset->SetText("Reset");
    this->ButtonsReset->SetBalloonHelpString("Reset Values"); 
  }

  this->Script("pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2", 
                this->ButtonsShow->GetWidgetName(),this->ButtonsReset->GetWidgetName());

  
  if (!this->FrameROIX)
    {
    this->FrameROIX = vtkKWFrame::New();
    }
  if (!this->FrameROIX->IsCreated())
    {
      this->FrameROIX->SetParent(this->FrameROI->GetFrame());
    this->FrameROIX->Create();
  }

  if (!this->LabelROIX)
    {
    this->LabelROIX = vtkKWLabel::New();
    }
  if (!this->LabelROIX->IsCreated())
    {

    this->LabelROIX->SetParent(this->FrameROIX);
    this->LabelROIX->Create();
    this->LabelROIX->SetText("X:");

    }
  if (!this->ROIX)
    {
    this->ROIX = vtkKWRange::New();
    }
  if (!this->ROIX->IsCreated())
    {

    this->ROIX->SetParent(this->FrameROIX);
    this->ROIX->Create();
    this->ROIX->SymmetricalInteractionOff();
    this->ROIX->SetCommand(this, "ROIXChangedCallback");    
    this->ROIX->SetWholeRange(-1, double(dimensions[0]-1));
    this->ROIX->SetResolution(1);
    }

  if (!this->FrameROIY)
    {
    this->FrameROIY = vtkKWFrame::New();
    }
  if (!this->FrameROIY->IsCreated())
    {
      this->FrameROIY->SetParent(this->FrameROI->GetFrame());
    this->FrameROIY->Create();
  }

  if (!this->LabelROIY)
    {
    this->LabelROIY = vtkKWLabel::New();
    }
  if (!this->LabelROIY->IsCreated())
    {

    this->LabelROIY->SetParent(this->FrameROIY);
    this->LabelROIY->Create();
    this->LabelROIY->SetText("Y:");

    }

 if (!this->ROIY)
    {
    this->ROIY = vtkKWRange::New();
    }
  if (!this->ROIY->IsCreated())
    {

    this->ROIY->SetParent(this->FrameROIY);

    this->ROIY->Create();
    this->ROIY->SymmetricalInteractionOff();
    this->ROIY->SetCommand(this, "ROIYChangedCallback");    
    this->ROIY->SetWholeRange(-1, double(dimensions[1]-1));
    this->ROIY->SetResolution(1);
    }

  if (!this->FrameROIZ)
    {
    this->FrameROIZ = vtkKWFrame::New();
    }
  if (!this->FrameROIZ->IsCreated())
    {
      this->FrameROIZ->SetParent(this->FrameROI->GetFrame());
    this->FrameROIZ->Create();
  }

  if (!this->LabelROIZ)
    {
    this->LabelROIZ = vtkKWLabel::New();
    }
  if (!this->LabelROIZ->IsCreated())
    {

    this->LabelROIZ->SetParent(this->FrameROIZ);
    this->LabelROIZ->Create();
    this->LabelROIZ->SetText("Z:");

    }

  if (!this->ROIZ)
    {
    this->ROIZ = vtkKWRange::New();
    }
  if (!this->ROIZ->IsCreated())
    {

    this->ROIZ->SetParent(this->FrameROIZ);

    this->ROIZ->Create();
    this->ROIZ->SymmetricalInteractionOff();
    this->ROIZ->SetCommand(this, "ROIZChangedCallback");    
    this->ROIZ->SetWholeRange(-1, double(dimensions[2]-1));
    this->ROIZ->SetResolution(1);
    }


//  this->Script("pack %s %s %s -side top -anchor nw -padx 0 -pady 3",this->FrameROIX->GetWidgetName(),this->FrameROIY->GetWidgetName(),this->FrameROIZ->GetWidgetName());
//  this->Script("pack %s %s -side left -anchor nw -padx 2 -pady 0",this->LabelROIX->GetWidgetName(),this->ROIX->GetWidgetName());
//  this->Script("pack %s %s -side left -anchor nw -padx 2 -pady 0",this->LabelROIY->GetWidgetName(),this->ROIY->GetWidgetName());
//  this->Script("pack %s %s -side left -anchor nw -padx 2 -pady 0",this->LabelROIZ->GetWidgetName(),this->ROIZ->GetWidgetName());
  

  // Set it up so it has default value from MRML file 
  this->ROIUpdateWithNode();
  {
   vtkKWWizardWidget *wizard_widget = this->GetGUI()->GetWizardWidget(); 
   wizard_widget->BackButtonVisibilityOn();
   wizard_widget->GetCancelButton()->EnabledOn();
  }
 
  if(!this->roiNode)
    {
    // Create ROI MRML node
    // see Base/GUI/vtkSlicerNodeSelectorWidget.cxx:ProcessNewNodeCommand
    vtkMRMLScene *scene = node->GetScene();
    vtkMRMLROINode *roi = 
      static_cast<vtkMRMLROINode*>(scene->CreateNodeByClass("vtkMRMLROINode"));
    scene->AddNode(roi);
    roi->SetName("ChangeTrackerROI");
    roi->SetVisibility(0);
    this->GetGUI()->ObserveMRMLROINode(roi);
    this->roiNode = roi;
    roi->Delete();
    }

  InitROIRender();
  ResetROIRender();
  this->MRMLUpdateROIFromROINode();
 
  if (!this->roiWidget)
    {
    this->roiWidget = vtkSlicerROIDisplayWidget::New();
    }

  if (!this->roiWidget->IsCreated())
    {
    vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
    vtkSlicerSliceLogic *sliceLogic = this->GetGUI()->GetSliceLogic();
    vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
    double rasDimensions[3];
    double rasCenter[3];
    double rasBounds[6];

    sliceLogic->GetVolumeRASBox(volumeNode, rasDimensions, rasCenter);

    rasBounds[0] = min(rasCenter[0]-rasDimensions[0]/2.,rasCenter[0]+rasDimensions[0]/2.);
    rasBounds[1] = min(rasCenter[1]-rasDimensions[1]/2.,rasCenter[1]+rasDimensions[1]/2.);
    rasBounds[2] = min(rasCenter[2]-rasDimensions[2]/2.,rasCenter[2]+rasDimensions[2]/2.);
    rasBounds[3] = max(rasCenter[0]-rasDimensions[0]/2.,rasCenter[0]+rasDimensions[0]/2.);
    rasBounds[4] = max(rasCenter[1]-rasDimensions[1]/2.,rasCenter[1]+rasDimensions[1]/2.);
    rasBounds[5] = max(rasCenter[2]-rasDimensions[2]/2.,rasCenter[2]+rasDimensions[2]/2.);
    
    this->roiWidget->SetParent(this->FrameROI->GetFrame());
    this->roiWidget->SetXRangeExtent(rasBounds[0],rasBounds[3]);
    this->roiWidget->SetYRangeExtent(rasBounds[1],rasBounds[4]);
    this->roiWidget->SetZRangeExtent(rasBounds[2],rasBounds[5]);

    this->roiWidget->Create();
    this->roiWidget->SetROINode(roiNode);
    }

  this->Script("pack %s -side top -anchor nw -padx 2 -pady 3 -fill x",
               this->roiWidget->GetWidgetName());

  // Very Important 
  this->AddGUIObservers();
  // Keep seperate bc GUIObserver is also called from vtkChangeTrackerGUI ! 
  // You only want to add the observers below when the step is active 
  this->AddROISamplingGUIObservers();
}

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::ROIXChangedCallback(double min, double max)  
{
  this->ROIChangedCallback(0,min, max);
}


//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::ROIYChangedCallback(double min, double max)  
{
  this->ROIChangedCallback(1,min, max);
}

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::ROIZChangedCallback(double min, double max)  
{
  this->ROIChangedCallback(2,min, max);
}  

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::ROIChangedCallback(int axis, double min, double max)  
{
  vtkMRMLChangeTrackerNode *mrmlNode = this->GetGUI()->GetNode();
  if (!mrmlNode) return;

  mrmlNode->SetROIMin(axis,int(min));  
  mrmlNode->SetROIMax(axis,int(max));  
  this->ROIMapUpdate();

}



//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::AddGUIObservers() 
{
  // cout << "vtkChangeTrackerROIStep::AddGUIObservers()" << endl; 
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

void vtkChangeTrackerROIStep::AddROISamplingGUIObservers() {
  vtkRenderWindowInteractor *rwi0 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Red")->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi0->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);

  // Slice GUI 1

  vtkRenderWindowInteractor *rwi1 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Yellow")->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi1->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent,this->WizardGUICallbackCommand);

  // Slice GUI 2

  vtkRenderWindowInteractor *rwi2 = vtkSlicerApplicationGUI::SafeDownCast(
    this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Green")->
    GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();

  rwi2->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);
} 


void vtkChangeTrackerROIStep::RemoveGUIObservers() 
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


void vtkChangeTrackerROIStep::RemoveROISamplingGUIObservers() {
  if (!this->GetGUI()) return;
  vtkSlicerApplicationGUI *ApplicationGUI = vtkSlicerApplicationGUI::SafeDownCast(this->GetGUI()->GetApplicationGUI());
  if (!ApplicationGUI) return; 
  for (int i = 0 ; i < 3 ; i ++ ) {
    vtkSlicerSliceGUI *MainGUI = NULL;
    if (i == 0) MainGUI = ApplicationGUI->GetMainSliceGUI("Red");
    if (i == 1) MainGUI = ApplicationGUI->GetMainSliceGUI("Yellow");
    if (i == 2) MainGUI = ApplicationGUI->GetMainSliceGUI("Green");
    if (!MainGUI) return;
    vtkRenderWindowInteractor *rwi = MainGUI->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();
    rwi->GetInteractorStyle()->RemoveObservers(vtkCommand::LeftButtonPressEvent, this->WizardGUICallbackCommand);
  }
}

void vtkChangeTrackerROIStep::WizardGUICallback(vtkObject *caller, unsigned long event, void *clientData, void *callData )
{
    vtkChangeTrackerROIStep *self = reinterpret_cast<vtkChangeTrackerROIStep *>(clientData);
    if (self) { self->ProcessGUIEvents(caller, event, callData); }


}

void vtkChangeTrackerROIStep::ROIReset() {
  // cout << "ROIReset Start" << endl;
  if (this->ROIX) this->ROIX->SetRange(-1,-1);
  if (this->ROIY) this->ROIY->SetRange(-1,-1);
  if (this->ROIZ) this->ROIZ->SetRange(-1,-1);
  if(this->roiNode){
    this->roiNode->SetXYZ(0., 0., 0.);
    this->roiNode->SetRadiusXYZ(10., 10., 10.);
    this->roiNode->SetVisibility(0);
  }
  this->ROIHideFlag = 0;
}


void vtkChangeTrackerROIStep::ROIUpdateAxisWithNewSample(vtkKWRange *ROIAxis, int Sample) {
  if (!ROIAxis) return;
  double *oldRange = ROIAxis->GetRange();
  double newRange[2];

  if ((Sample < oldRange[0]) || (oldRange[0] < 0)) newRange[0] = Sample;
  else  newRange[0] = oldRange[0]; 
  if ((Sample > oldRange[1]) || (oldRange[1] < 0)) newRange[1] = Sample;
  else newRange[1] = oldRange[1]; 
  ROIAxis->SetRange(newRange);
}

void vtkChangeTrackerROIStep::ROIUpdateWithNewSample(int ijkSample[3]) {
  // cout << "ROIUpdateWithNewSample start " << ijkSample[0] << " " << ijkSample[1] << " " << ijkSample[2] << " " << endl;
  this->ROIUpdateAxisWithNewSample(this->ROIX,ijkSample[0]);
  this->ROIUpdateAxisWithNewSample(this->ROIY,ijkSample[1]);
  this->ROIUpdateAxisWithNewSample(this->ROIZ,ijkSample[2]);
}

void vtkChangeTrackerROIStep::ROIUpdateAxisWithNode(vtkMRMLChangeTrackerNode* Node, vtkKWRange *ROIAxis, int Axis) {
  if (!Node || !ROIAxis) return;
  ROIAxis->SetRange(Node->GetROIMin(Axis),Node->GetROIMax(Axis));
}

void vtkChangeTrackerROIStep::ROIUpdateWithNode() {
  // cout << "ROIUpdateWithNode Start" << endl;
  vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
  this->ROIUpdateAxisWithNode(Node, this->ROIX,0); 
  this->ROIUpdateAxisWithNode(Node, this->ROIY,1); 
  this->ROIUpdateAxisWithNode(Node, this->ROIZ,2); 
  this->ROIMapUpdate();
}



// Return 1 if it is a valid ROI and zero otherwise
int vtkChangeTrackerROIStep::ROICheck() {
  // Define Variables
  vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
  if (!Node) return 0;

  vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
  if (!volumeNode) return 0;
  return this->GetGUI()->GetLogic()->CheckROI(volumeNode);
}

void vtkChangeTrackerROIStep::ROIMapUpdate() {

  vtkMRMLChangeTrackerNode* Node      =  this->GetGUI()->GetNode();
  if (!this->ROILabelMapNode || !this->ROILabelMap || !Node || !this->ROICheck()) return;
  int size[3]   = {Node->GetROIMax(0) - Node->GetROIMin(0) + 1, 
                   Node->GetROIMax(1) - Node->GetROIMin(1) + 1, 
                   Node->GetROIMax(2) - Node->GetROIMin(2) + 1};

  int center[3] = {(Node->GetROIMax(0) + Node->GetROIMin(0))/2,
                   (Node->GetROIMax(1) + Node->GetROIMin(1))/2, 
                   (Node->GetROIMax(2) + Node->GetROIMin(2))/2};

  this->ROILabelMap->SetCenter(center);
  this->ROILabelMap->SetSize(size);
  this->ROILabelMap->Update();
  this->ROILabelMapNode->Modified();

  // Update the roiNode
  if(roiNode && !roiUpdateGuard)
    {
    roiUpdateGuard = true;
    MRMLUpdateROINodeFromROI();
    roiNode->Modified();
    roiUpdateGuard = false;
    }
}


int vtkChangeTrackerROIStep::ROIMapShow() {
  // -----
  // Initialize
  if (!this->ROICheck()) {
    vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), 
                                     this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),
                                     "Change Tracker", 
                                     "Please define VOI correctly before pressing button", 
                                     vtkKWMessageDialog::ErrorIcon);
    return 0;
  }

  vtkMRMLChangeTrackerNode* Node      =  this->GetGUI()->GetNode();
  if (!Node) return 0;
  vtkMRMLScene* mrmlScene           =  Node->GetScene();
  vtkMRMLNode* mrmlFristScanRefNode =  mrmlScene->GetNodeByID(Node->GetScan1_Ref());
  vtkMRMLVolumeNode* volumeNode     =  vtkMRMLVolumeNode::SafeDownCast(mrmlFristScanRefNode);
  if (!volumeNode) return 0;
  int* dimensions = volumeNode->GetImageData()->GetDimensions();

  if (this->ROILabelMapNode || this->ROILabelMap) this->ROIMapRemove(); 

  // -----
  // Define LabelMap 
  this->ROILabelMap =  vtkImageRectangularSource::New();
  this->ROILabelMap->SetWholeExtent(0,dimensions[0] -1,0,dimensions[1] -1, 0,dimensions[2] -1); 
  this->ROILabelMap->SetOutputScalarTypeToShort();
  this->ROILabelMap->SetInsideGraySlopeFlag(0); 
  this->ROILabelMap->SetInValue(17);
  this->ROILabelMap->SetOutValue(0);
  this->ROILabelMap->Update();

  // Show map in Slicer 3 
  //  set scene [[$this GetLogic] GetMRMLScene]
  //  set volumesLogic [$::slicer3::VolumesGUI GetLogic]
  vtkSlicerApplication *application   = vtkSlicerApplication::SafeDownCast(this->GetApplication());
  vtkSlicerApplicationGUI *applicationGUI = this->GetGUI()->GetApplicationGUI();
//  vtkSlicerApplicationLogic *applicationLogic = this->GetGUI()->GetLogic()->GetApplicationLogic();

  vtkSlicerVolumesGUI  *volumesGUI    = vtkSlicerVolumesGUI::SafeDownCast(application->GetModuleGUIByName("Volumes")); 
  vtkSlicerVolumesLogic *volumesLogic = volumesGUI->GetLogic();
  // set labelNode [$volumesLogic CreateLabelVolume $scene $volumeNode $name]
  this->ROILabelMapNode = volumesLogic->CreateLabelVolume(mrmlScene,volumeNode, "TG_ROI");
  this->ROILabelMapNode->SetAndObserveImageData(this->ROILabelMap->GetOutput());

  // Now show in foreground 
  //  make the source node the active background, and the label node the active label
  // set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
  //$selectionNode SetReferenceActiveVolumeID [$volumeNode GetID]
  //$selectionNode SetReferenceActiveLabelVolumeID [$labelNode GetID] 
  //  applicationLogic->GetSelectionNode()->SetReferenceActiveVolumeID(volumeNode->GetID());
 
  // Reset to original slice location 
  double oldSliceSetting[3];
  oldSliceSetting[0] = double(applicationGUI->GetMainSliceGUI("Red")->GetSliceController()->GetOffsetScale()->GetValue());
  oldSliceSetting[1] = double(applicationGUI->GetMainSliceGUI("Yellow")->GetSliceController()->GetOffsetScale()->GetValue());
  oldSliceSetting[2] = double(applicationGUI->GetMainSliceGUI("Green")->GetSliceController()->GetOffsetScale()->GetValue());

  applicationGUI->GetMainSliceGUI("Red")->GetLogic()->GetSliceCompositeNode()->SetForegroundVolumeID(this->ROILabelMapNode->GetID());
  applicationGUI->GetMainSliceGUI("Yellow")->GetLogic()->GetSliceCompositeNode()->SetForegroundVolumeID(this->ROILabelMapNode->GetID());
  applicationGUI->GetMainSliceGUI("Green")->GetLogic()->GetSliceCompositeNode()->SetForegroundVolumeID(this->ROILabelMapNode->GetID());

  applicationGUI->GetMainSliceGUI("Red")->GetLogic()->GetSliceCompositeNode()->SetForegroundOpacity(0.6);
  applicationGUI->GetMainSliceGUI("Yellow")->GetLogic()->GetSliceCompositeNode()->SetForegroundOpacity(0.6);
  applicationGUI->GetMainSliceGUI("Green")->GetLogic()->GetSliceCompositeNode()->SetForegroundOpacity(0.6);

  // Reset to original slice location 
  applicationGUI->GetMainSliceGUI("Red")->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[0]);
  applicationGUI->GetMainSliceGUI("Yellow")->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[1]);
  applicationGUI->GetMainSliceGUI("Green")->GetSliceController()->GetOffsetScale()->SetValue(oldSliceSetting[2]);

  this->ROIMapUpdate();

  return 1;
}

void vtkChangeTrackerROIStep::ROIMapRemove() {
  
  if (this->ROILabelMapNode && this->GetGUI()) { 
    this->GetGUI()->GetMRMLScene()->RemoveNode(this->ROILabelMapNode);
  }
  this->ROILabelMapNode = NULL;

  if (this->ROILabelMap) { 
    this->ROILabelMap->Delete();
    this->ROILabelMap = NULL;
  }

  // Needs to be done otherwise when going backwards field is not correctly defined   
  if (this->ButtonsShow && this->ButtonsShow->IsCreated())  {
    this->ButtonsShow->SetText("Show render");
  }
}

void vtkChangeTrackerROIStep::RetrieveInteractorIJKCoordinates(vtkSlicerSliceGUI *sliceGUI, 
                                                               vtkRenderWindowInteractor *rwi,
                                                               int coords[3]) 
{
  coords[0] = coords[1] = coords[2] = -1;
  vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
  if (!Node) {
    cout << "ERROR: vtkChangeTrackerROIStep::RetrieveInteractorIJKCoordinates: No Node" << endl;
    return;
  } 

  if (!Node->GetScan1_Ref()) {
    cout << "ERROR: vtkChangeTrackerROIStep::RetrieveInteractorIJKCoordinates: No First Volume Defined" << endl;
    return;
  }
  vtkMRMLNode* mrmlNode =   Node->GetScene()->GetNodeByID(Node->GetScan1_Ref());
  vtkMRMLVolumeNode* volumeNode =  vtkMRMLVolumeNode::SafeDownCast(mrmlNode);

  if (!volumeNode)
    {
      cout << "ERROR: vtkChangeTrackerROIStep::RetrieveInteractorIJKCoordinates: No Scan1_Ref" << endl;
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
  coords[0] = int(0.5+(ijkPt[0]));  coords[1] = int(0.5+(ijkPt[1])); coords[2] = int(0.5+(ijkPt[2])); 

  //cout << "Sample:  " << rasPt[0] << " " <<  rasPt[1] << " " << rasPt[2] << " " << rasPt[3] << endl;
  //cout << "Coord: " << coords[0] << " " << coords[1] << " " << coords[2] << " " << coords[3] << endl;
  //cout << "Dimen: " << dimensions[0] << " " << dimensions[1] << " " << dimensions[2] << " " <<  endl;

}

void vtkChangeTrackerROIStep::ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData) {
  
  if (event == vtkKWPushButton::InvokedEvent) {
    vtkKWPushButton *button = vtkKWPushButton::SafeDownCast(caller);
    if (this->ButtonsShow && (button == this->ButtonsShow)) 
    { 
      if (this->ROILabelMapNode) {
        this->ButtonsShow->SetText("Show render");
        this->ROIMapRemove();
//        roiNode->SetVisibility(0);
        ResetROIRender();
        this->ROIHideFlag = 1;
      } else { 
        if (this->ROIMapShow()) { 
//          roiNode->SetVisibility(1);
          UpdateROIRender();
          this->ButtonsShow->SetText("Hide render");
        }
// FIXME: when feature complete
//        if (roiNode)  
//          roiNode->SetVisibility(1);
      }
    }
    if (this->ButtonsReset && (button == this->ButtonsReset)) 
    { 
      if (this->ROILabelMapNode) {
        this->ButtonsShow->SetText("Show render");
        this->ROIMapRemove();
        roiNode->SetVisibility(0);
        ResetROIRender();
      }
      this->ROIReset();
      this->MRMLUpdateROIFromROINode();
    }
    return;
  }

  vtkSlicerInteractorStyle *s = vtkSlicerInteractorStyle::SafeDownCast(caller);
  if (s && event == vtkCommand::LeftButtonPressEvent)
  {
    // Retrieve Coordinates and update ROI
    int index = 0; 
    vtkSlicerSliceGUI *sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(
      this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Red");

    vtkRenderWindowInteractor *rwi = sliceGUI->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();
    while (index < 2 && (s != rwi->GetInteractorStyle())) {
        index ++;
        if (index == 1) {
          sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Yellow");
        } else {
          sliceGUI = vtkSlicerApplicationGUI::SafeDownCast(this->GetGUI()->GetApplicationGUI())->GetMainSliceGUI("Green");
        }
        rwi = sliceGUI->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor();
    }
    int ijkCoords[3];
    this->RetrieveInteractorIJKCoordinates(sliceGUI, rwi, ijkCoords);
//    this->ResetROICenter(ijkCoords);
    this->ROIUpdateWithNewSample(ijkCoords);
    if (!this->ROILabelMapNode && !this->ROIHideFlag && this->ROICheck()) {
      if (this->ROIMapShow()) 
        {
        MRMLUpdateROINodeFromROI();
        roiNode->SetVisibility(1);
        this->ButtonsShow->SetText("Hide render");
        }
    }
    UpdateROIRender();
  }  
  // Define SHOW Button 
}


void vtkChangeTrackerROIStep::ProcessMRMLEvents(vtkObject *caller, unsigned long event, void *callData) {
//  if(event == vtkCommand::ModifiedEvent){
    vtkMRMLROINode *roiCaller = vtkMRMLROINode::SafeDownCast(caller);
    if(roiCaller && roiCaller == roiNode && event == vtkCommand::ModifiedEvent && !roiUpdateGuard)
      {
      
      roiUpdateGuard = true;
      MRMLUpdateROIFromROINode();
      this->ROIMapUpdate();
      if(this->Render_Filter->GetSize())
        this->UpdateROIRender();
      roiUpdateGuard = false;

      double *roiXYZ = roiNode->GetXYZ();
      vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication());
      app->GetApplicationGUI()->GetViewControlGUI()->MainViewSetFocalPoint(roiXYZ[0], roiXYZ[1], roiXYZ[2]);
//      cerr << "Resetting focal point to " << roiXYZ[0] << ", " << roiXYZ[1] << ", " << roiXYZ[2] << endl;
      }
}

// Propagate ROI changes in ChangeTracker MRML to ROINode MRML
void vtkChangeTrackerROIStep::MRMLUpdateROIFromROINode()
{
  vtkMRMLChangeTrackerNode* ctNode = this->GetGUI()->GetNode();
  // update roi to correspond to ROI widget
  double *roiXYZ = roiNode->GetXYZ();
  double *roiRadiusXYZ = roiNode->GetRadiusXYZ();

  double bbox0ras[4], bbox1ras[4];
  double bbox0ijk[4], bbox1ijk[4];

  // ROI bounding box in RAS coordinates
  bbox0ras[0] = roiXYZ[0]-roiRadiusXYZ[0];
  bbox0ras[1] = roiXYZ[1]-roiRadiusXYZ[1];
  bbox0ras[2] = roiXYZ[2]-roiRadiusXYZ[2];
  bbox0ras[3] = 1.;
  bbox1ras[0] = roiXYZ[0]+roiRadiusXYZ[0];
  bbox1ras[1] = roiXYZ[1]+roiRadiusXYZ[1];
  bbox1ras[2] = roiXYZ[2]+roiRadiusXYZ[2];
  bbox1ras[3] = 1.;

  vtkMatrix4x4 *rasToijk = vtkMatrix4x4::New();
  vtkMRMLVolumeNode *volumeNode = 
    vtkMRMLVolumeNode::SafeDownCast(ctNode->GetScene()->GetNodeByID(ctNode->GetScan1_Ref()));
  volumeNode->GetRASToIJKMatrix(rasToijk);
  rasToijk->MultiplyPoint(bbox0ras,bbox0ijk);
  rasToijk->MultiplyPoint(bbox1ras,bbox1ijk);
  rasToijk->Delete();
  for(int i=0;i<3;i++)
    {
    double tmp;
    if(bbox0ijk[i]>bbox1ijk[i])
      {
      tmp = bbox0ijk[i];
      bbox0ijk[i] = bbox1ijk[i];
      bbox1ijk[i] = tmp;
      }
    }
  ctNode->SetROIMin(0, (int)bbox0ijk[0]);
  ctNode->SetROIMax(0, (int)bbox1ijk[0]);
  ctNode->SetROIMin(1, (int)bbox0ijk[1]);
  ctNode->SetROIMax(1, (int)bbox1ijk[1]);
  ctNode->SetROIMin(2, (int)bbox0ijk[2]);
  ctNode->SetROIMax(2, (int)bbox1ijk[2]);
  
//  this->ROIX->SetRange(bbox0ijk[0], bbox1ijk[0]);
//  this->ROIY->SetRange(bbox0ijk[1], bbox1ijk[1]);
//  this->ROIZ->SetRange(bbox0ijk[2], bbox1ijk[2]);
}

// Propagate changes in ROINode MRML to ChangeTracker ROI MRML
void vtkChangeTrackerROIStep::MRMLUpdateROINodeFromROI()
{
  vtkMRMLChangeTrackerNode* Node      =  this->GetGUI()->GetNode();
  if (!this->ROILabelMapNode || !this->ROILabelMap || !Node || !this->ROICheck()) return;
  int size[3]   = {Node->GetROIMax(0) - Node->GetROIMin(0) + 1, 
                   Node->GetROIMax(1) - Node->GetROIMin(1) + 1, 
                   Node->GetROIMax(2) - Node->GetROIMin(2) + 1};

  int center[3] = {(Node->GetROIMax(0) + Node->GetROIMin(0))/2,
                   (Node->GetROIMax(1) + Node->GetROIMin(1))/2, 
                   (Node->GetROIMax(2) + Node->GetROIMin(2))/2};

  double pointRAS[4], pointIJK[4];
  double radius[3];
  vtkMatrix4x4 *ijkToras = vtkMatrix4x4::New();
  vtkMRMLVolumeNode *volumeNode = 
    vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
  volumeNode->GetIJKToRASMatrix(ijkToras);
  pointIJK[0] = (double)center[0];
  pointIJK[1] = (double)center[1];
  pointIJK[2] = (double)center[2];
  pointIJK[3] = 1.;
  ijkToras->MultiplyPoint(pointIJK,pointRAS);
  ijkToras->Delete();

  radius[0] = volumeNode->GetSpacing()[0]*(double)size[0]/2.;
  radius[1] = volumeNode->GetSpacing()[1]*(double)size[1]/2.;
  radius[2] = volumeNode->GetSpacing()[2]*(double)size[2]/2.;

  roiNode->SetXYZ(pointRAS[0], pointRAS[1], pointRAS[2]);
  roiNode->SetRadiusXYZ(radius[0], radius[1], radius[2]);
  roiNode->Modified();
}

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::TransitionCallback() 
{
  // cout << "vtkChangeTrackerROIStep::TransitionCallback() Start" << endl; 
  if (this->ROICheck()) { 
     // ----------------------------
     // Create SuperSampledVolume 
    vtkSlicerApplication *application   = vtkSlicerApplication::SafeDownCast(this->GetGUI()->GetApplication());
    vtkMRMLScalarVolumeNode *outputNode = this->GetGUI()->GetLogic()->CreateSuperSample(1);
    this->GetGUI()->GetLogic()->SaveVolume(application,outputNode); 

    if (outputNode) {
       // Prepare to update mrml node with results 
       vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
       if (!Node) return;
              
       // Delete old attached node first 
       this->GetGUI()->GetLogic()->DeleteSuperSample(1);

       // Update node 
       
       Node->SetScan1_SuperSampleRef(outputNode->GetID());
       //cout << "==============================" << endl;
       //cout << "vtkChangeTrackerROIStep::TransitionCallback " << Node->GetScan1_SuperSampleRef() << " " <<  Node->GetScan1_Ref() << endl;
       //cout << "==============================" << endl;

       // Remove blue ROI screen 
       this->ROIMapRemove();
       
       // remove the ROI widget
       if (roiNode)
         roiNode->SetVisibility(0);
       ResetROIRender();

       this->GUI->GetWizardWidget()->GetWizardWorkflow()->AttemptToGoToNextStep();
     } else {
       vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), 
                                        this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),
                                        "Change Tracker", 
                                        "Could not proceed to next step - scan1 might have disappeared", 
                                        vtkKWMessageDialog::ErrorIcon); 
     }
     // ---------------------------------
   } else {     
     vtkKWMessageDialog::PopupMessage(this->GUI->GetApplication(), 
                                      this->GUI->GetApplicationGUI()->GetMainSlicerWindow(),
                                      "Change Tracker", 
                                      "Please define VOI correctly before proceeding", 
                                      vtkKWMessageDialog::ErrorIcon);
   }
}


//----------------------------------------------------------------------------
void  vtkChangeTrackerROIStep::HideUserInterface()
{
  this->Superclass::HideUserInterface();
  this->RemoveROISamplingGUIObservers();
}

//----------------------------------------------------------------------------
void vtkChangeTrackerROIStep::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

void vtkChangeTrackerROIStep::ROIIntensityMinMaxUpdate(vtkImageData* image, double &intensityMin, double &intensityMax)
{
  if(!this->ROICheck())
  {
    return;
  }

  vtkMRMLChangeTrackerNode* ctNode = this->GetGUI()->GetNode();
  if(
    ctNode->GetROIMax(0)-ctNode->GetROIMin(0)<=0 ||
    ctNode->GetROIMax(1)-ctNode->GetROIMin(1)<=0 ||
    ctNode->GetROIMax(2)-ctNode->GetROIMin(2)<=0)
    {
    return;
    }

  // take a sparse sample of the ROI intensity, with 10 samples in each
  // dimension
  int ijk[3];
  int iInc, jInc, kInc;
  intensityMin = 1e10;
  intensityMax = -1.;

//  iInc = (ctNode->GetROIMax(0)-ctNode->GetROIMin(0))/3;
//  jInc = (ctNode->GetROIMax(1)-ctNode->GetROIMin(1))/3;
//  kInc = (ctNode->GetROIMax(2)-ctNode->GetROIMin(2))/3;
  iInc = 1;
  jInc = 1;
  kInc = 1;
  ijk[0] = ctNode->GetROIMin(0);
  ijk[1] = ctNode->GetROIMin(1);
  ijk[2] = ctNode->GetROIMin(2);

  for(;ijk[0]<ctNode->GetROIMax(0);ijk[0]++)
    {
    for(;ijk[1]<ctNode->GetROIMax(1);ijk[1]++)
      {
      for(;ijk[2]<ctNode->GetROIMax(2);ijk[2]++)
        {
        double intensity = 0;

        switch (image->GetScalarType())
             {
              case VTK_CHAR:{
                char *intensityPtr = (char*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_UNSIGNED_CHAR:{
                unsigned char *intensityPtr = (unsigned char*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_SHORT:{
                short *intensityPtr = (short*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_UNSIGNED_SHORT:{
                unsigned short *intensityPtr = (unsigned short*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_INT:{
                int *intensityPtr = (int*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_UNSIGNED_INT:{
                unsigned int *intensityPtr = (unsigned int*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_LONG:{
                long *intensityPtr = (long*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_UNSIGNED_LONG:{
                unsigned long *intensityPtr = (unsigned long*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_FLOAT:{
                float *intensityPtr = (float*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              case VTK_DOUBLE:{
                double *intensityPtr = (double*)image->GetScalarPointer(ijk[0], ijk[1], ijk[2]);
                intensity = (double)(*intensityPtr);
                break;}
              default:{
                cerr << "Unknown scalar type" << endl;
              }
             }
        if(intensityMin>intensity)
          intensityMin = intensity;
        if(intensityMax<intensity)
          intensityMax = intensity;
        }
      }
    }
}

void vtkChangeTrackerROIStep::InitROIRender()
{ 
  if(this->roiNode)
    {
    vtkMRMLChangeTrackerNode* Node = this->GetGUI()->GetNode();
    vtkMRMLVolumeNode* volumeNode =  
      vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
    if(volumeNode)
      {
      CreateRender(volumeNode, 0);
      }
    }
}

void vtkChangeTrackerROIStep::UpdateROIRender()
{
  vtkMRMLChangeTrackerNode* node = this->GetGUI()->GetNode();
  vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(node->GetScene()->GetNodeByID(node->GetScan1_Ref()));
  if(volumeNode)
    {
    float color0[3] = { 0.8, 0.8, 0.0 };
    float color1[3] = { 0.8, 0.8, 0.0 };
    double intensityMin, intensityMax;
    
    double* imgRange  =   volumeNode->GetImageData()->GetPointData()->GetScalars()->GetRange();
    intensityMin = imgRange[0];
    intensityMax = imgRange[1];
    this->ROIIntensityMinMaxUpdate(volumeNode->GetImageData(), intensityMin, intensityMax);
    this->SetRender_HighPassFilter((intensityMax+intensityMin)*.5, color0, color1);

    if(this->Render_RayCast_Mapper)
      {
      this->Render_Mapper->SetCroppingRegionPlanes(node->GetROIMin(0), node->GetROIMax(0),
                                                   node->GetROIMin(1), node->GetROIMax(1),
                                                   node->GetROIMin(2), node->GetROIMax(2));
      this->Render_Mapper->CroppingOn();
      this->GetGUI()->GetApplicationGUI()->GetViewerWidget()->RequestRender();
      }

    if(this->Render_Mapper)
      {
      // !!!  Cropping region is defined in voxel coordinates !!!
      this->Render_Mapper->SetCroppingRegionPlanes(node->GetROIMin(0), node->GetROIMax(0),
                                                   node->GetROIMin(1), node->GetROIMax(1),
                                                   node->GetROIMin(2), node->GetROIMax(2));
        
      this->Render_Mapper->CroppingOn();
      this->GetGUI()->GetApplicationGUI()->GetViewerWidget()->RequestRender();
      }
    }
}

void vtkChangeTrackerROIStep::ResetROIRender()
{
  this->Render_Filter->RemoveAllPoints();
}

void vtkChangeTrackerROIStep::ResetROICenter(int *center)
{
  vtkMRMLChangeTrackerNode* Node      =  this->GetGUI()->GetNode();
  double pointRAS[4], pointIJK[4];
  vtkMatrix4x4 *ijkToras = vtkMatrix4x4::New();
  vtkMRMLVolumeNode *volumeNode = 
    vtkMRMLVolumeNode::SafeDownCast(Node->GetScene()->GetNodeByID(Node->GetScan1_Ref()));
  volumeNode->GetIJKToRASMatrix(ijkToras);
  pointIJK[0] = (double)center[0];
  pointIJK[1] = (double)center[1];
  pointIJK[2] = (double)center[2];
  pointIJK[3] = 1.;
  ijkToras->MultiplyPoint(pointIJK,pointRAS);
  ijkToras->Delete();

  roiNode->SetXYZ(pointRAS[0], pointRAS[1], pointRAS[2]);

  CenterRYGSliceViews(pointRAS[0], pointRAS[1], pointRAS[2]);
}
