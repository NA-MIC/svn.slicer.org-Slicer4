#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkSlicerModuleGUI.h"
#include "vtkSlicerGUICollection.h"
#include "vtkCommandLineModuleGUI.h"
#include "vtkSlicerTractographyFiducialSeedingGUI.h"
//MRML nodes
#include "vtkMRMLNode.h"
#include "vtkMRMLCommandLineModuleNode.h"
#include "vtkMRMLDiffusionTensorVolumeNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLDiffusionWeightedVolumeNode.h"
#include "vtkMRMLFiberBundleNode.h"
#include "vtkMRMLFiducialListNode.h"
//logics
#include "vtkCommandLineModuleLogic.h"
#include "vtkSlicerTractographyFiducialSeedingLogic.h"
#include "vtkSlicerGradientEditorLogic.h"
//widgets
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerGradientEditorWidget.h"
#include "vtkSlicerMeasurementFrameWidget.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerGradientEditorWidget);
vtkCxxRevisionMacro (vtkSlicerGradientEditorWidget, "$Revision: 1.0 $");

//---------------------------------------------------------------------------
vtkSlicerGradientEditorWidget::vtkSlicerGradientEditorWidget(void)
  {
  this->Application = NULL;
  this->ActiveVolumeNode = NULL;
  this->OriginalNode = vtkMRMLDiffusionWeightedVolumeNode::New();
  this->MeasurementFrameWidget = NULL;
  this->GradientsWidget = NULL;
  this->RestoreButton = NULL;
  this->UndoButton = NULL;
  //Testframe maybe as an extra widget later
  this->TestFrame = NULL;
  this->RunButton = NULL;
  this->FiducialSelector = NULL;
  this->NumberOfChanges = 0;
  this->ModifiedForNewTensor = 1;
  this->TensorNode = NULL;
  this->FiberNode = NULL;
  }

//---------------------------------------------------------------------------
vtkSlicerGradientEditorWidget::~vtkSlicerGradientEditorWidget(void)
  {
  this->RemoveWidgetObservers();
  if (this->ActiveVolumeNode)
    {
    this->ActiveVolumeNode->Delete();
    this->ActiveVolumeNode = NULL;
    }
  if (this->OriginalNode)
    {
    this->OriginalNode->Delete();
    this->OriginalNode = NULL;
    }
  if (this->MeasurementFrameWidget)
    {
    this->MeasurementFrameWidget->Delete();
    this->MeasurementFrameWidget = NULL;
    }
  if (this->GradientsWidget)
    {
    this->GradientsWidget->Delete();
    this->GradientsWidget = NULL;
    }
  if (this->TestFrame)
    {
    this->TestFrame->SetParent (NULL);
    this->TestFrame->Delete();
    this->TestFrame = NULL;
    }
  if (this->FiducialSelector)
    {
    this->FiducialSelector->SetParent (NULL);
    this->FiducialSelector->Delete();
    this->FiducialSelector = NULL;
    }
  if (this->RunButton)
    {
    this->RunButton->SetParent (NULL);
    this->RunButton->Delete();
    this->RunButton = NULL;
    }
  if (this->RestoreButton)
    {
    this->RestoreButton->SetParent (NULL);
    this->RestoreButton->Delete();
    this->RestoreButton = NULL;
    }
  if (this->UndoButton)
    {
    this->UndoButton->Delete();
    this->UndoButton = NULL;
    }
  if (this->Application)
    {
    this->Application->Delete();
    this->Application = NULL;
    }
  if (this->TensorNode)
    {
    this->TensorNode->Delete();
    this->TensorNode = NULL;
    }
  if (this->FiberNode)
    {
    this->FiberNode->Delete();
    this->FiberNode = NULL;
    }
  this->NumberOfChanges = 0;
  this->ModifiedForNewTensor = 0;
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::AddWidgetObservers ( )
  {    
  this->RunButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->RestoreButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->UndoButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->MeasurementFrameWidget->AddObserver(vtkSlicerMeasurementFrameWidget::ChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->GradientsWidget->AddObserver(vtkSlicerGradientsWidget::ChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::RemoveWidgetObservers( )
  {
  this->RunButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->RestoreButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->UndoButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->MeasurementFrameWidget->RemoveObservers(vtkSlicerMeasurementFrameWidget::ChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->GradientsWidget->RemoveObservers(vtkSlicerGradientsWidget::ChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::PrintSelf (ostream& os, vtkIndent indent)
  {
  this->vtkObject::PrintSelf ( os, indent );
  os << indent << "vtkSlicerGradientEditorWidget: " << this->GetClassName ( ) << "\n";
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::ProcessWidgetEvents (vtkObject *caller, unsigned long event, void *callData)
  {

  //enable undo/restore button, when values were changed
  if((this->MeasurementFrameWidget == vtkSlicerMeasurementFrameWidget::SafeDownCast(caller) && 
    event == vtkSlicerMeasurementFrameWidget::ChangedEvent) ||( event == vtkSlicerGradientsWidget::ChangedEvent && 
    this->GradientsWidget == vtkSlicerGradientsWidget::SafeDownCast(caller)))
    {
    this->UndoButton->SetEnabled(1);
    this->RestoreButton->SetEnabled(1);
    this->NumberOfChanges++; //increment changes    
    this->ModifiedForNewTensor = 1; //tensor has to be estimated newly
    }

  //restore to original
  else if (this->RestoreButton == vtkKWPushButton::SafeDownCast(caller) && event == vtkKWPushButton::InvokedEvent)
    {
    this->MRMLScene->SaveStateForUndo();
    this->NumberOfChanges++; //increment changes
    this->ActiveVolumeNode->Copy(this->OriginalNode); //copy original back
    this->MeasurementFrameWidget->UpdateWidget(this->ActiveVolumeNode); //update GUI
    this->GradientsWidget->UpdateWidget(this->ActiveVolumeNode); //update GUI
    this->RestoreButton->SetEnabled(0); //disable restoreButton until next change
    this->UndoButton->SetEnabled(1); //enable undoButton
    this->ModifiedForNewTensor = 1; //tensor has to be estimated newly
    }

  //undo
  else if(event == vtkKWPushButton::InvokedEvent && this->UndoButton == vtkKWPushButton::SafeDownCast(caller))
    {
    //if there is a copy in the undoStack, that was made before loading
    if(this->NumberOfChanges>0)
      {
      this->MRMLScene->Undo(); //undo
      this->NumberOfChanges--; //decrement changes
      this->MeasurementFrameWidget->UpdateWidget(this->ActiveVolumeNode); //update GUI
      this->GradientsWidget->UpdateWidget(this->ActiveVolumeNode); //update GUI
      //disable buttons, when no changes are available
      if(this->NumberOfChanges==0)
        {
        this->UndoButton->SetEnabled(0);
        this->RestoreButton->SetEnabled(0);
        }
      }
    }

  //run test
  else if (this->RunButton == vtkKWPushButton::SafeDownCast(caller) && event == vtkKWPushButton::InvokedEvent)
    {
    this->RunButton->SetEnabled(0);
    if(this->ModifiedForNewTensor)
      {
      // create a command line module node
      vtkMRMLCommandLineModuleNode* module = vtkMRMLCommandLineModuleNode::SafeDownCast(
        this->MRMLScene->CreateNodeByClass("vtkMRMLCommandLineModuleNode"));

      // set its name  
      module->SetModuleDescription("Diffusion Tensor Estimation");
      module->SetName("GradientEditor: Tensor Estimation");

      // set the parameters
      module->SetParameterAsString("estimationMethod", "Least Squares");
      module->SetParameterAsDouble("otsuOmegaThreshold",0.5);
      module->SetParameterAsBool("removeIslands", 0);
      module->SetParameterAsBool("applyMask", 0);
      module->SetParameterAsString("inputVolume", this->ActiveVolumeNode->GetID());

      // create the output nodes
      if(this->TensorNode)
        {
        this->TensorNode->Delete(); //delete previous data
        }
      this->TensorNode = vtkMRMLDiffusionTensorVolumeNode::SafeDownCast(
        this->MRMLScene->CreateNodeByClass("vtkMRMLDiffusionTensorVolumeNode"));
      this->TensorNode->SetScene(this->GetMRMLScene());
      this->TensorNode->SetName("GradientenEditor: Tensor Node");
      this->MRMLScene->AddNode(this->TensorNode);

      vtkMRMLScalarVolumeNode* baseline= vtkMRMLScalarVolumeNode::SafeDownCast(
        this->MRMLScene->CreateNodeByClass("vtkMRMLScalarVolumeNode"));
      baseline->SetScene(this->GetMRMLScene());
      baseline->SetName("GradientenEditor: Baseline Node");
      this->MRMLScene->AddNode(baseline);

      vtkMRMLScalarVolumeNode* mask = vtkMRMLScalarVolumeNode::SafeDownCast(
        this->MRMLScene->CreateNodeByClass("vtkMRMLScalarVolumeNode"));
      mask->SetScene(this->GetMRMLScene());
      mask->SetName("GradientenEditor: Threshold Mask");
      this->MRMLScene->AddNode(mask);

      this->TensorNode->SetBaselineNodeID(baseline->GetID());
      this->TensorNode->SetMaskNodeID(mask->GetID());

      // set output parameters
      module->SetParameterAsString("outputTensor", this->TensorNode->GetID());
      module->SetParameterAsString("outputBaseline", baseline->GetID());
      module->SetParameterAsString("thresholdMask", mask->GetID());

      //get the existing gui of the "Diffusion Tensor Estimation Command Line Module" 
      vtkCommandLineModuleGUI *moduleGUI = vtkCommandLineModuleGUI::SafeDownCast(
        this->Application->GetModuleGUIByName("Diffusion Tensor Estimation"));
      moduleGUI->Enter();

      //set command line node to gui an logic
      moduleGUI->SetCommandLineModuleNode(module);
      moduleGUI->GetLogic()->SetCommandLineModuleNode(module);

      //estimate tensors
      moduleGUI->GetLogic()->Apply(module);

      //clean up
      module->Delete();
      baseline->Delete();
      mask->Delete();
      this->ModifiedForNewTensor = 0;
      }
    this->CreateTracts();  //start tractography seeding, with old or new tensor
    }
  }

void vtkSlicerGradientEditorWidget::CreateTracts ( )
  {
  //wait untill ImageData is set in other thread
  if(this->TensorNode->GetImageData() == NULL)
    {
    this->Script ( "update idletasks" );
    this->Script ( "after 5 \"%s CreateTracts \"",  this->GetTclName() );
    }
  else
    {
    //get fiducial list
    vtkMRMLFiducialListNode* fiducialListNode = vtkMRMLFiducialListNode::SafeDownCast(
      this->FiducialSelector->GetSelected());

    //create new fiber node
    if(this->FiberNode == NULL)
      {
      this->FiberNode = vtkMRMLFiberBundleNode::New();
      this->FiberNode->SetScene(this->GetMRMLScene());
      this->FiberNode->SetName("GradientenEditor: Fiber Node");
      this->MRMLScene->AddNode(this->FiberNode);
      }

    //get the existing gui of the "Tractography Fiducial Seeding Module"
    vtkSlicerTractographyFiducialSeedingGUI *moduleGUI = vtkSlicerTractographyFiducialSeedingGUI::SafeDownCast(
      this->Application->GetModuleGUIByName("FiducialSeeding"));    
    moduleGUI->Enter(); 

    //set the selectors to my nodes
    moduleGUI->SetVolumeSelector(this->TensorNode);
    moduleGUI->SetFiducialSelector(fiducialListNode);
    moduleGUI->SetOutFiberSelector(this->FiberNode);

    //create tracts
    moduleGUI->CreateTracts();
    this->RunButton->SetEnabled(1);
    }
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::UpdateWidget(vtkMRMLDiffusionWeightedVolumeNode *dwiNode)
  {
  if (dwiNode == NULL)
    {
    vtkErrorMacro(<< this->GetClassName() << ": dwiNode in UpdateWidget() is NULL");
    return;
    }
  vtkSetMRMLNodeMacro(this->ActiveVolumeNode, dwiNode); //set ActiveVolumeNode
  this->OriginalNode->Copy(this->ActiveVolumeNode); //make private copy before changing
  // update the measurement frame, gradients and bValues 
  // when the active node changes
  this->MeasurementFrameWidget->SetMRMLScene(this->GetMRMLScene());
  this->GradientsWidget->SetMRMLScene(this->GetMRMLScene());
  this->MeasurementFrameWidget->UpdateWidget(this->ActiveVolumeNode);
  this->GradientsWidget->UpdateWidget(this->ActiveVolumeNode);
  }

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::CreateWidget( )
  {
  //check if already created
  if (this->IsCreated()){
    vtkErrorMacro(<< this->GetClassName() << " already created");
    return;
    }

  //call the superclass to create the whole widget
  this->Superclass::CreateWidget();

  //create measurementFrame widget
  this->MeasurementFrameWidget = vtkSlicerMeasurementFrameWidget::New();
  this->MeasurementFrameWidget->SetParent(this->GetParent());
  this->MeasurementFrameWidget->Create();
  this->MeasurementFrameWidget->AddWidgetObservers();
  this->Script("pack %s -side top -anchor nw -fill x -padx 1 -pady 2", 
    this->MeasurementFrameWidget->GetWidgetName());

  //create gradient widget 
  this->GradientsWidget = vtkSlicerGradientsWidget::New();
  this->GradientsWidget->SetParent(this->GetParent());
  this->GradientsWidget->Create();
  this->GradientsWidget->AddWidgetObservers();
  this->Script("pack %s -side top -anchor nw -fill both -expand true -padx 1 -pady 2", 
    this->GradientsWidget->GetWidgetName());

  //create test frame 
  this->TestFrame = vtkKWFrameWithLabel::New();
  this->TestFrame->SetParent(this->GetParent());
  this->TestFrame->Create();
  this->TestFrame->SetLabelText("Test");
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
    this->TestFrame->GetWidgetName());

  //create run button
  this->RunButton = vtkKWPushButton::New();
  this->RunButton->SetParent(this->TestFrame->GetFrame());
  this->RunButton->Create();
  this->RunButton->SetText("Run");
  this->RunButton->SetWidth(7);
  this->RunButton->SetBalloonHelpString("Run test by computing tensors and tractography seeding.");

  //create fiducial list
  this->FiducialSelector = vtkSlicerNodeSelectorWidget::New();
  this->FiducialSelector->SetNodeClass("vtkMRMLFiducialListNode", NULL, NULL, NULL);
  this->FiducialSelector->SetNewNodeEnabled(0);
  this->FiducialSelector->NoneEnabledOn();
  this->FiducialSelector->SetShowHidden(1);
  this->FiducialSelector->SetParent(this->TestFrame->GetFrame());
  this->FiducialSelector->SetMRMLScene(this->GetMRMLScene());
  this->FiducialSelector->Create();  
  this->FiducialSelector->UpdateMenu();
  this->FiducialSelector->SetLabelText("Fiducial List:");
  this->FiducialSelector->SetBalloonHelpString("Set Fiducial List for tractography seeding.");

  this->Script("pack %s %s -side right -anchor ne -padx 3 -pady 2", 
    this->RunButton->GetWidgetName(),
    this->FiducialSelector->GetWidgetName());

  //create undoButton
  this->UndoButton = vtkKWPushButton::New();
  this->UndoButton->SetParent(this->GetParent());
  this->UndoButton->SetText("Undo");  
  this->UndoButton->Create();
  this->UndoButton->SetBalloonHelpString("");
  this->UndoButton->SetWidth(10);
  this->UndoButton->SetEnabled(0);

  //create restore  button
  this->RestoreButton = vtkKWPushButton::New();
  this->RestoreButton->SetParent(this->GetParent());
  this->RestoreButton->Create();
  this->RestoreButton->SetText("Restore");
  this->RestoreButton->SetBalloonHelpString("Restore to original values.");
  this->RestoreButton->SetWidth(10);
  this->RestoreButton->SetEnabled(0);

  //pack restoreButton and undoButton
  this->Script("pack %s %s -side right -anchor n -padx 4 -pady 2", 
    this->RestoreButton->GetWidgetName(),
    this->UndoButton->GetWidgetName());
  } 
