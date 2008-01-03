#include "vtkVolumeRenderingCudaModuleGUI.h"
#include "vtkVolumeRenderingCudaModuleLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWPushButton.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkMRMLScene.h"
#include "vtkVolume.h"

#include "vtkImageReader.h"
#include "vtkPNGReader.h"
#include "vtkImageViewer.h"
#include "vtkImageData.h"

#include "vtkVolumeCudaMapper.h"
#include "vtkKWTypeChooserBox.h"
#include "vtkKWMatrixWidget.h"



/// TEMPORARY
#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>


vtkVolumeRenderingCudaModuleGUI::vtkVolumeRenderingCudaModuleGUI()
{
    this->LoadButton = NULL;
    this->CreatePiplineTestButton = NULL;
    this->CudaMapper = NULL;
    this->CudaActor = NULL;
    
    this->ImageReader = NULL;
    this->ImageViewer = NULL;
    this->InputTypeChooser = NULL;
    this->InputResolutionMatrix = NULL;
}


vtkVolumeRenderingCudaModuleGUI::~vtkVolumeRenderingCudaModuleGUI()
{
    if (this->LoadButton != NULL)
    {
        this->LoadButton->SetParent(NULL);
        this->LoadButton->Delete();
        this->LoadButton = NULL; 
    }
    
    if (this->CreatePiplineTestButton != NULL)
    {
      this->CreatePiplineTestButton->SetParent(NULL);
      this->CreatePiplineTestButton->Delete();
      this->CreatePiplineTestButton = NULL;  
    }
    
    if (this->CudaMapper != NULL)
    {
       this->CudaMapper->Delete();
    }
    if (this->CudaActor != NULL)
    {
      this->CudaActor->Delete();  
    }
    
    if (this->ImageReader != NULL)
     this->ImageReader->Delete();
    
    if (this->ImageViewer != NULL)
      this->ImageViewer->Delete();
    
    DeleteWidget(this->InputTypeChooser);
    DeleteWidget(this->InputResolutionMatrix);
}

void vtkVolumeRenderingCudaModuleGUI::DeleteWidget(vtkKWWidget* widget)
{
  if (widget != NULL)
  {
    widget->SetParent(NULL);
      widget->Delete();
  }
}

vtkVolumeRenderingCudaModuleGUI* vtkVolumeRenderingCudaModuleGUI::New()
{
    vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVolumeRenderingCudaModuleGUI");
    if (ret)
        return (vtkVolumeRenderingCudaModuleGUI*)ret;
    // If the Factory was unable to create the object, we do it ourselfes.
    return new vtkVolumeRenderingCudaModuleGUI();
}


void vtkVolumeRenderingCudaModuleGUI::BuildGUI ( )
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    this->GetUIPanel()->AddPage("VolumeRenderingCuda","VolumeRenderingCuda",NULL);

    // Define your help text and build the help frame here.
    const char *help = "VolumeRenderingCuda. 3D Segmentation This module is currently a prototype and will be under active development throughout 3DSlicer's Beta release.";
    const char *about = "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details.";
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "VolumeRenderingCuda" );
    this->BuildHelpAndAboutFrame ( page, help, about );
    //
    //Load and save
    //
    vtkSlicerModuleCollapsibleFrame *loadSaveDataFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    loadSaveDataFrame->SetParent (page);
    loadSaveDataFrame->Create();
    loadSaveDataFrame->ExpandFrame();
    loadSaveDataFrame->SetLabelText("Load and Save");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        loadSaveDataFrame->GetWidgetName(), page->GetWidgetName());

    this->LoadButton = vtkKWPushButton::New();
    this->LoadButton->SetParent(loadSaveDataFrame->GetFrame());
    this->LoadButton->Create();
    this->LoadButton->SetText("Load new Model");
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->LoadButton->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());


    this->CreatePiplineTestButton = vtkKWPushButton::New();
    this->CreatePiplineTestButton->SetParent(loadSaveDataFrame->GetFrame());
    this->CreatePiplineTestButton->Create();
    this->CreatePiplineTestButton->SetText("Test Creating the pipeline");
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->CreatePiplineTestButton->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());


    this->InputTypeChooser = vtkKWTypeChooserBox::New();
    this->InputTypeChooser->SetParent(loadSaveDataFrame->GetFrame());
    this->InputTypeChooser->Create();
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->InputTypeChooser->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());
  
    this->InputResolutionMatrix = vtkKWMatrixWidget::New();
    this->InputResolutionMatrix->SetParent(loadSaveDataFrame->GetFrame());
    this->InputResolutionMatrix->Create();
    this->InputResolutionMatrix->SetRestrictElementValueToInteger();
    this->InputResolutionMatrix->SetNumberOfColumns(4);
    this->InputResolutionMatrix->SetNumberOfRows(1);
    this->InputResolutionMatrix->SetElementValueAsInt(0,0,256);
    this->InputResolutionMatrix->SetElementValueAsInt(0,1,256);
    this->InputResolutionMatrix->SetElementValueAsInt(0,2,1);    
    this->InputResolutionMatrix->SetElementValueAsInt(0,3,4);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->InputResolutionMatrix->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());
  
    ////Testing Pushbutton
    //this->PB_Testing= vtkKWPushButton::New();
    //this->PB_Testing->SetParent(loadSaveDataFrame->GetFrame());
    //this->PB_Testing->Create();
    //this->PB_Testing->SetText("Make All Models Invisible");
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->PB_Testing->GetWidgetName());


    ////NodeSelector  for Node from MRML Scene
    //this->NS_ImageData=vtkSlicerNodeSelectorWidget::New();
    //this->NS_ImageData->SetParent(loadSaveDataFrame->GetFrame());
    //this->NS_ImageData->Create();
    //this->NS_ImageData->NoneEnabledOn();
    //this->NS_ImageData->SetLabelText("Source Volume: ");
    //this->NS_ImageData->SetNodeClass("vtkMRMLScalarVolumeNode","LabelMap","0","");
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_ImageData->GetWidgetName());

    ////NodeSelector for VolumeRenderingNode Preset
    //this->NS_VolumeRenderingDataSlicer=vtkSlicerNodeSelectorVolumeRenderingWidget::New();
    //this->NS_VolumeRenderingDataSlicer->SetParent(loadSaveDataFrame->GetFrame());
    //this->NS_VolumeRenderingDataSlicer->Create();
    //this->NS_VolumeRenderingDataSlicer->SetLabelText("Use Existing Visualization Parameterset: ");
    //this->NS_VolumeRenderingDataSlicer->EnabledOff();//By default off
    //this->NS_VolumeRenderingDataSlicer->NoneEnabledOn();
    //this->NS_VolumeRenderingDataSlicer->SetNodeClass("vtkMRMLVolumeRenderingNode","","","");
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_VolumeRenderingDataSlicer->GetWidgetName());

    ////NodeSelector for VolumeRenderingNode Scene
    //this->NS_VolumeRenderingDataScene=vtkSlicerNodeSelectorVolumeRenderingWidget::New();
    //this->NS_VolumeRenderingDataScene->SetParent(loadSaveDataFrame->GetFrame());
    //this->NS_VolumeRenderingDataScene->Create();
    //this->NS_VolumeRenderingDataScene->NoneEnabledOn();
    //this->NS_VolumeRenderingDataScene->SetLabelText("Current Visualization Parameterset: ");
    //this->NS_VolumeRenderingDataScene->EnabledOff();//By default off
    //this->NS_VolumeRenderingDataScene->SetNodeClass("vtkMRMLVolumeRenderingNode","","","");
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->NS_VolumeRenderingDataScene->GetWidgetName());
    ////Missing: Load from file

    ////Create New Volume Rendering Node
    ////Entry With Label
    //this->EWL_CreateNewVolumeRenderingNode=vtkKWEntryWithLabel::New();
    //this->EWL_CreateNewVolumeRenderingNode->SetParent(loadSaveDataFrame->GetFrame());
    //this->EWL_CreateNewVolumeRenderingNode->Create();
    //this->EWL_CreateNewVolumeRenderingNode->SetLabelText("Name for Visualization Parameterset: ");
    //this->EWL_CreateNewVolumeRenderingNode->EnabledOff();
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2", this->EWL_CreateNewVolumeRenderingNode->GetWidgetName());


    //this->PB_CreateNewVolumeRenderingNode=vtkKWPushButton::New();
    //this->PB_CreateNewVolumeRenderingNode->SetParent(loadSaveDataFrame->GetFrame());
    //this->PB_CreateNewVolumeRenderingNode->Create();
    //this->PB_CreateNewVolumeRenderingNode->SetText("Create Visualization Parameterset");
    //app->Script("pack %s -side top -anchor e -padx 2 -pady 2",this->PB_CreateNewVolumeRenderingNode->GetWidgetName());

    ////Details frame
    //this->detailsFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    //this->detailsFrame->SetParent (this->UIPanel->GetPageWidget("VolumeRendering"));
    //this->detailsFrame->Create();
    //this->detailsFrame->ExpandFrame();
    //this->detailsFrame->SetLabelText("Details");
    //app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    //    this->detailsFrame->GetWidgetName(), this->UIPanel->GetPageWidget("VolumeRendering")->GetWidgetName());


    ////set subnodes
    ////Delete frames
    //if ( this->GetApplicationGUI() &&  this->GetApplicationGUI()->GetMRMLScene())
    //{
    //    this->GetApplicationGUI()->GetMRMLScene()->AddObserver( vtkMRMLScene::SceneCloseEvent, this->MRMLCallbackCommand );
    //}
    //loadSaveDataFrame->Delete();
    
    this->Built=true;
}

void vtkVolumeRenderingCudaModuleGUI::TearDownGUI ( )
{
    this->Exit();
    if ( this->Built )
    {
        this->RemoveGUIObservers();
    }
}

void vtkVolumeRenderingCudaModuleGUI::CreateModuleEventBindings ( )
{
    vtkDebugMacro("VolumeRenderingCudaModule: CreateModuleEventBindings: No ModuleEventBindings yet");
}
void vtkVolumeRenderingCudaModuleGUI::ReleaseModuleEventBindings ( )
{
    vtkDebugMacro("VolumeRenderingCudaModule: ReleaseModuleEventBindings: No ModuleEventBindings to remove yet");
}

void vtkVolumeRenderingCudaModuleGUI::AddGUIObservers ( )
{
    this->LoadButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->CreatePiplineTestButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputTypeChooser->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputResolutionMatrix->AddObserver(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
}

void vtkVolumeRenderingCudaModuleGUI::RemoveGUIObservers ( )
{
    this->LoadButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->CreatePiplineTestButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputTypeChooser->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputResolutionMatrix->RemoveObservers(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
}
void vtkVolumeRenderingCudaModuleGUI::RemoveMRMLNodeObservers ( )
{
}
void vtkVolumeRenderingCudaModuleGUI::RemoveLogicObservers ( )
{
}

void vtkVolumeRenderingCudaModuleGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                                        void *callData )
{
   vtkDebugMacro("vtkVolumeRenderingModuleGUI::ProcessGUIEvents: event = " << event);

   if (caller == this->LoadButton)
   {
     if (this->CudaMapper == NULL)
       this->CudaMapper = vtkVolumeCudaMapper::New();
     if (this->CudaActor == NULL)
       {
         this->CudaActor = vtkVolume::New();
         this->CudaActor->SetMapper(this->CudaMapper);
       }
       
    this->TestCudaViewer();
    this->CudaMapper->Render(NULL, NULL);
   }
   
   if (caller == this->CreatePiplineTestButton)
   {
      this->CreatePipelineTest();
   }
   
   
   /// INPUT TYPE OR SIZE CHANGED CHANGED
   if (caller == this->InputTypeChooser->GetMenu() || caller == this->InputResolutionMatrix)
   {
     cerr << "Type" << this->InputTypeChooser->GetSelectedName() << " " << this->InputTypeChooser->GetSelectedType() << 
     " X:" << this->InputResolutionMatrix->GetElementValueAsInt(0,0) << 
     " Y:" << this->InputResolutionMatrix->GetElementValueAsInt(0,1) <<
     " Z:" << this->InputResolutionMatrix->GetElementValueAsInt(0,2) <<
     " A:" << this->InputResolutionMatrix->GetElementValueAsInt(0,3) << endl;
     
     TestCudaViewer();
     this->ImageReader->SetDataScalarType(this->InputTypeChooser->GetSelectedType());
     this->ImageReader->SetDataExtent(0, this->InputResolutionMatrix->GetElementValueAsInt(0,0), 
         0, this->InputResolutionMatrix->GetElementValueAsInt(0,1), 
         0, this->InputResolutionMatrix->GetElementValueAsInt(0,2));
     this->ImageReader->SetNumberOfScalarComponents(this->InputResolutionMatrix->GetElementValueAsInt(0,3));
     
     
  
     try {
       cerr << "ALLOCATE" << endl;  
     //this->ImageData->SetScalarType(this->InputTypeChooser->GetSelectedType());
     this->ImageData->SetExtent(0, this->InputResolutionMatrix->GetElementValueAsInt(0,0), 
         0, this->InputResolutionMatrix->GetElementValueAsInt(0,1), 
         0, this->InputResolutionMatrix->GetElementValueAsInt(0,2));
     this->ImageData->SetNumberOfScalarComponents(this->InputResolutionMatrix->GetElementValueAsInt(0,3));
     this->ImageData->SetScalarTypeToUnsignedChar();
     this->ImageData->AllocateScalars();
     
      cerr << "READ" << endl;
       FILE *fp;
      fp=fopen("/projects/igtdev/bensch/svn/volrenSample/output.raw","r");
      fread(this->ImageData->GetScalarPointer(), sizeof(unsigned char), 
        this->InputResolutionMatrix->GetElementValueAsInt(0,0) *
        this->InputResolutionMatrix->GetElementValueAsInt(0,1) *
        this->InputResolutionMatrix->GetElementValueAsInt(0,2) * 
        this->InputResolutionMatrix->GetElementValueAsInt(0,3), fp);
      fclose(fp);
      this->ImageViewer->SetInput(this->ImageData);
      
      cerr << "FINISHED" << endl;
     }
     catch (...)
     {
       cerr << "ERROR READING" << endl;  
     }
     this->ImageViewer->Render();
   }
}


#include "vtkImageReader.h"

void vtkVolumeRenderingCudaModuleGUI::TestCudaViewer()
{
if (ImageViewer == NULL)
 {
  printf("START CREATING WINDOW\n");


/// ImageReader from a File
  ImageReader = vtkImageReader::New();
  ImageReader->SetFileDimensionality(2);
  ImageReader->SetDataScalarTypeToUnsignedChar();
  ImageReader->SetNumberOfScalarComponents(4);
  ImageReader->SetHeaderSize(0);
  ImageReader->SetFileName("/projects/igtdev/bensch/svn/volrenSample/output.raw");
  
/// ImageData (from File) in Memory.
  this->ImageData = vtkImageData::New();
  //this->ImageData->SetDimensions(256, 256, 1);
  this->ImageData->SetScalarTypeToUnsignedChar();
  this->ImageData->SetNumberOfScalarComponents(4);

/// Setup the ImageViewer  
  this->ImageViewer = vtkImageViewer::New();  
  //this->ImageViewer->SetInputConnection(ImageReader->GetOutputPort());
  
  this->ImageViewer->SetColorWindow(255);
  this->ImageViewer->SetColorLevel(128);
  
  printf("FINISHED CREATING WINDOW\n");
 }
  this->ImageViewer->Render();
}

void vtkVolumeRenderingCudaModuleGUI::CreatePipelineTest()
{
  vtkImageReader* reader = vtkImageReader::New();
  reader->SetFileName("/projects/igtdev/bensch/svn/volrenSample/heart256.raw");
  
  
  
  
  reader->Delete();
}

void vtkVolumeRenderingCudaModuleGUI::ProcessMRMLEvents ( vtkObject *caller, unsigned long event,
                                                         void *callData)
{
}


void vtkVolumeRenderingCudaModuleGUI::SetViewerWidget(vtkSlicerViewerWidget *viewerWidget)
{
}
void vtkVolumeRenderingCudaModuleGUI::SetInteractorStyle(vtkSlicerViewerInteractorStyle *interactorStyle)
{
}


void vtkVolumeRenderingCudaModuleGUI::Enter ( )
{
    vtkDebugMacro("Enter Volume Rendering Cuda Module");

    if ( this->Built == false )
    {
        this->BuildGUI();
        this->AddGUIObservers();
    }
    this->CreateModuleEventBindings();
    //this->UpdateGUI();
}
void vtkVolumeRenderingCudaModuleGUI::Exit ( )
{
    vtkDebugMacro("Exit: removeObservers for VolumeRenderingModule");
    this->ReleaseModuleEventBindings();
}


void vtkVolumeRenderingCudaModuleGUI::PrintSelf(ostream& os, vtkIndent indent)
{
    this->SuperClass::PrintSelf(os, indent);

    os<<indent<<"vtkVolumeRenderingCudaModuleGUI"<<endl;
    os<<indent<<"vtkVolumeRenderingCudaModuleLogic"<<endl;
    if(this->GetLogic())
    {
        this->GetLogic()->PrintSelf(os,indent.GetNextIndent());
    }
}
