#include "vtkVolumeRenderingCudaModuleGUI.h"
#include "vtkVolumeRenderingCudaModuleLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWPushButton.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkMRMLScene.h"
#include "vtkVolume.h"

#include "vtkImageData.h"

#include "vtkVolumeCudaMapper.h"
#include "vtkKWTypeChooserBox.h"
#include "vtkKWMatrixWidget.h"
#include "vtkKWLabel.h"

#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkTexture.h"

#include "vtkRenderer.h"

#include "vtkCudaMemory.h"

extern "C" {
#include "CUDA_renderAlgo.h"
}
/// TEMPORARY
#include <stdio.h>
#include <stdlib.h>
#include <cutil.h>


vtkVolumeRenderingCudaModuleGUI::vtkVolumeRenderingCudaModuleGUI()
{
    this->CudaMapper = NULL;
    this->CudaActor = NULL;

    this->InputTypeChooser = NULL;
    this->InputResolutionMatrix = NULL;
    this->Color = NULL;
}


vtkVolumeRenderingCudaModuleGUI::~vtkVolumeRenderingCudaModuleGUI()
{
    if (this->CudaMapper != NULL)
    {
        this->CudaMapper->Delete();
    }
    if (this->CudaActor != NULL)
    {
        this->CudaActor->Delete();  
    }

    DeleteWidget(this->InputTypeChooser);
    DeleteWidget(this->InputResolutionMatrix);
    DeleteWidget(this->Color);
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
    this->InputResolutionMatrix->SetElementValueAsInt(0,0,128);
    this->InputResolutionMatrix->SetElementValueAsInt(0,1,128);
    this->InputResolutionMatrix->SetElementValueAsInt(0,2,1);    
    this->InputResolutionMatrix->SetElementValueAsInt(0,3,4);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->InputResolutionMatrix->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());


    vtkKWLabel* label = vtkKWLabel::New();
    label->SetParent(loadSaveDataFrame->GetFrame());
    label->Create();
    label->SetText("Color:");
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        label->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());


    this->Color = vtkKWMatrixWidget::New();
    this->Color->SetParent(loadSaveDataFrame->GetFrame());
    this->Color->Create();
    this->Color->SetRestrictElementValueToInteger();
    this->Color->SetNumberOfColumns(3);
    this->Color->SetNumberOfRows(1);
    this->Color->SetElementValueAsInt(0,0,255);
    this->Color->SetElementValueAsInt(0,1,255);
    this->Color->SetElementValueAsInt(0,2,255);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->Color->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());  


    this->UpdateButton = vtkKWPushButton::New();
    this->UpdateButton->SetParent(loadSaveDataFrame->GetFrame());
    this->UpdateButton->Create();
    this->UpdateButton->SetText("Update Renderer");
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->UpdateButton->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());

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
    this->InputTypeChooser->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputResolutionMatrix->AddObserver(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->Color->AddObserver(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->UpdateButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
}

void vtkVolumeRenderingCudaModuleGUI::RemoveGUIObservers ( )
{
    this->InputTypeChooser->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputResolutionMatrix->RemoveObservers(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->Color->RemoveObservers(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->UpdateButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
}
void vtkVolumeRenderingCudaModuleGUI::RemoveMRMLNodeObservers ( )
{
}
void vtkVolumeRenderingCudaModuleGUI::RemoveLogicObservers ( )
{
}

#include "vtkImageReader.h"
void vtkVolumeRenderingCudaModuleGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                                        void *callData )
{
    vtkDebugMacro("vtkVolumeRenderingModuleGUI::ProcessGUIEvents: event = " << event);

    /// INPUT TYPE OR SIZE CHANGED CHANGED
    if (
        /*
        caller == this->InputTypeChooser->GetMenu() ||
        caller == this->InputResolutionMatrix ||
        caller == this->Color ||
        */
        caller == this->UpdateButton
        )
    {
        cerr << "Type" << this->InputTypeChooser->GetSelectedName() << " " << this->InputTypeChooser->GetSelectedType() << 
            " X:" << this->InputResolutionMatrix->GetElementValueAsInt(0,0) << 
            " Y:" << this->InputResolutionMatrix->GetElementValueAsInt(0,1) <<
            " Z:" << this->InputResolutionMatrix->GetElementValueAsInt(0,2) <<
            " A:" << this->InputResolutionMatrix->GetElementValueAsInt(0,3) << endl;


        // create required objects
        if (this->CudaMapper == NULL)
        {
            this->CudaMapper = vtkVolumeCudaMapper::New();

            // Reading in the Data using a ImageReader
            vtkImageReader* reader = vtkImageReader::New();
            reader->SetDataScalarTypeToUnsignedChar();
            reader->SetNumberOfScalarComponents(1);
            reader->SetDataExtent(0, 255, 
                0, 255, 
                0, 255);
            reader->SetFileDimensionality(3);
            // reader->SetNumberOfScalarComponents(1);

            reader->SetFileName("C:\\Documents and Settings\\bensch\\Desktop\\svn\\orxonox\\subprojects\\volrenSample\\heart256.raw");
            reader->Update();

            vtkImageData* data = reader->GetOutput();
            this->CudaMapper->SetInput(data);
        }
        if (this->CudaActor == NULL)
        {
            this->CudaActor = vtkVolume::New();
            this->CudaActor->SetMapper(this->CudaMapper);
        }
        this->CudaMapper->SetColor(this->Color->GetElementValueAsInt(0,0), this->Color->GetElementValueAsInt(0,1), this->Color->GetElementValueAsInt(0,2));
        this->CudaMapper->Render(
            this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderer(),
            this->CudaActor);
        this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
    }
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
