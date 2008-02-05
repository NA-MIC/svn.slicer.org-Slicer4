#include "vtkVolumeRenderingCudaModuleGUI.h"
#include "vtkVolumeRenderingCudaModuleLogic.h"
#include "vtkSlicerApplication.h"
#include "vtkKWWidget.h"
#include "vtkKWPushButton.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkMRMLScene.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include "vtkImageData.h"

#include "vtkVolumeCudaMapper.h"
#include "vtkKWTypeChooserBox.h"
#include "vtkKWMatrixWidget.h"
#include "vtkKWLabel.h"
#include "vtkKWVolumePropertyWidget.h"
#include "vtkKWEvent.h"
#include "vtkKWRange.h"

#include "vtkKWHistogramSet.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkKWHistogram.h"

#include "vtkCellArray.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkTexture.h"

// Transfer functions:
#include "vtkColorTransferFunction.h"

#include "vtkRenderer.h"

#include "CudappMemory.h"

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
    this->CudaVolume = NULL;
    this->CudaVolumeProperty = NULL;

    this->Histograms = NULL;

    this->InputTypeChooser = NULL;
    this->InputResolutionMatrix = NULL;
    this->RenderModeChooser = NULL;
    this->VolumePropertyWidget = NULL;

    this->ThresholdRange = NULL;

    this->RenderScheduled = false;
}


vtkVolumeRenderingCudaModuleGUI::~vtkVolumeRenderingCudaModuleGUI()
{
    if (this->CudaMapper != NULL)
    {
        this->CudaMapper->Delete();
    }
    if (this->CudaVolume != NULL)
    {
        this->CudaVolume->Delete();  
    }

    if (this->Histograms != NULL)
        this->Histograms->Delete();

    if (this->CudaVolumeProperty != NULL)
        this->CudaVolumeProperty->Delete();

    DeleteWidget(this->InputTypeChooser);
    DeleteWidget(this->InputResolutionMatrix);
    DeleteWidget(this->RenderModeChooser);
    DeleteWidget(this->VolumePropertyWidget);
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
    this->InputResolutionMatrix->SetElementValueAsInt(0,0,256);
    this->InputResolutionMatrix->SetElementValueAsInt(0,1,256);
    this->InputResolutionMatrix->SetElementValueAsInt(0,2,1);    
    this->InputResolutionMatrix->SetElementValueAsInt(0,3,4);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->InputResolutionMatrix->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());


    this->VolumePropertyWidget = vtkKWVolumePropertyWidget::New();
    this->VolumePropertyWidget->SetParent(loadSaveDataFrame->GetFrame());
    this->VolumePropertyWidget->Create();
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->VolumePropertyWidget->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName()); 

    this->ThresholdRange = vtkKWRange::New();
    this->ThresholdRange->SetParent(loadSaveDataFrame->GetFrame());
    this->ThresholdRange->Create();
    this->ThresholdRange->SetWholeRange(0, 255);
    this->ThresholdRange->SetRange(90, 255);
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->ThresholdRange->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName()); 

    this->RenderModeChooser = vtkKWMenuButton::New();
    this->RenderModeChooser->SetParent(loadSaveDataFrame->GetFrame());
    this->RenderModeChooser->Create();
    this->RenderModeChooser->GetMenu()->AddRadioButton("To Texture");
    this->RenderModeChooser->GetMenu()->AddRadioButton("To Memory");
    this->RenderModeChooser->GetMenu()->SelectItem(0);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->RenderModeChooser->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());  



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
    this->UpdateButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);

    this->RenderModeChooser->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);

    this->VolumePropertyWidget->AddObserver(vtkKWEvent::VolumePropertyChangedEvent, (vtkCommand*)this->GUICallbackCommand);

    this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::StartEvent,(vtkCommand *)this->GUICallbackCommand);
    this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::EndEvent,(vtkCommand *)this->GUICallbackCommand);

    this->ThresholdRange->AddObserver(vtkKWRange::RangeValueChangingEvent, (vtkCommand*)this->GUICallbackCommand);
}

void vtkVolumeRenderingCudaModuleGUI::RemoveGUIObservers ( )
{
    this->InputTypeChooser->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->InputResolutionMatrix->RemoveObservers(vtkKWMatrixWidget::ElementChangedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->UpdateButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand*)this->GUICallbackCommand);
    this->RenderModeChooser->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
}
void vtkVolumeRenderingCudaModuleGUI::RemoveMRMLNodeObservers ( )
{
}

void vtkVolumeRenderingCudaModuleGUI::RemoveLogicObservers ( )
{
}

#include "vtkImageReader.h"
#include "vtkOpenGLExtensionManager.h"
#include <sstream>

void vtkVolumeRenderingCudaModuleGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                                        void *callData )
{
    vtkDebugMacro("vtkVolumeRenderingModuleGUI::ProcessGUIEvents: event = " << event);

    /// INPUT TYPE OR SIZE CHANGED CHANGED
    if (

        caller == this->InputTypeChooser->GetMenu() ||
        caller == this->InputResolutionMatrix ||
        caller == this->ThresholdRange ||

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
            vtkImageReader* reader[5];
            for (unsigned int i = 0; i < 1; i++ ) 
            {
                reader[i]= vtkImageReader::New();
                reader[i]->SetDataScalarTypeToUnsignedChar();
                reader[i]->SetNumberOfScalarComponents(1);
                reader[i]->SetDataExtent(0, 255,
                    0, 255, 
                    0, 255);
                reader[i]->SetFileDimensionality(3);

                std::stringstream s;
                s << "C:\\heart256-" << i+1 << ".raw";

                reader[i]->SetFileName(s.str().c_str());
                reader[i]->Update();

                //this->CudaMapper->MultiInput[i] = reader[i]->GetOutput();
            }
            this->CudaMapper->SetInput(reader[0]->GetOutput());


        }
        if (this->CudaVolume == NULL)
        {
            this->CudaVolume = vtkVolume::New();
            this->CudaVolume->SetMapper(this->CudaMapper);

            this->CudaVolumeProperty = vtkVolumeProperty::New();
            this->CudaVolume->SetProperty(this->CudaVolumeProperty);
            this->VolumePropertyWidget->SetVolumeProperty(this->CudaVolumeProperty);
            this->VolumePropertyWidget->ScalarOpacityUnitDistanceVisibilityOff ();
            //this->VolumePropertyWidget->SetDataSet(this->CudaMapper->GetInput()->GetImageData());
            this->Histograms = vtkKWHistogramSet::New();

            //Add Histogram for image data
            //this->Histograms->AddHistograms(this->CudaMapper->GetInput()->GetPointData()->GetScalars());
            vtkKWHistogram *histo = vtkKWHistogram::New();
            histo->BuildHistogram(this->CudaMapper->GetInput()->GetPointData()->GetScalars(),0);
            this->Histograms->AddHistogram(histo,"0");

            
            //Build the gradient histogram
            vtkImageGradientMagnitude *grad = vtkImageGradientMagnitude::New();
            grad->SetDimensionality(3);
            grad->SetInput(this->CudaMapper->GetInput());
            grad->Update();
            vtkKWHistogram *gradHisto = vtkKWHistogram::New();
            gradHisto->BuildHistogram(grad->GetOutput()->GetPointData()->GetScalars(),0);
            this->Histograms->AddHistogram(gradHisto,"0gradient");

            //Delete      
            this->VolumePropertyWidget->SetHistogramSet(this->Histograms);

            grad->Delete();
            gradHisto->Delete();

            this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderer()->AddVolume(this->CudaVolume);
        }
        this->CudaMapper->SetThreshold(this->ThresholdRange->GetRange());

        this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
        // this->UpdateVolume();
        //this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
    }
    else if (caller == this->RenderModeChooser->GetMenu())
    {
        if (this->CudaMapper != NULL)
            if (!strcmp (this->RenderModeChooser->GetValue(), "To Texture"))
                this->CudaMapper->SetRenderMode(vtkVolumeCudaMapper::RenderToTexture);
            else
                this->CudaMapper->SetRenderMode(vtkVolumeCudaMapper::RenderToMemory);
    }

    else if (caller == this->VolumePropertyWidget)
    {
        if (this->CudaMapper != NULL);
        this->CudaMapper->Update();
        this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
    }

    else if(this->CudaMapper != NULL &&
        caller==this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow() && event==vtkCommand::EndEvent)
    {

        if (this->RenderScheduled == false)
        {    this->RenderScheduled=true;
        this->Script("after 20 %s ScheduleRender",this->GetTclName());//[[[$::slicer3::ApplicationGUI GetViewerWidget] GetMainViewer] GetRenderWindow] Render");
        }

    }


}

void vtkVolumeRenderingCudaModuleGUI::ScheduleRender()
{
    this->RenderScheduled = false;
    this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->Render();
}

void vtkVolumeRenderingCudaModuleGUI::ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData)
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
