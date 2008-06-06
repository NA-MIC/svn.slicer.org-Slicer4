#include "vtkVolumeRenderingCudaModuleGUI.h"
#include "vtkVolumeRenderingCudaModuleLogic.h"
#include "vtkSlicerApplication.h"

#include "vtkImageGradientMagnitude.h"
#include "vtkColorTransferFunction.h"
#include "vtkVolumeProperty.h"
#include "vtkImageData.h"
#include "vtkVolume.h"
#include "vtkRenderer.h"
#include "vtkPointData.h"

// Widgets
#include "vtkKWPushButton.h"
#include "vtkKWTypeChooserBox.h"
#include "vtkKWMatrixWidget.h"
#include "vtkKWLabel.h"
#include "vtkKWVolumePropertyWidget.h"
#include "vtkKWEvent.h"
#include "vtkKWScale.h"
#include "vtkKWRange.h"
#include "vtkKWHistogramSet.h"
#include "vtkKWHistogram.h"

// MRML
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkMRMLScalarVolumeNode.h"

// vtkCuda
#include "vtkCudaVolumeMapper.h"
#include "vtkCudaMemoryTexture.h"
extern "C" {
#include "CUDA_renderAlgo.h"
}


vtkVolumeRenderingCudaModuleGUI::vtkVolumeRenderingCudaModuleGUI()
{
    this->CudaMapper = NULL;
    this->CudaVolume = NULL;
    this->CudaVolumeProperty = NULL;

    this->Histograms = NULL;

    this->RenderModeChooser = NULL;
    this->VolumePropertyWidget = NULL;

    this->ScaleFactorScale = NULL;

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

    DeleteWidget(this->RenderModeChooser);
    DeleteWidget(this->VolumePropertyWidget);
    DeleteWidget(this->ScaleFactorScale);
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

    //NodeSelector  for Node from MRML Scene
    this->NS_ImageData=vtkSlicerNodeSelectorWidget::New();
    this->NS_ImageData->SetParent(loadSaveDataFrame->GetFrame());
    this->NS_ImageData->Create();
    this->NS_ImageData->NoneEnabledOn();
    this->NS_ImageData->SetLabelText("Source Volume: ");
    this->NS_ImageData->SetNodeClass("vtkMRMLScalarVolumeNode","","","");
    app->Script("pack %s -side top -fill x -anchor nw -padx 2 -pady 2",this->NS_ImageData->GetWidgetName());

    this->VolumePropertyWidget = vtkKWVolumePropertyWidget::New();
    this->VolumePropertyWidget->SetParent(loadSaveDataFrame->GetFrame());
    this->VolumePropertyWidget->Create();
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->VolumePropertyWidget->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName()); 

    this->RenderModeChooser = vtkKWMenuButton::New();
    this->RenderModeChooser->SetParent(loadSaveDataFrame->GetFrame());
    this->RenderModeChooser->Create();
    this->RenderModeChooser->GetMenu()->AddRadioButton("To Texture");
    this->RenderModeChooser->GetMenu()->AddRadioButton("To Memory");
    this->RenderModeChooser->GetMenu()->SelectItem(0);
    app->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        this->RenderModeChooser->GetWidgetName(), loadSaveDataFrame->GetFrame()->GetWidgetName());  

    
    this->ScaleFactorScale = vtkKWScale::New();
    this->ScaleFactorScale->SetParent(loadSaveDataFrame->GetFrame());
    this->ScaleFactorScale->Create();
    this->ScaleFactorScale->SetRange(1.0f, 10.0f);
    this->ScaleFactorScale->SetResolution(.5f);
    this->ScaleFactorScale->SetValue(1.0f);
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->ScaleFactorScale->GetWidgetName()); 



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
    this->NS_ImageData->AddObserver(vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->RenderModeChooser->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);

    this->VolumePropertyWidget->AddObserver(vtkKWEvent::VolumePropertyChangedEvent, (vtkCommand*)this->GUICallbackCommand);

    this->ScaleFactorScale->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)this->GUICallbackCommand);

    this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::StartEvent,(vtkCommand *)this->GUICallbackCommand);
    this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderWindow()->AddObserver(vtkCommand::EndEvent,(vtkCommand *)this->GUICallbackCommand);
}

void vtkVolumeRenderingCudaModuleGUI::RemoveGUIObservers ( )
{
    this->RenderModeChooser->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
}
void vtkVolumeRenderingCudaModuleGUI::RemoveMRMLNodeObservers ( )
{
}

void vtkVolumeRenderingCudaModuleGUI::RemoveLogicObservers ( )
{
}


void vtkVolumeRenderingCudaModuleGUI::test()
{

    // Reading in the Data using a ImageReader
    //vtkImageReader* reader[5];
    for (unsigned int i = 0; i < 1; i++ ) 
    {
        //reader[i]= vtkImageReader::New();
        //reader[i]->SetDataScalarTypeToUnsignedChar();
        //reader[i]->SetNumberOfScalarComponents(1);
        //reader[i]->SetDataExtent(0, 255,
        //    0, 255, 
        //    0, 255);
        //reader[i]->SetFileDimensionality(3);

        //std::stringstream s;
        //s << "C:\\heart256-" << i+1 << ".raw";

        //reader[i]->SetFileName(s.str().c_str());
        //reader[i]->Update();

        ////this->CudaMapper->MultiInput[i] = reader[i]->GetOutput();
    }
    //this->CudaMapper->SetInput(reader[0]->GetOutput());
}

void vtkVolumeRenderingCudaModuleGUI::CreateMapper()
{
    if (this->CudaMapper == NULL)
    {
        this->CudaMapper = vtkCudaVolumeMapper::New();
    }
    if (this->CudaVolume == NULL)
    {
        this->CudaVolume = vtkVolume::New();
        this->CudaVolume->SetMapper(this->CudaMapper);

        this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderer()->AddVolume(this->CudaVolume);
    }

}

void vtkVolumeRenderingCudaModuleGUI::DeleteMapper()
{
    if (this->CudaVolume != NULL)
    {
        this->GetApplicationGUI()->GetViewerWidget()->GetMainViewer()->GetRenderer()->RemoveVolume(this->CudaVolume);
        this->CudaVolume->Delete();
        this->CudaVolume = NULL;
    }
    if (this->CudaMapper != NULL)
    {
        this->CudaMapper->Delete();
        this->CudaMapper = NULL;
    }
}

void vtkVolumeRenderingCudaModuleGUI::ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                                        void *callData )
{
    vtkDebugMacro("vtkVolumeRenderingModuleGUI::ProcessGUIEvents: event = " << event);

    if (caller == this->RenderModeChooser->GetMenu())
    {
        if (this->CudaMapper != NULL)
            if (!strcmp (this->RenderModeChooser->GetValue(), "To Texture"))
                this->CudaMapper->SetRenderMode(vtkCudaMemoryTexture::RenderToTexture);
            else
                this->CudaMapper->SetRenderMode(vtkCudaMemoryTexture::RenderToMemory);
    }

    else if (caller == this->NS_ImageData)
    {
        vtkMRMLScalarVolumeNode *selectedImageData=vtkMRMLScalarVolumeNode::SafeDownCast(this->NS_ImageData->GetSelected());
        if (selectedImageData != NULL && selectedImageData->GetImageData() != NULL)
        {
            this->CreateMapper();
            selectedImageData->GetImageData()->SetSpacing(selectedImageData->GetSpacing());

            this->CudaMapper->SetInput(selectedImageData->GetImageData());

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
        }
        else
            this->DeleteMapper();
    }

    else if (caller == this->VolumePropertyWidget)
    {
        if (this->CudaMapper != NULL)
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
    else if (caller == this->ScaleFactorScale)
    {
        if (this->CudaMapper != NULL)
            this->CudaMapper->SetRenderOutputScaleFactor(this->ScaleFactorScale->GetValue());
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

    this->NS_ImageData->SetMRMLScene(this->GetLogic()->GetMRMLScene());
    this->NS_ImageData->UpdateMenu();

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
