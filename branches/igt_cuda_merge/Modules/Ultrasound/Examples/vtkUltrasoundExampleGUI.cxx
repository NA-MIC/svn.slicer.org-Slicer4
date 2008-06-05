#include "vtkUltrasoundExampleGUI.h"
#include "vtkObjectFactory.h"


#include "vtkKWRenderWidget.h"
#include "vtkRenderWindow.h"
#include "vtkKWApplication.h"

#include "vtkKWVolumePropertyWidget.h"
#include "vtkCallbackCommand.h"
#include "vtkKWEvent.h"
#include "vtkKWCheckButton.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenu.h"


#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkImageReader.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include "vtkImageData.h"

#include "vtkInteractorStyleTrackballCamera.h"

#include "vtkUltrasoundStreamerGUI.h"
#include "vtkUltrasoundStreamSource.h"

#include "vtkUltrasoundToolGUI.h"

#include "vtkCudaVolumeMapper.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include "vtkVolumeRayCastMapper.h"
#include "vtkVolumeTextureMapper2D.h"
#include "vtkVolumeTextureMapper3D.h"

vtkCxxRevisionMacro(vtkUltrasoundExampleGUI, "$ Revision 1.0$");
vtkStandardNewMacro(vtkUltrasoundExampleGUI);


vtkUltrasoundExampleGUI::vtkUltrasoundExampleGUI()
{
    // Initialize Values
    this->Volume = NULL;
    this->VolumeMapper = NULL;
    this->VolumePropertyWidget = NULL;
    this->UltrasoundStreamerGUI = NULL;
    this->UltrasoundToolGUI = NULL;
    this->ImageData = NULL;

    this->renderScheduled = false;
    this->isRendering = false;
}

void vtkUltrasoundExampleGUI::CreateWidget()
{
    this->Superclass::CreateWidget();

    this->ImageData = vtkImageData::New();
    this->ImageData->SetScalarTypeToUnsignedChar();
    this->ImageData->SetExtent(0, 9, 0, 9, 0, 9);
    this->ImageData->SetNumberOfScalarComponents(1);
    this->ImageData->SetSpacing(1.0f, 1.0f, 0.74f);
    this->ImageData->AllocateScalars();
    memset(this->ImageData->GetScalarPointer(), 0, 10*10*10*this->ImageData->GetScalarSize());;

    vtkKWApplication* app = this->GetApplication();
    // Add a render widget, attach it to the view frame, and pack
    this->renderWidget = vtkKWRenderWidget::New();
    this->renderWidget->SetParent(this->GetViewFrame());
    this->renderWidget->Create();
    this->renderWidget->GetRenderer()->SetBackground(0, 0, 0);
    //this->GetApplication()->Script("pack %s -expand y -fill both -anchor nw", 
    //    this->renderWidget->GetWidgetName());

    this->renderWidget2 = vtkKWRenderWidget::New();
    this->renderWidget2->SetParent(this->GetViewFrame());
    this->renderWidget2->Create();
    this->GetApplication()->Script("pack %s %s -expand y -fill both -anchor w", 
        this->renderWidget->GetWidgetName(),
        this->renderWidget2->GetWidgetName());

    /// GUI EVENT
    this->GUICallbackCommand = vtkCallbackCommand::New ( );
    this->GUICallbackCommand->SetCallback( vtkUltrasoundExampleGUI::GuiEventStatic );
    this->GUICallbackCommand->SetClientData(this);

    this->StartCallbackCommand = vtkCallbackCommand::New ( );
    this->StartCallbackCommand->SetCallback( vtkUltrasoundExampleGUI::RenderBeginStatic );
    this->StartCallbackCommand->SetClientData(this);

    this->StopCallbackCommand = vtkCallbackCommand::New ( );
    this->StopCallbackCommand->SetCallback( vtkUltrasoundExampleGUI::RenderEndStatic );
    this->StopCallbackCommand->SetClientData(this);

    this->UltrasoundCommand = vtkCallbackCommand::New ( );
    this->UltrasoundCommand->SetCallback(vtkUltrasoundExampleGUI::UltrasoundEventStatic);
    this->UltrasoundCommand->SetClientData(this);

    // RenderWindow Setup    
    this->renderWidget->GetRenderer()->GetActiveCamera()->SetPosition(500, 500, 500);
    this->renderWidget->GetRenderer()->GetActiveCamera()->SetClippingRange(10, 1000);
    this->renderWidget->GetRenderer()->GetActiveCamera()->ParallelProjectionOff();

    vtkInteractorStyle* interactorStyle = vtkInteractorStyleTrackballCamera::New();
    renderWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(interactorStyle);
    renderWidget->SetRenderModeToInteractive();

    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::StartEvent, (vtkCommand*)StartCallbackCommand);
    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::EndEvent, (vtkCommand*)StopCallbackCommand);

    this->CreateUltrasoundWidget();
    this->CreateToolWidget();
    this->CreateVolumeRenderingWidget();
}
void vtkUltrasoundExampleGUI::CreateUltrasoundWidget()
{
    this->UltrasoundStreamerGUI = vtkUltrasoundStreamerGUI::New();
    this->UltrasoundStreamerGUI->SetParent(this->GetMainPanelFrame());
    this->UltrasoundStreamerGUI->Create();
    this->GetApplication()->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->UltrasoundStreamerGUI->GetWidgetName());
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::EnablingEvent, (vtkCommand*)this->UltrasoundCommand);
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::DisabledEvent, (vtkCommand*)this->UltrasoundCommand);
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamerGUI::DataUpdatedEvent, (vtkCommand*)this->UltrasoundCommand);
    this->UltrasoundStreamerGUI->AddObserver(vtkUltrasoundStreamSource::ConnectionEstablished, (vtkCommand*)this->UltrasoundCommand);
}

void vtkUltrasoundExampleGUI::CreateToolWidget()
{
    this->UltrasoundToolGUI = vtkUltrasoundToolGUI::New();
    this->UltrasoundToolGUI->SetParent(this->GetMainPanelFrame());
    this->UltrasoundToolGUI->Create();
    this->GetApplication()->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->UltrasoundToolGUI->GetWidgetName());
    this->UltrasoundToolGUI->SetRenderer(this->renderWidget->GetRenderer());

}

void vtkUltrasoundExampleGUI::CreateVolumeRenderingWidget()
{
    /////////////////////////////
    /// VOLUME RENDERING PART ///
    /////////////////////////////
    // Create the mapper and actor
    this->Volume = vtkVolume::New();
    //this->VolumeMapper = vtkVolumeTextureMapper2D::New();
    this->VolumeMapper = vtkCudaVolumeMapper::New();
    this->VolumeMapper->SetInput(this->ImageData);
    this->Volume->SetMapper(VolumeMapper);


    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    this->Volume->SetProperty(prop);
    this->renderWidget->GetRenderer()->AddVolume(Volume);


    //mb_Mapper = vtkKWMenuButton::New();
    //mb_Mapper->SetParent(win->GetMainPanelFrame());
    //mb_Mapper->Create();
    //mb_Mapper->GetMenu()->AddRadioButton("Software_Ray_Caster");
    //mb_Mapper->GetMenu()->AddRadioButton("Texture_2D");
    //mb_Mapper->GetMenu()->AddRadioButton("Texture_3D");
    //mb_Mapper->GetMenu()->AddRadioButton("CUDA");
    //mb_Mapper->SetValue("CUDA");
    //mb_Mapper->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)MapperCallbackCommand);

    //app->Script("pack %s -side top -anchor nw -expand n -fill x -pady 2",
    //    mb_Mapper->GetWidgetName()); 


    //Volume Property
    VolumePropertyWidget = vtkKWVolumePropertyWidget::New();
    VolumePropertyWidget->SetParent(this->GetMainPanelFrame());
    VolumePropertyWidget->Create();
    this->GetApplication()->Script( "pack %s -side top -anchor nw -expand n -fill x -pady 2",
        VolumePropertyWidget->GetWidgetName()); 
    VolumePropertyWidget->AddObserver(vtkKWEvent::VolumePropertyChangedEvent, (vtkCommand*)GUICallbackCommand);

    VolumePropertyWidget->SetVolumeProperty(prop);
    prop->Delete();


    //ScaleFactorScale = vtkKWScale::New();
    //ScaleFactorScale->SetParent(win->GetMainPanelFrame());
    //ScaleFactorScale->Create();
    //ScaleFactorScale->SetRange(1.0f, 10.0f);
    //ScaleFactorScale->SetResolution(.5f);
    //ScaleFactorScale->SetValue(1.0f);
    //this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
    //    ScaleFactorScale->GetWidgetName()); 
    //ScaleFactorScale->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)GUICallbackCommand);
}

vtkUltrasoundExampleGUI::~vtkUltrasoundExampleGUI()
{
    this->StartCallbackCommand->Delete();
    this->StopCallbackCommand->Delete();
    this->GUICallbackCommand->Delete();

    this->UltrasoundToolGUI->Delete();

    this->Volume->Delete();
    this->VolumeMapper->Delete();

    this->VolumePropertyWidget->Delete();
    this->renderWidget->Delete();
    this->renderWidget2->Delete();
}


void vtkUltrasoundExampleGUI::ScheduleRender()
{
    if (this->renderScheduled == false)
    {
        //if (cb_Animate->GetSelectedState() == 1)
        //{
        //    if (++frameNumber >= readers.size())
        //        frameNumber = 0;
        //    VolumeMapper->SetInput(readers[frameNumber]->GetOutput());
        //}
        this->renderScheduled = true;
        this->renderWidget->Render();
    }
}

void vtkUltrasoundExampleGUI::GuiEventStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundExampleGUI::SafeDownCast((vtkObject*)clientData)->GuiEvent(caller);
}
void vtkUltrasoundExampleGUI::RenderBeginStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundExampleGUI::SafeDownCast((vtkObject*)clientData)->RenderBegin();
}
void vtkUltrasoundExampleGUI::RenderEndStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundExampleGUI::SafeDownCast((vtkObject*)clientData)->RenderEnd();
}

void vtkUltrasoundExampleGUI::UltrasoundEventStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundExampleGUI::SafeDownCast((vtkObject*)clientData)->UltrasoundEvent(eid);
}

void vtkUltrasoundExampleGUI::GuiEvent(vtkObject* caller)
{
    this->ScheduleRender();
}

void vtkUltrasoundExampleGUI::RenderBegin()
{
    this->renderScheduled = true;
}

void vtkUltrasoundExampleGUI::RenderEnd()
{
    this->renderScheduled=false;
    //this->Script("after idle %s ScheduleRender", this->GetTclName());
    //[[[$::slicer3::ApplicationGUI GetViewerWidget] GetMainViewer] GetRenderWindow] Render");
}

void vtkUltrasoundExampleGUI::UltrasoundEvent(unsigned long ev)
{
    switch (ev)
    {
    case vtkUltrasoundStreamerGUI::DataUpdatedEvent:
        this->UltrasoundToolGUI->UpdateTracker();
        this->ScheduleRender();
        break;
    case vtkUltrasoundStreamerGUI::EnablingEvent:
        this->UltrasoundStreamerGUI->SetImageData(this->ImageData);
        break;
    case vtkUltrasoundStreamerGUI::DisabledEvent:
        break;
    case vtkUltrasoundStreamSource::ConnectionEstablished:
        //this->VolumeMapper->Delete();
        //this->VolumeMapper = vtkCudaVolumeMapper::New();
        //this->Volume->SetMapper(VolumeMapper);

        this->VolumeMapper->SetInput(this->ImageData);
        break;
    default:
        break;
    }
}
