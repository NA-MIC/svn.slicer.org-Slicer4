
#include "vtkUltrasoundGUI.h"
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

#include "vtkCudaVolumeMapper.h"
#include "vtkFixedPointVolumeRayCastMapper.h"
#include "vtkVolumeRayCastMapper.h"
#include "vtkVolumeTextureMapper2D.h"
#include "vtkVolumeTextureMapper3D.h"

vtkCxxRevisionMacro(vtkUltrasoundGUI, "$ Revision 1.0$");
vtkStandardNewMacro(vtkUltrasoundGUI);


vtkUltrasoundGUI::vtkUltrasoundGUI()
{
    // Initialize Values
    this->Volume = NULL;
    this->VolumeMapper = NULL;
    this->VolumePropertyWidget = NULL;

    this->frameNumber = 0;
    this->readers;
    this->renderScheduled = false;
    this->isRendering = false;


    

}

void vtkUltrasoundGUI::CreateWidget()
{
    this->Superclass::CreateWidget();

    vtkKWApplication* app = this->GetApplication();
    // Add a render widget, attach it to the view frame, and pack
    this->renderWidget = vtkKWRenderWidget::New();
    this->renderWidget->SetBackgroundColor(200, 200, 200);
    this->renderWidget->SetParent(this->GetViewFrame());
    this->renderWidget->Create();
    this->GetApplication()->Script("pack %s -expand y -fill both -anchor c -expand y", 
        this->renderWidget->GetWidgetName());

    // Create the mapper and actor
    this->Volume = vtkVolume::New();
    this->VolumeMapper = vtkVolumeTextureMapper2D::New();
    this->Volume->SetMapper(VolumeMapper);

    this->LoadUltrasoundHeartSeries("D:\\Volumes\\4DUltrasound\\3DDCM002.raw");
    if (!readers.empty())
        this->VolumeMapper->SetInput(readers[0]->GetOutput());

    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    this->Volume->SetProperty(prop);
    this->renderWidget->GetRenderer()->AddVolume(Volume);

    /// GUI EVENT
    this->GUICallbackCommand = vtkCallbackCommand::New ( );
    GUICallbackCommand->SetCallback( vtkUltrasoundGUI::GuiEventStatic );
    GUICallbackCommand->SetClientData(this);

    this->StartCallbackCommand = vtkCallbackCommand::New ( );
    StartCallbackCommand->SetCallback( vtkUltrasoundGUI::RenderBeginStatic );
    StartCallbackCommand->SetClientData(this);

    this->StopCallbackCommand = vtkCallbackCommand::New ( );
    StopCallbackCommand->SetCallback( vtkUltrasoundGUI::RenderEndStatic );
    StopCallbackCommand->SetClientData(this);



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

    this->cb_Animate = vtkKWCheckButton::New();
    this->cb_Animate->SetParent(this->GetMainPanelFrame());
    this->cb_Animate->Create();
    this->GetApplication()->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        this->cb_Animate->GetWidgetName());
    this->cb_Animate->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)GUICallbackCommand);


    this->renderWidget->GetRenderer()->GetActiveCamera()->SetPosition(500, 500, 500);
    this->renderWidget->GetRenderer()->GetActiveCamera()->SetClippingRange(10, 1000);
    this->renderWidget->GetRenderer()->GetActiveCamera()->ParallelProjectionOff();

    vtkInteractorStyle* interactorStyle = vtkInteractorStyleTrackballCamera::New();
    renderWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(interactorStyle);
    renderWidget->SetRenderModeToInteractive();

    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::StartEvent, (vtkCommand*)StartCallbackCommand);
    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::EndEvent, (vtkCommand*)StopCallbackCommand);
}

vtkUltrasoundGUI::~vtkUltrasoundGUI()
{
//    
    // Deallocate and exit
    StartCallbackCommand->Delete();
    StopCallbackCommand->Delete();
    GUICallbackCommand->Delete();

    Volume->Delete();
    VolumeMapper->Delete();

    while(!readers.empty())
    {
        readers.back()->Delete();
        readers.pop_back();
    }

    VolumePropertyWidget->Delete();
    cb_Animate->Delete();
    renderWidget->Delete();

}


void vtkUltrasoundGUI::LoadUltrasoundHeartSeries(void* data)
{
    const char* fileName = (const char*)data;
    std::cout << "Loading " << std::flush;
    int j = 92;
    for (unsigned int i = 0; i < 50; i ++)
    {
        readers.push_back(vtkImageReader::New());
        readers[i]->SetDataScalarTypeToUnsignedChar();
        readers[i]->SetNumberOfScalarComponents(1);
        readers[i]->SetDataExtent(0, 79, 0, 79, 0, 159);
        readers[i]->SetFileDimensionality(3);
        readers[i]->SetDataSpacing(1.0f, 1.0f, 0.74f);
        readers[i]->SetHeaderSize(i * 80 * 80 * 160 * sizeof(unsigned char));

        readers[i]->SetFileName(fileName);
        readers[i]->Update();
        std::cout << "." << std::flush;
    }
    std::cout << "done" <<  std::endl;
}

void vtkUltrasoundGUI::ReRender()
{
    if (!renderScheduled)
    {
        renderScheduled = true;
        this->GetApplication()->Script("after 100 %s Render", renderWidget->GetTclName());
    }
}


void vtkUltrasoundGUI::GuiEventStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundGUI::SafeDownCast((vtkObject*)clientData)->GuiEvent(caller);
}
void vtkUltrasoundGUI::RenderBeginStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundGUI::SafeDownCast((vtkObject*)clientData)->RenderBegin();
}
void vtkUltrasoundGUI::RenderEndStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    vtkUltrasoundGUI::SafeDownCast((vtkObject*)clientData)->RenderEnd();
}


void vtkUltrasoundGUI::GuiEvent(vtkObject* caller)
{
    if (caller == cb_Animate && cb_Animate->GetSelectedState() == 1) 
    {
        ReRender();
    }
    else if (caller == VolumePropertyWidget)
    {
        ReRender();
    }
}

void vtkUltrasoundGUI::RenderBegin()
{
    if (isRendering == true)
        printf("DAMN\n");
    isRendering = true;
}

void vtkUltrasoundGUI::RenderEnd()
{
    renderScheduled = false;
    if (!renderScheduled && cb_Animate->GetSelectedState() == 1)
    {
        if (++frameNumber >= readers.size())
            frameNumber = 0;
        VolumeMapper->SetInput(readers[frameNumber]->GetOutput());
        ReRender();
    }
    isRendering = false;
}
