#include "vtkKWApplication.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWWindow.h"
#include "vtkRenderWindow.h"

#include "vtkKWVolumePropertyWidget.h"
#include "vtkKWRange.h"
#include "vtkCallbackCommand.h"
#include "vtkKWEvent.h"
#include "vtkKWCheckButton.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenu.h"

#include <vtksys/SystemTools.hxx>
#include <vtksys/CommandLineArguments.hxx>

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

vtkKWApplication *app;
vtkKWRenderWidget* renderWidget;
vtkKWCheckButton* cb_Animate;


vtkVolume* Volume = NULL;
vtkVolumeMapper* VolumeMapper = NULL;
vtkKWVolumePropertyWidget* VolumePropertyWidget = NULL;

unsigned int frameNumber = 0;
std::vector<vtkImageReader*> readers;
bool renderScheduled = false;
bool isRendering = false;

void LoadUltrasoundHeartSeries(void* data)
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

void ReRender()
{
    if (!renderScheduled)
    {
        renderScheduled = true;
        app->Script("after 100 %s Render", renderWidget->GetTclName());
    }
}

void GuiEvent(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
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

void RenderBegin(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    if (isRendering == true)
        printf("DAMN\n");
    isRendering = true;
}

void RenderEnd(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
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

int my_main(int argc, char *argv[])
{
    // Initialize Tcl
    Tcl_Interp *interp = vtkKWApplication::InitializeTcl(argc, argv, &cerr);
    if (!interp)
    {
        cerr << "Error: InitializeTcl failed" << endl ;
        return 1;
    }

    // Process some command-line arguments
    // The --test option here is used to run this example as a non-interactive 
    // test for software quality purposes. You can ignore it.
    int option_test = 0;
    vtksys::CommandLineArguments args;
    args.Initialize(argc, argv);
    args.AddArgument(
        "--test", vtksys::CommandLineArguments::NO_ARGUMENT, &option_test, "");
    args.Parse();

    app = vtkKWApplication::New();
    app->SetName("Ultrasound Heart Viewer");
    if (option_test)
    {
        app->SetRegistryLevel(0);
        app->PromptBeforeExitOff();
    }
    app->RestoreApplicationSettingsFromRegistry();

    // Set a help link. Can be a remote link (URL), or a local file

    app->SetHelpDialogStartingPage("http://www.kwwidgets.org");

    // Add a window
    // Set 'SupportHelp' to automatically add a menu entry for the help link
    vtkKWWindow *win = vtkKWWindow::New();
    app->AddWindow(win);
    win->Create();
    win->SecondaryPanelVisibilityOff();

    // Add a render widget, attach it to the view frame, and pack
    renderWidget = vtkKWRenderWidget::New();
    renderWidget->SetBackgroundColor(200, 200, 200);
    renderWidget->SetParent(win->GetViewFrame());
    renderWidget->Create();
    app->Script("pack %s -expand y -fill both -anchor c -expand y", 
        renderWidget->GetWidgetName());


    // Create the mapper and actor
    Volume = vtkVolume::New();
    VolumeMapper = vtkVolumeTextureMapper2D::New();
    Volume->SetMapper(VolumeMapper);

    LoadUltrasoundHeartSeries("D:\\Volumes\\4DUltrasound\\3DDCM002.raw");
    if (!readers.empty())
        VolumeMapper->SetInput(readers[0]->GetOutput());

    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    Volume->SetProperty(prop);
    renderWidget->GetRenderer()->AddVolume(Volume);

    /// GUI EVENT
    vtkCallbackCommand* GUICallbackCommand = vtkCallbackCommand::New ( );
    GUICallbackCommand->SetCallback( GuiEvent );

    vtkCallbackCommand* StartCallbackCommand = vtkCallbackCommand::New ( );
    StartCallbackCommand->SetCallback( RenderBegin );

    vtkCallbackCommand* StopCallbackCommand = vtkCallbackCommand::New ( );
    StopCallbackCommand->SetCallback( RenderEnd );



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
    VolumePropertyWidget->SetParent(win->GetMainPanelFrame());
    VolumePropertyWidget->Create();
    app->Script( "pack %s -side top -anchor nw -expand n -fill x -pady 2",
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
    //app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
    //    ScaleFactorScale->GetWidgetName()); 
    //ScaleFactorScale->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)GUICallbackCommand);

    cb_Animate = vtkKWCheckButton::New();
    cb_Animate->SetParent(win->GetMainPanelFrame());
    cb_Animate->Create();
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        cb_Animate->GetWidgetName());
    cb_Animate->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)GUICallbackCommand);


    renderWidget->GetRenderer()->GetActiveCamera()->SetPosition(500, 500, 500);
    renderWidget->GetRenderer()->GetActiveCamera()->SetClippingRange(10, 1000);
    renderWidget->GetRenderer()->GetActiveCamera()->ParallelProjectionOff();

    vtkInteractorStyle* interactorStyle = vtkInteractorStyleTrackballCamera::New();
    renderWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(interactorStyle);
    renderWidget->SetRenderModeToInteractive();

    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::StartEvent, (vtkCommand*)StartCallbackCommand);
    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::EndEvent, (vtkCommand*)StopCallbackCommand);

    int ret = 0;
    win->Display();
    if (!option_test)
    {
        app->Start(argc, argv);
        ret = app->GetExitStatus();
    }
    renderWidget->GetRenderer()->RemoveVolume(Volume);
    win->Close();

    interactorStyle->Delete();
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
    win->Delete();
    app->Delete();

    return ret;
}

/*#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
int __stdcall WinMain(HINSTANCE, HINSTANCE, LPSTR lpCmdLine, int)
{
int argc;
char **argv;
vtksys::SystemTools::ConvertWindowsCommandLineToUnixArguments(
lpCmdLine, &argc, &argv);
int ret = my_main(argc, argv);
for (int i = 0; i < argc; i++) { delete [] argv[i]; }
delete [] argv;
return ret;
}
#else */
int main(int argc, char *argv[])
{
    return my_main(argc, argv);
}

