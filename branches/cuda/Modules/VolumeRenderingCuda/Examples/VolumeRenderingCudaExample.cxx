#include "vtkActor.h"
#include "vtkKWApplication.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWWindow.h"
#include "vtkRenderWindow.h"
#include "vtkKWSimpleAnimationWidget.h"


#include "vtkKWVolumePropertyWidget.h"
#include "vtkKWRange.h"
#include "vtkCallbackCommand.h"
#include "vtkKWEvent.h"


#include <vtksys/SystemTools.hxx>
#include <vtksys/CommandLineArguments.hxx>


#include "vtkVolumeCudaMapper.h"
#include "vtkImageReader.h"
#include <sstream>
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include "vtkImageViewer.h"
#include "vtkImageWriter.h"
#include "vtkImageShiftScale.h"
#include "vtkImageData.h"

vtkKWRenderWidget* renderWidget;
vtkKWRange* ThresholdRange;
vtkKWVolumePropertyWidget* VolumePropertyWidget;
vtkVolumeCudaMapper* VolumeMapper;

void UpdateRenderer(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    VolumeMapper->SetThreshold(ThresholdRange->GetRange());
    renderWidget->Render();
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

    // Create the application
    // If --test was provided, ignore all registry settings, and exit silently
    // Restore the settings that have been saved to the registry, like
    // the geometry of the user interface so far.

    vtkKWApplication *app = vtkKWApplication::New();
    app->SetName("KWPolygonalObjectViewerExample");
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
    win->SupportHelpOn();
    app->AddWindow(win);
    win->Create();
    win->SecondaryPanelVisibilityOff();

    // Add a render widget, attach it to the view frame, and pack

    renderWidget = vtkKWRenderWidget::New();
    renderWidget->SetBackgroundColor(255, 255, 255);
    renderWidget->SetParent(win->GetViewFrame());
    renderWidget->Create();

    app->Script("pack %s -expand y -fill both -anchor c -expand y", 
        renderWidget->GetWidgetName());

    // Create the mapper and actor


    vtkVolume* volume = vtkVolume::New();
    VolumeMapper = vtkVolumeCudaMapper::New();


    // Reading in the Data using a ImageReader
    vtkImageReader* reader[5];
    for (unsigned int i = 0; i < 5; i++ ) 
    {
        reader[i] = NULL;
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

//        volumeMapper->MultiInput[i] = reader[i]->GetOutput();
    }

    
        reader[0]->Delete();
        reader[0]= vtkImageReader::New();
        reader[0]->SetDataScalarTypeToShort();
        reader[0]->SetNumberOfScalarComponents(1);
        reader[0]->SetDataExtent(0, 255,
            0, 255, 
            0, 53);
        reader[0]->SetFileDimensionality(3);


        reader[0]->SetFileName("C:\\prostate.raw");
        reader[0]->Update();


        vtkImageShiftScale* scaler = vtkImageShiftScale::New();
        scaler->SetOutputScalarTypeToUnsignedChar();
        scaler->SetInput(reader[0]->GetOutput());
        scaler->Update();

    reader[0]->Update();
    VolumeMapper->SetInput(scaler->GetOutput());

    VolumeMapper->SetRenderMode(vtkVolumeCudaMapper::RenderToTexture);

    volume->SetMapper(VolumeMapper);
    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    volume->SetProperty(prop);
    renderWidget->AddViewProp(volume);

    /// GUI EVENT
    vtkCallbackCommand* GUICallbackCommand = vtkCallbackCommand::New ( );
    //GUICallbackCommand->SetClientData( reinterpret_cast<void *>(this) );
    GUICallbackCommand->SetCallback( UpdateRenderer);

    /// SETUP THE GUI
    VolumePropertyWidget = vtkKWVolumePropertyWidget::New();
    VolumePropertyWidget->SetParent(win->GetMainPanelFrame());
    VolumePropertyWidget->Create();
    app->Script( "pack %s -side top -anchor nw -expand n -fill x -pady 2",
        VolumePropertyWidget->GetWidgetName()); 
    VolumePropertyWidget->AddObserver(vtkKWEvent::VolumePropertyChangedEvent, (vtkCommand*)GUICallbackCommand);

    VolumePropertyWidget->SetVolumeProperty(prop);
    prop->Delete();

    ThresholdRange = vtkKWRange::New();
    ThresholdRange->SetParent(win->GetMainPanelFrame());
    ThresholdRange->Create();
    ThresholdRange->SetWholeRange(0, 255);
    ThresholdRange->SetRange(90, 255);
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        ThresholdRange->GetWidgetName()); 
    ThresholdRange->AddObserver(vtkKWRange::RangeValueChangingEvent, (vtkCommand*)GUICallbackCommand);



    // Add the actor to the scene
    renderWidget->ResetCamera();

    // Create a material property editor

    // Create a simple animation widget

    vtkKWFrameWithLabel *animation_frame = vtkKWFrameWithLabel::New();
    animation_frame->SetParent(win->GetMainPanelFrame());
    animation_frame->Create();
    animation_frame->SetLabelText("Movie Creator");

    app->Script("pack %s -side top -anchor nw -expand n -fill x -pady 2",
        animation_frame->GetWidgetName());

    vtkKWSimpleAnimationWidget *animation_widget = 
        vtkKWSimpleAnimationWidget::New();
    animation_widget->SetParent(animation_frame->GetFrame());
    animation_widget->Create();
    animation_widget->SetRenderWidget(renderWidget);
    animation_widget->SetAnimationTypeToCamera();

    app->Script("pack %s -side top -anchor nw -expand n -fill x",
        animation_widget->GetWidgetName());

    // Start the application
    // If --test was provided, do not enter the event loop and run this example
    // as a non-interactive test for software quality purposes.


    int ret = 0;
    win->Display();
    if (!option_test)
    {
        app->Start(argc, argv);
        ret = app->GetExitStatus();
    }
    /*
    vtkImageViewer* viewer = vtkImageViewer::New();
    while (true)
    {
    volumeMapper->Render(renderWidget->GetRenderer(), volume);
    viewer->SetInput(volumeMapper->GetOutput());
    viewer->Render();
    Sleep(100);
    }*/
    win->Close();

    // Deallocate and exit


    volume->Delete();
    VolumeMapper->Delete();
    for (unsigned int i = 0; i < 5; i++)
        if (reader[i] != NULL)
            reader[i]->Delete();

    animation_frame->Delete();
    animation_widget->Delete();
    VolumePropertyWidget->Delete();
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
//#endif
