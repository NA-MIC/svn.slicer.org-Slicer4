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
#include "vtkKWCheckButton.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenu.h"
#include "vtkKWScale.h"

#include <vtksys/SystemTools.hxx>
#include <vtksys/CommandLineArguments.hxx>

#include "vtkRenderer.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkImageReader.h"
#include <sstream>
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"

#include "vtkImageWriter.h"
#include "vtkCudaImageDataFilter.h"
#include "vtkCudaImageData.h"
#include "vtkImageData.h"


vtkKWApplication *app;
vtkKWRenderWidget* renderWidget;
vtkKWRange* ThresholdRange;
vtkKWScale* SteppingSizeScale;
vtkKWScale* ScaleFactorScale;
vtkKWVolumePropertyWidget* VolumePropertyWidget;
vtkCudaVolumeMapper* VolumeMapper;
vtkKWCheckButton* cb_Animate;
vtkKWMenuButton*  mb_Model;

int frameNumber = 0;
vtkImageReader* reader[5];
bool renderScheduled = false;

void Clear()
{
    VolumeMapper->SetInput(NULL);
}

void LoadHead()
{
    reader[0]->SetDataScalarTypeToUnsignedChar();
    reader[0]->SetNumberOfScalarComponents(1);
    reader[0]->SetDataExtent(0, 255,
        0, 255, 
        0, 93);
    reader[0]->SetFileDimensionality(3);


    std::stringstream s;
    s << "D:\\fullhead94.raw";

    reader[0]->SetFileName(s.str().c_str());
    reader[0]->Update();

    VolumeMapper->SetInput(reader[0]->GetOutput());
}

void LoadHeart()
{
    reader[0]->SetDataScalarTypeToUnsignedChar();
    reader[0]->SetNumberOfScalarComponents(1);
    reader[0]->SetDataExtent(0, 255,
        0, 255, 
        0, 255);
    reader[0]->SetFileDimensionality(3);


    std::stringstream s;
    s << "D:\\heart512.raw";

    reader[0]->SetFileName(s.str().c_str());
    reader[0]->Update();

    VolumeMapper->SetInput(reader[0]->GetOutput());
}

void LoadHeartSeries()
{
//    Reading in the Data using a ImageReader
    for (unsigned int i = 0; i < 5; i++ ) 
    {
        reader[i]->SetDataScalarTypeToUnsignedChar();
        reader[i]->SetNumberOfScalarComponents(1);
        reader[i]->SetDataExtent(0, 255,
            0, 255, 
            0, 255);
        reader[i]->SetFileDimensionality(3);


        std::stringstream s;
        s << "C:\\heart256-" << i+1 << ".raw";
        //s << "C:\\fullhead94.raw";

        reader[i]->SetFileName(s.str().c_str());
        reader[i]->Update();

        //        volumeMapper->MultiInput[i] = reader[i]->GetOutput();
    }
}

void LoadLung()
{
    reader[0]->SetDataScalarTypeToShort();
    reader[0]->SetNumberOfScalarComponents(1);
    reader[0]->SetDataExtent(0, 127,
        0, 127, 
        0, 29);
    reader[0]->SetFileDimensionality(3);

    std::stringstream s;
    s << "D:\\lung128x128x30.raw";

    reader[0]->SetFileName(s.str().c_str());
    reader[0]->Update();

    VolumeMapper->SetInput(reader[0]->GetOutput());
}

void ChangeModel(vtkObject* caller, unsigned long eid, void* clientData, void* callData)
{
        if (! strcmp(mb_Model->GetValue(), "Head"))
            LoadHead();
        else if (!strcmp(mb_Model->GetValue(), "Heart"))
            LoadHeart();
        else if (!strcmp(mb_Model->GetValue(), "Lung"))
            LoadLung();
        else
            Clear();
        renderWidget->Render();
}

void UpdateRenderer(vtkObject *caller, unsigned long eid, void *clientData, void *callData)
{
    if (caller == ThresholdRange)
      VolumeMapper->SetThreshold(ThresholdRange->GetRange());
    else if (caller == SteppingSizeScale)
        VolumeMapper->SetSampleDistance(SteppingSizeScale->GetValue());
    else if (caller == ScaleFactorScale)
        VolumeMapper->SetRenderOutputScaleFactor(ScaleFactorScale->GetValue());
    renderWidget->GetRenderer()->Render();
}

void Animate(vtkObject* caller, unsigned long eid, void* clientData, void* callData)
{
    int enabled = cb_Animate->GetSelectedState();
    if (cb_Animate->GetSelectedState() == 1)
    {
        if (++frameNumber == 5)
            frameNumber = 0;
        VolumeMapper->SetInput(reader[frameNumber]->GetOutput());
        if (renderScheduled == false)
        {    
            renderScheduled=true;
            renderWidget->SetRenderModeToInteractive();
            cerr << *renderWidget;
            app->Script("after 100 %s Render",renderWidget->GetTclName());
        }
    }
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
    app->SetName("CUDA/VTK Volume Rendering Viewer");
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
    VolumeMapper = vtkCudaVolumeMapper::New();

    for (unsigned int i = 0; i < 5; i++)
        reader[i] = NULL;
    reader[0] = vtkImageReader::New();
    LoadHead();

    //reader[0]->Delete();
    //reader[0]= vtkImageReader::New();
    //reader[0]->SetDataScalarTypeToShort();
    //reader[0]->SetNumberOfScalarComponents(1);
    //reader[0]->SetDataExtent(0, 127,
    //    0, 127, 
    //    0, 29);
    //reader[0]->SetFileDimensionality(3);


    //reader[0]->SetFilePattern("C:\\Ultrasound_Prostate\\US.*");
    //reader[0]->SetFileName("C:\\lung128x128x30.raw");
    //reader[0]->Update();

    //vtkImageShiftScale* scaler = vtkImageShiftScale::New();
    //scaler->SetOutputScalarTypeToUnsignedChar();
    //scaler->SetInput(reader[0]->GetOutput());
    //scaler->Update();

    //vtkCudaImageDataFilter* filter = vtkCudaImageDataFilter::New();
    //filter->SetInput(reader[0]->GetOutput());
    //filter->Update();

    VolumeMapper->SetInput(reader[0]->GetOutput());

    VolumeMapper->SetRenderMode(0/*vtkCudaMemoryTexture::RenderToTexture*/);

    volume->SetMapper(VolumeMapper);
    vtkVolumeProperty* prop = vtkVolumeProperty::New();
    volume->SetProperty(prop);
    renderWidget->GetRenderer()->AddVolume(volume);

    /// GUI EVENT
    vtkCallbackCommand* GUICallbackCommand = vtkCallbackCommand::New ( );
    GUICallbackCommand->SetCallback( UpdateRenderer);
    vtkCallbackCommand* AnimCallbackCommand = vtkCallbackCommand::New ( );
    AnimCallbackCommand->SetCallback( Animate);
    vtkCallbackCommand* ModelCallbackCommand = vtkCallbackCommand::New();
    ModelCallbackCommand->SetCallback( ChangeModel );

    /// SETUP THE GUI
    mb_Model = vtkKWMenuButton::New();
    mb_Model->SetParent(win->GetMainPanelFrame());
    mb_Model->Create();
    mb_Model->GetMenu()->AddRadioButton("Heart");
    mb_Model->GetMenu()->AddRadioButton("Head");
    mb_Model->GetMenu()->AddRadioButton("Lung");
    mb_Model->GetMenu()->AddRadioButton("Empty");
    mb_Model->SetValue("Heart");
    mb_Model->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)ModelCallbackCommand);

    app->Script("pack %s -side top -anchor nw -expand n -fill x -pady 2",
        mb_Model->GetWidgetName()); 
    
    //Volume Property
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

    SteppingSizeScale = vtkKWScale::New();
    SteppingSizeScale->SetParent(win->GetMainPanelFrame());
    SteppingSizeScale->Create();
    SteppingSizeScale->SetRange(0.1f, 3.0f);
    SteppingSizeScale->SetResolution(.1f);
    SteppingSizeScale->SetValue(1.0f);
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        SteppingSizeScale->GetWidgetName()); 
    SteppingSizeScale->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)GUICallbackCommand);


    ScaleFactorScale = vtkKWScale::New();
    ScaleFactorScale->SetParent(win->GetMainPanelFrame());
    ScaleFactorScale->Create();
    ScaleFactorScale->SetRange(1.0f, 10.0f);
    ScaleFactorScale->SetResolution(.5f);
    ScaleFactorScale->SetValue(1.0f);
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        ScaleFactorScale->GetWidgetName()); 
    ScaleFactorScale->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand*)GUICallbackCommand);



    cb_Animate = vtkKWCheckButton::New();
    cb_Animate->SetParent(win->GetMainPanelFrame());
    cb_Animate->Create();
    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
        cb_Animate->GetWidgetName());


    // Add the actor to the scene
    renderWidget->ResetCamera();

    renderWidget->GetRenderWindow()->AddObserver(vtkCommand::EndEvent,  (vtkCommand*)AnimCallbackCommand);
    int ret = 0;
    win->Display();
    if (!option_test)
    {
        app->Start(argc, argv);
        ret = app->GetExitStatus();
    }
    renderWidget->GetRenderer()->RemoveVolume(volume);
    win->Close();

    // Deallocate and exit

    SteppingSizeScale->Delete();
    ScaleFactorScale->Delete();
    ModelCallbackCommand->Delete();
    AnimCallbackCommand->Delete();
    GUICallbackCommand->Delete();
    //    filter->Delete();
    mb_Model->Delete();
    volume->Delete();
    VolumeMapper->Delete();
    for (unsigned int i = 0; i < 5; i++)
        if (reader[i] != NULL)
            reader[i]->Delete();

    VolumePropertyWidget->Delete();
    cb_Animate->Delete();
    ThresholdRange->Delete();
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
