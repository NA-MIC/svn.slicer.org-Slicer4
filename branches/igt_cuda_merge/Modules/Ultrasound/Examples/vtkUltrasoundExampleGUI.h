#ifndef __ULTRASOUND_EXAMPLE_GUI_H_
#define __ULTRASOUND_EXAMPLE_GUI_H_

#include "vtkKWWindow.h"
#include "vtkUltrasoundExampleModule.h"

#include <vtkstd/vector>
class vtkCallbackCommand;
class vtkVolume;
class vtkVolumeMapper;
class vtkImageReader;
class vtkKWVolumePropertyWidget;
class vtkKWCheckButton;
class vtkKWRenderWidget;

class VTK_ULTRASOUNDEXAMPLEGUILIBRARY_EXPORT vtkUltrasoundExampleGUI : public vtkKWWindow
{
public:
    vtkTypeRevisionMacro(vtkUltrasoundExampleGUI, vtkKWWindow);
    static vtkUltrasoundExampleGUI *New();

    void LoadUltrasoundHeartSeries(void* data);

    void ScheduleRender();

    //EVENTS: VTK is not supporting Object Calls.
    static void GuiEventStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData);
    static void RenderBeginStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData);
    static void RenderEndStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData);
    void GuiEvent(vtkObject* caller);
    void RenderBegin();
    void RenderEnd();

    vtkCallbackCommand* GUICallbackCommand;
    vtkCallbackCommand* StartCallbackCommand;
    vtkCallbackCommand* StopCallbackCommand;

protected:
    vtkUltrasoundExampleGUI();
    virtual ~vtkUltrasoundExampleGUI();

    virtual void CreateWidget();


    private:
    vtkUltrasoundExampleGUI operator=(const vtkUltrasoundExampleGUI&);
    vtkUltrasoundExampleGUI(const vtkUltrasoundExampleGUI&);

private:
    vtkKWRenderWidget* renderWidget;
    vtkKWCheckButton* cb_Animate;


    vtkVolume* Volume;
    vtkVolumeMapper* VolumeMapper;
    vtkKWVolumePropertyWidget* VolumePropertyWidget;

    unsigned int frameNumber;
    //BTX
    vtkstd::vector<vtkImageReader*> readers;
    //ETX
    bool renderScheduled;
    bool isRendering;
};

#endif /* __ULTRASOUND_EXAMPLE_GUI_H_ */
