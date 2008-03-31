#ifndef __ULTRASOUND_EXAMPLE_GUI_H_
#define __ULTRASOUND_EXAMPLE_GUI_H_

#include "vtkKWWindow.h"
#include "vtkUltrasoundModule.h"

#include <vtkstd/vector>
class vtkCallbackCommand;
class vtkVolume;
class vtkVolumeMapper;
class vtkImageReader;
class vtkKWVolumePropertyWidget;
class vtkKWCheckButton;
class vtkKWRenderWidget;

class VTK_ULTRASOUNDLIBRARY_EXPORT vtkUltrasoundGUI : public vtkKWWindow
{
    vtkTypeRevisionMacro(vtkUltrasoundGUI, vtkKWWindow);
    static vtkUltrasoundGUI *New();

    void LoadUltrasoundHeartSeries(void* data);

    void ScheduleRender();


    //EVENTS: VTK is not supporting Object Calls.
    void ReRender();
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
    vtkUltrasoundGUI();
    virtual ~vtkUltrasoundGUI();

    virtual void CreateWidget();


    private:
    vtkUltrasoundGUI operator=(const vtkUltrasoundGUI&);
    vtkUltrasoundGUI(const vtkUltrasoundGUI&);

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
