#ifndef __ULTRASOUND_EXAMPLE_GUI_H_
#define __ULTRASOUND_EXAMPLE_GUI_H_

#include "vtkKWWindow.h"
#include "vtkUltrasoundExampleModule.h"

#include <vtkstd/vector>
class vtkCallbackCommand;
class vtkVolume;
class vtkVolumeMapper;
class vtkImageData;
class vtkKWVolumePropertyWidget;
class vtkKWCheckButton;
class vtkKWRenderWidget;

class vtkUltrasoundStreamerGUI;

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
    static void UltrasoundEventStatic(vtkObject *caller, unsigned long eid, void *clientData, void *callData);
    void GuiEvent(vtkObject* caller);
    void RenderBegin();
    void RenderEnd();
    void UltrasoundEvent(unsigned long event);

    vtkCallbackCommand* GUICallbackCommand;
    vtkCallbackCommand* StartCallbackCommand;
    vtkCallbackCommand* StopCallbackCommand;
    vtkCallbackCommand* UltrasoundCommand;

protected:
    vtkUltrasoundExampleGUI();
    virtual ~vtkUltrasoundExampleGUI();

    virtual void CreateWidget();
    void CreateUltrasoundWidget();
    void CreateVolumeRenderingWidget();


    private:
    vtkUltrasoundExampleGUI operator=(const vtkUltrasoundExampleGUI&);
    vtkUltrasoundExampleGUI(const vtkUltrasoundExampleGUI&);

private:
    vtkKWRenderWidget*              renderWidget;

    vtkImageData*                   ImageData;
    vtkVolume*                      Volume;
    vtkVolumeMapper*                VolumeMapper;
    vtkKWVolumePropertyWidget*      VolumePropertyWidget;
    
    vtkUltrasoundStreamerGUI*       UltrasoundStreamerGUI;    

    bool renderScheduled;
    bool isRendering;
};

#endif /* __ULTRASOUND_EXAMPLE_GUI_H_ */
