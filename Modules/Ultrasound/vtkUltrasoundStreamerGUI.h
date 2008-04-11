#ifndef __ULTRASOUND_ULTRASOUND_STREAMER_GUI_H_
#define __ULTRASOUND_ULTRASOUND_STREAMER_GUI_H_

#include "vtkKWFrame.h"
#include "vtkUltrasoundModule.h"

class vtkKWCheckButton;
class vtkKWScaleWithLabel;

#include "vtkImageData.h"
class vtkUltrasoundStreamSource;
class vtkCallbackCommand;

class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundStreamerGUI : public vtkKWFrame
{
public:
    //BTX
    enum {
        VolumeUpdatedEvent = 10001,
        EnablingEvent,
        EnabledEvent,
        DisablingEvent,
        DisabledEvent,
    };
    //ETX

    vtkTypeRevisionMacro(vtkUltrasoundStreamerGUI, vtkKWFrame);
    static vtkUltrasoundStreamerGUI* New();

    bool IsEnabled() const;
    int GetRefreshRate() const;

    vtkGetObjectMacro(StreamSource, vtkUltrasoundStreamSource);
    vtkGetObjectMacro(ImageData, vtkImageData);
    vtkSetObjectMacro(ImageData, vtkImageData);

    static void ProcessGUIEventsStatic (vtkObject *caller, unsigned long event, void *callData, void* object);

    void ProcessGUIEvents ( vtkObject *caller, unsigned long event, void *callData );
    void UpdateInput();

    vtkCallbackCommand* GUICallbackCommand;

protected:
    vtkUltrasoundStreamerGUI();
    ~vtkUltrasoundStreamerGUI();

    virtual void CreateWidget();
    virtual void AddGUIObservers();

private:
    vtkUltrasoundStreamerGUI(const vtkUltrasoundStreamerGUI&); // Not Defined
    vtkUltrasoundStreamerGUI& operator=(const vtkUltrasoundStreamerGUI&); // Not Defined

    vtkUltrasoundStreamSource*  StreamSource;
    vtkKWCheckButton*           cb_Enabled;
    vtkKWScaleWithLabel*        sc_RefreshRate;

    vtkImageData*               ImageData;
};

#endif /* __ULTRASOUND_ULTRASOUND_STREAMER_GUI_H_ */
