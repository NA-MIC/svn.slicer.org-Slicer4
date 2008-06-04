#ifndef __ULTRASOUND_TOOL_GUI_H_
#define __ULTRASOUND_TOOL_GUI_H_

#include "vtkKWFrame.h"
#include "vtkUltrasoundModule.h"

class vtkKWCheckButton;
class vtkKWScaleWithLabel;

class vtkRenderer;
class vtkMatrix4x4;
class vtkCylinderSource;
class vtkCallbackCommand;
class vtkActor;
class vtkPolyDataMapper;
class vtkKWMatrixWidget;
class vtkMiniBirdInstrumentTracker;

class vtkKWScaleWithEntry;



class VTK_ULTRASOUNDMODULE_EXPORT vtkUltrasoundToolGUI : public vtkKWFrame
{
public:
    vtkTypeRevisionMacro(vtkUltrasoundToolGUI, vtkKWFrame);
    static vtkUltrasoundToolGUI *New();


    void SetRenderer(vtkRenderer* renderer) { this->Renderer = renderer; }
    void UpdateTracker();

    static void ProcessGUIEventsStatic (vtkObject *caller, unsigned long event, void *callData, void* object);
    void ProcessGUIEvents ( vtkObject *caller, unsigned long event, void *callData );


protected:
    vtkUltrasoundToolGUI();
    ~vtkUltrasoundToolGUI();

    virtual void CreateWidget();
    virtual void AddGUIObservers();

    vtkCallbackCommand* GUICallbackCommand;

private:
    vtkUltrasoundToolGUI(const vtkUltrasoundToolGUI&); // Not Defined
    vtkUltrasoundToolGUI& operator=(const vtkUltrasoundToolGUI&); // Not Defined

    vtkRenderer*            Renderer;
    vtkCylinderSource*      Rod;
    vtkPolyDataMapper*      RodMapper;
    vtkActor*               RodActor;

    vtkKWCheckButton*       cb_Enabled;
    vtkMatrix4x4*           Transform;
    vtkKWMatrixWidget*      TransformWidget;

    vtkKWScaleWithEntry*    ToolAdjustmentScales[3];
    vtkKWScaleWithEntry*    ProbeAdjustmentScales[3];

    vtkMiniBirdInstrumentTracker*    Tracker;
};

#endif /* __VTK_ULTRASOUND_TOOL_GUI_H_ */
