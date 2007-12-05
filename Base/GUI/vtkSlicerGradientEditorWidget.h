#ifndef __vtkSlicerGradientEditorWidget_h
#define __vtkSlicerGradientEditorWidget_h

#include "vtkSlicerWidget.h"

class vtkKWFrameWithLabel;
class vtkKWLoadSaveButton;
class vtkKWEntrySet;
class vtkKWCheckButtonSet;
class vtkKWPushButton;
class vtkKWLabel;
class vtkKWComboBox;
class vtkKWText;
class vtkSlicerNodeSelectorWidget;
class vtkKWCheckButton;

class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerGradientEditorWidget : public vtkSlicerWidget
{
public:
    static vtkSlicerGradientEditorWidget* New();
    vtkTypeRevisionMacro(vtkSlicerGradientEditorWidget,vtkKWCompositeWidget);
    //void PrintSelf(ostream& os, vtkIndent indent);

protected:
    vtkSlicerGradientEditorWidget(void);
    virtual ~vtkSlicerGradientEditorWidget(void);

    // Description:
    // Create the widget.
    virtual void CreateWidget();

    vtkKWLoadSaveButton *SaveButton;
    vtkKWLoadSaveButton *LoadButton;
    vtkKWFrameWithLabel *MeasurementFrame;
    vtkKWFrameWithLabel *TestFrame;
    vtkKWFrameWithLabel *GradientsFrame;
    vtkKWFrameWithLabel *LoadsaveFrame;
    vtkKWEntrySet *Matrix;
    vtkKWCheckButtonSet *Checkbuttons;
    vtkKWPushButton *Negative;
    vtkKWPushButton *Swap;
    vtkKWPushButton *Rotate;
    vtkKWPushButton *Run;
    vtkKWLabel *LabelAngle;
    vtkKWComboBox *Angle;
    vtkKWText *Gradients;
    vtkSlicerNodeSelectorWidget *Mask;
    vtkKWCheckButton *Checkbutton1;
    vtkKWCheckButton *Checkbutton2;
    vtkKWCheckButton *Checkbutton3;

private:
    vtkSlicerGradientEditorWidget ( const vtkSlicerGradientEditorWidget& ); // Not implemented.
    void operator = ( const vtkSlicerGradientEditorWidget& ); //Not implemented.
};

#endif 
