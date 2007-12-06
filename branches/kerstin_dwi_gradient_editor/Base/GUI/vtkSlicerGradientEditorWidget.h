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
class vtkKWTextWithScrollbars;
class vtkSlicerNodeSelectorWidget;
class vtkKWCheckButton;
class vtkKWLoadSaveButtonWithLabel;

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

    vtkKWLoadSaveButtonWithLabel *SaveButton;
    vtkKWLoadSaveButtonWithLabel *LoadButton;
    vtkKWLoadSaveButtonWithLabel *LoadGradientsButton;
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
    vtkKWTextWithScrollbars *Gradients;
    vtkSlicerNodeSelectorWidget *Mask;
    vtkKWCheckButton *Checkbutton1;
    vtkKWCheckButton *Checkbutton2;
    vtkKWCheckButton *Checkbutton3;
    vtkKWCheckButton *EnableMatrix;

private:
    vtkSlicerGradientEditorWidget ( const vtkSlicerGradientEditorWidget& ); // Not implemented.
    void operator = ( const vtkSlicerGradientEditorWidget& ); //Not implemented.
};

#endif 
