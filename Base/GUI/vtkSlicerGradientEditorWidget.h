#ifndef __vtkSlicerGradientEditorWidget_h
#define __vtkSlicerGradientEditorWidget_h

#define ARRAY_LENGTH 3
#define NUMBER_MATRIX_VALUES 9
#include "vtkSlicerWidget.h"
class vtkKWFrameWithLabel;
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
    vtkTypeRevisionMacro(vtkSlicerGradientEditorWidget,vtkSlicerWidget);
    //void PrintSelf(ostream& os, vtkIndent indent);

    // Description:
    // Add/Remove observers on widgets in the GUI
    virtual void AddWidgetObservers ( );
    virtual void RemoveWidgetObservers ( );

    // Description:
    // Method to propagate events generated in GUI to logic / mrml
    virtual void ProcessWidgetEvents(vtkObject *caller, unsigned long event, void *callData );

protected:
    vtkSlicerGradientEditorWidget(void);
    virtual ~vtkSlicerGradientEditorWidget(void);

    // Description:
    // Create the widget.
    virtual void CreateWidget();
    void PrintSelf (ostream& os, vtkIndent indent );

    vtkKWLoadSaveButtonWithLabel *SaveButton;
    vtkKWLoadSaveButtonWithLabel *LoadButton;
    vtkKWLoadSaveButtonWithLabel *LoadGradientsButton;
    vtkKWFrameWithLabel *MeasurementFrame;
    vtkKWFrameWithLabel *TestFrame;
    vtkKWFrameWithLabel *GradientsFrame;
    vtkKWFrameWithLabel *LoadsaveFrame;
    vtkKWEntrySet *Matrix;
    vtkKWPushButton *NegativeButton;
    vtkKWPushButton *SwapButton;    
    vtkKWPushButton *RunButton;
    vtkKWPushButton *RotateButton;
    vtkKWLabel *LabelAngle;
    vtkKWComboBox *Angle;
    vtkKWTextWithScrollbars *Gradients;
    vtkSlicerNodeSelectorWidget *Mask;
    vtkKWCheckButton *EnableMatrixButton;
    vtkKWCheckButton* Checkbuttons[ARRAY_LENGTH];

private:
    vtkSlicerGradientEditorWidget ( const vtkSlicerGradientEditorWidget& ); // Not implemented.
    void operator = ( const vtkSlicerGradientEditorWidget& ); //Not implemented.
};

#endif 
