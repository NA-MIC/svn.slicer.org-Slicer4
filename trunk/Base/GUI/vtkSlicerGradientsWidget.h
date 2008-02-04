#ifndef __vtkSlicerGradientsWidget_h
#define __vtkSlicerGradientsWidget_h

#include "vtkSlicerWidget.h"

class vtkMRMLDiffusionWeightedVolumeNode;
class vtkKWFrameWithLabel;
class vtkKWFrame;
class vtkKWLoadSaveButtonWithLabel;
class vtkKWCheckButton;
class vtkKWTextWithScrollbars ;
class vtkKWLabel;

// Description:
// This class implements Slicer's DWI Gradients GUI.
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerGradientsWidget : public vtkSlicerWidget
  {
  public:

    // Description:
    // Usual vtk class functions
    static vtkSlicerGradientsWidget* New();
    vtkTypeRevisionMacro(vtkSlicerGradientsWidget,vtkSlicerWidget);
    void PrintSelf (ostream& os, vtkIndent indent );

    // Description:
    // Add/Remove observers on widgets in the GUI
    virtual void AddWidgetObservers ( );
    virtual void RemoveWidgetObservers ( );

    // Description:
    // Method to propagate events generated in GUI to logic / mrml
    void ProcessWidgetEvents(vtkObject *caller, unsigned long event, void *callData );

    // Description:
    // Method to update the widget when a new node is loaded.
    void UpdateWidget(vtkMRMLDiffusionWeightedVolumeNode *dwiNode);

    // Description:
    // Method to propagate keypress-events generated in the textbox of gradients.
    void TextFieldModifiedCallback();

  protected:
    vtkSlicerGradientsWidget(void);
    virtual ~vtkSlicerGradientsWidget(void);

    // Description:
    // Method to create the widget.
    virtual void CreateWidget();

    // Description:
    // Method to update the gradientsTextbox containing bValue and gradients (GUI).
    void UpdateGradients();

    vtkMRMLDiffusionWeightedVolumeNode *ActiveVolumeNode;
    vtkKWFrameWithLabel *GradientsFrame;
    vtkKWFrame *ButtonsFrame;
    vtkKWLoadSaveButtonWithLabel *LoadGradientsButton;
    vtkKWCheckButton *EnableGradientsButton;
    vtkKWTextWithScrollbars *GradientsTextbox;
    vtkKWLabel *StatusLabel;

  private:
    vtkSlicerGradientsWidget ( const vtkSlicerGradientsWidget& ); // Not implemented.
    void operator = ( const vtkSlicerGradientsWidget& ); //Not implemented.
  };

#endif 
