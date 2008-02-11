// .NAME vtkSlicerGradientsWidget 
// .SECTION Description
// This class implements Slicer's DWI Gradients widget, part of the GradientEditor GUI.
// Inherits most behavior from vtkSlicerWidget.
#ifndef __vtkSlicerGradientsWidget_h
#define __vtkSlicerGradientsWidget_h

#include "vtkSlicerWidget.h"

class vtkMRMLDiffusionWeightedVolumeNode;
class vtkDoubleArray;
//widgets
class vtkKWFrameWithLabel;
class vtkKWFrame;
class vtkKWLoadSaveButtonWithLabel;
class vtkKWCheckButton;
class vtkKWTextWithScrollbars ;
class vtkKWLabel;
class vtkKWMessageDialog;

class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerGradientsWidget : public vtkSlicerWidget
  {
  public:

    // Description:
    // Usual vtk class functions.
    static vtkSlicerGradientsWidget* New();
    vtkTypeRevisionMacro(vtkSlicerGradientsWidget,vtkSlicerWidget);
    void PrintSelf(ostream& os, vtkIndent indent);

    // Description:
    // Add/Remove observers on widgets in the GUI.
    virtual void AddWidgetObservers();
    virtual void RemoveWidgetObservers();

    // Description:
    // Method to propagate events generated in GUI to logic / mrml.
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

    // Description:
    // Method to update the status label for the gradients (GUI).
    void UpdateStatusLabel(int status);

    // Description:
    // Method to display a message dialog to the user (GUI).
    void DisplayMessageDialog(const char* message);

    // Description:
    // Method to save changes to the activeVolumeNode.
    void SaveGradients();

    vtkMRMLDiffusionWeightedVolumeNode *ActiveVolumeNode;
    vtkDoubleArray *Gradients;
    vtkDoubleArray *BValues;
    //widgets (GUI)
    vtkKWFrameWithLabel *GradientsFrame;
    vtkKWFrame *ButtonsFrame;
    vtkKWLoadSaveButtonWithLabel *LoadGradientsButton;
    vtkKWCheckButton *EnableGradientsButton;
    vtkKWTextWithScrollbars *GradientsTextbox;
    vtkKWLabel *StatusLabel;
    vtkKWMessageDialog *MessageDialog;

  private:
    vtkSlicerGradientsWidget (const vtkSlicerGradientsWidget&); // Not implemented.
    void operator = (const vtkSlicerGradientsWidget&); //Not implemented.
  };

#endif 
