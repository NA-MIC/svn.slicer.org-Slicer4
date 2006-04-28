#ifndef __vtkSlicerSliceControlGUI_h
#define __vtkSlicerSliceControlGUI_h

#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerApplicationLogic.h"

#include "vtkKWCompositeWidget.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWSpinBoxWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"

class vtkCallbackCommand;

// Description:
// Manages a slice view
//
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerSliceControlGUI : public vtkKWCompositeWidget
{
public:
  static vtkSlicerSliceControlGUI* New ( );
  vtkTypeRevisionMacro ( vtkSlicerSliceControlGUI, vtkKWCompositeWidget );
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Get the sub widgets
  vtkGetObjectMacro(OffsetScale, vtkKWScaleWithEntry);
  vtkGetObjectMacro(FieldOfViewEntry, vtkKWSpinBoxWithLabel);
  vtkGetObjectMacro(OrientationMenu, vtkKWMenuButtonWithLabel);

  // Description:
  // the slice node is the mrml repository of the persistant and doable/undoble 
  // state of the 
  vtkGetObjectMacro (SliceNode, vtkMRMLSliceNode);
  void SetSliceNode (vtkMRMLSliceNode *SliceNode);

  // Description:
  // the mrml scene - observed to know undo status
  vtkGetObjectMacro (MRMLScene, vtkMRMLScene);
  void SetMRMLScene (vtkMRMLScene *MRMLScene);

  // Description:
  // Gets called by the command callback when the mrml node is modified
  void UpdateWidgets();


  // Description:
  // Method to invoke from the OrientationMenu
  void SetOrientationFromMenu ();

  // Description:
  // Transfers the state of the widgets into the mrml node 
  // -- sets undo state before applying 
  void Apply(double val) {this->Apply();}; // accept the scale value as arg
  void Apply();

  // Description:
  // Transfers the state of the widgets into the mrml node
  // -- not undoable
  void TransientApply(double val) {this->TransientApply();}; // accept the scale value as arg
  void TransientApply();
    
protected:
  vtkSlicerSliceControlGUI ( );
  ~vtkSlicerSliceControlGUI ( );

  // Description:
  // Create the widget.
  virtual void CreateWidget();
  
  // sub widgets.
  vtkKWScaleWithEntry *OffsetScale;
  vtkKWSpinBoxWithLabel *FieldOfViewEntry;
  vtkKWMenuButtonWithLabel *OrientationMenu;

  vtkMRMLScene *MRMLScene;
  vtkMRMLSliceNode *SliceNode;
    
  vtkCallbackCommand *MRMLCallbackCommand;
  vtkCallbackCommand *WidgetCallbackCommand;


private:
  vtkSlicerSliceControlGUI ( const vtkSlicerSliceControlGUI& ); //Not implemented.
  void operator = ( const vtkSlicerSliceControlGUI& ); //Not implemented.
};

#endif
