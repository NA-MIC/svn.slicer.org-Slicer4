#ifndef __vtkSlicerWaitMessageWidget_h
#define __vtkSlicerWaitMessageWidget_h

#include "vtkSlicerWidget.h"
#include "vtkKWTopLevel.h"
#include "vtkKWLabel.h"


class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerWaitMessageWidget : public vtkSlicerWidget
{
  
public:
  static vtkSlicerWaitMessageWidget* New();
  vtkTypeRevisionMacro(vtkSlicerWaitMessageWidget,vtkSlicerWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Getting the window in which to display message.
  vtkGetObjectMacro ( MessageWindow, vtkKWTopLevel );

  // Description:
  // Getting the label that contains the message
  vtkGetObjectMacro (TextLabel, vtkKWLabel );

  // Description:
  // Getting the label that contains the icon
  vtkGetObjectMacro (ImageLabel, vtkKWLabel );

  // Description:
  // Internal Callbacks. do not use.
  void DisplayWindow ( );
  void WithdrawWindow ( );
  void SetText ( const char *text);
  void ResetText ( );
  
 protected:
  vtkSlicerWaitMessageWidget();
  virtual ~vtkSlicerWaitMessageWidget();

  // Description:
  // Create the widget.
  virtual void CreateWidget();

  vtkKWTopLevel *MessageWindow;
  vtkKWLabel *ImageLabel;
  vtkKWLabel *TextLabel;
  
private:

  vtkSlicerWaitMessageWidget(const vtkSlicerWaitMessageWidget&); // Not implemented
  void operator=(const vtkSlicerWaitMessageWidget&); // Not Implemented
};

#endif

