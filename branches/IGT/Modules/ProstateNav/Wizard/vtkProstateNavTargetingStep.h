#ifndef __vtkProstateNavTargetingStep_h
#define __vtkProstateNavTargetingStep_h

#include "vtkProstateNavStep.h"

class vtkKWFrame;
class vtkKWPushButton;
class vtkKWEntryWithLabel;
class vtkKWMultiColumnListWithScrollbars;
class vtkKWMatrixWidgetWithLabel;

class VTK_PROSTATENAV_EXPORT vtkProstateNavTargetingStep : public vtkProstateNavStep
{
public:
  static vtkProstateNavTargetingStep *New();
  vtkTypeRevisionMacro(vtkProstateNavTargetingStep,vtkProstateNavStep);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void ShowUserInterface();
  virtual void ProcessGUIEvents(vtkObject *caller, unsigned long event, void *callData);  

protected:
  vtkProstateNavTargetingStep();
  ~vtkProstateNavTargetingStep();

  // GUI Widgets
  vtkKWFrame *MainFrame;
  vtkKWFrame *TargetListFrame;
  vtkKWFrame *TargetControlFrame;

  vtkKWMultiColumnListWithScrollbars* TargetListColumnList;
  vtkKWMatrixWidgetWithLabel* NeedlePositionMatrix;
  vtkKWMatrixWidgetWithLabel* NeedleNormalMatrix;
  vtkKWPushButton *MoveButton;
  vtkKWPushButton *StopButton;

private:
  vtkProstateNavTargetingStep(const vtkProstateNavTargetingStep&);
  void operator=(const vtkProstateNavTargetingStep&);
};

#endif
