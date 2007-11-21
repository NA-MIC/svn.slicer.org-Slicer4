#ifndef __vtkProstateNavTargetingStep_h
#define __vtkProstateNavTargetingStep_h

#include "vtkProstateNavStep.h"

class vtkKWFrame;
class vtkKWEntry;
class vtkKWCheckButton;
class vtkKWEntryWithLabel;
class vtkKWMultiColumnListWithScrollbars;

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
  vtkKWEntryWithLabel *PosREntry;
  vtkKWEntryWithLabel *PosAEntry;
  vtkKWEntryWithLabel *PosSEntry;
  vtkKWEntryWithLabel *NormREntry;
  vtkKWEntryWithLabel *NormAEntry;
  vtkKWEntryWithLabel *NormSEntry;

  vtkKWMultiColumnListWithScrollbars*  TargetListColumnList;
  

private:
  vtkProstateNavTargetingStep(const vtkProstateNavTargetingStep&);
  void operator=(const vtkProstateNavTargetingStep&);
};

#endif
