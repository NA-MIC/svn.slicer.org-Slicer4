#ifndef __vtkQueryAtlasSearchTermWidget_h
#define __vtkQueryAtlasSearchTermWidget_h

#include "vtkQueryAtlasWin32Header.h"
#include "vtkSlicerWidget.h"

class vtkKWPushButton;
class vtkKWMultiColumnListWithScrollbars;
class vtkQueryAtlasIcons;
class vtkKWFrame;

class VTK_QUERYATLAS_EXPORT vtkQueryAtlasSearchTermWidget : public vtkSlicerWidget
{
  
public:
  static vtkQueryAtlasSearchTermWidget* New();
  vtkTypeRevisionMacro(vtkQueryAtlasSearchTermWidget,vtkSlicerWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Get methods on class members ( no Set methods required. )
  vtkGetObjectMacro ( AddNewButton, vtkKWPushButton);
  vtkGetObjectMacro ( ClearAllButton, vtkKWPushButton);
  vtkGetObjectMacro ( ClearSelectedButton, vtkKWPushButton);
  vtkGetObjectMacro ( MultiColumnList, vtkKWMultiColumnListWithScrollbars );
  vtkGetObjectMacro ( QueryAtlasIcons, vtkQueryAtlasIcons );
  vtkGetObjectMacro ( ContainerFrame, vtkKWFrame );
  vtkGetMacro (NumberOfColumns, int );
  
  // Description:
  // Method that gets all terms in the multicolumn list
  virtual void GetAllSearchTerms ( );

  // Description:
  // alternative method to propagate events generated in GUI to logic / mrml
  virtual void ProcessWidgetEvents ( vtkObject *caller, unsigned long event, void *callData );
  
  // Description:
  // alternative method to propagate events generated in GUI to logic / mrml
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );
  
  // Description:
  // removes observers on widgets in the class
  virtual void RemoveWidgetObservers ( );

  // Description:
  // add observers on color node
  virtual void AddMRMLObservers ( );

  // Description:
  // remove observers on color node
  virtual void RemoveMRMLObservers ( );

  
 protected:
  vtkQueryAtlasSearchTermWidget();
  virtual ~vtkQueryAtlasSearchTermWidget();

  vtkKWPushButton *AddNewButton;
  vtkKWPushButton *ClearAllButton;
  vtkKWPushButton *ClearSelectedButton;
  vtkKWMultiColumnListWithScrollbars *MultiColumnList;
  vtkQueryAtlasIcons *QueryAtlasIcons;
  vtkKWFrame *ContainerFrame;
  
  // Description:
  // Create the widget.
  virtual void CreateWidget();

  // Description:
  // Update the widget, used when the color node id changes
  void UpdateWidget();
  
  void UpdateMRML();

  // Description:
  // Called when the selected row changes, just update the label, called from UpdateWidget
  void UpdateSelectedColor();
  
  int NumberOfColumns;

  vtkQueryAtlasSearchTermWidget(const vtkQueryAtlasSearchTermWidget&); // Not implemented
  void operator=(const vtkQueryAtlasSearchTermWidget&); // Not Implemented
};

#endif

