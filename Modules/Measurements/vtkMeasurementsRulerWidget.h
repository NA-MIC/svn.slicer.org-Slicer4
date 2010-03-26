/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerNodeSelectorWidget.h,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

=========================================================================auto=*/

///  vtkSlicerNodeSelectorWidget - menu to select volumes from current mrml scene
/// 
/// Inherits most behavior from kw widget, but is specialized to observe
/// the current mrml scene and update the entries of the pop up menu to correspond
/// to the currently available volumes.  This widget also has a notion of the current selection
/// that can be observed or set externally
//


#ifndef __vtkMeasurementsRulerWidget_h
#define __vtkMeasurementsRulerWidget_h

#include "vtkMeasurementsWin32Header.h"

#include "vtkSlicerWidget.h"

  
class vtkSlicerNodeSelectorWidget;
class vtkKWCheckButtonWithLabel;
class vtkKWScaleWithLabel;
class vtkKWChangeColorButton;
class vtkKWMenuButton;
class vtkKWMenuButtonWithLabel;
class vtkKWLabel;
class vtkKWEntry;
class vtkKWEntryWithLabel;
class vtkMRMLMeasurementsRulerNode;
class vtkSlicerViewerWidget;
class vtkMeasurementsDistanceWidgetClass;
class VTK_MEASUREMENTS_EXPORT vtkMeasurementsRulerWidget : public vtkSlicerWidget
{
  
public:
  static vtkMeasurementsRulerWidget* New();
  vtkTypeRevisionMacro(vtkMeasurementsRulerWidget,vtkSlicerWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// access methods
  vtkGetObjectMacro(VisibilityButton, vtkKWCheckButtonWithLabel);
  vtkGetObjectMacro (AllVisibilityMenuButton, vtkKWMenuButton);

  /// 
  /// Getting the mrml ruler node id
  vtkGetStringMacro(RulerNodeID);
  
  /// 
  /// Set the fid list id, and update the widgets
  void SetRulerNodeID(char *id);
    
  /// 
  /// alternative method to propagate events generated in GUI to logic / mrml
  virtual void ProcessWidgetEvents ( vtkObject *caller, unsigned long event, void *callData );
  
  /// 
  /// alternative method to propagate events generated in GUI to logic / mrml
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

  /// 
  /// adds observes on widgets in the class
  virtual void AddWidgetObservers();
  

  /// 
  /// removes observers on widgets in the class
  virtual void RemoveWidgetObservers ( );

  /// 
  /// add observers on node
  virtual void AddMRMLObservers ( );

  /// 
  /// remove observers on node
  virtual void RemoveMRMLObservers ( );

  /// 
  /// Get/set the viewer widget so can add a the ruler widget to it
  vtkGetObjectMacro(ViewerWidget, vtkSlicerViewerWidget);
  virtual void SetViewerWidget(vtkSlicerViewerWidget *viewerWidget);
//BTX
  /// 
  /// encapsulated 3d widgets for each ruler node
  std::map<std::string, vtkMeasurementsDistanceWidgetClass *> DistanceWidgets;
//ETX
  /// 
  /// get a distance widget by ruler node id
  vtkMeasurementsDistanceWidgetClass *GetDistanceWidget(const char * nodeID);

  /// 
  /// set up a new distance widget for this node
  void AddDistanceWidget(vtkMRMLMeasurementsRulerNode *rulerNode);
  /// 
  /// remove distance widget for this node
  void RemoveDistanceWidget(vtkMRMLMeasurementsRulerNode *rulerNode);
  /// 
  /// check scene to make sure that have a widget for each ruler node, and no extra widgets...
  void Update3DWidgetsFromMRML();

protected:
  vtkMeasurementsRulerWidget();
  virtual ~vtkMeasurementsRulerWidget();

  /// 
  /// Create the widget.
  virtual void CreateWidget();

  /// 
  /// update the widget GUI from the settings in the passed in rulerNode
  void UpdateWidget(vtkMRMLMeasurementsRulerNode *rulerNode);

  ///
  /// update the distance label with the value from the ruler widget
  void UpdateDistanceLabel(vtkMRMLMeasurementsRulerNode *activeRulerNode);

  /// 
  /// update the 3d distance widget from the settings in teh passed in
  /// rulerNode
  void Update3DWidget(vtkMRMLMeasurementsRulerNode *rulerNode);
  
  /// 
  /// update the mrml node passed in, or the selected in the node selector if
  /// null, to match the 3d widget
  void UpdateMRMLFromWidget(vtkMRMLMeasurementsRulerNode *rulerNode = NULL);

  /// 
  /// Set visibility on all ruler nodes in the scene to the passed in visibility flag
  void ModifyAllRulerVisibility (int visibilityState);

  /// 
  /// the id of the mrml node currently displayed in the widget
  char* RulerNodeID;
  /// 
  /// to select a measurements ruler node from the scene, and create new ones
  vtkSlicerNodeSelectorWidget* RulerSelectorWidget;

  /// 
  /// visibility check button
  vtkKWCheckButtonWithLabel *VisibilityButton;
  /// 
  /// Change the colour of the end points
  vtkKWChangeColorButton *PointColourButton;
  /// 
  /// Change the colour of the line
  vtkKWChangeColorButton *LineColourButton;
  /// 
  /// Change the colour of the distance annotation text
  vtkKWChangeColorButton *TextColourButton;
  
  /// 
  /// point position entry
  vtkKWLabel *Position1Label;
  vtkKWEntry *Position1XEntry;
  vtkKWEntry *Position1YEntry;
  vtkKWEntry *Position1ZEntry;
  vtkKWLabel *Position2Label;
  vtkKWEntry *Position2XEntry;
  vtkKWEntry *Position2YEntry;
  vtkKWEntry *Position2ZEntry;

  ///
  /// distance label
  vtkKWLabel *DistanceLabel;

  /// 
  /// distance annotation option entries
  vtkKWEntryWithLabel *DistanceAnnotationFormatEntry;
  vtkKWEntryWithLabel *DistanceAnnotationScaleEntry;
  vtkKWCheckButtonWithLabel *DistanceAnnotationVisibilityButton;

  /// 
  /// number of subdivisions on the line
  vtkKWEntryWithLabel *ResolutionEntry;

  /// 
  /// Ruler GUI elements to select models upon which to constrain the ruler end points
  vtkSlicerNodeSelectorWidget* RulerModel1SelectorWidget;
  vtkSlicerNodeSelectorWidget* RulerModel2SelectorWidget;

  /// 
  /// menu button whose menu exposes options for
  /// setting visibility of all ruler nodes
  vtkKWMenuButton *AllVisibilityMenuButton;

  /// 
  /// flag set to 1 when updating a 3d widget (todo: set it to the index of the
  /// ruler node / 3d widget once have more than one)
  int Updating3DWidget;

  /// 
  /// pointer to the viewer widget so can add props, request renders 
  vtkSlicerViewerWidget *ViewerWidget;
  
  ///
  /// Sets Slicer's interaction mode to "pick and manipulate"
  /// when a widget is being manipulated, and
  void UpdateInteractionModeAtStartInteraction();
  ///
  /// Resets Slicer's interaction mode back to original
  /// interaction mode when manipulation is finished.
  void UpdateInteractionModeAtEndInteraction();

private:


  vtkMeasurementsRulerWidget(const vtkMeasurementsRulerWidget&); /// Not implemented
  void operator=(const vtkMeasurementsRulerWidget&); /// Not Implemented
};

#endif

