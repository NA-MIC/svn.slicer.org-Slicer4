/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxSelectCellsWidget.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMimxSelectCellsWidget - 3D widget for selecting cells from 
// any mesh using rubberband style picking
// .SECTION Description
// 3DWidget contains a rubberbandstyleinteractor for the selection of a part
// of a dataset. The user has to input the dataset and its properties
// from which a subset has to be chosen. When once the subset is chosen, the 
// subset is highlighted. The output is an unstructuredgrid.
//

#ifndef __vtkMimxSelectCellsWidget_h
#define __vtkMimxSelectCellsWidget_h

#include "vtkAbstractWidget.h"

#include "vtkActor.h"

class vtkActor;
class vtkDataSetMapper;
class vtkInteractorObserver;
class vtkInteractorStyleRubberBandPick;
class vtkInteractorStyleSwitch;
class vtkRenderedAreaPicker;
class vtkUnstructuredGrid;

class vtkMimxSelectCellsWidget : public vtkAbstractWidget
{
public:
  // Description:
  // Instantiate the object.
  static vtkMimxSelectCellsWidget *New();

  vtkTypeRevisionMacro(vtkMimxSelectCellsWidget,vtkAbstractWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Methods that satisfy the superclass' API.
  virtual void SetEnabled(int);
  virtual void SetEnabled(int, vtkMimxSelectCellsWidget *Self);

  virtual void PlaceWidget(double bounds[6]){};
  vtkUnstructuredGrid *GetPickedSubset();
  void Initialize();
 /* vtkSetObjectMacro(InputActor, vtkActor);
  vtkGetObjectMacro(InputActor, vtkActor);*/
  //vtkGetObjectMacro(PreviousSelectedSubsetActor, vtkActor);
  vtkGetObjectMacro(CurrentSelectedSubsetActor, vtkActor);
  //vtkGetObjectMacro(PreviousSelectedSubset, vtkUnstructuredGrid);
  void AcceptSelectedMesh(vtkMimxSelectCellsWidget *Self);
  void SetInputAndCurrentSelectedMesh(
          vtkUnstructuredGrid * Input, vtkUnstructuredGrid *PrevSelSubset);
  void CreateDefaultRepresentation(){}
  void SetInput(vtkUnstructuredGrid *Input);
  vtkUnstructuredGrid *GetCurrentSelectedSubset();
  vtkUnstructuredGrid *Inputset;
protected:
  vtkMimxSelectCellsWidget();
  ~vtkMimxSelectCellsWidget();

  vtkInteractorStyleRubberBandPick *RubberBandStyle;
  vtkRenderedAreaPicker *AreaPicker;
  //vtkActor *InputActor;

  //vtkUnstructuredGrid *CurrentSelectedSubset;
  //vtkDataSetMapper *CurrentSelectedSubsetMapper;
  vtkActor *CurrentSelectedSubsetActor;

 /* vtkUnstructuredGrid *PreviousSelectedSubset;
  vtkDataSetMapper *PreviousSelectedSubsetMapper;
  vtkActor *PreviousSelectedSubsetActor;*/
 
  //vtkInteractorStyleSwitch *CurrentStyle;
//BTX - manage the WidgetEvent of the widget
  int WidgetEvent;
  enum WidgetEventIds 
  {
          Start = 0,
          Outside,
          LeftMouseButtonDown,
          LeftMouseButtonUp,
          MouseMove
  };
//ETX

  static void LeftButtonUpCallback            (vtkAbstractWidget* );
  static void LeftButtonDownCallback            (vtkAbstractWidget* );
  static void MouseMoveCallback                         (vtkAbstractWidget* );
  int PickX0, PickY0, PickX1, PickY1;
private:
  vtkMimxSelectCellsWidget(const vtkMimxSelectCellsWidget&);  //Not implemented
  void operator=(const vtkMimxSelectCellsWidget&);  //Not implemented
};

#endif
