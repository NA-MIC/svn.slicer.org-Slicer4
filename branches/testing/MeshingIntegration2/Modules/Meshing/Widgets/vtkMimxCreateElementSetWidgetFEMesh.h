/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxCreateElementSetWidgetFEMesh.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMimxCreateElementSetWidgetFEMesh - a widget to manipulate 3D parallelopipeds
//
// .SECTION Description
// This widget was designed with the aim of visualizing / probing cuts on
// a skewed image data / structured grid. 
//
// .SECTION Interaction
// The widget allows you to create a parallelopiped (defined by 8 handles).
// The widget is initially placed by using the "PlaceWidget" method in the
// representation class. After the widget has been created, the following
// interactions may be used to manipulate it :
// 1) Click on a handle and drag it around moves the handle in space, while
//    keeping the same axis alignment of the parallelopiped
// 2) Dragging a handle with the shift button pressed resizes the piped
//    along an axis.
// 3) Control-click on a handle creates a chair at that position. (A chair
//    is a depression in the piped that allows you to visualize cuts in the
//    volume). 
// 4) Clicking on a chair and dragging it around moves the chair within the
//    piped.
// 5) Shift-click on the piped enables you to translate it.
//
// .SECTION Caveats
// .SECTION See Also vtkParallelopipedRepresentation

#ifndef __vtkMimxCreateElementSetWidgetFEMesh_h
#define __vtkMimxCreateElementSetWidgetFEMesh_h

#include "vtkAbstractWidget.h"
#include "vtkUnstructuredGrid.h"
#include "vtkActor.h"

class vtkActor;
class vtkCellPicker;
class vtkDataSet;
class vtkIdList;
class vtkInteractorStyleRubberBandPick;
class vtkPointLocator;
class vtkPolyDataMapper;
class vtkRenderedAreaPicker;
class vtkUnstructuredGrid;
class vtkGeometryFilter;
class vtkPolyData;

class vtkMimxCreateElementSetWidgetFEMesh : public vtkAbstractWidget
{
 public:
  // Description:
  // Instantiate the object.
  static vtkMimxCreateElementSetWidgetFEMesh *New();

  vtkTypeRevisionMacro(vtkMimxCreateElementSetWidgetFEMesh,vtkAbstractWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Override the superclass method. This is a composite widget, (it internally
  // consists of handle widgets). We will override the superclass method, so
  // that we can pass the enabled state to the internal widgets as well.
  virtual void SetEnabled(int);
  
  // Description:
  // Methods to change the whether the widget responds to interaction.
  // Overridden to pass the state to component widgets.
  vtkSetObjectMacro(InputActor, vtkActor);
  //
  vtkGetObjectMacro(SelectedCellIds, vtkIdList);
  vtkGetObjectMacro(SelectedPointIds, vtkIdList);

  void CreateDefaultRepresentation(){}
  void ComputeSelectedPointIds(vtkDataSet *ExtractedUGrid, vtkMimxCreateElementSetWidgetFEMesh *Self);
  void AcceptSelectedMesh(vtkMimxCreateElementSetWidgetFEMesh *Self);
  void SetInput(vtkUnstructuredGrid *input);
  void SetCellSelectionState(int Selection);
protected:
  vtkMimxCreateElementSetWidgetFEMesh();
  ~vtkMimxCreateElementSetWidgetFEMesh();

  static void CrtlMouseMoveCallback               (vtkAbstractWidget* );
  static void ShiftMouseMoveCallback               (vtkAbstractWidget* );
  static void CrtlLeftButtonDownCallback                        (vtkAbstractWidget* );
  static void CrtlLeftButtonUpCallback            (vtkAbstractWidget* );
  static void ShiftLeftButtonDownCallback                       (vtkAbstractWidget* );
  static void ShiftLeftButtonUpCallback            (vtkAbstractWidget* );
  static void LeftButtonUpCallback                      (vtkAbstractWidget* );
  static void SelectCellsOnSurfaceFunction(vtkMimxCreateElementSetWidgetFEMesh *Self);
  static void SelectVisibleCellsOnSurfaceFunction(vtkMimxCreateElementSetWidgetFEMesh *Self);
  static void SelectCellsThroughFunction(vtkMimxCreateElementSetWidgetFEMesh *Self);
  static void SelectIndividualCellFunction(vtkMimxCreateElementSetWidgetFEMesh *Self);
  void ComputeSelectedCellIds(
        vtkMimxCreateElementSetWidgetFEMesh *self, vtkDataSet *DataSet);
  int ComputeOriginalCellNumber(
          vtkMimxCreateElementSetWidgetFEMesh *self, vtkIdList *PtIds);
  void DeleteSelectedCells(vtkMimxCreateElementSetWidgetFEMesh *Self);
  int DoesCellBelong(int CellNum, vtkMimxCreateElementSetWidgetFEMesh *Self);
  int GetCellNumGivenFaceIds(vtkIdList *PtIds, vtkMimxCreateElementSetWidgetFEMesh *Self);
  int GetFaceNumGivenCellNumFaceIds(int CellNum, 
          vtkIdList *PtIds, vtkPolyData *Surface, vtkMimxCreateElementSetWidgetFEMesh *Self);
  static void ExtractElementsBelongingToAFace(vtkMimxCreateElementSetWidgetFEMesh *Self);
  vtkIdList *SelectedCellIds;
  vtkIdList *SelectedPointIds;
  vtkIdList *DeleteCellIds;
  //BTX
  // Description:
  // Events invoked by this widget
  int WidgetEvent;
  enum WidgetEventIds 
    {
    Start = 0,
        Outside,
        CrtlLeftMouseButtonDown,
        CrtlLeftMouseButtonUp,
        CrtlLeftMouseButtonMove,
        VTKLeftButtonDown,
        VTKMouseMove,
        CrtlRightMouseButtonDown,
        ShiftLeftMouseButtonDown,
        ShiftLeftMouseButtonUp,
        ShiftMouseMove,
        LeftMouseButtonUp
    };
  //ETX
  enum CellSelectionType
  {
          SelectCellsThrough = 0,
          SelectCellsOnSurface,
          SelectVisibleCellsOnSurface,
          SelectIndividualCell,
          SelectMultipleCells
  };
   int CellSelectionState;
  vtkInteractorStyleRubberBandPick *RubberBandStyle;
  vtkRenderedAreaPicker *AreaPicker;

  vtkUnstructuredGrid *Input;
  vtkUnstructuredGrid *OriginalInput;

  vtkIdType PickX0;
  vtkIdType PickY0;
  vtkIdType PickX1;
  vtkIdType PickY1;
  vtkIdType PickStatus;
  vtkActor *InputActor;
  vtkUnstructuredGrid *ExtractedGrid;
  vtkActor *ExtractedActor;
  vtkActor *SelectedActor;
  vtkPointLocator *PointLocator;
  vtkPoints *LocatorPoints;
  vtkPointLocator *InputLocator;
  vtkPoints *InputPoints;
  vtkGeometryFilter *SurfaceFilter;
  vtkPolyDataMapper *SurfaceMapper;
  vtkActor *SurfaceActor;
private:
  vtkMimxCreateElementSetWidgetFEMesh(const vtkMimxCreateElementSetWidgetFEMesh&);  //Not implemented
  void operator=(const vtkMimxCreateElementSetWidgetFEMesh&);  //Not implemented
};

#endif
