/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxExtractFaceWidget.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMimxExtractFaceWidget - a widget to manipulate 3D parallelopipeds

#ifndef __vtkMimxExtractFaceWidget_h
#define __vtkMimxExtractFaceWidget_h

#include "vtkAbstractWidget.h"
#include "vtkUnstructuredGrid.h"
#include "vtkActor.h"

class vtkActor;
class vtkCellPicker;
class vtkCollection;
class vtkPoints;
class vtkUnstructuredGrid;
class vtkDataSetMapper;
class vtkPolyData;
class vtkPolyDataMapper;
class vtkProp;
class vtkProperty;
class vtkActorCollection;
class vtkIdList;

class vtkMimxExtractFaceWidget : public vtkAbstractWidget
{
 public:
  // Description:
  // Instantiate the object.
  static vtkMimxExtractFaceWidget *New();

  vtkTypeRevisionMacro(vtkMimxExtractFaceWidget,vtkAbstractWidget);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Override the superclass method. This is a composite widget, (it internally
  // consists of handle widgets). We will override the superclass method, so
  // that we can pass the enabled state to the internal widgets as well.
  virtual void SetEnabled(int);

  virtual void SetInput(vtkDataSet*);
  void CreateDefaultRepresentation(){}

  //  vtkGetMacro(FacePoints, vtkIdList*);
  vtkSetObjectMacro(CompleteUGrid, vtkUnstructuredGrid);
  vtkSetObjectMacro(InputActor, vtkActor);

  vtkGetObjectMacro(PickedCellList, vtkIdList);
  vtkGetObjectMacro(PickedFaceList, vtkIdList);
  vtkGetObjectMacro(CompletePickedCellList, vtkIdList);
  vtkGetObjectMacro(CompletePickedFaceList, vtkIdList);
 protected:
  vtkMimxExtractFaceWidget();
  ~vtkMimxExtractFaceWidget();

  vtkUnstructuredGrid *UGrid;
  //BTX - manage the state of the widget
 
  //BTX - manage the state of the widget
  int State;
  enum WidgetEventIds 
  {
          Start = 0,
          Outside,
          ShiftLeftMouseButtonDown,
          ShiftLeftMouseButtonUp
  };

  enum WidgetState
  {
          StartWidget = 0,
          PickMultipleFace
  };
  //ETX

  void ExtractFace();
  void Initialize();

  static void ShiftLeftButtonDownCallback                       (vtkAbstractWidget* );
  static void ShiftLeftButtonUpCallback            (vtkAbstractWidget* );

  // the hexahedron (6 faces)
  vtkActor          *FaceActor;
  vtkActor                      *InputActor;
  vtkPolyDataMapper *FaceMapper;
  vtkPolyData       *FacePolyData;
  vtkActor          *SelectedFaceActor;
  vtkPolyDataMapper *SelectedFaceMapper;
  vtkCellPicker *FacePicker;

  vtkActorCollection *FaceGeometry;
  vtkActorCollection *InteriorFaceGeometry;
  vtkUnstructuredGrid *CompleteUGrid;
  vtkIdList *PickedCellList;
  vtkIdList *PickedFaceList;
  vtkIdList *CompletePickedCellList;
  vtkIdList *CompletePickedFaceList;
  vtkIdList *SurfaceCellList;
  int GetInputPickedCellAndFace(int PickedFace, int &CellFace);
  int GetInputPickedCompleteFace(
          int CellNum, int CellFace, int &CompleteCell, int &CompleteFace);
  double LastPickPosition[3];
  void RemoveHighlightedFaces(vtkMimxExtractFaceWidget *Self);
  void ShowInteriorHighlightedFaces();
  void HideInteriorHighlightedFaces();
  void DeleteInteriorHighlightedFaces();
  void ComputeInteriorHighlightedFaces();
private:
  vtkMimxExtractFaceWidget(const vtkMimxExtractFaceWidget&);  //Not implemented
  void operator=(const vtkMimxExtractFaceWidget&);  //Not implemented
};

#endif
