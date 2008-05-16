/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxEditBBGroup.h,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.26 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/
 
Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
// .NAME vtkKWMimxEditBBGroup - a tabbed notebook of UI pages
// .SECTION Description
// The class is derived from vtkKWCompositeWidget. It contains 5 pages 1) Image
// 2) Surface 3) Building Block 4) F E Mesh 5) Mesh quality. Each page inturn
// contains a list of options specific to a page.

#ifndef __vtkKWMimxEditBBGroup_h
#define __vtkKWMimxEditBBGroup_h

#include "vtkKWMimxGroupBase.h"
#include "vtkKWMimxFEMeshMenuGroup.h"
#include "vtkKWMimxMainWindow.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

class vtkKWMenuButtonWithLabel;
class vtkKWFrameWithLabel;
class vtkKWPushButton;
class vtkKWPushButtonSet;
class vtkKWCheckButtonSet;

class vtkLinkedListWrapper;
class vtkKWRadioButton;
class vtkKWRadioButtonSet;
class vtkKWComboBoxWithLabel;
class vtkKWEntryWithLabel;
class vtkKWRenderWidget;
class vtkKWMimxMergeBBGroup;
class vtkKWMimxMirrorBBGroup;

class vtkMimxUnstructuredGridWidget;
class vtkMimxExtractEdgeWidget;
class vtkMimxExtractFaceWidget;
class vtkMimxExtractCellWidget;
class vtkMimxSelectCellsWidget;



class vtkKWMimxEditBBGroup : public vtkKWMimxGroupBase
{
public:
  static vtkKWMimxEditBBGroup* New();
  vtkTypeRevisionMacro(vtkKWMimxEditBBGroup,vtkKWCompositeWidget);
  void PrintSelf(ostream& os, vtkIndent indent);
  virtual void Update();
  virtual void UpdateEnableState();
  vtkGetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  vtkSetObjectMacro(MimxMainWindow, vtkKWMimxMainWindow);
  vtkGetObjectMacro(ObjectListComboBox, vtkKWComboBoxWithLabel);
  vtkGetObjectMacro(DoUndoButtonSet, vtkKWPushButtonSet);
  void EditBBAddCellCallback(int Mode);
  void EditBBSplitCellCallback(int Mode);
  void EditBBDeleteCellCallback(int Mode);
  void EditBBDoneCallback();
  void EditBBCancelCallback();
  int EditBBApplyCallback();
  void EditBBMoveCellCallback(int Mode);
  void EditBBVtkInteractionCallback(int Mode);
  void EditBBMirrorCallback(int Mode);
  void EditBBMergeCallback(int Mode);
  void EditConvertToHBBCallback(int Mode);
  void SelectionChangedCallback(const char*);
  void RadiusChangeCallback(const char*);
  void UpdateObjectLists();
  void DoBBCallback();
  void UndoBBCallback();
  void SelectFullSetCallback();
  void ApplySelectSubsetCallback();
  void SelectSubsetCallback();
  void CancelSelectSubsetCallback();
  void AddEditedBB(int BBNum, vtkUnstructuredGrid *output, 
          const char* name, vtkIdType& count);
  void DeselectAllButtons();
  void SetDoUndoButtonSelectSubsetButton();
  void RepackEntryFrame();
  void RepackMirrorFrame();

  void PlaceMirroringPlaneAboutX();
  void PlaceMirroringPlaneAboutY();
  void PlaceMirroringPlaneAboutZ();

protected:
        vtkKWMimxEditBBGroup();
        ~vtkKWMimxEditBBGroup();
        virtual void CreateWidget();
  vtkKWComboBoxWithLabel *ObjectListComboBox;
  vtkKWMimxMainWindow *MimxMainWindow;
  //vtkKWRadioButton *MoveButton;
  //vtkKWRadioButton *AddButton;
  //vtkKWRadioButton *DeleteButton;
  //vtkKWRadioButton *SplitButton;
  //vtkKWRadioButton *VtkInteractionButton;
  vtkKWCheckButtonSet *RadioButtonSet;
  vtkKWPushButtonSet *DoUndoButtonSet;
  vtkKWRadioButtonSet *SelectSubsetRadiobuttonSet;
  vtkKWFrame *ButtonFrame;
  vtkKWFrame *EntryFrame;
  vtkMimxUnstructuredGridWidget *UnstructuredGridWidget;
  vtkMimxExtractEdgeWidget *ExtractEdgeWidget;
  vtkMimxExtractFaceWidget *ExtractFaceWidget;
  vtkMimxExtractCellWidget *ExtractCellWidget;
  vtkMimxExtractCellWidget *ExtractCellWidgetHBB;
  vtkIdType AddButtonState;
  vtkIdType DeleteButtonState;
  vtkIdType MoveButtonState;
  vtkIdType RegularButtonState;
  vtkIdType SplitButtonState;
  vtkIdType MirrorButtonState;
  vtkIdType ConvertToHBBButtonState;
//  vtkIdType SelectionState;
  vtkIdType SplitCount;
  vtkIdType AddCount;
  vtkIdType DeleteCount;
  vtkIdType MirrorCount;
  vtkIdType ConvertToHBBCount;
  vtkIdType MergeCount;
  vtkIdType CancelStatus;
  vtkKWEntryWithLabel *RadiusEntry;
 // vtkKWFrameWithLabel *EditButtonFrame;
  vtkKWMimxMergeBBGroup *MergeBBGroup;
  vtkKWMimxMirrorBBGroup *MirrorBBGroup;
  vtkMimxSelectCellsWidget *SelectCellsWidget;

  vtkKWFrameWithLabel *MirrorFrame;
  vtkKWCheckButtonWithLabel *TypeOfMirroring;
  vtkKWRadioButtonSet *AxisSelection;
  vtkPlaneWidget *MirrorPlaneWidget;

  double defaultRadiusEntry;
  double defaultExtrusionLength;
  double defaultMergeTolerance;
  double Parameter;
  double OriginalRadius;
private:
  vtkKWMimxEditBBGroup(const vtkKWMimxEditBBGroup&); // Not implemented
  void operator=(const vtkKWMimxEditBBGroup&); // Not implemented
 };

#endif

