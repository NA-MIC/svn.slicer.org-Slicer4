/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxCreateElementSetWidgetFEMesh.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMimxCreateElementSetWidgetFEMesh.h"

#include "vtkActor.h"
#include "vtkAssemblyPath.h"
#include "vtkCallbackCommand.h"
#include "vtkCamera.h"
#include "vtkCell.h"
#include "vtkCellData.h"
#include "vtkCellLocator.h"
#include "vtkCleanPolyData.h"
#include "vtkCommand.h"
#include "vtkDataSet.h"
#include "vtkDataSetMapper.h"
#include "vtkExtractSelectedFrustum.h"
#include "vtkExtractSelection.h"
#include "vtkExtractSelectedIds.h"
#include "vtkExtractSelectedPolyDataIds.h"
#include "vtkEvent.h"
#include "vtkGarbageCollector.h"
#include "vtkGeometryFilter.h"
#include "vtkHandleWidget.h"
#include "vtkIdFilter.h"
#include "vtkIdList.h"
#include "vtkIdTypeArray.h"
#include "vtkInformation.h"
#include "vtkInteractorObserver.h"
#include "vtkInteractorStyleRubberBandPick.h"
#include "vtkLookupTable.h"
#include "vtkObjectFactory.h"
#include "vtkPointLocator.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderedAreaPicker.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkRenderWindow.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataSource.h"
#include "vtkSelection.h"
#include "vtkVisibleCellSelector.h"
#include "vtkWidgetCallbackMapper.h" 
#include "vtkWidgetEvent.h"
#include "vtkWidgetEventTranslator.h"
#include "vtkCellPicker.h"
#include "vtkGeometryFilter.h"
#include "vtkMimxUnstructuredToStructuredGrid.h"
#include "vtkMimxExtractStructuredGridFace.h"
#include "vtkStructuredGridGeometryFilter.h"
#include "vtkUnstructuredGrid.h"

#include "vtkUnstructuredGridWriter.h"
#include "vtkPolyDataWriter.h"
#include "vtkStructuredGridWriter.h"


vtkCxxRevisionMacro(vtkMimxCreateElementSetWidgetFEMesh, "$Revision: 1.9 $");
vtkStandardNewMacro(vtkMimxCreateElementSetWidgetFEMesh);

//----------------------------------------------------------------------
vtkMimxCreateElementSetWidgetFEMesh::vtkMimxCreateElementSetWidgetFEMesh()
{
  this->CallbackMapper->SetCallbackMethod(
            vtkCommand::LeftButtonPressEvent,
            vtkEvent::ControlModifier, 0, 1, NULL,
                        vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonDown,
                        this, vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftButtonDownCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::MouseMoveEvent,
          vtkEvent::ControlModifier, 0, 1, NULL,
          vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonMove,
          this, vtkMimxCreateElementSetWidgetFEMesh::CrtlMouseMoveCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::MouseMoveEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxCreateElementSetWidgetFEMesh::ShiftMouseMove,
          this, vtkMimxCreateElementSetWidgetFEMesh::ShiftMouseMoveCallback);

  this->CallbackMapper->SetCallbackMethod(
            vtkCommand::LeftButtonReleaseEvent,
                        vtkEvent::ControlModifier, 0, 1, NULL,
            vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonUp,
            this, vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftButtonUpCallback);

  this->CallbackMapper->SetCallbackMethod(vtkCommand::LeftButtonReleaseEvent,
          vtkMimxCreateElementSetWidgetFEMesh::LeftMouseButtonUp,
          this, vtkMimxCreateElementSetWidgetFEMesh::LeftButtonUpCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::LeftButtonPressEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftMouseButtonDown,
          this, vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftButtonDownCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::LeftButtonReleaseEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftMouseButtonUp,
          this, vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftButtonUpCallback);

  this->RubberBandStyle =  vtkInteractorStyleRubberBandPick::New();
  this->AreaPicker = vtkRenderedAreaPicker::New();
  this->Input = vtkUnstructuredGrid::New();
  this->InputActor = NULL;
  this->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Start;
  this->SelectedCellIds = vtkIdList::New();
  this->SelectedPointIds = vtkIdList::New();
  this->DeleteCellIds = vtkIdList::New();
  this->PickX0 = -1;    this->PickY0 = -1;      this->PickX1 = -1;      this->PickY1 = -1;
  this->CellSelectionState = 0;
  this->ExtractedGrid = vtkUnstructuredGrid::New();
  vtkDataSetMapper *extractedmapper = vtkDataSetMapper::New();
  extractedmapper->SetInput(this->ExtractedGrid);
  this->ExtractedActor = vtkActor::New();
  this->ExtractedActor->SetMapper(extractedmapper);
  extractedmapper->Delete();
  this->ExtractedActor->GetProperty()->SetColor(1.0,0.0,0.0);
  this->ExtractedActor->GetProperty()->SetRepresentationToWireframe();
  this->ExtractedActor->GetProperty()->SetLineWidth(2.0);
  vtkDataSetMapper *mapper = vtkDataSetMapper::New();
  mapper->SetInput(this->Input);
  this->SelectedActor = vtkActor::New() ;
  this->SelectedActor->SetMapper(mapper);
  mapper->Delete();
  this->OriginalInput = NULL;
  this->PointLocator = NULL;
  this->LocatorPoints = NULL;
  this->InputLocator = NULL;
  this->InputPoints = NULL;
  this->PickStatus = 0;
  this->SurfaceFilter = vtkGeometryFilter::New();
  this->SurfaceMapper = vtkPolyDataMapper::New();
  this->SurfaceActor = vtkActor::New();
  this->SurfaceActor->SetMapper(this->SurfaceMapper);
}

//----------------------------------------------------------------------
vtkMimxCreateElementSetWidgetFEMesh::~vtkMimxCreateElementSetWidgetFEMesh()
{
        this->RubberBandStyle->Delete();
        this->AreaPicker->Delete();

        if(this->SelectedCellIds)
                this->SelectedCellIds->Delete();
        
        if (this->SelectedPointIds)
        {
                this->SelectedPointIds->Delete();
        }
        this->DeleteCellIds->Delete();
        if(this->ExtractedActor)
        {
                this->ExtractedActor->Delete();
        }
        if(this->SelectedActor)
                this->SelectedActor->Delete();
        if(this->PointLocator)
                this->PointLocator->Delete();
        if(this->LocatorPoints)
                this->LocatorPoints->Delete();  
        if(this->InputLocator)
                this->InputLocator->Delete();
        if(this->InputPoints)
                this->InputPoints->Delete();
        this->SurfaceFilter->Delete();
        this->SurfaceActor->Delete();
        this->SurfaceMapper->Delete();
}

//----------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SetEnabled(int enabling)
{  
  if ( enabling ) //----------------
    {
    vtkDebugMacro(<<"Enabling widget");
  
    if ( this->Enabled ) //already enabled, just return
      {
      return;
      }
  
    if ( ! this->Interactor )
      {
      vtkErrorMacro(<<"The interactor must be set prior to enabling the widget");
      return;
      }
        if(!this->Input)
        {
                vtkErrorMacro(<<"Input Should be Set");
                return;
        }

    int X=this->Interactor->GetEventPosition()[0];
    int Y=this->Interactor->GetEventPosition()[1];
  
    if ( ! this->CurrentRenderer )
      {
                  this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(X,Y));

                  if (this->CurrentRenderer == NULL)
                  {
                          return;
                  }

                  if(this->InputActor)
                  {
                          this->CurrentRenderer->RemoveActor(this->InputActor);
                  }
                  if (this->SelectedActor)
                  {
                          this->CurrentRenderer->AddActor(this->SelectedActor);
                  }
      }
  
    // We're ready to enable
    this->Enabled = 1;
  
    // listen for the events found in the EventTranslator
    if ( ! this->Parent )
      {
      this->EventTranslator->AddEventsToInteractor(this->Interactor,
        this->EventCallbackCommand,this->Priority);
      }
    else
      {
      this->EventTranslator->AddEventsToParent(this->Parent,
        this->EventCallbackCommand,this->Priority);
      }
        
        if ( !this->Interactor )
        {
                vtkErrorMacro(<<"The interactor must be set prior to enabling/disabling widget");
                return;
        }

        this->Interactor->SetInteractorStyle(this->RubberBandStyle);
//      this->RubberBandStyle->SetEnabled(1);

        this->Interactor->SetPicker(this->AreaPicker);
    }
  
  else //disabling------------------
    {
    vtkDebugMacro(<<"Disabling widget");
  
    if ( ! this->Enabled ) //already disabled, just return
      {
      return;
      }
  
    this->Enabled = 0;
  
    // don't listen for events any more
    if ( ! this->Parent )
      {
      this->Interactor->RemoveObserver(this->EventCallbackCommand);
      }
    else
      {
      this->Parent->RemoveObserver(this->EventCallbackCommand);
      }

//      this->RubberBandStyle->OnChar();
        if (this->InputActor)
        {
                this->CurrentRenderer->AddActor(this->InputActor);
        }
        if(this->ExtractedActor)
        {
                this->CurrentRenderer->RemoveActor(this->ExtractedActor);
        }
        if (this->SelectedActor)
        {
                this->CurrentRenderer->RemoveActor(this->SelectedActor);
        }
        this->CurrentRenderer->RemoveActor(this->SelectedActor);
        if(this->SurfaceActor)
                this->CurrentRenderer->RemoveActor(this->SurfaceActor);

        this->SelectedCellIds->Initialize();
        this->SelectedPointIds->Initialize();
        this->DeleteCellIds->Initialize();
    this->InvokeEvent(vtkCommand::DisableEvent,NULL);
    this->SetCurrentRenderer(NULL);
    }
  
  // Should only render if there is no parent
  if ( this->Interactor && !this->Parent )
    {
    this->Interactor->Render();
    }
}
//--------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftButtonDownCallback(vtkAbstractWidget *w)
{
        vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
        if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectMultipleCells)        return;
        self->PickStatus = 1;
        if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectIndividualCell)       
        {
                self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonDown;
                self->EventCallbackCommand->SetAbortFlag(1);
                self->StartInteraction();
                self->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
                self->Interactor->Render();
                return;
        }
        if(self->RubberBandStyle->GetEnabled())
        {
                self->RubberBandStyle->SetEnabled(0);
                self->RubberBandStyle->OnChar();
        }
        self->RubberBandStyle->SetEnabled(1);
        int *size;
        size = self->Interactor->GetRenderWindow()->GetSize();
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];
//      cout <<X<<"  "<<Y<<endl;
        self->PickX0 = X;
        self->PickY0 = Y;
        // Okay, make sure that the pick is in the current renderer
        if ( !self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y) )
        {
                self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Outside;
                return;
        }

        self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonDown;
        if(self->Input->GetCellData()->GetArray("Mimx_Scalars"))
        {
                self->Input->GetCellData()->RemoveArray("Mimx_Scalars");
                self->Input->Modified();
        }
        self->RubberBandStyle->GetInteractor()->SetKeyCode('r');
        self->RubberBandStyle->OnChar();
        self->RubberBandStyle->OnLeftButtonDown();
        self->EventCallbackCommand->SetAbortFlag(1);
        self->StartInteraction();
        self->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
        self->Interactor->Render();
}
//----------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftButtonUpCallback(vtkAbstractWidget *w)
{
  vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
  if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectMultipleCells)      return;

  int *size;
  size = self->Interactor->GetRenderWindow()->GetSize();

  int X = self->Interactor->GetEventPosition()[0];
  int Y = self->Interactor->GetEventPosition()[1];

 // cout <<X<<"  "<<Y<<endl;

  self->PickX1 = X;
  self->PickY1 = Y;

  if ( self->WidgetEvent == vtkMimxCreateElementSetWidgetFEMesh::Outside ||
          self->WidgetEvent == vtkMimxCreateElementSetWidgetFEMesh::Start )
  {
          return;
  }

  if(self->ExtractedActor)
          self->CurrentRenderer->RemoveActor(self->ExtractedActor);
  if(self->WidgetEvent == vtkMimxCreateElementSetWidgetFEMesh::CrtlLeftMouseButtonDown)
  {
          self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Start;

          if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectVisibleCellsOnSurface)
          {
                  vtkMimxCreateElementSetWidgetFEMesh::SelectVisibleCellsOnSurfaceFunction(self);
          }
          if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectCellsOnSurface)
          {
                  vtkMimxCreateElementSetWidgetFEMesh::SelectCellsOnSurfaceFunction(self);
          }
          if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectCellsThrough)
          {
                  vtkMimxCreateElementSetWidgetFEMesh::SelectCellsThroughFunction(self);
          }
          if (self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectIndividualCell)
          {
                        vtkMimxCreateElementSetWidgetFEMesh::ExtractElementsBelongingToAFace(self);
                        return;
          }
  }
  self->RubberBandStyle->GetInteractor()->SetKeyCode('r');
  self->RubberBandStyle->OnChar();
  self->RubberBandStyle->OnLeftButtonUp();
  self->RubberBandStyle->SetEnabled(0);
  self->EventCallbackCommand->SetAbortFlag(1);
  self->EndInteraction();
  self->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
  self->PickStatus = 0;
  self->Interactor->Render();
}
//----------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::CrtlMouseMoveCallback(vtkAbstractWidget *w)
{
  vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
  if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectMultipleCells)      return;

  if(self->CellSelectionState != vtkMimxCreateElementSetWidgetFEMesh::SelectIndividualCell)
  {
          self->RubberBandStyle->OnMouseMove();
  }
  else{
          self->SelectIndividualCellFunction(self);
  }
}
//---------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ComputeSelectedPointIds(
        vtkDataSet *ExtractedUGrid, vtkMimxCreateElementSetWidgetFEMesh *self)
{
        int i;
        vtkPoints *polypoints ;
        if(ExtractedUGrid->GetDataObjectType() == VTK_POLY_DATA)
                polypoints = vtkPolyData::SafeDownCast(ExtractedUGrid)->GetPoints();
        else
                polypoints = vtkUnstructuredGrid::SafeDownCast(ExtractedUGrid)->GetPoints();

        vtkPoints *inputpoints = self->OriginalInput->GetPoints();

        if(self->SelectedPointIds)
        {
                self->SelectedPointIds->Initialize();
        }
        else{
                self->SelectedPointIds = vtkIdList::New();
        }

        for (i=0; i<polypoints->GetNumberOfPoints(); i++)
        {
                int location = PointLocator->IsInsertedPoint(polypoints->GetPoint(i));
                if(location == -1)
                {
                        vtkErrorMacro("Point sets do not match");
                        self->SelectedPointIds->Initialize();
                        return;
                }
                else
                {
                        self->SelectedPointIds->InsertNextId(location);
                }
        }
}
//----------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);  
}
//-----------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SelectVisibleCellsOnSurfaceFunction(
        vtkMimxCreateElementSetWidgetFEMesh *self)
{
        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();

        vtkPolyDataMapper *filmapper = vtkPolyDataMapper::New();
        vtkActor *filactor = vtkActor::New();
        filmapper->SetInput(fil->GetOutput());
        filactor->SetMapper(filmapper);
        filactor->PickableOn();
        self->CurrentRenderer->AddActor(filactor);
        self->CurrentRenderer->Render();

        vtkVisibleCellSelector *select = vtkVisibleCellSelector::New();
        select->SetRenderer(self->CurrentRenderer);

        double x0 = self->PickX0;
        double y0 = self->PickY0;
        double x1 = self->PickX1;
        double y1 = self->PickY1;

        select->SetRenderPasses(0,1,0,1,1);
        select->SetArea(static_cast<int>(x0),static_cast<int>(y1),static_cast<int>(x1),
                static_cast<int>(y0));
        select->Select();

        vtkSelection *res = vtkSelection::New();
        select->GetSelectedIds(res);

        vtkSelection *cellidssel = res->GetChild(0);
        vtkExtractSelectedPolyDataIds *extr = vtkExtractSelectedPolyDataIds::New();
        if (cellidssel)
        {
                extr->SetInput(0, fil->GetOutput());
                extr->SetInput(1, cellidssel);
                extr->Update();
        }

        if (extr->GetOutput()->GetNumberOfCells() < 1)
        {
                select->Delete();
                res->Delete();
                extr->Delete();
                filmapper->Delete();
                fil->Delete();
                self->CurrentRenderer->RemoveActor(filactor);
                filactor->Delete();
                return;
        }
        self->ComputeSelectedCellIds(self, extr->GetOutput());
        select->Delete();
        res->Delete();
        extr->Delete();
        filmapper->Delete();
        fil->Delete();
        self->CurrentRenderer->RemoveActor(filactor);
        filactor->Delete();

//      vtkSelection *selection = vtkSelection::New();
//      vtkExtractSelection *ext = vtkExtractSelection::New();
//      ext->SetInput(0, self->Input);
//      ext->SetInput(1, selection);
//
//      vtkIdTypeArray *globalids = vtkIdTypeArray::New();
//      globalids->SetNumberOfComponents(1);
//      globalids->SetName("GIds");
////    globalids->SetNumberOfTuples(self->Input->GetNumberOfCells());
//      for (i=0; i<self->Input->GetNumberOfCells(); i++)
//      {
//              globalids->InsertNextValue(i);
//      }
//      self->Input->GetCellData()->AddArray(globalids);
//      self->Input->GetCellData()->SetGlobalIds(globalids);
//
//      selection->Clear();
//      selection->GetProperties()->Set(
//              vtkSelection::CONTENT_TYPE(), vtkSelection::GLOBALIDS);
//      vtkIdTypeArray *cellIds = vtkIdTypeArray::New();
//      cellIds->SetNumberOfComponents(1);
//      cellIds->SetNumberOfTuples(self->SelectedCellIds->GetNumberOfIds());
//      for (i=0; i<self->SelectedCellIds->GetNumberOfIds(); i++)
//      {
//              cellIds->SetTuple1(i, self->SelectedCellIds->GetId(i));
//      }
//      selection->SetSelectionList(cellIds);
//      cellIds->Delete();
//      
//      ext->Update();
//
//      self->Input->GetCellData()->RemoveArray("GIds");
//      globalids->Delete();
//      if(self->ExtractedActor)
//      {
//              self->CurrentRenderer->RemoveActor(self->ExtractedActor);
//              self->ExtractedActor->Delete();
//      }
//      self->ExtractedActor = vtkActor::New();
//      vtkDataSetMapper *mapper = vtkDataSetMapper::New();
//      mapper->SetInput(ext->GetOutput());
//      self->ExtractedActor->SetMapper(mapper);
//      mapper->Delete();
//      if(self->InputActor)
//              self->CurrentRenderer->AddActor(self->InputActor);
//      self->CurrentRenderer->AddActor(self->ExtractedActor);
//      self->ExtractedActor->GetProperty()->SetColor(1.0, 0.0,0.0);
//      self->ExtractedActor->GetProperty()->SetRepresentationToWireframe();
//      selection->Delete();
//      ext->Delete();
}
//-----------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SelectCellsOnSurfaceFunction(
        vtkMimxCreateElementSetWidgetFEMesh *self)
{
        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();

        vtkPolyDataMapper *filmapper = vtkPolyDataMapper::New();
        vtkActor *filactor = vtkActor::New();
        filmapper->SetInput(fil->GetOutput());
        filactor->SetMapper(filmapper);
        filactor->PickableOn();
        self->CurrentRenderer->AddActor(filactor);
        if(self->InputActor)
                self->CurrentRenderer->RemoveActor(self->InputActor);
        self->CurrentRenderer->Render();

        double x0 = self->PickX0;
        double y0 = self->PickY0;
        double x1 = self->PickX1;
        double y1 = self->PickY1;
        self->AreaPicker->AreaPick(static_cast<int>(x0), static_cast<int>(y0), 
                static_cast<int>(x1), static_cast<int>(y1), self->CurrentRenderer);
        vtkExtractSelectedFrustum *Extract = vtkExtractSelectedFrustum::New();
        Extract->SetInput(fil->GetOutput());
//      Extract->PassThroughOff();
        Extract->SetFrustum(self->AreaPicker->GetFrustum());
        Extract->Update();
        vtkUnstructuredGrid *exugrid = vtkUnstructuredGrid::SafeDownCast(
                Extract->GetOutput());  
        if (exugrid->GetNumberOfCells() < 1)
        {
                Extract->Delete();
                filmapper->Delete();
                fil->Delete();
                self->CurrentRenderer->RemoveActor(filactor);
                filactor->Delete();
                return;
        }
        self->ComputeSelectedCellIds(self, exugrid);
        Extract->Delete();
        filmapper->Delete();
        fil->Delete();
        self->CurrentRenderer->RemoveActor(filactor);
        filactor->Delete();

        //vtkSelection *selection = vtkSelection::New();
        //vtkExtractSelection *ext = vtkExtractSelection::New();
        //ext->SetInput(0, self->Input);
        //ext->SetInput(1, selection);

        //vtkIdTypeArray *globalids = vtkIdTypeArray::New();
        //globalids->SetNumberOfComponents(1);
        //globalids->SetName("GIds");
        ////    globalids->SetNumberOfTuples(self->Input->GetNumberOfCells());
        //for (i=0; i<self->Input->GetNumberOfCells(); i++)
        //{
        //      globalids->InsertNextValue(i);
        //}
        //self->Input->GetCellData()->AddArray(globalids);
        //self->Input->GetCellData()->SetGlobalIds(globalids);

        //selection->Clear();
        //selection->GetProperties()->Set(
        //      vtkSelection::CONTENT_TYPE(), vtkSelection::GLOBALIDS);
        //vtkIdTypeArray *cellIds = vtkIdTypeArray::New();
        //cellIds->SetNumberOfComponents(1);
        //cellIds->SetNumberOfTuples(self->SelectedCellIds->GetNumberOfIds());
        //for (i=0; i<self->SelectedCellIds->GetNumberOfIds(); i++)
        //{
        //      cellIds->SetTuple1(i, self->SelectedCellIds->GetId(i));
        //}
        //selection->SetSelectionList(cellIds);
        //cellIds->Delete();

        //ext->Update();

        //self->Input->GetCellData()->RemoveArray("GIds");
        //globalids->Delete();
        //if(self->ExtractedActor)
        //{
        //      self->CurrentRenderer->RemoveActor(self->ExtractedActor);
        //      self->ExtractedActor->Delete();
        //}
        //self->ExtractedActor = vtkActor::New();
        //vtkDataSetMapper *mapper = vtkDataSetMapper::New();
        //mapper->SetInput(ext->GetOutput());
        //self->ExtractedActor->SetMapper(mapper);
        //mapper->Delete();
        //if(self->InputActor)
        //      self->CurrentRenderer->AddActor(self->InputActor);
        //self->CurrentRenderer->AddActor(self->ExtractedActor);
        //self->ExtractedActor->GetProperty()->SetColor(1.0, 0.0,0.0);
        //self->ExtractedActor->GetProperty()->SetLineWidth(2);
        //self->ExtractedActor->GetProperty()->SetRepresentationToWireframe();
        //selection->Delete();
        //ext->Delete();
}
//-----------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SelectCellsThroughFunction(
        vtkMimxCreateElementSetWidgetFEMesh *self)
{
        double x0 = self->PickX0;
        double y0 = self->PickY0;
        double x1 = self->PickX1;
        double y1 = self->PickY1;
        self->AreaPicker->AreaPick(static_cast<int>(x0), static_cast<int>(y0), 
                static_cast<int>(x1), static_cast<int>(y1), self->CurrentRenderer);
        vtkExtractSelectedFrustum *Extract = vtkExtractSelectedFrustum::New();
        Extract->SetInput(self->Input);
//      Extract->PassThroughOff();
        Extract->SetFrustum(self->AreaPicker->GetFrustum());
        Extract->Update();
        vtkUnstructuredGrid *exugrid = vtkUnstructuredGrid::SafeDownCast(
                Extract->GetOutput());  
        if(exugrid->GetNumberOfCells() <1)
        {
                Extract->Delete();
                return;
        }
        self->ComputeSelectedCellIds(self, exugrid);
        Extract->Delete();
}
//-----------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SelectIndividualCellFunction(
        vtkMimxCreateElementSetWidgetFEMesh *self)
{
        //int X = self->Interactor->GetEventPosition()[0];
        //int Y = self->Interactor->GetEventPosition()[1];

        //// Okay, we can process this. Try to pick handles first;
        //// if no handles picked, then pick the bounding box.
        //if (!self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y))
        //{
        //      return;
        //}
        //int i;
        //vtkAssemblyPath *path;
        //vtkCellPicker *CellPicker = vtkCellPicker::New();
        //CellPicker->SetTolerance(0.01);
        //CellPicker->AddPickList(self->InputActor);
        //CellPicker->PickFromListOn();
        //CellPicker->Pick(X,Y,0.0,self->CurrentRenderer);
        //path = CellPicker->GetPath();
        //if ( path != NULL )
        //{
        //      vtkIdType PickedCell = CellPicker->GetCellId();
        //      if(PickedCell != -1)
        //      {
        //              self->SelectedCellIds->Initialize();
        //              self->SelectedCellIds->SetNumberOfIds(1);
        //              self->SelectedCellIds->SetId(0, PickedCell);
        //              if(self->SingleCellActor)
        //              {
        //                      self->CurrentRenderer->RemoveActor(self->SingleCellActor);      
        //                      self->SingleCellActor->Delete();
        //              }
        //              self->SingleCellActor = vtkActor::New();
        //              vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::New();
        //              vtkIdList *ptids = vtkIdList::New();
        //              self->Input->GetCellPoints(PickedCell,ptids);
        //              vtkPoints *points = vtkPoints::New();
        //              for (i=0; i<ptids->GetNumberOfIds(); i++)
        //              {
        //                      points->InsertNextPoint(self->Input->GetPoint(ptids->GetId(i)));
        //                      ptids->SetId(i,i);
        //              }
        //              ugrid->Allocate(1,1);
        //              ugrid->InsertNextCell(self->Input->GetCellType(PickedCell), ptids);
        //              ugrid->SetPoints(points);
        //              points->Delete();
        //              vtkDataSetMapper *mapper = vtkDataSetMapper::New();
        //              mapper->SetInput(ugrid);
        //              ugrid->Delete();
        //              self->SingleCellActor->SetMapper(mapper);
        //              mapper->Delete();
        //              ptids->Delete();
        //              self->SingleCellActor->GetProperty()->SetColor(1.0,0.66,0.33);
        //              self->SingleCellActor->GetProperty()->SetLineWidth(
        //                      self->InputActor->GetProperty()->GetLineWidth()*2.0);
        //              self->SingleCellActor->GetProperty()->SetRepresentationToWireframe();
        //              self->CurrentRenderer->AddActor(self->SingleCellActor); 
        //      }
        //}
        //self->Interactor->Render();
        //CellPicker->Delete();

        int *size;
        size = self->Interactor->GetRenderWindow()->GetSize();
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];

        // Okay, make sure that the pick is in the current renderer
        if ( !self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y) )
        {
                self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Outside;
                return;
        }

        self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::ShiftMouseMove;

        int i;
        vtkAssemblyPath *path;
        vtkCellPicker *CellPicker = vtkCellPicker::New();
        CellPicker->SetTolerance(0.01);
        CellPicker->AddPickList(self->SurfaceActor);
        CellPicker->PickFromListOn();
        CellPicker->Pick(X,Y,0.0,self->CurrentRenderer);
        path = CellPicker->GetPath();
        vtkPolyData *Surface = self->SurfaceFilter->GetOutput();
        if ( path != NULL )
        {
                vtkIdType PickedCell = CellPicker->GetCellId();
                if(PickedCell != -1)
                {
                        vtkLookupTable *lut = vtkLookupTable::New();
                        lut->SetNumberOfColors(2);
                        lut->Build();
                        lut->SetTableValue(0, 1.0, 1.0, 1.0, 1.0);
                        lut->SetTableValue(1, 1.0, 0.0, 0.0, 1.0);
                        lut->SetTableRange(0,1);
                        self->SurfaceActor->GetMapper()->SetLookupTable(lut);
                        self->SurfaceActor->GetMapper()->SetScalarRange(0,1);
                        lut->Delete();
                        vtkIntArray *intarray = vtkIntArray::New();
                        intarray->SetNumberOfValues(Surface->GetNumberOfCells());
                        for (i=0; i<Surface->GetNumberOfCells(); i++)
                        {
                                intarray->SetValue(i, 0);
                        }
                        intarray->SetValue(PickedCell, 1);
                        intarray->SetName("Mimx_Scalars");
                        Surface->GetCellData()->SetScalars(intarray);
                        intarray->Delete();
                        Surface->Modified();
                }
        }
        self->Interactor->Render();
        CellPicker->Delete();
}
//--------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ComputeSelectedCellIds(
        vtkMimxCreateElementSetWidgetFEMesh *Self, vtkDataSet *DataSet)
{
        Self->ComputeSelectedPointIds(DataSet, Self);
        Self->SelectedCellIds->Initialize();
        Self->DeleteCellIds->Initialize();
        // first compute the cells selected
        int i, j, CellNum;
        for (i=0; i<DataSet->GetNumberOfCells(); i++)
        {
                vtkIdList *PtIds = DataSet->GetCell(i)->GetPointIds();
                vtkIdList *OrIds = vtkIdList::New();
                OrIds->SetNumberOfIds(PtIds->GetNumberOfIds());
                for (j=0; j<PtIds->GetNumberOfIds(); j++)
                {
                        OrIds->SetId(j, Self->SelectedPointIds->GetId(PtIds->GetId(j)));
                }
                CellNum = Self->ComputeOriginalCellNumber(Self, OrIds);
                OrIds->Delete();
                if(CellNum == -1)
                {
                        Self->SelectedCellIds->Initialize();
                        vtkErrorMacro("Cells chosen do not belong to the original set");
                        return;
                }
                else
                {
                        if(Self->DoesCellBelong(CellNum, Self))
                        {
                                Self->SelectedCellIds->InsertUniqueId(CellNum);
                        }
                }
        }

        // create a new selected element set based on computed selected cells
        Self->ExtractedGrid->Initialize();
        Self->ExtractedGrid->Allocate(DataSet->GetNumberOfCells(), DataSet->GetNumberOfCells());
        vtkPointLocator *locator = vtkPointLocator::New();
        vtkPoints *points = vtkPoints::New();
        points->Allocate(DataSet->GetNumberOfPoints()*4);
        locator->InitPointInsertion(points, Self->OriginalInput->GetPoints()->GetBounds());
        vtkIdList *idlist;
        double x[3];
        vtkIdType tempid;
        int numIds = Self->SelectedCellIds->GetNumberOfIds();
        Self->ExtractedGrid->Initialize();
        Self->ExtractedGrid->Allocate(numIds, numIds);
        for (i=0; i<numIds; i++)
        {
                idlist = vtkIdList::New();
                idlist->DeepCopy(Self->OriginalInput->GetCell(Self->SelectedCellIds->GetId(i))->GetPointIds());
                for(int j=0; j <8; j++)
                {
                        Self->OriginalInput->GetPoints()->GetPoint(idlist->GetId(j),x);          
                        locator->InsertUniquePoint(x,tempid);
                        idlist->SetId(j,tempid);
                }
                Self->ExtractedGrid->InsertNextCell(12, idlist);
                idlist->Delete();
        }

        points->Squeeze();
        Self->ExtractedGrid->SetPoints(points);
        Self->ExtractedGrid->Squeeze();
        points->Delete();
        locator->Delete();
        Self->ExtractedGrid->Modified();
        Self->CurrentRenderer->AddActor(Self->ExtractedActor); 
        Self->Interactor->Render();
        //vtkUnstructuredGridWriter *writer = vtkUnstructuredGridWriter::New();
        //writer->SetInput(Self->ExtractedGrid);
        //writer->SetFileName("ExtractUGrid.vtk");
        //writer->Write();
        //writer->Delete();
}
//---------------------------------------------------------------------------------------
int vtkMimxCreateElementSetWidgetFEMesh::ComputeOriginalCellNumber(
        vtkMimxCreateElementSetWidgetFEMesh *self, vtkIdList *PtIds)
{
        int i,j, k;
        if(PtIds->GetNumberOfIds() == 8)
        {
                for (i=0; i<self->OriginalInput->GetNumberOfCells(); i++)
                {
                        vtkIdList *InputIds = self->OriginalInput->GetCell(i)->GetPointIds();
                        int NumIds = InputIds->GetNumberOfIds();
                        bool status = true;
                        for (k=0; k<NumIds; k++)
                        {
                                if(PtIds->IsId(InputIds->GetId(k)) == -1)
                                {
                                        status = false;
                                        break;
                                }
                        }
                        if(status)
                                return i;
                        status = true;
                }
        }
        else
        {
                for (i=0; i<self->OriginalInput->GetNumberOfCells(); i++)
                {
                        vtkCell *Cell = self->OriginalInput->GetCell(i);
                        int NumFaces = Cell->GetNumberOfFaces();
                        bool status = true;
                        for (j=0; j<NumFaces; j++)
                        {
                                vtkIdList *InputIds = Cell->GetFace(j)->GetPointIds();
                                int NumIds = InputIds->GetNumberOfIds();
                                for (k=0; k<NumIds; k++)
                                {
                                        if(PtIds->IsId(InputIds->GetId(k)) == -1)
                                        {
                                                status = false;
                                                break;
                                        }
                                }
                                if(status)
                                        return i;
                                status = true;
                        }
                }
        }
        return -1;
}
//----------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::AcceptSelectedMesh(vtkMimxCreateElementSetWidgetFEMesh *Self)
{
        if(!Self->ExtractedGrid)        return;

        if(Self->WidgetEvent == vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftMouseButtonDown)
        {
                if(Self->DeleteCellIds->GetNumberOfIds())
                {
                        Self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Start;
                        Self->DeleteSelectedCells(Self);
                        vtkUnstructuredGrid *tempgrid = vtkUnstructuredGrid::New();
                        tempgrid->DeepCopy(Self->ExtractedGrid);
                        Self->ComputeSelectedCellIds(Self, tempgrid);
                        tempgrid->Delete();
                }
        }

        if(!Self->ExtractedGrid->GetNumberOfCells())    return;
        if(!Self->ExtractedGrid->GetNumberOfPoints())   return;
        if(Self->ExtractedActor)
        {
                Self->CurrentRenderer->RemoveActor(Self->ExtractedActor);
        }
        Self->Input->Initialize();
        Self->Input->DeepCopy(Self->ExtractedGrid);
        Self->Input->Modified();
        //
        if(Self->InputLocator)
                Self->InputLocator->Initialize();
        else
                Self->InputLocator = vtkPointLocator::New();
        if(Self->InputPoints)
                Self->InputPoints->Initialize();
        else
                Self->InputPoints = vtkPoints::New();
        Self->InputLocator->InitPointInsertion(Self->InputPoints, Self->Input->GetBounds());
        for (int i=0; i<Self->Input->GetNumberOfPoints(); i++)
                Self->InputLocator->InsertNextPoint(Self->Input->GetPoint(i));
        //
        Self->CurrentRenderer->AddActor(Self->SelectedActor); 
        if(Self->ExtractedActor)
                Self->CurrentRenderer->RemoveActor(Self->ExtractedActor); 
        if(Self->SurfaceActor)
                Self->CurrentRenderer->RemoveActor(Self->SurfaceActor);
        Self->ExtractedGrid->Initialize();
        Self->ExtractedGrid->Modified();
        Self->Interactor->Render();
}
//-------------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SetInput(vtkUnstructuredGrid *input)
{
        int i;
        if(this->Input)
                this->Input->Initialize();
        else
                this->Input = vtkUnstructuredGrid::New();
        this->Input->DeepCopy(input);
        vtkDataSetMapper::SafeDownCast(
                this->SelectedActor->GetMapper())->SetInput(this->Input);
        this->Input->Modified();
        //
//      this->CurrentRenderer->AddActor(this->SelectedActor); 
        this->SelectedActor->PickableOn();
        this->OriginalInput = input;
        //
        this->SurfaceFilter->SetInput(this->OriginalInput);
        this->SurfaceFilter->Update();
        this->SurfaceMapper->SetInput(this->SurfaceFilter->GetOutput());
        this->SurfaceMapper->Modified();
        //

        // build the locator
        if(this->PointLocator)
                this->PointLocator->Initialize();
        else
                this->PointLocator = vtkPointLocator::New();
        if(this->LocatorPoints)
                this->LocatorPoints->Initialize();
        else
                this->LocatorPoints = vtkPoints::New();
        this->PointLocator->InitPointInsertion(this->LocatorPoints, input->GetBounds());
        for (i=0; i<input->GetNumberOfPoints(); i++)
                this->PointLocator->InsertNextPoint(input->GetPoint(i));
        //
        if(this->InputLocator)
                this->InputLocator->Initialize();
        else
                this->InputLocator = vtkPointLocator::New();
        if(this->InputPoints)
                this->InputPoints->Initialize();
        else
                this->InputPoints = vtkPoints::New();
        this->InputLocator->InitPointInsertion(this->InputPoints, input->GetBounds());
        for (i=0; i<input->GetNumberOfPoints(); i++)
                this->InputLocator->InsertNextPoint(input->GetPoint(i));
        //
        this->SelectedCellIds->Initialize();
        this->SelectedPointIds->Initialize();
        this->DeleteCellIds->Initialize();
}
//--------------------------------------------------------------------------------------------
//void vtkMimxCreateElementSetWidgetFEMesh::CrtlRightButtonDownCallback(vtkAbstractWidget *w)
//{
//      vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
//      int *size;
//      size = self->Interactor->GetRenderWindow()->GetSize();
//      int X = self->Interactor->GetEventPosition()[0];
//      int Y = self->Interactor->GetEventPosition()[1];
//      //      cout <<X<<"  "<<Y<<endl;
//      self->PickX0 = X;
//      self->PickY0 = Y;
//      // Okay, make sure that the pick is in the current renderer
//      if ( !self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y) )
//      {
//              self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Outside;
//              return;
//      }
//      self->AcceptSelectedMesh(self);
//}
//-----------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftButtonDownCallback(vtkAbstractWidget *w)
{
        vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
        self->EventCallbackCommand->SetAbortFlag(1);
        self->StartInteraction();
        self->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
        self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftMouseButtonDown;
        self->Interactor->Render();
}
//-----------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ShiftLeftButtonUpCallback(vtkAbstractWidget *w)
{
        vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
        if(self->CellSelectionState != vtkMimxCreateElementSetWidgetFEMesh::SelectMultipleCells)        return;
        
        vtkIntArray *scalararray = vtkIntArray::SafeDownCast(
                self->Input->GetCellData()->GetArray("Mimx_Scalars"));
        int i;

        int numCells = self->Input->GetNumberOfCells();
        if(!scalararray)
        {
                scalararray = vtkIntArray::New();
                scalararray->SetNumberOfValues(numCells);
                scalararray->SetName("Mimx_Scalars");
                self->Input->GetCellData()->AddArray(scalararray);
                for (i=0; i<numCells; i++)
                {
                        scalararray->SetValue(i, 0);
                }
        }

        int *size;
        size = self->Interactor->GetRenderWindow()->GetSize();
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];

        // Okay, make sure that the pick is in the current renderer
        if ( !self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y) )
        {
                self->WidgetEvent = vtkMimxCreateElementSetWidgetFEMesh::Outside;
                return;
        }

        vtkAssemblyPath *path;
        vtkCellPicker *CellPicker = vtkCellPicker::New();
        CellPicker->SetTolerance(0.01);
        CellPicker->AddPickList(self->SelectedActor);
        CellPicker->PickFromListOn();
        CellPicker->Pick(X,Y,0.0,self->CurrentRenderer);
        path = CellPicker->GetPath();
        if ( path != NULL )
        {
                vtkIdType PickedCell = CellPicker->GetCellId();
                if(PickedCell != -1)
                {
                        int location = self->DeleteCellIds->IsId(PickedCell);
                        if(location == -1)
                        {
                                self->DeleteCellIds->InsertNextId(PickedCell);
                        }
                        else
                        {
                                self->DeleteCellIds->DeleteId(PickedCell);
                        }
                        vtkLookupTable *lut = vtkLookupTable::New();
                        lut->SetNumberOfColors(2);
                        lut->Build();
                        lut->SetTableValue(0, 1.0, 1.0, 1.0, 1.0);
                        lut->SetTableValue(1, 1.0, 0.0, 0.0, 1.0);
                        lut->SetTableRange(0,1);
                        self->SelectedActor->GetMapper()->SetLookupTable(lut);
                        self->SelectedActor->GetMapper()->SetScalarRange(0,1);
                        lut->Delete();
                        for (i=0; i<numCells; i++)
                        {
                                scalararray->SetValue(i,0);
                                if(self->DeleteCellIds->IsId(i) != -1)
                                {
                                        scalararray->SetValue(i,1);
                                }
                        }
                        self->Input->GetCellData()->SetScalars(scalararray);
                        self->Input->Modified();
                }
        }
        self->Interactor->Render();
        CellPicker->Delete();
}
//------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::DeleteSelectedCells(vtkMimxCreateElementSetWidgetFEMesh *Self)
{
        // create a new selected element set based on computed selected cells
        Self->ExtractedGrid->Initialize();
        vtkPointLocator *locator = vtkPointLocator::New();
        vtkPoints *points = vtkPoints::New();
        points->Allocate(Self->Input->GetNumberOfPoints());
        locator->InitPointInsertion(points, Self->OriginalInput->GetPoints()->GetBounds());
        vtkIdList *idlist;
        int i;
        double x[3];
        vtkIdType tempid;
        int numIds = Self->Input->GetNumberOfCells();
        Self->ExtractedGrid->Allocate(numIds, numIds);
        for (i=0; i<numIds; i++)
        {
                if(Self->DeleteCellIds->IsId(i) == -1)
                {
                        idlist = vtkIdList::New();
                        idlist->DeepCopy(Self->Input->GetCell(i)->GetPointIds());
                        for(int j=0; j <idlist->GetNumberOfIds(); j++)
                        {
                                Self->Input->GetPoints()->GetPoint(idlist->GetId(j),x);          
                                locator->InsertUniquePoint(x,tempid);
                                idlist->SetId(j,tempid);
                        }
                        Self->ExtractedGrid->InsertNextCell(12, idlist);
                        idlist->Delete();
                }
        }

        points->Squeeze();
        Self->ExtractedGrid->SetPoints(points);
        Self->ExtractedGrid->Squeeze();
        points->Delete();
        locator->Delete();
        Self->ExtractedGrid->Modified();        
}
//-------------------------------------------------------------------------------------------
int vtkMimxCreateElementSetWidgetFEMesh::DoesCellBelong(
        int CellNum, vtkMimxCreateElementSetWidgetFEMesh *Self)
{
        int i, j;
        vtkIdList *OrIds = Self->OriginalInput->GetCell(CellNum)->GetPointIds();
        vtkIdList *SelectedIds = vtkIdList::New();
        SelectedIds->SetNumberOfIds(OrIds->GetNumberOfIds());

        for (i = 0; i < OrIds->GetNumberOfIds(); i++)
        {
                int location = Self->InputLocator->IsInsertedPoint(
                        Self->OriginalInput->GetPoint(OrIds->GetId(i)));
                if( location == -1)
                {
                        SelectedIds->Delete();
                        return 0;
                }
                else
                {
                        SelectedIds->SetId(i, location);
                }
        }
        // loop through all the cells in the Self->Input to locate the cell connectivity
        // if found return 1 else return 0;
        bool status;
        for (i=0; i< Self->Input->GetNumberOfCells(); i++)
        {
                status = true;
                vtkIdList *InputIds = Self->Input->GetCell(i)->GetPointIds();
                for (j=0; j<InputIds->GetNumberOfIds(); j++)
                {
                        if(SelectedIds->IsId(InputIds->GetId(j)) == -1)
                        {
                                status = false;
                                break;
                        }
                }
                if(status)
                {
                        SelectedIds->Delete();
                        return 1;
                }
        }
        SelectedIds->Delete();
        return 0;
}
//-------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::LeftButtonUpCallback(vtkAbstractWidget* w)
{
        vtkMimxCreateElementSetWidgetFEMesh *Self = 
                reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
        
        if(Self->PickStatus)
        {
                Self->CrtlLeftButtonUpCallback(w);
                return;
        }
        if(Self->CellSelectionState == 
                vtkMimxCreateElementSetWidgetFEMesh::SelectMultipleCells)
        {
                Self->ShiftLeftButtonUpCallback(w);
        }
}
//--------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::SetCellSelectionState(int Selection)
{
        this->CellSelectionState = Selection;
        if(this->CellSelectionState == 3)
        {
                this->CurrentRenderer->RemoveActor(this->SelectedActor);
                this->CurrentRenderer->AddActor(this->SurfaceActor);
        }
        else
        {
                this->CurrentRenderer->AddActor(this->SelectedActor);
                this->CurrentRenderer->RemoveActor(this->SurfaceActor);
        }
        this->CurrentRenderer->Render();
}
//---------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ShiftMouseMoveCallback(vtkAbstractWidget *w)
{
        //vtkMimxCreateElementSetWidgetFEMesh *self = reinterpret_cast<vtkMimxCreateElementSetWidgetFEMesh*>(w);
        //if(self->CellSelectionState == vtkMimxCreateElementSetWidgetFEMesh::SelectIndividualCell)
        //{
        //      self->SelectIndividualCellFunction(self);
        //}
}
//-------------------------------------------------------------------------------------------------
int vtkMimxCreateElementSetWidgetFEMesh::GetCellNumGivenFaceIds(
        vtkIdList *PtIds, vtkMimxCreateElementSetWidgetFEMesh *Self)
{
        int i,j,k;
        int numcells = Self->OriginalInput->GetNumberOfCells();
        for (i=0; i<numcells; i++)
        {
                vtkCell *cell = Self->OriginalInput->GetCell(i);
                int numfaces = cell->GetNumberOfFaces();
                for (j=0; j<numfaces; j++)
                {
                        vtkCell *face = cell->GetFace(j);
                        vtkIdList *faceids = face->GetPointIds();
                        bool status = true;
                        for (k=0; k<faceids->GetNumberOfIds(); k++)
                        {
                                if(PtIds->IsId(faceids->GetId(k)) == -1)
                                {
                                        status = false;
                                        break;
                                }
                        }
                        if(status)      
                        {
                                return i;
                        }
                }
        }
        return -1;
}
//---------------------------------------------------------------------------------------------
int vtkMimxCreateElementSetWidgetFEMesh::GetFaceNumGivenCellNumFaceIds(
        int CellNum, vtkIdList *PtIds, vtkPolyData *Surface, vtkMimxCreateElementSetWidgetFEMesh *Self)
{
        int i,j,k;
        vtkUnstructuredGrid *bbox = vtkUnstructuredGrid::New();
        vtkDataArray *dataarray = Self->OriginalInput->GetFieldData()->GetArray("Mesh_Seed");
        bbox->GetCellData()->AddArray(dataarray);
        bbox->Allocate(dataarray->GetNumberOfTuples(), dataarray->GetNumberOfTuples());
        for(i=0; i< dataarray->GetNumberOfTuples(); i++)
        {
                vtkIdList *idlist = vtkIdList::New();
                idlist->SetNumberOfIds(8);
                bbox->InsertNextCell(12, idlist);
                idlist->Delete();
        }
        vtkMimxUnstructuredToStructuredGrid *utosgrid = vtkMimxUnstructuredToStructuredGrid::New();
        utosgrid->SetInput(Self->OriginalInput);
        utosgrid->SetBoundingBox(bbox);
        utosgrid->SetStructuredGridNum(CellNum);
        utosgrid->Update();
        vtkStructuredGrid *sgrid = utosgrid->GetOutput();
        bbox->Delete();
        for (i=0; i<6; i++)
        {
                vtkMimxExtractStructuredGridFace *exface =
                        vtkMimxExtractStructuredGridFace::New();
                exface->SetInput(sgrid);
                exface->SetFaceNum(i);
                exface->Update();
                vtkStructuredGrid *facesgrid = exface->GetOutput();
                vtkPointLocator *localPointLocator = vtkPointLocator::New();
                vtkPoints *localPoints = vtkPoints::New();
                localPointLocator->InitPointInsertion(localPoints, Self->OriginalInput->GetBounds());
                for (j=0; j<facesgrid->GetNumberOfPoints(); j++)
                        localPointLocator->InsertNextPoint(facesgrid->GetPoint(j));

                // check if the points list from pointids match
                bool status = true;
                for (k=0; k<PtIds->GetNumberOfIds(); k++)
                {
                        int location = localPointLocator->IsInsertedPoint(
                                Self->OriginalInput->GetPoint(PtIds->GetId(k)));
                        if(location == -1)
                        {
                                status = false;
                                break;
                        }
                }
                localPointLocator->Delete();
                localPoints->Delete();

                if(status)
                {
                        //vtkStructuredGridWriter *writer = vtkStructuredGridWriter::New();
                        //writer->SetFileName("Structured.vtk");
                        //writer->SetInput(exface->GetOutput());
                        //writer->Write();
                        //writer->Delete();
                        vtkStructuredGridGeometryFilter *fil = vtkStructuredGridGeometryFilter::New();
                        fil->SetInput(exface->GetOutput());
                        fil->Update();
                        Surface->DeepCopy(fil->GetOutput());
                        fil->Delete();
                        exface->Delete();
                        utosgrid->Delete();
                        return i;
                }
                exface->Delete();
        }
        utosgrid->Delete();
        return -1;
}
//----------------------------------------------------------------------------------------------
void vtkMimxCreateElementSetWidgetFEMesh::ExtractElementsBelongingToAFace(
        vtkMimxCreateElementSetWidgetFEMesh *self)
{
        vtkPolyData *polydata = self->SurfaceFilter->GetOutput();
        vtkIntArray *scalararray = vtkIntArray::SafeDownCast(
                polydata->GetCellData()->GetArray("Mimx_Scalars"));
        if(!scalararray)        return;
        int i;
        for (i=0; i<polydata->GetNumberOfCells(); i++)
        {
                if(scalararray->GetValue(i))    break;
        }
        vtkIdList *ptids = vtkIdList::New();
        polydata->GetCellPoints(i, ptids);
        int cellnum = self->GetCellNumGivenFaceIds(ptids, self);
        if(cellnum == -1)       return;
        int dim[3];
        vtkIntArray *meshseed = vtkIntArray::SafeDownCast(
                self->OriginalInput->GetFieldData()->GetArray("Mesh_Seed"));
        if(!meshseed)   return;
        int StartEleNum = 0;
        int EndEleNum = -1;
        int bblocknum = -1;
        for (i=0; i<meshseed->GetNumberOfTuples(); i++)
        {
                StartEleNum = EndEleNum +1;
                meshseed->GetTupleValue(i, dim);
                EndEleNum = EndEleNum + (dim[0]-1)*(dim[1]-1)*(dim[2]-1)-1;
                if(cellnum >= StartEleNum && cellnum <= EndEleNum)
                {
                        bblocknum = i;
                        break;
                }
        }
        if(bblocknum == -1)
        {
                ptids->Delete();
                return;
        }
        vtkPolyData *surface = vtkPolyData::New();
        int FaceNum =  self->GetFaceNumGivenCellNumFaceIds(bblocknum, ptids, surface, self);
        if(FaceNum == -1)
        {
                ptids->Delete();
                surface->Delete();
                return;
        }
        //vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
        //writer->SetFileName("Extract.vtk");
        //writer->SetInput(surface);
        //writer->Write();
        //writer->Delete();
        self->ComputeSelectedCellIds(self, surface);
        surface->Delete();
        ptids->Delete();
        self->EventCallbackCommand->SetAbortFlag(1);
        self->EndInteraction();
        self->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
        self->PickStatus = 0;
        self->Interactor->Render();
}
