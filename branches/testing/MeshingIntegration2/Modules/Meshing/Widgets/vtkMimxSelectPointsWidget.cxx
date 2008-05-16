/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxSelectPointsWidget.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMimxSelectPointsWidget.h"

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
#include "vtkExtractSelectedPolyDataIds.h"
#include "vtkEvent.h"
#include "vtkGarbageCollector.h"
#include "vtkGeometryFilter.h"
#include "vtkGlyph3D.h"
#include "vtkHandleWidget.h"
#include "vtkIdFilter.h"
#include "vtkIdList.h"
#include "vtkIntArray.h"
#include "vtkInteractorObserver.h"
#include "vtkInteractorStyleRubberBandPick.h"
#include "vtkMath.h"
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
#include "vtkSphereSource.h"
#include "vtkStructuredGrid.h"
#include "vtkVisibleCellSelector.h"
#include "vtkSelectVisiblePoints.h"
#include "vtkWidgetCallbackMapper.h" 
#include "vtkWidgetEvent.h"
#include "vtkWidgetEventTranslator.h"
#include "vtkPointPicker.h"
#include "vtkPointLocator.h"
#include "vtkFieldData.h"

//#include "vtkStructuredGridWriter.h"
#include "vtkPolyDataWriter.h"

#include "vtkMimxUnstructuredToStructuredGrid.h"
#include "vtkMimxExtractStructuredGridFace.h"

vtkCxxRevisionMacro(vtkMimxSelectPointsWidget, "$Revision: 1.7 $");
vtkStandardNewMacro(vtkMimxSelectPointsWidget);

//----------------------------------------------------------------------
vtkMimxSelectPointsWidget::vtkMimxSelectPointsWidget()
{
  this->CallbackMapper->SetCallbackMethod(
            vtkCommand::LeftButtonPressEvent,
            vtkEvent::ControlModifier, 0, 1, NULL,
                        vtkMimxSelectPointsWidget::CrtlLeftMouseButtonDown,
                        this, vtkMimxSelectPointsWidget::CrtlLeftButtonDownCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::MouseMoveEvent,
          vtkEvent::ControlModifier, 0, 1, NULL,
          vtkMimxSelectPointsWidget::CrtlLeftMouseButtonMove,
          this, vtkMimxSelectPointsWidget::CrtlMouseMoveCallback);

  this->CallbackMapper->SetCallbackMethod(
            vtkCommand::LeftButtonReleaseEvent,
                        vtkEvent::ControlModifier, 0, 1, NULL,
            vtkMimxSelectPointsWidget::CrtlLeftMouseButtonUp,
            this, vtkMimxSelectPointsWidget::CrtlLeftButtonUpCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::LeftButtonPressEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxSelectPointsWidget::ShiftLeftMouseButtonDown,
          this, vtkMimxSelectPointsWidget::ShiftLeftButtonDownCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::LeftButtonReleaseEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxSelectPointsWidget::ShiftLeftMouseButtonUp,
          this, vtkMimxSelectPointsWidget::ShiftLeftButtonUpCallback);

  this->CallbackMapper->SetCallbackMethod(
          vtkCommand::MouseMoveEvent,
          vtkEvent::ShiftModifier, 0, 1, NULL,
          vtkMimxSelectPointsWidget::ShiftLeftMouseButtonMove,
          this, vtkMimxSelectPointsWidget::ShiftMouseMoveCallback);

  this->CallbackMapper->SetCallbackMethod(vtkCommand::LeftButtonReleaseEvent,
          vtkMimxSelectPointsWidget::LeftMouseButtonUp,
          this, vtkMimxSelectPointsWidget::LeftButtonUpCallback);

  this->RubberBandStyle =  vtkInteractorStyleRubberBandPick::New();
  this->AreaPicker = vtkRenderedAreaPicker::New();
  this->Input = NULL;
  this->InputActor = NULL;
  this->WidgetEvent = vtkMimxSelectPointsWidget::Start;
  this->Sphere = vtkSphereSource::New();
  this->Glyph = NULL;
  this->GlyphActor = NULL;
  this->Mapper = NULL;
  this->SphereActor = NULL;
  this->SelectedPointIds = vtkIdList::New();
  this->PickX0 = -1;    this->PickY0 = -1;      this->PickX1 = -1;      this->PickY1 = -1;
  this->PointSelectionState = 4;
  this->PickStatus = 0;
}

//----------------------------------------------------------------------
vtkMimxSelectPointsWidget::~vtkMimxSelectPointsWidget()
{
        this->RubberBandStyle->Delete();
        this->AreaPicker->Delete();
        this->Sphere->Delete();
        if(this->Glyph)
                this->Glyph->Delete();

        if(this->GlyphActor)
                this->GlyphActor->Delete();
        if(this->Mapper)
                this->Mapper->Delete();

        if(this->SelectedPointIds)
                this->SelectedPointIds->Delete();

        if(this->InputActor)
        {
                this->InputActor->Delete();
        }
        if(this->SingleSphere)
        {
                this->SingleSphere->Delete();
        }
        if(this->SphereActor)
                this->SphereActor->Delete();
}

//----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SetEnabled(int enabling)
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
        vtkDataSetMapper *mapper = vtkDataSetMapper::New();
        mapper->SetInput(this->Input);
        if(!this->InputActor)
                this->InputActor = vtkActor::New();
        this->InputActor->SetMapper(mapper);

    int X=this->Interactor->GetEventPosition()[0];
    int Y=this->Interactor->GetEventPosition()[1];
  
    if ( ! this->CurrentRenderer )
      {
      this->SetCurrentRenderer(this->Interactor->FindPokedRenderer(X,Y));
  
      if (this->CurrentRenderer == NULL)
        {
        return;
        }
          this->CurrentRenderer->AddActor(this->InputActor);
          this->InputActor->PickableOn();
          mapper->Delete();

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
                this->CurrentRenderer->RemoveActor(this->InputActor);
                this->InputActor->Delete();
                this->InputActor = NULL;
        }
        
        if(this->GlyphActor)
        {
                this->CurrentRenderer->RemoveActor(this->GlyphActor);
        }
        if(this->SphereActor)
        {
                this->CurrentRenderer->RemoveActor(this->SphereActor);
        }
        this->SelectedPointIds->Initialize();
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
void vtkMimxSelectPointsWidget::CrtlLeftButtonDownCallback(vtkAbstractWidget *w)
{
        vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
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
                self->WidgetEvent = vtkMimxSelectPointsWidget::Outside;
                return;
        }

        self->WidgetEvent = vtkMimxSelectPointsWidget::CrtlLeftMouseButtonDown;

        self->RubberBandStyle->GetInteractor()->SetKeyCode('r');
        self->RubberBandStyle->OnChar();
        self->RubberBandStyle->OnLeftButtonDown();
        // Okay, we can process this. Try to pick handles first;
        // if no handles picked, then try to pick the line.
        self->EventCallbackCommand->SetAbortFlag(1);
        self->StartInteraction();
        self->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
        self->PickStatus = 1;
        self->Interactor->Render();
}
//----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::CrtlLeftButtonUpCallback(vtkAbstractWidget *w)
{
  vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
  int *size;
  size = self->Interactor->GetRenderWindow()->GetSize();

  int X = self->Interactor->GetEventPosition()[0];
  int Y = self->Interactor->GetEventPosition()[1];

 // cout <<X<<"  "<<Y<<endl;

  self->PickX1 = X;
  self->PickY1 = Y;

  if ( self->WidgetEvent == vtkMimxSelectPointsWidget::Outside ||
          self->WidgetEvent == vtkMimxSelectPointsWidget::Start )
  {
          return;
  }

  if(self->WidgetEvent == vtkMimxSelectPointsWidget::CrtlLeftMouseButtonDown)
  {
          self->WidgetEvent = vtkMimxSelectPointsWidget::Start;

          if(self->PointSelectionState != vtkMimxSelectPointsWidget::SelectSinglePoint &&
                  self->PointSelectionState != vtkMimxSelectPointsWidget::SelectPointsBelongingToAFace)
          {
                  if(self->GlyphActor)
                  {
                          self->CurrentRenderer->RemoveActor(self->GlyphActor);
                          self->GlyphActor->Delete(); 
                  }
                  self->GlyphActor = vtkActor::New();

                  if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectVisiblePointsOnSurface)
                  {
                          vtkMimxSelectPointsWidget::SelectVisiblePointsOnSurfaceFunction(self);
                  }
                  if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectPointsOnSurface)
                  {
                          vtkMimxSelectPointsWidget::SelectPointsOnSurfaceFunction(self);
                  }
                  if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectPointsThrough)
                  {
                          vtkMimxSelectPointsWidget::SelectPointsThroughFunction(self);
                  }

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
void vtkMimxSelectPointsWidget::CrtlMouseMoveCallback(vtkAbstractWidget *w)
{
  vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
  if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectSinglePoint ||
          self->PointSelectionState == vtkMimxSelectPointsWidget::SelectPointsBelongingToAFace)
  {
        self->SelectSinglePointFunction(self);
  }
  else
  {
          self->RubberBandStyle->OnMouseMove();
  }
}
//---------------------------------------------------------------------
void vtkMimxSelectPointsWidget::ComputeSelectedPointIds(
        vtkDataSet *ExtractedUGrid, vtkMimxSelectPointsWidget *self)
{

        int i;
        vtkPoints *polypoints;
        if(ExtractedUGrid->GetDataObjectType() == VTK_POLY_DATA)
                polypoints = vtkPolyData::SafeDownCast(ExtractedUGrid)->GetPoints();

        if (ExtractedUGrid->GetDataObjectType() == VTK_STRUCTURED_GRID)
                polypoints = vtkStructuredGrid::SafeDownCast(ExtractedUGrid)->GetPoints();

        if (ExtractedUGrid->GetDataObjectType() == VTK_UNSTRUCTURED_GRID)
                polypoints = vtkUnstructuredGrid::SafeDownCast(ExtractedUGrid)->GetPoints();

        vtkPoints *inputpoints = self->Input->GetPoints();
    
        vtkPoints *newpts = vtkPoints::New();
        vtkPointLocator *locator = vtkPointLocator::New();
        locator->InitPointInsertion(newpts, self->Input->GetBounds());
        
        for (i=0; i< inputpoints->GetNumberOfPoints(); i++)
        {
                locator->InsertNextPoint(inputpoints->GetPoint(i));
        }
         if(self->SelectedPointIds)
         {
                 self->SelectedPointIds->Initialize();
         }
         else{
                 self->SelectedPointIds = vtkIdList::New();
         }

         for (i=0; i<polypoints->GetNumberOfPoints(); i++)
         {
                 int location = locator->IsInsertedPoint(polypoints->GetPoint(i));
                 if(location == -1)
                 {
                         vtkErrorMacro("Point sets do not match");
                         locator->Delete();
                         self->SelectedPointIds->Initialize();
                         return;
                 }
                 else
                 {
                        self->SelectedPointIds->InsertNextId(location);
                 }
         }
         newpts->Delete();
         locator->Delete();
}
//----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);  
}
//-----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SelectVisiblePointsOnSurfaceFunction(
        vtkMimxSelectPointsWidget *self)
{
        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();

        vtkCleanPolyData *clean = vtkCleanPolyData::New();
        clean->SetInput(fil->GetOutput());
        clean->Update();
        vtkPolyDataMapper *cleanpolydatamapper = vtkPolyDataMapper::New();
        vtkActor *cleanactor = vtkActor::New();
        cleanpolydatamapper->SetInput(clean->GetOutput());
        cleanactor->SetMapper(cleanpolydatamapper);
        cleanactor->PickableOn();
        self->CurrentRenderer->AddActor(cleanactor);
        if(self->InputActor)
                self->CurrentRenderer->RemoveActor(self->InputActor);
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

        vtkSelection *cellids = res->GetChild(0);
        vtkExtractSelectedPolyDataIds *extr = vtkExtractSelectedPolyDataIds::New();
        if (cellids)
        {
                extr->SetInput(0, clean->GetOutput());
                extr->SetInput(1, cellids);
                extr->Update();
        }
        vtkCleanPolyData *cleanextr = vtkCleanPolyData::New();
        cleanextr->SetInput(extr->GetOutput());
        cleanextr->Update();

        vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
        writer->SetInput(extr->GetOutput());
        writer->SetFileName("Extract.vtk");
        writer->Write();
        writer->Delete();

        self->ComputeSelectedPointIds(cleanextr->GetOutput(), self );
        //sphere sizing calculations
        self->Sphere->SetRadius(self->ComputeSphereRadius(cleanextr->GetOutput()));
        //

        if(self->Glyph)
                self->Glyph->Delete();
        self->Glyph = vtkGlyph3D::New();

        self->Glyph->SetInput(cleanextr->GetOutput());
        self->Glyph->SetSource(self->Sphere->GetOutput());
        self->Glyph->Update();
        if(self->Mapper)
                self->Mapper->Delete();
        self->Mapper = vtkPolyDataMapper::New();
        self->Mapper->SetInput(self->Glyph->GetOutput());
        self->GlyphActor->SetMapper(self->Mapper);
        self->GlyphActor->GetProperty()->SetColor(1.0,0.66,0.33);
        self->CurrentRenderer->AddActor(self->GlyphActor);  

        self->CurrentRenderer->RemoveActor(cleanactor);
        cleanactor->Delete();
        cleanextr->Delete();
        select->Delete();
        res->Delete();
        extr->Delete();
        fil->Delete();
        clean->Delete();
        cleanpolydatamapper->Delete();

        if(self->InputActor)
                self->CurrentRenderer->AddActor(self->InputActor);
        self->CurrentRenderer->Render();

}
//-----------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SelectPointsOnSurfaceFunction(
        vtkMimxSelectPointsWidget *self)
{
        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();

        vtkCleanPolyData *clean = vtkCleanPolyData::New();
        clean->SetInput(fil->GetOutput());
        clean->Update();
        vtkPolyDataMapper *cleanpolydatamapper = vtkPolyDataMapper::New();
        vtkActor *cleanactor = vtkActor::New();
        cleanpolydatamapper->SetInput(clean->GetOutput());
        cleanactor->SetMapper(cleanpolydatamapper);
        cleanactor->PickableOn();
        self->CurrentRenderer->AddActor(cleanactor);
        self->CurrentRenderer->Render();
        double x0 = self->PickX0;
        double y0 = self->PickY0;
        double x1 = self->PickX1;
        double y1 = self->PickY1;
        self->AreaPicker->AreaPick(static_cast<int>(x0), static_cast<int>(y0), 
                static_cast<int>(x1), static_cast<int>(y1), self->CurrentRenderer);
        vtkExtractSelectedFrustum *Extract = vtkExtractSelectedFrustum::New();
        Extract->SetInput(clean->GetOutput());
//      Extract->PassThroughOff();
        Extract->SetFrustum(self->AreaPicker->GetFrustum());
        Extract->Update();
        vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(
                Extract->GetOutput());
        self->CurrentRenderer->RemoveActor(cleanactor);
        self->CurrentRenderer->Render();
        cleanactor->Delete();
        cleanpolydatamapper->Delete();
        if(ugrid->GetNumberOfPoints() < 1)
        {
                Extract->Delete();
                fil->Delete();
                clean->Delete();
                return;
        }
        self->ComputeSelectedPointIds(ugrid, self );

        //sphere sizing calculations
        //double bounds[6];
        //Extract->GetOutput()->GetBounds(bounds);
        //double cuberoot = pow(Extract->GetOutput()->GetNumberOfPoints(), 0.5);
        //double edgelength = ((bounds[1] - bounds[0]) + (bounds[3] - bounds[2]) +
        //      (bounds[5] - bounds[4]))/3.0;
        //double radius = 0.25*edgelength/cuberoot;
        //self->Sphere->SetRadius(radius);
        self->Sphere->SetRadius(self->ComputeSphereRadius(ugrid));
        //

        if(self->Glyph)
                self->Glyph->Delete();
        self->Glyph = vtkGlyph3D::New();

        self->Glyph->SetInput(ugrid);
        self->Glyph->SetSource(self->Sphere->GetOutput());
        self->Glyph->Update();
        Extract->Delete();
        clean->Delete();
        fil->Delete();
        if(self->Mapper)
                self->Mapper->Delete();
        self->Mapper = vtkPolyDataMapper::New();
        self->Mapper->SetInput(self->Glyph->GetOutput());
        self->GlyphActor->SetMapper(self->Mapper);
        self->GlyphActor->GetProperty()->SetColor(1.0,0.66,0.33);
        self->CurrentRenderer->AddActor(self->GlyphActor);  
}
//-----------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SelectPointsThroughFunction(
        vtkMimxSelectPointsWidget *self)
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
        vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(
                Extract->GetOutput());

        if(ugrid->GetNumberOfPoints() < 1)
        {
                Extract->Delete();
                return;
        }
        self->ComputeSelectedPointIds(ugrid, self);

        //sphere sizing calculations
        self->Sphere->SetRadius(self->ComputeSphereRadius(ugrid));
        //

        if(self->Glyph)
                self->Glyph->Delete();
        self->Glyph = vtkGlyph3D::New();

        self->Glyph->SetInput(ugrid);
        self->Glyph->SetSource(self->Sphere->GetOutput());
        self->Glyph->Update();
        Extract->Delete();

        if(self->Mapper)
                self->Mapper->Delete();
        self->Mapper = vtkPolyDataMapper::New();
        self->Mapper->SetInput(self->Glyph->GetOutput());
        self->GlyphActor->SetMapper(self->Mapper);
        self->GlyphActor->GetProperty()->SetColor(1.0,0.66,0.33);
        self->CurrentRenderer->AddActor(self->GlyphActor);  
}
//-----------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SelectSinglePointFunction(
        vtkMimxSelectPointsWidget *self)
{
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];

        // Okay, we can process this. Try to pick handles first;
        // if no handles picked, then pick the bounding box.
        if (!self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y))
        {
                return;
        }

        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();
        vtkCleanPolyData *clean = vtkCleanPolyData::New();
        clean->SetInput(fil->GetOutput());
        clean->Update();

        vtkPolyDataMapper *cleanmapper = vtkPolyDataMapper::New();
        cleanmapper->SetInput(clean->GetOutput());

        vtkActor *cleanactor = vtkActor::New();
        cleanactor->SetMapper(cleanmapper);
        self->CurrentRenderer->AddActor(cleanactor);  
        self->Interactor->Render();

        vtkAssemblyPath *path;
        vtkPointPicker *PointPicker = vtkPointPicker::New();
        PointPicker->SetTolerance(0.01);
        PointPicker->AddPickList(cleanactor);
        PointPicker->PickFromListOn();
        PointPicker->Pick(X,Y,0.0,self->CurrentRenderer);
        path = PointPicker->GetPath();
        if ( path != NULL )
        {
                vtkIdType PickedPoint = PointPicker->GetPointId();
                if(PickedPoint != -1)
                {
                        if(!self->SphereActor)
                        {
                                self->SphereActor = vtkActor::New();
                                self->SingleSphere = vtkSphereSource::New();
                                self->SingleSphere->SetRadius(0.5);
                                vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
                                mapper->SetInput(self->SingleSphere->GetOutput());
                                self->SphereActor->SetMapper(mapper);
                                mapper->Delete();
                        }
                        self->SingleSphere->SetCenter(clean->GetOutput()->GetPoint(PickedPoint));
                        self->SingleSphere->Modified();
                        self->SphereActor->GetProperty()->SetColor(1.0,0.66,0.33);
                        self->CurrentRenderer->RemoveActor(self->SphereActor);  
                        self->CurrentRenderer->AddActor(self->SphereActor); 
                        self->ComputeSelectedPointIds(self);
                }
        }
        self->CurrentRenderer->RemoveActor(cleanactor);  
        self->Interactor->Render();
        fil->Delete();
        clean->Delete();
        cleanmapper->Delete();
        cleanactor->Delete();
        PointPicker->Delete();
}
//--------------------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::ComputeSelectedPointIds(vtkMimxSelectPointsWidget *self)
{

        int i;

        vtkPoints *inputpoints = self->Input->GetPoints();

        vtkPoints *newpts = vtkPoints::New();
        vtkPointLocator *locator = vtkPointLocator::New();
        locator->InitPointInsertion(newpts, self->Input->GetBounds());

        for (i=0; i< inputpoints->GetNumberOfPoints(); i++)
        {
                locator->InsertNextPoint(inputpoints->GetPoint(i));
        }
        if(self->SelectedPointIds)
        {
                self->SelectedPointIds->Initialize();
        }
        else{
                self->SelectedPointIds = vtkIdList::New();
        }

        int location = locator->IsInsertedPoint(
                self->SingleSphere->GetCenter());
        if(location == -1)
        {
                vtkErrorMacro("Invalid Sphere Location");
                locator->Delete();
                self->SelectedPointIds->Initialize();
                return;
        }
        else
        {
                self->SelectedPointIds->InsertNextId(location);
        }
        newpts->Delete();
        locator->Delete();
}
//----------------------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::ShiftLeftButtonUpCallback(vtkAbstractWidget *w)
{
        vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);

        if ( self->WidgetEvent == vtkMimxSelectPointsWidget::Outside ||
                self->WidgetEvent == vtkMimxSelectPointsWidget::Start )
        {
                return;
        }

        if(self->WidgetEvent == vtkMimxSelectPointsWidget::ShiftLeftMouseButtonDown)
        {
                self->WidgetEvent = vtkMimxSelectPointsWidget::Start;

                        if(self->GlyphActor)
                        {
                                self->CurrentRenderer->RemoveActor(self->GlyphActor);
                                self->GlyphActor->Delete(); 
                        }
                        self->GlyphActor = vtkActor::New();

                        if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectPointsBelongingToAFace)
                        {
                                vtkMimxSelectPointsWidget::SelectPointsOnAFaceFunction(self);
                        }
                }
        self->EndInteraction();
        self->InvokeEvent(vtkCommand::EndInteractionEvent,NULL);
        self->Interactor->Render();
}
//----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::ShiftMouseMoveCallback(vtkAbstractWidget *w)
{
        vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
        if(self->PointSelectionState == vtkMimxSelectPointsWidget::SelectSinglePoint ||
                self->PointSelectionState == vtkMimxSelectPointsWidget::SelectPointsBelongingToAFace)
        {
                self->SelectSinglePointFunction(self);
        }
}
//---------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SelectPointsOnAFaceFunction(
        vtkMimxSelectPointsWidget *self)
{
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];

        // Okay, we can process this. Try to pick handles first;
        // if no handles picked, then pick the bounding box.
        if (!self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y))
        {
                return;
        }
        
        if(!self->Input->GetFieldData()->GetArray("Mesh_Seed"))
        {
//              vtkErrorMacro("Mesh Seed data should be present for the input mesh");
                return ;
        }
        vtkGeometryFilter *fil = vtkGeometryFilter::New();
        fil->SetInput(self->Input);
        fil->Update();
        vtkCleanPolyData *clean = vtkCleanPolyData::New();
        clean->SetInput(fil->GetOutput());
        clean->Update();

        // create a building block with just the mesh seeds.
        vtkUnstructuredGrid *bblock = vtkUnstructuredGrid::New();
        vtkIntArray *intarray = vtkIntArray::New();
        intarray->SetNumberOfComponents(3);
        vtkIntArray *meshseed = vtkIntArray::SafeDownCast(
                self->Input->GetFieldData()->GetArray("Mesh_Seed"));

        int i,j, dim[3];
        // create a copy of mesh seed values with no junk values.
        for (i=0; i<meshseed->GetNumberOfTuples(); i++)
        {
                meshseed->GetTupleValue(i, dim);
                intarray->InsertNextTupleValue(dim);
        }
        intarray->SetName("Mesh_Seed");
        bblock->GetCellData()->AddArray(intarray);
        bblock->Allocate(intarray->GetNumberOfTuples(), 
                intarray->GetNumberOfTuples());
        for(i=0; i< intarray->GetNumberOfTuples(); i++)
        {
                vtkIdList *idlist = vtkIdList::New();
                idlist->SetNumberOfIds(8);
                bblock->InsertNextCell(12, idlist);
                idlist->Delete();
        }
        self->SelectedPointIds->Initialize();
        for (i=0; i<intarray->GetNumberOfTuples();i++)
        {
                vtkMimxUnstructuredToStructuredGrid *utosgrid = 
                        vtkMimxUnstructuredToStructuredGrid::New();
                utosgrid->SetInput(self->Input);
                utosgrid->SetBoundingBox(bblock);
                utosgrid->SetStructuredGridNum(i);
                utosgrid->Update();
                vtkStructuredGrid *solid = vtkStructuredGrid::New();
                solid->DeepCopy(utosgrid->GetOutput());
                if(self->CheckGivenPointBelongsToStructuredGrid(utosgrid->GetOutput(), 
                        self->SingleSphere->GetCenter(), self->Input->GetBounds()))
                {
                        for (j=0; j<6; j++)
                        {
                                vtkMimxExtractStructuredGridFace *exface =
                                        vtkMimxExtractStructuredGridFace::New();
                                exface->SetInput(utosgrid->GetOutput());
                                exface->SetFaceNum(j);
                                exface->Update();
                                vtkStructuredGrid *exfacegrid = vtkStructuredGrid::New();
                                exfacegrid->DeepCopy(exface->GetOutput());
                                if(self->CheckGivenPointBelongsToStructuredGrid(
                                        exfacegrid, self->SingleSphere->GetCenter(), self->Input->GetBounds()))
                                {
                                        if(self->DoAllPointsLieOnSurface(exfacegrid, clean->GetOutput(), self->Input->GetBounds()))
                                        {
                                                vtkStructuredGrid *sgrid = vtkStructuredGrid::New();
                                                sgrid->DeepCopy(exface->GetOutput());
                                                self->ComputeSelectedPointIds(sgrid, self);
                                                //sphere sizing calculations
                                                vtkGeometryFilter *fil1 = vtkGeometryFilter::New();
                                                fil1->SetInput(sgrid);
                                                fil1->Update();
                                                self->Sphere->SetRadius(self->ComputeSphereRadius(fil1->GetOutput()));
                                                fil1->Delete();
                                                //
                                                if(self->Glyph)
                                                        self->Glyph->Delete();
                                                self->Glyph = vtkGlyph3D::New();
                                                self->Glyph->SetInput(sgrid);
                                                self->Glyph->SetSource(self->Sphere->GetOutput());
                                                self->Glyph->Update();
                                                sgrid->Delete();
        //                                              self->Glyph->SetScaleFactor(self->Glyph->GetScaleFactor()*0.5);
                                                if(self->Mapper)
                                                        self->Mapper->Delete();
                                                self->Mapper = vtkPolyDataMapper::New();
                                                self->Mapper->SetInput(self->Glyph->GetOutput());
                                                self->GlyphActor->SetMapper(self->Mapper);
                                                self->GlyphActor->GetProperty()->SetColor(1.0,0.66,0.33);
                                                self->CurrentRenderer->AddActor(self->GlyphActor);                                      
                                                exface->Delete();
                                                utosgrid->Delete();
                                                bblock->Delete();
                                                intarray->Delete();
                                                fil->Delete();
                                                clean->Delete();
                                                return;
                                        }
                                }
                                exface->Delete();
                                exfacegrid->Delete();
                        }
                }
                utosgrid->Delete();
                solid->Delete();
        }
        bblock->Delete();
        intarray->Delete();
        fil->Delete();
        clean->Delete();                        
}
//--------------------------------------------------------------------------------------
int vtkMimxSelectPointsWidget::CheckGivenPointBelongsToStructuredGrid(
        vtkStructuredGrid *SGrid, double x[3], double *bounds)
{
        int i;
        vtkPoints *newpts = vtkPoints::New();
        vtkPointLocator *locator = vtkPointLocator::New();
        locator->InitPointInsertion(newpts, bounds);
        
        for (i=0; i< SGrid->GetNumberOfPoints(); i++)
        {
                locator->InsertNextPoint(SGrid->GetPoint(i));
        }
        int location = locator->IsInsertedPoint(x);
        if(location == -1)
        {
                locator->Initialize();
                locator->Delete();
                newpts->Delete();
                return 0;
        }
        locator->Initialize();
        locator->Delete();
        newpts->Delete();
        return 1;       
}
//--------------------------------------------------------------------------------------
int vtkMimxSelectPointsWidget::DoAllPointsLieOnSurface(
        vtkStructuredGrid *SGrid, vtkPolyData *Surface, double *bounds)
{
        int i;
        vtkPoints *newpts = vtkPoints::New();
        vtkPointLocator *locator = vtkPointLocator::New();
        locator->InitPointInsertion(newpts, bounds);

        for (i=0; i< Surface->GetNumberOfPoints(); i++)
        {
                        locator->InsertNextPoint(Surface->GetPoint(i));
        }
        for (i=0; i<SGrid->GetNumberOfPoints(); i++)
        {
                int location = locator->IsInsertedPoint(SGrid->GetPoint(i));
                if(location == -1)
                {
                        locator->Initialize();
                        locator->Delete();
                        newpts->Delete();
                        return 0;
                }       
        }
        locator->Initialize();
        locator->Delete();
        newpts->Delete();
        return 1;       
}
//--------------------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::ShiftLeftButtonDownCallback(vtkAbstractWidget *w)
{
        vtkMimxSelectPointsWidget *self = reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
        int X = self->Interactor->GetEventPosition()[0];
        int Y = self->Interactor->GetEventPosition()[1];
        //      cout <<X<<"  "<<Y<<endl;
        // Okay, make sure that the pick is in the current renderer
        if ( !self->CurrentRenderer || !self->CurrentRenderer->IsInViewport(X, Y) )
        {
                self->WidgetEvent = vtkMimxSelectPointsWidget::Outside;
                return;
        }
        self->WidgetEvent = vtkMimxSelectPointsWidget::ShiftLeftMouseButtonDown;
        self->EventCallbackCommand->SetAbortFlag(1);
        self->StartInteraction();
        self->InvokeEvent(vtkCommand::StartInteractionEvent,NULL);
        self->Interactor->Render();
}
//----------------------------------------------------------------------
void vtkMimxSelectPointsWidget::LeftButtonUpCallback(vtkAbstractWidget* w)
{
        vtkMimxSelectPointsWidget *Self = 
                reinterpret_cast<vtkMimxSelectPointsWidget*>(w);
        if(Self->PickStatus)
        {
                Self->CrtlLeftButtonUpCallback(w);
        }
}
//-------------------------------------------------------------------------------------------
void vtkMimxSelectPointsWidget::SetSphereRadius(double Radius)
{
        this->Sphere->SetRadius(Radius);
        this->Sphere->Update();
}
//--------------------------------------------------------------------------------------------
double vtkMimxSelectPointsWidget::GetSphereRadius()
{
        return this->Sphere->GetRadius();
}
//--------------------------------------------------------------------------------------------
double vtkMimxSelectPointsWidget::ComputeSphereRadius(vtkDataSet *DataSet)
{
        double cumdist = 0.0;
        int count = 0;
        double radius = 0.0;
        if(DataSet->GetDataObjectType() == VTK_UNSTRUCTURED_GRID)
        {
                vtkUnstructuredGrid *ugrid = vtkUnstructuredGrid::SafeDownCast(DataSet);
                int i,j;
                for (i=0; i<ugrid->GetNumberOfCells(); i++)
                {
                        vtkCell *cell = ugrid->GetCell(i);
                        for (j=0; j<cell->GetNumberOfEdges(); j++)
                        {
                                vtkCell *edge = cell->GetEdge(j);
                                vtkIdList *ptids = edge->GetPointIds();
                                int pt1 = ptids->GetId(0);
                                int pt2 = ptids->GetId(1);
                                double p1[3], p2[3];
                                ugrid->GetPoint(pt1, p1);       ugrid->GetPoint(pt2, p2);
                                cumdist = cumdist + sqrt(vtkMath::Distance2BetweenPoints(p1, p2));
                                count ++;
                        }
                }
                radius = 0.25*cumdist/count;
                return radius;
        }
        else
        {
                vtkPolyData *polydata = vtkPolyData::SafeDownCast(DataSet);
                int i,j;
                for (i=0; i<polydata->GetNumberOfCells(); i++)
                {
                        vtkCell *cell = polydata->GetCell(i);
                        for (j=0; j<cell->GetNumberOfEdges(); j++)
                        {
                                vtkCell *edge = cell->GetEdge(j);
                                vtkIdList *ptids = edge->GetPointIds();
                                int pt1 = ptids->GetId(0);
                                int pt2 = ptids->GetId(1);
                                double p1[3], p2[3];
                                polydata->GetPoint(pt1, p1);    polydata->GetPoint(pt2, p2);
                                cumdist = cumdist + sqrt(vtkMath::Distance2BetweenPoints(p1, p2));
                                count ++;
                        }
                }
                radius = 0.25*cumdist/count;    
                return radius;
        }
}
