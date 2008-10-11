/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxMeshActor.cxx,v $
Language:  C++
Date:      $Date: 2008/05/05 19:29:58 $
Version:   $Revision: 1.15 $

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

#include "vtkMimxMeshActor.h"

#include "mimxLinkedList.h"
#include "vtkActor.h"
#include "vtkCellData.h"
#include "vtkCellTypes.h"
#include "vtkDataSetMapper.h"
#include "vtkGenericCell.h"
#include "vtkHexahedron.h"
#include "vtkIdList.h"
#include "vtkIntArray.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkProperty.h"
#include "vtkUnstructuredGrid.h"
#include "vtkFeatureEdges.h"
#include "vtkShrinkFilter.h"
#include "vtkGeometryFilter.h"
#include "vtkTubeFilter.h"
#include "vtkStringArray.h"
#include "vtkPolyDataMapper.h"
#include "vtkExtractCells.h"
#include "vtkRenderer.h"
#include "vtkPointData.h"
#include "vtkScalarBarActor.h"
#include "vtkLookupTable.h"
#include "vtkTextProperty.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkPlaneWidget.h"
#include "vtkExtractGeometry.h"
#include "vtkPlane.h"
#include "vtkCommand.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"
#include "vtkPointSet.h"


vtkCxxRevisionMacro(vtkMimxMeshActor, "$Revision: 1.15 $");
vtkStandardNewMacro(vtkMimxMeshActor);

// Create a callback function so the selection plane can invoke this when it is
// moved by the operator.  The corresponding implicit plane function is passed 
// back into a pre-allocated vtkPlane instance for use elsewhere in the application

class vtkPlaneWidgetEventCallback : public vtkCommand
{
public:
  static vtkPlaneWidgetEventCallback *New() 
    { return new vtkPlaneWidgetEventCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkPlaneWidget *planeWidget = reinterpret_cast<vtkPlaneWidget*>(caller);
      planeWidget->GetPlane(this->PlaneInstance);
    }
  
  vtkPlane *PlaneInstance;
};

//----------------------------------------------------------------------------------------
vtkMimxMeshActor::vtkMimxMeshActor()
{
  this->ElementShrinkFactor = 1.0;
  this->DataType = ACTOR_FE_MESH;
        this->ElementSetName = NULL;
        this->IsVisible = true;
        this->ElementSetDisplayList.clear();
        this->DisplayMode = vtkMimxMeshActor::DisplayMesh;
        this->DisplayType = vtkMimxMeshActor::DisplaySurfaceAndOutline;
        this->NumberOfElementSets = 0;
        this->Interactor = NULL;
        this->Renderer = NULL;
        this->CuttingPlaneEnabled = false;
        this->TextColor[0] = this->TextColor[1] = this->TextColor[2] = 1.0;
        
        /* Setup the Pipeline for the Mesh */
  vtkPoints *points = vtkPoints::New();
        this->UnstructuredGrid = vtkUnstructuredGrid::New();
  this->UnstructuredGrid->SetPoints(points);
  points->Delete();
  
  this->ShrinkFilter = vtkShrinkFilter::New();
  this->ShrinkFilter->SetInput(this->UnstructuredGrid);
  this->ShrinkFilter->SetShrinkFactor(this->ElementShrinkFactor);

  vtkDataSetMapper*  meshMapper = vtkDataSetMapper::New();
  meshMapper->SetInputConnection(this->ShrinkFilter->GetOutputPort());
  this->UnstructuredGridMapper = meshMapper;

  this->Actor = vtkActor::New();
  this->Actor->SetMapper(this->UnstructuredGridMapper);
        
        /* Setup the Pipeline for the Wireframe */
  this->OutlineGeometryFilter = vtkGeometryFilter::New();
  this->OutlineGeometryFilter->SetInput( this->UnstructuredGrid );

  vtkFeatureEdges* featureEdges = vtkFeatureEdges::New();
  featureEdges->SetInput( this->OutlineGeometryFilter->GetOutput() );
  featureEdges->BoundaryEdgesOn();
  featureEdges->ManifoldEdgesOn();
  featureEdges->FeatureEdgesOff();
  featureEdges->ColoringOff();
  
  this->TubeFilter = vtkTubeFilter::New();
  this->TubeFilter->SetInputConnection(featureEdges->GetOutputPort());
  this->TubeFilter->SetRadius(0.03);
   
  this->OutlineMapper = vtkPolyDataMapper::New();
  this->OutlineMapper->SetInputConnection( /*this->OutlineGeometryFilter->GetOutput()*/ this->TubeFilter->GetOutputPort() );
  this->OutlineMapper->SetScalarVisibility( 0 );
  
  this->OutlineActor = vtkActor::New();
  this->OutlineActor->SetMapper( this->OutlineMapper );
  this->OutlineActor->GetProperty()->SetColor(0.0,0.0,0.0);
  this->OutlineActor->GetProperty()->SetRepresentationToSurface();
  this->OutlineActor->GetProperty()->SetAmbient(0);
  this->OutlineActor->GetProperty()->SetLineWidth(1.0);
  
  /* Setup the Pipeline for Interior Display */
  /*** NOTE: This is not currently being used ***/
  this->InteriorShrinkFilter = vtkShrinkFilter::New();
  this->InteriorShrinkFilter->SetInput( this->UnstructuredGrid );
  this->InteriorShrinkFilter->SetShrinkFactor(0.995);
  
  this->InteriorMapper = vtkDataSetMapper::New();
  this->InteriorMapper->SetInputConnection(this->InteriorShrinkFilter->GetOutputPort());
  this->InteriorMapper->ScalarVisibilityOff();
  
  this->InteriorActor = vtkActor::New();
  this->InteriorActor->SetMapper( this->InteriorMapper );
  this->InteriorActor->GetProperty()->SetColor(0.5,0.5,0.5);
  this->InteriorActor->GetProperty()->SetOpacity(0.25);
  this->InteriorActor->GetProperty()->SetRepresentationToWireframe();
  this->InteriorActor->GetProperty()->SetAmbient(1.0);
  this->InteriorActor->SetVisibility(0);
  /* This is used to display a Portion of the Mesh when the Cutting Plane is enabled */
  
  this->CuttingPlaneWidget = vtkPlaneWidget::New();
  this->CuttingPlaneWidget->SetInput( this->UnstructuredGrid );
  this->CuttingPlaneWidget->SetRepresentationToSurface();
  this->CuttingPlaneWidget->GetPlaneProperty()->SetColor(0.2,0.2,0);
  this->CuttingPlaneWidget->GetPlaneProperty()->SetOpacity(0.2);
  //this->CuttingPlaneWidget->SetEnabled( 0 );
  //this->CuttingPlaneWidget->SetInteractor(this->RenderWidget->GetRenderWindowInteractor());
  //this->CuttingPlaneWidget->AddObserver(vtkCommand::InteractionEvent,PlaneMoveCallback);
  
  
  this->CuttingPlane = vtkPlane::New();
  
  this->ClipPlaneGeometryFilter = vtkExtractGeometry::New();
        this->ClipPlaneGeometryFilter->SetInput( this->UnstructuredGrid );
        //this->ClipPlaneGeometryFilter->SetImplicitFunction( this->CuttingPlane );
        
        /* This is used for Scale Display */
        this->lutFilter = vtkLookupTable::New();
  this->lutFilter->SetAlphaRange(1.0, 1.0);
  this->lutFilter->SetTableRange(0.0, 1.0);
  
        this->LegendActor = vtkScalarBarActor::New();
  vtkTextProperty *textProperty = this->LegendActor->GetTitleTextProperty();
  textProperty->SetFontFamilyToArial();
  textProperty->SetColor( TextColor );
  textProperty->ShadowOff();
  textProperty->ItalicOff();
  textProperty->SetFontSize(40);
  
  this->LegendActor->SetTitleTextProperty( textProperty );
  this->LegendActor->SetLabelTextProperty( textProperty );
  this->LegendActor->SetLookupTable((this->lutFilter));
  this->LegendActor->SetTitle("Quality");
  this->LegendActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  this->LegendActor->GetPositionCoordinate()->SetValue(0.1,0.05);
  this->LegendActor->SetOrientationToVertical();
  this->LegendActor->SetWidth(0.1);
  this->LegendActor->SetHeight(0.9);
  this->LegendActor->SetPosition(0.01,0.1);
  this->LegendActor->SetLabelFormat("%9.3f");
  this->LegendActor->SetVisibility( 0 );
  this->IsAverageEdgeLengthCalculated = 0;
  this->AverageEdgeLength = 0.0;
  this->PointSetOfNodeSet = NULL;
}

//----------------------------------------------------------------------------------------
vtkMimxMeshActor::~vtkMimxMeshActor()
{
        this->UnstructuredGrid->Delete();
        this->UnstructuredGridMapper->Delete();
        this->Actor->Delete();
        this->OutlineMapper->Delete();
        this->OutlineActor->Delete();
        this->InteriorMapper->Delete();
        this->InteriorActor->Delete();
        this->ShrinkFilter->Delete();
        if(this->PointSetOfNodeSet)
                this->PointSetOfNodeSet->Delete();
}

//----------------------------------------------------------------------------------------
vtkUnstructuredGrid* vtkMimxMeshActor::GetDataSet()
{
  return this->UnstructuredGrid;
}

//----------------------------------------------------------------------------------------
void vtkMimxMeshActor::SetDataSet(vtkUnstructuredGrid *mesh)
{
  this->UnstructuredGrid->DeepCopy( mesh );
  this->CreateElementSetList();
  this->UnstructuredGrid->Modified();
}


//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DeleteNodeSet(const char *Name)
{
        vtkFieldData *fielddata = this->UnstructuredGrid->GetFieldData();
        vtkPointData *pointdata = this->UnstructuredGrid->GetPointData();

        vtkStringArray *stringarray = vtkStringArray::SafeDownCast(
                fielddata->GetAbstractArray("Node_Set_Names"));

        if(!stringarray)        return;

        vtkDataArray *datasetarray;

        datasetarray = pointdata->GetArray(Name);
        if(datasetarray)        pointdata->RemoveArray(Name);
        
        // for displacement related field data
        char DispX[256], DispY[256], DispZ[256];

        strcpy(DispX, Name);
        strcpy(DispY, Name);
        strcpy(DispZ, Name);

        strcat(DispX, "_Displacement_X");
        strcat(DispY, "_Displacement_Y");
        strcat(DispZ, "_Displacement_Z");

        datasetarray = fielddata->GetArray(DispX);
        if(datasetarray)        fielddata->RemoveArray(DispX);

        datasetarray = fielddata->GetArray(DispY);
        if(datasetarray)        fielddata->RemoveArray(DispY);

        datasetarray = fielddata->GetArray(DispZ);
        if(datasetarray)        fielddata->RemoveArray(DispZ);

        // for Force related field data
        char ForceX[256], ForceY[256], ForceZ[256];

        strcpy(ForceX, Name);
        strcpy(ForceY, Name);
        strcpy(ForceZ, Name);

        strcat(ForceX, "_Force_X");
        strcat(ForceY, "_Force_Y");
        strcat(ForceZ, "_Force_Z");

        datasetarray = fielddata->GetArray(ForceX);
        if(datasetarray)        fielddata->RemoveArray(ForceX);

        datasetarray = fielddata->GetArray(ForceY);
        if(datasetarray)        fielddata->RemoveArray(ForceY);

        datasetarray = fielddata->GetArray(ForceZ);
        if(datasetarray)        fielddata->RemoveArray(ForceZ);

        // for Rotation related field data
        char RotationX[256], RotationY[256], RotationZ[256];

        strcpy(RotationX, Name);
        strcpy(RotationY, Name);
        strcpy(RotationZ, Name);

        strcat(RotationX, "_Rotation_X");
        strcat(RotationY, "_Rotation_Y");
        strcat(RotationZ, "_Rotation_Z");

        datasetarray = fielddata->GetArray(RotationX);
        if(datasetarray)        fielddata->RemoveArray(RotationX);

        datasetarray = fielddata->GetArray(RotationY);
        if(datasetarray)        fielddata->RemoveArray(RotationY);

        datasetarray = fielddata->GetArray(RotationZ);
        if(datasetarray)        fielddata->RemoveArray(RotationZ);

        // for Moment related field data
        char MomentX[256], MomentY[256], MomentZ[256];

        strcpy(MomentX, Name);
        strcpy(MomentY, Name);
        strcpy(MomentZ, Name);

        strcat(MomentX, "_Moment_X");
        strcat(MomentY, "_Moment_Y");
        strcat(MomentZ, "_Moment_Z");

        datasetarray = fielddata->GetArray(MomentX);
        if(datasetarray)        fielddata->RemoveArray(MomentX);

        datasetarray = fielddata->GetArray(MomentY);
        if(datasetarray)        fielddata->RemoveArray(MomentY);

        datasetarray = fielddata->GetArray(MomentZ);
        if(datasetarray)        fielddata->RemoveArray(MomentZ);

        vtkStringArray *temparray = vtkStringArray::New();
        temparray->DeepCopy(stringarray);
        stringarray->Initialize();
        int i;

        for (i=0; i<temparray->GetNumberOfTuples(); i++)
        {
                if(strcmp(temparray->GetValue(i), Name))
                {
                        stringarray->InsertNextValue(temparray->GetValue(i));
                }
        }
        temparray->Delete();
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DeleteElementSet(const char *Name)
{
        vtkFieldData *fielddata = this->UnstructuredGrid->GetFieldData();
        vtkCellData *celldata = this->UnstructuredGrid->GetCellData();

        vtkStringArray *stringarray = vtkStringArray::SafeDownCast(
                fielddata->GetAbstractArray("Element_Set_Names"));

        if(!stringarray)        return;

        vtkDataArray *datasetarray;
        
        datasetarray = celldata->GetArray(Name);
        if(datasetarray)        celldata->RemoveArray(Name);

        char Young[256];
        strcpy(Young, Name);
        strcat(Young, "_Constant_Youngs_Modulus");
        datasetarray = fielddata->GetArray(Young);
        if(datasetarray)        fielddata->RemoveArray(Young);

        char Poisson[256];
        strcpy(Poisson, Name);
        strcat(Poisson, "_Constant_Poissons_Ratio");
        datasetarray = fielddata->GetArray(Poisson);
        if(datasetarray)        fielddata->RemoveArray(Poisson);
        
        char ImageBased[256];
        strcpy(ImageBased, Name);
        strcat(ImageBased, "_Image_Based_Material_Property");
        datasetarray = celldata->GetArray(ImageBased);
        if(datasetarray)        celldata->RemoveArray(ImageBased);
        
        strcat(ImageBased, "_ReBin");
        datasetarray = celldata->GetArray(ImageBased);
        if(datasetarray)        celldata->RemoveArray(ImageBased);

        vtkStringArray *temparray = vtkStringArray::New();
        temparray->DeepCopy(stringarray);
        stringarray->Initialize();
        int i;

        for (i=0; i<temparray->GetNumberOfTuples(); i++)
        {
                if(strcmp(temparray->GetValue(i), Name))
                {
                        stringarray->InsertNextValue(temparray->GetValue(i));
                }
        }
        temparray->Delete();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetDisplayMode( int mode )
{

  switch ( mode )
  {
    case vtkMimxMeshActor::DisplayMesh:
      this->DisplayMode = mode;
      UpdateElementSetDisplay();
      UpdateMeshDisplay();
      break;
    case vtkMimxMeshActor::DisplayElementSets:
      this->DisplayMode = mode;
      UpdateMeshDisplay();
      UpdateElementSetDisplay();
      break;
  }
  
}

//----------------------------------------------------------------------------------
int vtkMimxMeshActor::GetDisplayMode( )
{
  return this->DisplayMode;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::UpdateMeshDisplay()
{
  if ((this->DisplayMode == vtkMimxMeshActor::DisplayMesh) && (this->IsVisible == true))
  {
    switch ( this->DisplayType )
    {
      case vtkMimxMeshActor::DisplaySurface:
        this->Actor->SetVisibility( 1 );
        //this->InteriorActor->SetVisibility( 1 );
        this->OutlineActor->SetVisibility( 0 );
        break;
      case vtkMimxMeshActor::DisplayOutline:
        this->Actor->SetVisibility( 0 );
        //this->InteriorActor->SetVisibility( 0 );
        this->OutlineActor->SetVisibility( 1 );
        break;
      case vtkMimxMeshActor::DisplaySurfaceAndOutline:
        this->Actor->SetVisibility( 1 );
        //this->InteriorActor->SetVisibility( 1 );
        this->OutlineActor->SetVisibility( 1 );
        break;
    }
  }
  else
  {
    this->Actor->SetVisibility( 0 );
    this->InteriorActor->SetVisibility( 0 );
    this->OutlineActor->SetVisibility( 0 );
  }
    
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::UpdateElementSetDisplay()
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin(); it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    if (( currentSet->IsVisible == true ) && 
        (this->DisplayMode == vtkMimxMeshActor::DisplayElementSets) && 
        (this->IsVisible == true))
    {
      switch ( currentSet->DisplayType )
      {
        case vtkMimxMeshActor::DisplaySurface:
          currentSet->SurfaceActor->SetVisibility( 1 );
          //currentSet->InteriorActor->SetVisibility( 1 );
          currentSet->OutlineActor->SetVisibility( 0 );
          break;
        case vtkMimxMeshActor::DisplayOutline:
          currentSet->SurfaceActor->SetVisibility( 0 );
          //currentSet->InteriorActor->SetVisibility( 0 );
          currentSet->OutlineActor->SetVisibility( 1 );
          break;
        case vtkMimxMeshActor::DisplaySurfaceAndOutline:
          currentSet->SurfaceActor->SetVisibility( 1 );
          //currentSet->InteriorActor->SetVisibility( 1 );
          currentSet->OutlineActor->SetVisibility( 1 );
          break;
      }
    }
    else
    {
      currentSet->SurfaceActor->SetVisibility( 0 );
      currentSet->InteriorActor->SetVisibility( 0 );
      currentSet->OutlineActor->SetVisibility( 0 );
    }
  }
  
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshDisplayType( int type)
{
  switch ( type )
  {
    case vtkMimxMeshActor::DisplaySurface:
    case vtkMimxMeshActor::DisplayOutline:
    case vtkMimxMeshActor::DisplaySurfaceAndOutline:
      this->DisplayType = type;
      break;
  }
  UpdateElementSetDisplay();
  UpdateMeshDisplay();
}


//----------------------------------------------------------------------------------
int vtkMimxMeshActor::GetMeshDisplayType( )
{
  return this->DisplayType;
}


//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetMeshOutlineColor(double &red, double &green, double &blue)
{
  this->OutlineActor->GetProperty()->GetColor(red, green, blue);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetMeshOutlineColor(double rgb[3])
{
  this->OutlineActor->GetProperty()->GetColor(rgb);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshOutlineColor(double red, double green, double blue)
{
  this->OutlineActor->GetProperty()->SetColor(red, green, blue);
  this->OutlineActor->GetProperty()->SetEdgeColor(red, green, blue);
  this->OutlineActor->Modified();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshOutlineColor(double rgb[3])
{
  this->SetMeshOutlineColor(rgb[0], rgb[1], rgb[2]);
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetMeshOutlineRadius( )
{
  return this->OutlineActor->GetProperty()->GetLineWidth( );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshOutlineRadius(double radius)
{
  this->OutlineActor->GetProperty()->SetLineWidth(radius);
  this->OutlineActor->Modified();
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetMeshShrinkFactor( )
{
  return this->ShrinkFilter->GetShrinkFactor();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshShrinkFactor(double shrinkFactor)
{
  this->ShrinkFilter->SetShrinkFactor( shrinkFactor );
  this->Actor->Modified();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshColor(double red, double green, double blue)
{
  this->Actor->GetProperty()->SetColor(red, green, blue);
  this->Actor->Modified();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshColor(double rgb[3])
{
  this->SetMeshColor(rgb[0], rgb[1], rgb[2]);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetMeshColor(double &red, double &green, double &blue)
{
  this->Actor->GetProperty()->GetColor(red, green, blue);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetMeshColor(double rgb[3])
{
  this->Actor->GetProperty()->GetColor(rgb);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshOpacity(double opacity)
{
  this->Actor->GetProperty()->SetOpacity( opacity );
  this->Actor->Modified();
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetMeshOpacity( )
{
  return this->Actor->GetProperty()->GetOpacity( );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshScalarVisibility(bool visibility)
{
  this->UnstructuredGridMapper->SetScalarVisibility(static_cast<int>(visibility));
  this->Actor->Modified();
}

//----------------------------------------------------------------------------------
bool vtkMimxMeshActor::GetMeshScalarVisibility( )
{
  return static_cast<bool>( this->UnstructuredGridMapper->GetScalarVisibility( ) );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshLegendVisibility(bool visible)
{
  this->LegendActor->SetVisibility(static_cast<int>(visible));
  this->LegendActor->Modified();
}

//----------------------------------------------------------------------------------
bool vtkMimxMeshActor::GetMeshLegendVisibility( )
{
  return static_cast<bool>( this->LegendActor->GetVisibility( ) );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetMeshScalarName(std::string attributeName)
{
  this->UnstructuredGrid->GetCellData()->SetActiveAttribute(
      attributeName.c_str(),vtkDataSetAttributes::SCALARS);
  this->activeAttribute = attributeName;
  this->Actor->Modified();
  this->LegendActor->SetTitle( attributeName.c_str() );
}

//----------------------------------------------------------------------------------
std::string vtkMimxMeshActor::GetMeshScalarName( )
{
  return this->activeAttribute;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::EnableMeshCuttingPlane( )
{
  /*
  this->ExtractSelectedGeometryFilter->Modified();
  this->ExtractSelectedGeometryFilter->Update();
  
  this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->ExtractSelectedGeometryFilter->GetOutput());
  this->VTKQualityFilterPtr->Modified();
  */
  //this->CuttingPlaneWidget->SetInput( this->UnstructuredGrid );
  this->CuttingPlaneWidget->UpdatePlacement( );
  this->CuttingPlaneWidget->PlaceWidget();
  this->CuttingPlaneWidget->SetEnabled( 1 );
  this->CuttingPlaneWidget->GetPlane( this->CuttingPlane );
  this->CuttingPlaneEnabled = true;
  
  this->ShrinkFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
  this->OutlineGeometryFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
  this->InteriorShrinkFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DisableMeshCuttingPlane( )
{
        /*
        this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->InitialMesh);
        this->VTKQualityFilterPtr->Modified();
        this->CuttingPlaneEnabled = 0;
        */
  this->CuttingPlaneWidget->SetEnabled( 0 );
  this->CuttingPlaneEnabled = false;
  
  this->ShrinkFilter->SetInput(this->UnstructuredGrid);
  this->OutlineGeometryFilter->SetInput( this->UnstructuredGrid );
  this->InteriorShrinkFilter->SetInput( this->UnstructuredGrid );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetInvertCuttingPlane( bool invert )
{
        this->ClipPlaneGeometryFilter->SetExtractInside( static_cast<int> ( invert ) ); 
}

//----------------------------------------------------------------------------------
bool vtkMimxMeshActor::GetInvertCuttingPlane( )
{
        return static_cast<bool> ( this->ClipPlaneGeometryFilter->GetExtractInside( ) ); 
}


//----------------------------------------------------------------------------------
void vtkMimxMeshActor::ShowMesh()
{
  this->IsVisible = true;
  UpdateElementSetDisplay();
  UpdateMeshDisplay();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::HideMesh()
{
  this->IsVisible = false;
  UpdateMeshDisplay();
  UpdateElementSetDisplay();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetDisplayType( std::string setName, int type )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      switch ( type )
      {
        case vtkMimxMeshActor::DisplaySurface:
        case vtkMimxMeshActor::DisplayOutline:
        case vtkMimxMeshActor::DisplaySurfaceAndOutline:
          currentSet->DisplayType = type;
          break;
      }
      break;
    }
  }
  UpdateMeshDisplay();
  UpdateElementSetDisplay();
}


//----------------------------------------------------------------------------------
int vtkMimxMeshActor::GetElementSetDisplayType( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return currentSet->DisplayType;
    }
  }
  return vtkMimxMeshActor::DisplaySurfaceAndOutline;
}


//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetElementSetOutlineColor(std::string setName, double &red, double &green, double &blue)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    if (setName == currentSet->name)
    {
      currentSet->OutlineActor->GetProperty()->GetColor(red, green, blue);
      return;
    }
  }
  red = green = blue = 0.0;
  return;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetElementSetOutlineColor(std::string setName, double rgb[3])
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    if (setName == currentSet->name)
    {
      currentSet->OutlineActor->GetProperty()->GetColor(rgb);
      return;
    }
  }
  rgb[0] = rgb[1] = rgb[2] = 0.0;
  return;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetOutlineColor(std::string setName, double red, double green, double blue)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->OutlineActor->GetProperty()->SetColor(red, green, blue);
      currentSet->OutlineActor->Modified();
      return;
    }
  }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetOutlineColor(std::string setName, double rgb[3])
{
  this->SetElementSetOutlineColor(setName, rgb[0], rgb[1], rgb[2]);
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetElementSetShrinkFactor( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return currentSet->ShrinkFilter->GetShrinkFactor();
    }
  }
  
  return 0.0;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetShrinkFactor(std::string setName, double shrinkFactor)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->ShrinkFilter->SetShrinkFactor( shrinkFactor );
      currentSet->SurfaceActor->Modified();
      return;
    }
  }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetColor(std::string setName, double red, double green, double blue)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->SurfaceActor->GetProperty()->SetColor(red, green, blue);
      currentSet->SurfaceActor->Modified();
      return;
    }
  }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetColor(std::string setName, double rgb[3])
{
  this->SetElementSetColor(setName, rgb[0], rgb[1], rgb[2]);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetElementSetColor(std::string setName, double &red, double &green, double &blue)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->SurfaceActor->GetProperty()->GetColor(red, green, blue);
      return;
    }
  }
  red = green = blue = 0.0;
  return;  
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::GetElementSetColor(std::string setName, double rgb[3])
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->SurfaceActor->GetProperty()->GetColor(rgb);
      return;
    }
  }
  rgb[0] = rgb[1] = rgb[2] = 0.0;
  return;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetOpacity(std::string setName, double opacity)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->SurfaceActor->GetProperty()->SetOpacity( opacity );
      currentSet->SurfaceActor->Modified();
      return;
    }
  }
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetElementSetOpacity( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return currentSet->SurfaceActor->GetProperty()->GetOpacity( );
    }
  }
  
  return 0.0;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::ShowElementSet( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->IsVisible = true;
      break;
    }
  }
  
  UpdateMeshDisplay();
  UpdateElementSetDisplay();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::HideElementSet( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->IsVisible = false;
      break;
    }
  }
  
  UpdateMeshDisplay();
  UpdateElementSetDisplay();
}

//----------------------------------------------------------------------------------
bool vtkMimxMeshActor::GetElementSetVisibility( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return currentSet->IsVisible;
      break;
    }
  }
  
  return false;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetOutlineRadius( std::string setName, double radius )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->TubeFilter->SetRadius(radius);
      currentSet->OutlineActor->Modified();
      break;
    }
  }  
}

//----------------------------------------------------------------------------------
double vtkMimxMeshActor::GetElementSetOutlineRadius( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return currentSet->TubeFilter->GetRadius();
      break;
    }
  }
  
  return 0.0;
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::CreateElementSetList( )
{
  while (!ElementSetDisplayList.empty())
  {
    MeshDisplayProperty *currentSet = ElementSetDisplayList.front();
    if (currentSet->SurfaceActor) currentSet->SurfaceActor->Delete();
    if (currentSet->OutlineActor) currentSet->OutlineActor->Delete();
    if (currentSet->InteriorActor) currentSet->InteriorActor->Delete();
    if (currentSet->SurfaceMapper) currentSet->SurfaceMapper->Delete();
    if (currentSet->OutlineMapper) currentSet->OutlineMapper->Delete();
    if (currentSet->InteriorMapper) currentSet->InteriorMapper->Delete();
    if (currentSet->ShrinkFilter) currentSet->ShrinkFilter->Delete();
    ElementSetDisplayList.pop_front();
  }
  
  this->NumberOfElementSets = 0;
  vtkFieldData *fielddata = this->UnstructuredGrid->GetFieldData();
  if ( fielddata )
  {
        vtkStringArray *stringarray = vtkStringArray::SafeDownCast(
                fielddata->GetAbstractArray("Element_Set_Names"));
        if ( stringarray )
        {
        this->NumberOfElementSets = stringarray->GetNumberOfValues();
        for (int i=0;i<stringarray->GetNumberOfValues();i++)
        {
          vtkStdString name = stringarray->GetValue(i);  
          AddElementSetListItem( name );
        }
        }
        }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DeleteElementSetListItem( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      if (currentSet->SurfaceActor) currentSet->SurfaceActor->Delete();
      if (currentSet->OutlineActor) currentSet->OutlineActor->Delete();
      if (currentSet->InteriorActor) currentSet->InteriorActor->Delete();
      if (currentSet->SurfaceMapper) currentSet->SurfaceMapper->Delete();
      if (currentSet->OutlineMapper) currentSet->OutlineMapper->Delete();
      if (currentSet->InteriorMapper) currentSet->InteriorMapper->Delete();
      if (currentSet->ShrinkFilter) currentSet->ShrinkFilter->Delete();
      ElementSetDisplayList.erase(it);
      return;
    }
  }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::AddElementSetListItem( std::string setName )
{
  MeshDisplayProperty *elementSetProperty =  new MeshDisplayProperty;   
  elementSetProperty->name = setName;
  elementSetProperty->IsVisible = true;
  elementSetProperty->DisplayType = vtkMimxMeshActor::DisplaySurfaceAndOutline;
  
  /* Extract Cells */
        vtkCellData *celldata = this->UnstructuredGrid->GetCellData();
  vtkIntArray *datasetarray = vtkIntArray::SafeDownCast(celldata->GetArray(setName.c_str()));
  vtkIdList *cellIds = vtkIdList::New();
  for (int i=0;i<datasetarray->GetNumberOfTuples();i++)
  {
    if (datasetarray->GetValue(i) > 0 ) cellIds->InsertNextId(i);
  }
  elementSetProperty->ExtractCellsFilter = vtkExtractCells::New();
  elementSetProperty->ExtractCellsFilter->SetInput( this->UnstructuredGrid );
  elementSetProperty->ExtractCellsFilter->SetCellList( cellIds );
  
  elementSetProperty->ShrinkFilter = vtkShrinkFilter::New();
  elementSetProperty->ShrinkFilter->SetInput(elementSetProperty->ExtractCellsFilter->GetOutput());
  elementSetProperty->ShrinkFilter->SetShrinkFactor(1.0);

  elementSetProperty->SurfaceMapper = vtkDataSetMapper::New();
  elementSetProperty->SurfaceMapper->SetInputConnection(elementSetProperty->ShrinkFilter->GetOutputPort());
  
  elementSetProperty->SurfaceActor = vtkActor::New();
  elementSetProperty->SurfaceActor->SetMapper(elementSetProperty->SurfaceMapper);
  double redRand, greenRand, blueRand;
  redRand = static_cast<double> ( rand() ) / static_cast<double> ( RAND_MAX );
  greenRand = static_cast<double> ( rand() ) / static_cast<double> ( RAND_MAX );
  blueRand = static_cast<double> ( rand() ) / static_cast<double> ( RAND_MAX );
  elementSetProperty->SurfaceActor->GetProperty()->SetColor(redRand,greenRand,blueRand);
  
  /* Create the Outlines */
  elementSetProperty->GeometryFilter = vtkGeometryFilter::New();
  elementSetProperty->GeometryFilter->SetInput( elementSetProperty->ExtractCellsFilter->GetOutput() );

  vtkFeatureEdges* featureEdges = vtkFeatureEdges::New();
  featureEdges->SetInput( elementSetProperty->GeometryFilter->GetOutput() );
  featureEdges->BoundaryEdgesOn();
  featureEdges->ManifoldEdgesOn();
  featureEdges->FeatureEdgesOff();
  featureEdges->ColoringOff();
  
  elementSetProperty->TubeFilter = vtkTubeFilter::New();
  elementSetProperty->TubeFilter->SetInputConnection(featureEdges->GetOutputPort());
  elementSetProperty->TubeFilter->SetRadius(0.03);
  
  elementSetProperty->OutlineMapper = vtkPolyDataMapper::New();
  elementSetProperty->OutlineMapper->SetInputConnection( elementSetProperty->TubeFilter->GetOutputPort() );
  elementSetProperty->OutlineMapper->SetScalarVisibility( 0 );
  
  elementSetProperty->OutlineActor = vtkActor::New();
  elementSetProperty->OutlineActor->SetMapper( elementSetProperty->OutlineMapper );
  elementSetProperty->OutlineActor->GetProperty()->SetColor(0.0,0.0,0.0);
  elementSetProperty->OutlineActor->GetProperty()->SetRepresentationToSurface();
  elementSetProperty->OutlineActor->GetProperty()->SetAmbient(0);

  /* Create the Interior View */
  elementSetProperty->InteriorShrinkFilter = vtkShrinkFilter::New();
  elementSetProperty->InteriorShrinkFilter->SetInput( elementSetProperty->ExtractCellsFilter->GetOutput() );
  elementSetProperty->InteriorShrinkFilter->SetShrinkFactor(0.995);
  
  elementSetProperty->InteriorMapper = vtkDataSetMapper::New();
  elementSetProperty->InteriorMapper->SetInputConnection(elementSetProperty->InteriorShrinkFilter->GetOutputPort());
  elementSetProperty->InteriorMapper->ScalarVisibilityOff();
  
  elementSetProperty->InteriorActor = vtkActor::New();
  elementSetProperty->InteriorActor->SetMapper( elementSetProperty->InteriorMapper );
  elementSetProperty->InteriorActor->GetProperty()->SetColor(0.5,0.5,0.5);
  elementSetProperty->InteriorActor->GetProperty()->SetOpacity(0.25);
  elementSetProperty->InteriorActor->GetProperty()->SetRepresentationToWireframe();
  elementSetProperty->InteriorActor->GetProperty()->SetAmbient(1.0);
  elementSetProperty->InteriorActor->SetVisibility(0);
  
  /* Add the Display to the List */
  ElementSetDisplayList.push_back( elementSetProperty );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetRenderer(vtkRenderer *renderer)
{
  this->Renderer = renderer;
  this->Renderer->AddViewProp(this->Actor);
  this->Renderer->AddViewProp(this->OutlineActor);
  this->Renderer->AddViewProp(this->InteriorActor);
  this->Renderer->AddViewProp(this->LegendActor);
  
  //this->CuttingPlaneWidget->SetInteractor(this->RenderWidget->GetRenderWindowInteractor());
  
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  { 
    MeshDisplayProperty *currentSet = *it;
    this->Renderer->AddViewProp(currentSet->SurfaceActor);
    this->Renderer->AddViewProp(currentSet->OutlineActor);
    this->Renderer->AddViewProp(currentSet->InteriorActor);
  }
  
  UpdateElementSetDisplay();
  UpdateMeshDisplay(); 
}

//----------------------------------------------------------------------------------
vtkRenderer* vtkMimxMeshActor::GetRenderer( )
{
  return (this->Renderer);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetInteractor(vtkRenderWindowInteractor *interactor)
{
  this->Interactor = interactor;
  this->CuttingPlaneWidget->SetInteractor( interactor );
  this->CuttingPlaneWidget->SetEnabled( 0 );
  
  /***VAM ***/
  vtkPlaneWidgetEventCallback* PlaneMoveCallback = vtkPlaneWidgetEventCallback::New();
  PlaneMoveCallback->PlaneInstance = this->CuttingPlane;
  this->CuttingPlaneWidget->AddObserver(vtkCommand::InteractionEvent,PlaneMoveCallback);  
  this->ClipPlaneGeometryFilter->SetImplicitFunction(this->CuttingPlane);
}

//----------------------------------------------------------------------------------
vtkRenderWindowInteractor* vtkMimxMeshActor::GetInteractor( )
{
  return (this->Interactor);
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetLegendRange(double min, double max)
{
  this->lutFilter->SetTableRange(min, max);
  this->lutFilter->ForceBuild();
  this->UnstructuredGridMapper->SetScalarRange(min, max);

  this->LegendActor->SetLookupTable(this->lutFilter);
  this->LegendActor->Modified();
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetScalarName(std::string setName, std::string scalarName)
{
        vtkDoubleArray *doubleArray = vtkDoubleArray::SafeDownCast(
                this->UnstructuredGrid->GetCellData()->GetArray(scalarName.c_str()));
        if(!doubleArray)        return;

        double* range;
        range = this->ComputeElementSetScalarRange(setName.c_str(), scalarName.c_str());
        this->SetLegendRange(range[0], range[1]);
    this->SetMeshScalarName( scalarName );
}

//----------------------------------------------------------------------------------
std::string vtkMimxMeshActor::GetElementSetScalarName( std::string setName )
{
  return this->GetMeshScalarName( );
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::EnableElementSetCuttingPlane( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      this->ClipPlaneGeometryFilter->SetInput( currentSet->ExtractCellsFilter->GetOutput() );
      this->CuttingPlaneWidget->UpdatePlacement( );
      this->CuttingPlaneWidget->PlaceWidget();
      this->CuttingPlaneWidget->SetEnabled( 1 );
      this->CuttingPlaneWidget->GetPlane( this->CuttingPlane );
      this->CuttingPlaneEnabled = true;
      
      currentSet->ShrinkFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
      currentSet->GeometryFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
      currentSet->InteriorShrinkFilter->SetInput( this->ClipPlaneGeometryFilter->GetOutput() );
      break;
    }
  }  
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DisableElementSetCuttingPlane( std::string setName )
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      this->CuttingPlaneWidget->SetEnabled( 0 );
      this->CuttingPlaneEnabled = false;
      this->ClipPlaneGeometryFilter->SetInput( this->UnstructuredGrid );
      
      currentSet->ShrinkFilter->SetInput( currentSet->ExtractCellsFilter->GetOutput() );
      currentSet->GeometryFilter->SetInput( currentSet->ExtractCellsFilter->GetOutput() );
      currentSet->InteriorShrinkFilter->SetInput( currentSet->ExtractCellsFilter->GetOutput() );
      this->Actor->Modified();
      break;
    }
  }  
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetElementSetScalarVisibility(std::string setName, bool visibility)
{
  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      currentSet->SurfaceMapper->SetScalarVisibility(static_cast<int>(visibility));
      this->Actor->Modified();
      break;
    }
  }
}
//----------------------------------------------------------------------------------
bool vtkMimxMeshActor::GetElementSetScalarVisibility( std::string setName )
{

  std::list<MeshDisplayProperty*>::iterator it;
  
  for ( it=this->ElementSetDisplayList.begin() ; it != this->ElementSetDisplayList.end(); it++ )
  {
    MeshDisplayProperty *currentSet = *it;
    
    if (setName == currentSet->name)
    {
      return static_cast<bool>( currentSet->SurfaceMapper->GetScalarVisibility( ) );
      break;
    }
  }
  return false;

}
      
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::DeleteBoundaryConditionStep(int StepNum)
{
        int i,j;
        if(StepNum < 1) return;
        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;
        vtkIntArray *boundCond = vtkIntArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray("Boundary_Condition_Number_Of_Steps"));
        if(!boundCond)  return;
        int numSteps = boundCond->GetValue(0);
        if(StepNum > numSteps)  return;
        
        char charStepNum[10];
        sprintf(charStepNum, "%d", StepNum);

        vtkStringArray *nodesetnamestring = vtkStringArray::SafeDownCast(
                ugrid->GetFieldData()->GetAbstractArray("Node_Set_Names"));
        if(!nodesetnamestring)  return;

        vtkFloatArray *floatarray;
        char Concatenate[256];
        vtkFieldData *fieldData = ugrid->GetFieldData();
        for(i=0; i<nodesetnamestring->GetNumberOfValues(); i++)
        {
                const char* nodesetname =  nodesetnamestring->GetValue(i);

                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "X", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "Y", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "Z", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "X", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "Y", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "Z", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "X", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "Y", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "Z", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "X", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "Y", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
                //
                Concatenate[256];
                this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "Z", Concatenate);
                floatarray = vtkFloatArray::SafeDownCast(
                        ugrid->GetFieldData()->GetArray(Concatenate));
                if(floatarray)
                {
                        fieldData->RemoveArray(Concatenate);
                }
        }
        // loop through all the higher numbered steps to change the step numbers
        for (j=StepNum+1; j<=numSteps;j++)
        {
                charStepNum[10];
                char charStepNumNew[10];
                sprintf(charStepNum, "%d", j);
                sprintf(charStepNumNew, "%d", j-1);
                //itoa(j, charStepNum, 10);
                //itoa(j-1, charStepNumNew, 10);
                char ConcatenateNew[256];
                for(i=0; i<nodesetnamestring->GetNumberOfValues(); i++)
                {
                        const char* nodesetname =  nodesetnamestring->GetValue(i);
                        
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "X", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Force", "X", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "Y", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Force", "Y", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Force", "Z", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Force", "Z", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "X", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Displacement", "X", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "Y", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Displacement", "Y", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Displacement", "Z", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Displacement", "Z", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "X", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Rotation", "X", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "Y", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Rotation", "Y", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Rotation", "Z", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Rotation", "Z", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "X", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Moment", "X", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "Y", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Moment", "Y", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                        Concatenate[256];
                        ConcatenateNew[256];
                        this->ConcatenateStrings("Step", charStepNum, nodesetname, "Moment", "Z", Concatenate);
                        this->ConcatenateStrings("Step", charStepNumNew, nodesetname, "Moment", "Z", ConcatenateNew);
                        floatarray = vtkFloatArray::SafeDownCast(
                                ugrid->GetFieldData()->GetArray(Concatenate));
                        if(floatarray)
                        {
                                floatarray->SetName(ConcatenateNew);
                        }
                        //
                }
        }
        boundCond->SetValue(0, numSteps-1);
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::ConcatenateStrings(const char* Step, const char* Num, 
                        const char* NodeSetName, const char* Type, const char* Direction, char *Name)
{
        strcpy(Name, Step);
        strcat(Name, "_");
        strcat(Name,Num);
        strcat(Name, "_");
        strcat(Name, NodeSetName);
        strcat(Name, "_");
        strcat(Name,Type);
        strcat(Name, "_");
        strcat(Name, Direction);
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::ChangeElementSetNumbers(const char *ElSetName, int StartEleNum)
{
        vtkCellData *cellData = this->UnstructuredGrid->GetCellData();
        vtkDataArray *dataArray = cellData->GetArray(ElSetName);
        if(!dataArray)  return;
        vtkIntArray *elsetValues = vtkIntArray::SafeDownCast(dataArray);
        if(!elsetValues)        return;
        vtkDataArray *elementNumbers = cellData->GetArray("Element_Numbers");
        if(!elementNumbers)     return;
        vtkIntArray *elementNumbersInt = vtkIntArray::SafeDownCast(elementNumbers);
        if(!elementNumbersInt)  return;

        int i;
        int numCells = this->UnstructuredGrid->GetNumberOfCells();
        for (i=0; i<numCells; i++)
        {
                int belongs = elsetValues->GetValue(i);
                if(belongs)
                {
                        elementNumbersInt->SetValue(i, StartEleNum++);
                }
        }
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::ChangeNodeSetNumbers(const char *NodeSetName, int StartNodeNum)
{
        vtkPointData *pointData = this->UnstructuredGrid->GetPointData();
        vtkDataArray *dataArray = pointData->GetArray(NodeSetName);
        if(!dataArray)  return;
        vtkIntArray *nodesetValues = vtkIntArray::SafeDownCast(dataArray);
        if(!nodesetValues)      return;
        vtkDataArray *nodeNumbers = pointData->GetArray("Node_Numbers");
        if(!nodeNumbers)        return;
        vtkIntArray *nodeNumbersInt = vtkIntArray::SafeDownCast(nodeNumbers);
        if(!nodeNumbersInt)     return;

        int i;
        int numPoints = this->UnstructuredGrid->GetNumberOfPoints();
        for (i=0; i<numPoints; i++)
        {
                int belongs = nodesetValues->GetValue(i);
                if(belongs)
                {
                        nodeNumbersInt->SetValue(i, StartNodeNum++);
                }
        }
}

//----------------------------------------------------------------------------------
void vtkMimxMeshActor::SetLegendTextColor(double color[3])
{
  TextColor[0] = color[0];
  TextColor[1] = color[1];
  TextColor[2] = color[2];
  
  vtkTextProperty *textProperty = this->LegendActor->GetTitleTextProperty();
  textProperty->SetColor( TextColor );
  
  this->LegendActor->SetTitleTextProperty( textProperty );
  this->LegendActor->SetLabelTextProperty( textProperty );
}

//----------------------------------------------------------------------------------
double *vtkMimxMeshActor::GetLegendTextColor( )
{
  return TextColor;
}
  
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------------
void vtkMimxMeshActor::CalculateAverageEdgeLength()
{
        if(!this->UnstructuredGrid)     return;
        int numNodes, numCells;
        numNodes = this->UnstructuredGrid->GetNumberOfPoints();
        numCells = this->UnstructuredGrid->GetNumberOfCells();
        if(!numCells || !numNodes)      return;

        double cumdist = 0.0;
        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;
        int i,j;
        int count = 0;
        for (i=0; i<numCells; i++)
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
        this->AverageEdgeLength = cumdist/count;
        this->IsAverageEdgeLengthCalculated = 1;
}
//----------------------------------------------------------------------------------
vtkPointSet* vtkMimxMeshActor::GetPointSetOfNodeSet(const char* NodeSetName)
{
        if(!this->UnstructuredGrid->GetPointData()->GetArray(NodeSetName))      return NULL;
        vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                this->UnstructuredGrid->GetPointData()->GetArray(NodeSetName));

        vtkPoints *points = vtkPoints::New();
        if(!this->PointSetOfNodeSet)    this->PointSetOfNodeSet = vtkUnstructuredGrid::New();
        else    this->PointSetOfNodeSet->Initialize();
        
        int numPoints = this->UnstructuredGrid->GetNumberOfPoints();

        for (int i=0; i<numPoints; i++)
        {
                if(intarray->GetValue(i))
                {
                        points->InsertNextPoint(this->UnstructuredGrid->GetPoint(i));
                }
        }
        this->PointSetOfNodeSet->SetPoints(points);
        points->Delete();
        return this->PointSetOfNodeSet;
}
//-----------------------------------------------------------------------------------
void vtkMimxMeshActor::StoreConstantMaterialProperty(
        const char* ElSetName, double YoungMod)
{
        int i;

        char imagebased[256];
        strcpy(imagebased, ElSetName);
        strcat(imagebased, "_Image_Based_Material_Property");

        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;

        vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(ElSetName));
        if(!intarray)   return;
        vtkDoubleArray *matarray = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(imagebased));

        if(matarray)
        {
                ugrid->GetCellData()->RemoveArray(imagebased);
                strcat(imagebased, "_ReBin");

                matarray = vtkDoubleArray::SafeDownCast(ugrid->GetCellData()->GetArray(imagebased));

                if(matarray)    ugrid->GetCellData()->RemoveArray(imagebased);
        }

        matarray = vtkDoubleArray::SafeDownCast(ugrid->GetCellData()->GetArray("Youngs_Modulus"));
        int numCells = ugrid->GetNumberOfCells();
        if(!matarray)
        {
                matarray = vtkDoubleArray::New();
                matarray->SetNumberOfValues(numCells);
                matarray->SetName("Youngs_Modulus");
                for (i=0; i<numCells; i++)
                {
                        matarray->SetValue(i, -1);
                }
                ugrid->GetCellData()->AddArray(matarray);
                matarray->Delete();
        }
        matarray = vtkDoubleArray::SafeDownCast(ugrid->GetCellData()->GetArray("Youngs_Modulus"));
        
        for (i=0; i< numCells; i++)
        {
                if(intarray->GetValue(i))
                        matarray->SetValue(i, YoungMod);
        }

        char young[256];
        strcpy(young, ElSetName);
        strcat(young, "_Constant_Youngs_Modulus");

        vtkDoubleArray *Earray = vtkDoubleArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(young));        

        if(Earray)
        {
                Earray->SetValue(0, YoungMod);
        }
        else
        {
                Earray = vtkDoubleArray::New();
                Earray->SetName(young);
                Earray->InsertNextValue(YoungMod);
                ugrid->GetFieldData()->AddArray(Earray);
                Earray->Delete();
        }
}
//------------------------------------------------------------------------------------
void vtkMimxMeshActor::StoreImageBasedMaterialProperty(const char *ElSetName)
{
        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;
        char str[256];
        int i;
        strcpy(str, ElSetName);
        strcat(str, "_Image_Based_Material_Property");
        vtkDoubleArray *matarray = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(str));
        if(!matarray)   return;

        vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(ElSetName));
        if(!intarray)   return;

        vtkDoubleArray *youngsmodulus = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray("Youngs_Modulus"));
        int numCells = ugrid->GetNumberOfCells();
        if(!youngsmodulus)
        {
                youngsmodulus = vtkDoubleArray::New();
                youngsmodulus->SetNumberOfValues(numCells);
                youngsmodulus->SetName("Youngs_Modulus");
                for (i=0; i<numCells; i++)
                {
                        matarray->SetValue(i, -1);
                }
                ugrid->GetCellData()->AddArray(youngsmodulus);
                youngsmodulus->Delete();
        }
        youngsmodulus = vtkDoubleArray::SafeDownCast(ugrid->GetCellData()->GetArray("Youngs_Modulus")); 

        for (i=0; i< numCells; i++)
        {
                if(intarray->GetValue(i))
                        youngsmodulus->SetValue(i, matarray->GetValue(i));
        }
}
//-------------------------------------------------------------------------------------
void vtkMimxMeshActor::StoreConstantPoissonsRatio(
        const char *ElSetName, double PoissonRatio)
{
        char poisson[256];
        strcpy(poisson, ElSetName);
        strcat(poisson, "_Constant_Poissons_Ratio");

        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;

        vtkFloatArray *Nuarray = vtkFloatArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(poisson));

        if(Nuarray)
        {
                Nuarray->SetValue(0, PoissonRatio);
        }
        else
        {
                Nuarray = vtkFloatArray::New();
                Nuarray->SetName(poisson);
                Nuarray->InsertNextValue(PoissonRatio);
                ugrid->GetFieldData()->AddArray(Nuarray);
                Nuarray->Delete();
        }
}
//-------------------------------------------------------------------------------------
void vtkMimxMeshActor::StoreImageBasedMaterialPropertyReBin(const char *ElSetName)
{
        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;
        char str[256];
        int i;
        strcpy(str, ElSetName);
        strcat(str, "_Image_Based_Material_Property_ReBin");
        vtkDoubleArray *matarray = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(str));
        if(!matarray)   return;

        vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(ElSetName));
        if(!intarray)   return;

        vtkDoubleArray *youngsmodulus = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray("Youngs_Modulus"));
        int numCells = ugrid->GetNumberOfCells();
        if(!youngsmodulus)
        {
                youngsmodulus = vtkDoubleArray::New();
                youngsmodulus->SetNumberOfValues(numCells);
                youngsmodulus->SetName("Youngs_Modulus");
                for (i=0; i<numCells; i++)
                {
                        matarray->SetValue(i, -1);
                }
                ugrid->GetCellData()->AddArray(youngsmodulus);
                youngsmodulus->Delete();
        }
        youngsmodulus = vtkDoubleArray::SafeDownCast(ugrid->GetCellData()->GetArray("Youngs_Modulus")); 

        for (i=0; i< numCells; i++)
        {
                if(intarray->GetValue(i))
                        youngsmodulus->SetValue(i, matarray->GetValue(i));
        }
}
//-------------------------------------------------------------------------------------
double* vtkMimxMeshActor::ComputeElementSetScalarRange(
        const char* ElSetName, const char* ArrayName)
{
        double range[2];
        double min = VTK_FLOAT_MAX;
        double max = VTK_FLOAT_MIN;

        vtkUnstructuredGrid *ugrid = this->UnstructuredGrid;

        vtkDoubleArray *doublearray = vtkDoubleArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(ArrayName));
        if(!doublearray)        return NULL;

        vtkIntArray *intarray = vtkIntArray::SafeDownCast(
                ugrid->GetCellData()->GetArray(ElSetName));

        if(!intarray)   return NULL;

        int i;
        int numCells = ugrid->GetNumberOfCells();
        for (i=0; i<numCells; i++)
        {
                if(intarray->GetValue(i))
                {
                        double val = doublearray->GetValue(i);
                        if(val > max)   max = val;
                        if(val < min)   min = val;
                }
        }
        range[0] = min;
        range[1] = max;
        return range;
}
