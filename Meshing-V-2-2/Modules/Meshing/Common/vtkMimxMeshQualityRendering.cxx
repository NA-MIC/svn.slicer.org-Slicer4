/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxMeshQualityRendering.cxx,v $
Language:  C++
Date:      $Date: 2008/10/17 03:37:39 $
Version:   $Revision: 1.25.2.2 $


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



#include "vtkMimxMeshQualityRendering.h"
#include "vtkObjectFactory.h"
#include "vtkRearrangeFields.h"
#include "vtkMeshQuality.h"
#include "vtkDataSet.h"
#include "vtkAssignAttribute.h"
#include "vtkShrinkFilter.h"
#include "vtkLookupTable.h"
#include "vtkCellData.h"
#include "vtkDoubleArray.h"
#include "vtkDataSetMapper.h"
#include "vtkUnstructuredGridReader.h"
#include "vtkUnstructuredGridWriter.h"
#include "vtkMath.h"
#include "vtkProperty.h"
#include "vtkLogLookupTable.h"
#include "vtkUnstructuredGrid.h"
#include "vtkThreshold.h"
#include "vtkRenderer.h"
#include "vtkExtractEdges.h"
#include "vtkTubeFilter.h"
#include "vtkPolyDataMapper.h"
#include "vtkGeometryFilter.h"
#include "vtkFeatureEdges.h"
#include "vtkHedgeHog.h"
#include "vtkFaceHedgeHog.h"
#include "vtkPolyDataNormals.h"
#include "vtkTextProperty.h"
#include "vtkScalarBarActor.h"
#include "vtkExtractCells.h"
#include "vtkIdList.h"
#include "vtkPointData.h"
#include "vtkPassThroughFilter.h"

#include "vtkFloatArray.h"

//----------------------------------------------------------------------------

vtkStandardNewMacro( vtkMimxMeshQualityRendering );
vtkCxxRevisionMacro(vtkMimxMeshQualityRendering, "$Revision: 1.4 $");


//----------------------------------------------------------------
vtkMimxMeshQualityRendering::vtkMimxMeshQualityRendering(void)
{
  this->InitVariables();
}


//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::InitVariables(void)
{

  // default shrink size,  opacity of each element, and other initial states

  this->ElementShrinkFactor = 1.0;
  this->ElementOpacity = 1.0;
  this->thresholdValue = 9999;
  this->SavedQualityMetric = 0;
  this->ShowFilledElements = 1;
  this->ShowClippedOutline = 1;
  this->ShowOutline = 0;
  this->ShowInteriorOutlines = 0;
  this->ShowSurfaceHedgehog = 0;
  this->CuttingPlaneEnabled = 0;

  // initialize to a standard state, if the user doesn't modify the
  // parameters.  Display only the outline initially and pick the volume test
  //
  this->IsInitialized = 0;
  this->InitialMesh = NULL;
  this->SavedMesh = NULL;
  this->SavedElementMapper = NULL;
  this->SavedColorLegendActor = NULL;
  this->ReportingCanvas = NULL;
  this->SavedHedgehogFilter = NULL;
  this->SavedMeshPolyDataForRendering = NULL;
  this-> SavedOutlinePolyDataForRendering = NULL;
  this->SavedHedgehogPolyDataForRendering = NULL;

  //  this is set when the user is overriding the lookup table autocolor through
  // the interface

  this->OverrideMeshColorRange = 0;

  // setup a re-used list to identify interesting cells that might be highlighted
  // to the user
  this->SavedCellList = vtkIdList::New();
  this->SavedCellList->Allocate(100);

  this->SavedDisplayPrecision = 3;

  // instantiate this fileter early, so the clipping plane can be set before or after the geometry

  this->ExtractSelectedGeometryFilter = vtkExtractGeometry::New();

}



//----------------------------------------------------------------
vtkMimxMeshQualityRendering::~vtkMimxMeshQualityRendering(void)
{
  cerr << "need to clean up memory here" << endl;
}



void vtkMimxMeshQualityRendering::SetThresholdValue (double thresh)
{
   this->thresholdValue = thresh;
   if (this->IsInitialized)
    this->UpdatePipeline();
}



//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetElementShrinkFactor(double shrink)
{
  this->ElementShrinkFactor = shrink;

  // test  to determine if the pipeline is initialized first
  // and only call update afterwards
  if (this->IsInitialized)
    this->UpdatePipeline();
}

//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetElementOpacity(double opacity)
{
  this->ElementOpacity = opacity;
  // test  to determine if the pipeline is initialized first
  // and only call update afterwards
  if (this->IsInitialized)
    this->UpdatePipeline();
}




//----------------------------------------------------------------
vtkActor*  vtkMimxMeshQualityRendering::ReturnFilledElementsActor(void)
{
  return this->SavedElementActor;
}

//----------------------------------------------------------------
vtkActor*  vtkMimxMeshQualityRendering::ReturnInternalOutlinesActor(void)
{
  return this->SavedInteriorOutlinesActor;
}

//----------------------------------------------------------------
vtkActor*  vtkMimxMeshQualityRendering::ReturnOutlineActor(void)
{
  return this->SavedOutlineActor;
}

//----------------------------------------------------------------
vtkActor*  vtkMimxMeshQualityRendering::ReturnClippedOutlineActor(void)
{
  return this->SavedClippedOutlineActor;
}


//----------------------------------------------------------------
vtkActor*  vtkMimxMeshQualityRendering::ReturnHedgehogActor(void)
{
  return this->SavedSurfaceHedgehogActor;
}


//----------------------------------------------------------------
vtkActor2D*  vtkMimxMeshQualityRendering::ReturnColorLegendActor(void)
{
  return this->SavedColorLegendActor;
}


void vtkMimxMeshQualityRendering::EnableCuttingPlane(void)
{
  // use the output of the extract geometry filter to draw the filled elements.  This
  // way, the implicit plane can "cut away" some elements.  A method call to enable is
  // required to minimize the amount of re-processing of the VTK pipeline needed.  A disable
  // event is required to reset the pipeline input

  this->ExtractSelectedGeometryFilter->Modified();
  this->ExtractSelectedGeometryFilter->Update();
  this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->ExtractSelectedGeometryFilter->GetOutput());
  this->VTKQualityFilterPtr->Modified();
  this->CuttingPlaneEnabled = 1;

}

void vtkMimxMeshQualityRendering::SaveSelectionAsNewMeshFile(char *filename)
{

  if (CuttingPlaneEnabled) {
    // use the output of the extract geometry filter to draw a selection, according to the
    // current cutting plane.  Save the output of the filter to a new file, according to the filename
    // passed as an argument
    vtkUnstructuredGridWriter *writer = vtkUnstructuredGridWriter::New();
    writer->SetFileName(filename);
    // CRL - changed to output of threshold so user has better control over
    // part of mesh to save
    //writer->SetInput(this->ExtractSelectedGeometryFilter->GetOutput());
    //writer->SetInput(this->thresholdFilter->GetOutput());
    writer->SetInput(this->shrinkFilter->GetOutput());
    writer->Write();
    writer->Delete();
  } else
  {
    cerr << "Error: cutting plane must be enabled" << endl;
  }
}



void vtkMimxMeshQualityRendering::SetInvertCuttingPlaneSelection(int invert)
{
 this->ExtractSelectedGeometryFilter->SetExtractInside(invert);
   // CRL - commented out 7/24/07 seemed in wrong method (belongs in disable method)
   //this->CuttingPlaneEnabled = 0;
}


void vtkMimxMeshQualityRendering::DisableCuttingPlane(void)
{
  this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->InitialMesh);
  this->VTKQualityFilterPtr->Modified();
  this->CuttingPlaneEnabled = 0;
}


void vtkMimxMeshQualityRendering::HighlightFocusElements(void)
{
        // there is a selection filter which passes only cells maintained on a list
        // of cell IDs.  Since the user is opting for viewing only the interesting elements,
        // we will modify the pipeline to look at the output of a selection filter
        this->CellSelectionFilter->SetInput(this->VTKQualityFilterPtr->GetInput());
        this->VTKQualityFilterPtr->SetInput(this->CellSelectionFilter->GetOutput());
        this->CellSelectionFilter->SetCellList(this->SavedCellList);
        this->VTKQualityFilterPtr->Update();
        this->SetElementsHighlighted(1);
}


void vtkMimxMeshQualityRendering::CancelElementHighlight(void)
{
        // this cancels highlight by removing the cell selection filter from
        // the pipeline
        //this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->InitialMesh);
        this->VTKQualityFilterPtr->SetInput(this->CellSelectionFilter->GetInput());
        this->VTKQualityFilterPtr->Update();
        this->SetElementsHighlighted(0);
}




void vtkMimxMeshQualityRendering::SetDisplayPrecision(int precision)
{
        // invoke change on color actor.  if other things need to be changed, we
        // would add them here.  There is no state recorded as an instance variable currently
        //this->DisplayedPrecision = precision;
        this->SavedDisplayPrecision = precision;
        int mantissa = 13-precision;
        char labelformat[20];
        sprintf(labelformat,"%%%d.%df",mantissa,precision);
        cout << "setting precision to: " << labelformat << endl;
        ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetLabelFormat(labelformat);
        this->VTKQualityFilterPtr->Update();
}






//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetCuttingPlaneFunction(vtkPlane *cuttingPlane)
{
  // this function is called with a user-allocated vtkPlane instance that
  // specifies a plane equation that should be used

 this->ExtractSelectedGeometryFilter->SetImplicitFunction(cuttingPlane);
 this->SavedCuttingPlaneImplicitFunction = cuttingPlane;
}



void vtkMimxMeshQualityRendering::SetHedgehogScale(double value)
{
         if (!this->IsInitialized)
          {
                 this->SavedHedgehogFilter->SetScaleFactor(value);
                 this->SavedHedgehogFilter->Update();
          }
}

//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::InitializeFromFile(char* filestr)
{
  this->InitVariables();
  //cout << "file read beginning" << endl;
  vtkUnstructuredGridReader* reader = vtkUnstructuredGridReader::New();
  reader->SetFileName(filestr);
  reader->Update();
  //cout << "file read completed" << endl;
  this->InitialMesh = (reader->GetOutput());
  this->SetupVTKPipeline();
  this->UpdatePipeline();
}

 //----------------------------------------------------------------
void vtkMimxMeshQualityRendering::InitializeFromExternalMesh(vtkUnstructuredGrid *grid)
{

        // first, check if the mesh is empty and skip action if it is
//      if (grid)
//      {
//         int numberOfCells = grid->GetNumberOfCells();
//
//         // now, maybe the grid did exist, but it is empty.  If so, lets still skip processing
//         if (numberOfCells > 0)
//         {
       this->InitialMesh = grid;

                  if (!this->IsInitialized)
                  {
                        SetupVTKPipeline();
                  }
                  else
                  {
                        this->CreateProcessedMesh();
                        this->SetQualityMeasureToVolume();
                        this->UpdatePipeline();
                  }
//         }
//      }
}

//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetQualityMeasure(int qualityTest)
{


  // store this because the user could set it before the pipeline is invoked
  this->SavedQualityMetric = qualityTest;
  if (this->IsInitialized)
   {
     this->InvokeQualityMeasure();
     // it seems that calling this again before a renderer refresh causes a crash
     //this->UpdatePipeline();
    }

}



//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::InvokeQualityMeasure(void)
{
  // here include a switch to set the most likely tests to be selected
  // using the enumerated values 0..4

  switch (this->SavedQualityMetric)
  {

    case 0: {this->VTKQualityFilterPtr->SetHexQualityMeasureToVolume();
                 if (this->IsInitialized) ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("Volume");
                 break;}
    case 1: {this->VTKQualityFilterPtr->SetHexQualityMeasureToEdgeCollapse();
                if (this->IsInitialized)  ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("EdgeCollapse");
                break;}
    case 2: {this->VTKQualityFilterPtr->SetHexQualityMeasureToJacobian();
                if (this->IsInitialized)  ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("Jacobian");
                break;}
    case 3: {this->VTKQualityFilterPtr->SetHexQualityMeasureToSkew();
                if (this->IsInitialized) ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("Skew");
                break;}
    case 4: {this->VTKQualityFilterPtr->SetHexQualityMeasureToAngleOutOfBounds();
                if (this->IsInitialized) ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("AngleBounds");
                break;}
    default: {this->VTKQualityFilterPtr->SetHexQualityMeasureToEdgeCollapse();
                if (this->IsInitialized) ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("EdgeCollapse");}
  }
  //this->VTKQualityFilterPtr->Modified();
  if (this->IsInitialized) {
    this->UpdatePipeline();
  }

}


void vtkMimxMeshQualityRendering::SetQualityMeasureToVolume(void)
{
   this->VTKQualityFilterPtr->SetHexQualityMeasureToVolume();
   this->VTKQualityFilterPtr->SetTetQualityMeasureToVolume();
   if (this->IsInitialized) ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("Volume");

}

void vtkMimxMeshQualityRendering::SetQualityMeasureToShape(void)
{
  this->VTKQualityFilterPtr->SetHexQualityMeasureToShape();
  this->VTKQualityFilterPtr->SetTetQualityMeasureToShape();

}

void vtkMimxMeshQualityRendering::SetQualityMeasureToJacobian(void)
{
  this->VTKQualityFilterPtr->SetHexQualityMeasureToJacobian();
  this->VTKQualityFilterPtr->SetTetQualityMeasureToJacobian();
  if (this->IsInitialized)  ((vtkScalarBarActor*)(this->SavedColorLegendActor))->SetTitle("Jacobian");

}

void vtkMimxMeshQualityRendering::SetQualityMeasureToFrobenius(void)
{
  this->VTKQualityFilterPtr->SetHexQualityMeasureToMedAspectFrobenius();
  this->VTKQualityFilterPtr->SetTetQualityMeasureToAspectFrobenius();
}

void vtkMimxMeshQualityRendering::SetQualityMeasureToEdgeRatio(void)
{
  this->VTKQualityFilterPtr->SetHexQualityMeasureToEdgeRatio();
  this->VTKQualityFilterPtr->SetTetQualityMeasureToEdgeRatio();
}




//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::CalculateMeshQuality(void)
{
  if (!this->IsInitialized)
  {
    SetupVTKPipeline();
  }
  //this->thresholdFilter->Modified();
  this->UpdatePipeline();
}


//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::FindMinimumAndMaximumQualityForMesh(vtkUnstructuredGrid *mesh,
                                                                double *minQualityFound,
                                                                double *maxQualityFound)
{

    char warningText[128];
   // initialize to values that will be overwritten
   *minQualityFound = 9.9e10;
   *maxQualityFound = -9.9e10;
   int thisID;

  // clear out the interesting cell list
   this->SavedCellList->Reset();

  // go through the cell data and record the smallest and largest values found
  // so we can use these values to autoscale the colors
  long numCells = ((mesh->GetCellData())->GetArray("Quality"))->GetNumberOfTuples();
  // cout << "QualityRender: examining " << numCells << " cells" << endl;

  // if there is a canvas, be sure to clear it for display
  if (this->ReportingCanvas != NULL)
     this->ReportingCanvas->GetWidget()->DeleteAll();

  // initialize that no bad cells have been found, update these values during the loop below
  bool foundBadValue = false;
  int badValueCount = 0;

  for (int i=0; i< numCells; i++)
  {
    double thisQ = ((vtkDoubleArray*)(mesh->GetCellData())->GetArray("Quality"))->GetValue(i);
    if (thisQ > *maxQualityFound) *maxQualityFound = thisQ;
    if (thisQ < *minQualityFound) *minQualityFound = thisQ;

    // negative value is significant, so lets start snooping on values
    if (thisQ < 0) {

        // record the occurance of at least one bad element
        foundBadValue |= true;
        badValueCount++;

      //cout << "element ID: " << i << " has quality: " << thisQ << endl;


      // pull the element ID out of the field data, because we can't depend on the loop index. Added test
      // for array existence because this pipeline is called with a zero element mesh once during initialization
      // and this case caused a crash during that situation.

//        if (numCells >0 && mesh->GetCellData()->HasArray("ELEMENT_ID"))
//                  thisID = ((vtkIntArray*)mesh->GetCellData()->GetArray("ELEMENT_ID"))->GetValue(i);
//       else
                    thisID = i;

      // to support dynamic resolution changes, a level of indirection
      // is needed, where the format string has to be created first
      int mantissa = 14-this->SavedDisplayPrecision;
      char formatstring[80];
      sprintf(formatstring, "ElementID: %%06d, quality: %%%d.%df",mantissa,this->SavedDisplayPrecision);
      //cout << "report frame format string:" << formatstring << endl;
      sprintf(warningText,formatstring,thisID,thisQ);
      //cout << warningText << endl;

      // keep a record in a cell list of this ID so we can highlight it later
      // if the user wants to
      this->SavedCellList->InsertNextId(i);

      // if there is a canvas, report the value of out of bounds elements
      if (this->ReportingCanvas != NULL)
        this->ReportingCanvas->GetWidget()->Append(warningText);
    }

  }

  // if there is a canvas, add the summary report at the end and display the report
   if ((this->ReportingCanvas != NULL)) {
        sprintf(warningText,"%d out of %d elements were distorted",badValueCount,((int) numCells));
       this->ReportingCanvas->GetWidget()->Append(warningText);
       this->ReportingCanvas->GetWidget()->Modified();
   }

  //cout << "Min quality found: " << *minQualityFound << " max quality found: " ;
  //cout << *maxQualityFound << endl;

  // we are displaying with logarithmetic coloring schemes, so force the lookup values
  // to be greater than zero

 // if (*minQualityFound < 0.001)
 // {
 //   *minQualityFound = 0.001;
 // cout << "clamping min quality to 0.001" << endl;
 // }
}

void vtkMimxMeshQualityRendering::SetMeshColorRange(double low, double high)
{
        // this method is called as an override to the general practice of
        // letting the VTK lookup table auto-scale.  The user is requesting the
        // lookup table range to be set specifically to look at specific range of the
        // mesh values.  Generally this will be a subrange, but no boundary checking is
        // performed
        if (this->OverrideMeshColorRange)
        {
                this->lutFilter->SetTableRange(low,high);
                this->SavedElementMapper->SetScalarRange (low, high);
                this->UpdatePipeline();
        }
}

//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetupVTKPipeline(void)
{


  // setup the pipeline here

  this->SetupMeshFilterPipeline();
  this->SetupOutlineVTKPipeline();
  this->SetupClippedOutlineVTKPipeline();
  this->SetupInteriorVTKPipeline();
  this->SetupHedgeHogVTKPipeline();
  this->SetupColorLegendActor();

  // the VTK pipeline exists, now it is OK to change values on the filters
  this->IsInitialized = 1;

}



void vtkMimxMeshQualityRendering::SetupOutlineVTKPipeline(void)
{

  //vtkExtractEdges* extractOutline = (vtkExtractEdges*) vtkExtractEdges::New();
  //  extractOutline->SetInput(this->InitialMesh);
  vtkGeometryFilter* outlineGeo = (vtkGeometryFilter*) vtkGeometryFilter::New();
    outlineGeo->SetInput(this->InitialMesh);
  vtkFeatureEdges* featureEdges = (vtkFeatureEdges*) vtkFeatureEdges::New();
    featureEdges->SetInput(outlineGeo->GetOutput());
    featureEdges->BoundaryEdgesOn();
    featureEdges->ManifoldEdgesOn();
    featureEdges->FeatureEdgesOff();
  vtkTubeFilter* outlineTube = (vtkTubeFilter*) vtkTubeFilter::New();
    outlineTube->SetInputConnection(featureEdges->GetOutputPort());
    outlineTube->SetRadius(0.03);
  vtkPolyDataMapper* outlineMap = (vtkPolyDataMapper*)vtkPolyDataMapper::New();
    outlineMap->SetInputConnection(outlineTube->GetOutputPort());
  vtkActor* outlineActor = (vtkActor*)vtkActor::New();
    outlineActor->SetMapper(outlineMap);
    (outlineActor->GetProperty())->SetColor(0.0,0.0,0.0);
    (outlineActor->GetProperty())->SetRepresentationToSurface();
    (outlineActor->GetProperty())->SetAmbient(0);
  this->SavedOutlineActor = outlineActor;
}


void vtkMimxMeshQualityRendering::SetupClippedOutlineVTKPipeline(void)
{

  //vtkExtractEdges* extractOutline = (vtkExtractEdges*) vtkExtractEdges::New();
    //extractOutline->SetInput(this->InitialMesh);
  vtkGeometryFilter* outlineGeo = (vtkGeometryFilter*) vtkGeometryFilter::New();
    outlineGeo->SetInput(this->SavedMesh);
  vtkFeatureEdges* featureEdges = (vtkFeatureEdges*) vtkFeatureEdges::New();
    featureEdges->SetInput(outlineGeo->GetOutput());
    featureEdges->BoundaryEdgesOn();
    featureEdges->ManifoldEdgesOn();
    featureEdges->FeatureEdgesOff();
  vtkTubeFilter* outlineTube = (vtkTubeFilter*) vtkTubeFilter::New();
    outlineTube->SetInputConnection(featureEdges->GetOutputPort());
    outlineTube->SetRadius(0.03);
  vtkPolyDataMapper* outlineMap = (vtkPolyDataMapper*)vtkPolyDataMapper::New();
    outlineMap->SetInputConnection(outlineTube->GetOutputPort());
  vtkActor* outlineActor = (vtkActor*)vtkActor::New();
    outlineActor->SetMapper(outlineMap);
    (outlineActor->GetProperty())->SetColor(0.0,0.0,0.0);
    (outlineActor->GetProperty())->SetRepresentationToSurface();
    (outlineActor->GetProperty())->SetAmbient(0);
  this->SavedClippedOutlineActor = outlineActor;

  // add output PolyData creation for Slicer integration
  this->SavedOutlinePolyDataForRendering = outlineTube->GetOutput();
}



void vtkMimxMeshQualityRendering::SetupInteriorVTKPipeline(void)
{
  // this will create a small boundary between the elements by shrinking,
  // so we will see the interior elements.  This shrinking step is needed
  // because the VTK Mappers display only boundary elements, not the internal
  // elements

 vtkShrinkFilter* interiorShrink = (vtkShrinkFilter*) vtkShrinkFilter::New();
   interiorShrink->SetInput(this->SavedMesh);
   interiorShrink->SetShrinkFactor(0.995);
 vtkDataSetMapper* interiorMapper = (vtkDataSetMapper*)vtkDataSetMapper::New();
   interiorMapper->SetInputConnection(interiorShrink->GetOutputPort());
   interiorMapper->ScalarVisibilityOff();
 vtkActor* interiorActor = (vtkActor*)vtkActor::New();
   interiorActor->SetMapper(interiorMapper);
   (interiorActor->GetProperty())->SetColor(0.5,0.5,0.5);
   (interiorActor->GetProperty())->SetOpacity(0.25);
   (interiorActor->GetProperty())->SetRepresentationToWireframe();
   // assign ambient lighting so the lines are always the same color regardless or orientation
   (interiorActor->GetProperty())->SetAmbient(1.0);
 // keep a pointer to this actor so it can be referenced later. *** fix
 // to do this reference according to the VTK rules instead
 this->SavedInteriorOutlinesActor = interiorActor;
}


void vtkMimxMeshQualityRendering::SetupHedgeHogVTKPipeline(void)
{
  // this method takes the unstructured grid elements and pipes them through a
  // VTK hedgehog filter to render arrows aligned with the normals from the outside
  // surfaces

  vtkGeometryFilter* geom = (vtkGeometryFilter*) vtkGeometryFilter::New();
   geom->SetInput(this->SavedMesh);
   geom->MergingOff();

  // set parameters to normal filter so there is a single vector at the corner
  // of each element drawn.  This isn't as good as having a normal on the center of
  // the boundary polygon, but we don't seem to be able to generate Cell Normals unless
  // we make polygons of the boundary surface.  This will do for now.

  // *** later:  mesh -> contour-> normalsWithCellNormalsOnly

  vtkPolyDataNormals* normals = (vtkPolyDataNormals*)vtkPolyDataNormals::New();
   normals->SetInput(geom->GetOutput());
   normals->SplittingOff();
   //normals->ComputePointNormalsOff();
   normals->ComputeCellNormalsOn();

  vtkFaceHedgeHog* hedge = (vtkFaceHedgeHog*) vtkFaceHedgeHog::New();
   hedge->SetInput(normals->GetOutput());
   hedge->SetScaleFactor(1.0);
   hedge->SetVectorModeToUseNormal();
   this->SavedHedgehogFilter = hedge;

  vtkPolyDataMapper* hedgeMapper = (vtkPolyDataMapper*)vtkPolyDataMapper::New();
   hedgeMapper->SetInputConnection(hedge->GetOutputPort());
   //hedgeMapper->ScalarVisibilityOff();
  vtkActor* hedgeActor = (vtkActor*)vtkActor::New();
   hedgeActor->SetMapper(hedgeMapper);
   (hedgeActor->GetProperty())->SetColor(0.5,0.2,0.5);
   (hedgeActor->GetProperty())->SetRepresentationToWireframe();
   // assign ambient lighting so the lines are always the same color regardless or orientation
   (hedgeActor->GetProperty())->SetAmbient(1.0);
  // keep a pointer to this actor so it can be referenced later. *** fix
  // to do this reference according to the VTK rules instead
  this->SavedSurfaceHedgehogActor = hedgeActor;
}

void vtkMimxMeshQualityRendering::CreateProcessedMesh(void)
{
        // Since we may want to cut away some elements during interaction, we can't use the position
        // in the rendering dataset as a unique id.  Instead, each element must have an index number added as
        // a custom attribute.  Once each element has a unique ID, then this id can be used to identify
        // elements by number, regardless of whether only some of them are currently being displayed
        // by the interface.   The first step is to find out the total number of elements and make an
        // array to hold the indices

        int numberOfCells = this->InitialMesh->GetNumberOfCells();
        cout << "initializing pipeline with " << numberOfCells << " cells" << endl;


        vtkFieldData* fieldData = this->InitialMesh->GetCellData();
        vtkIntArray* elementIdNumbers = vtkIntArray::New();
        elementIdNumbers->SetName("ELEMENT_ID");

        //  Fill the element id array with increasing, unique values

        for (int i=0; i< numberOfCells; i++) {
                  elementIdNumbers->InsertNextValue(i);
        }

        // add a cell extraction operation here.  Under control of the interface, a user can decide to display only a
        // subset of cells by selecting the cell IDs and updating the pipeline to put this filter in the pipeline.  For normal
        // display cases, this filter is not in the VTK pipeline.

        this->CellSelectionFilter->SetInput(this->InitialMesh);

        // the dataset might have point data scalars associated with it.  This
        // causes problems with some elements not displaying, so remove the
        // point data

//      vtkPointData *pointData = this->InitialMesh->GetPointData();
//      for (int count = 0; count < pointData->GetNumberOfArrays(); count++)
//      {
//              cout << "found data array on points.  Removing it." << endl;
//              pointData->RemoveArray(pointData->GetArrayName(count));
//      }

        // the dataset might have cell data scalars associated with it.  This
        // causes problems with some elements not displaying, so remove the
        // cell data

//      vtkCellData *cellData = this->InitialMesh->GetCellData();
//      for (int count = 0; count < cellData->GetNumberOfArrays(); count++)
//      {
//              cout << "found  data array on cells.  Removing it." << endl;
//              cellData->RemoveArray(cellData->GetArrayName(count));
//      }

        // Assign the element ID array to the mesh by putting the array in the field data
        // then there will be a new attribute with unique element ID assigned to each element

        if (!fieldData->HasArray("ELEMENT_ID"))
        {
          fieldData->AddArray(elementIdNumbers);
        } else {
          cout << "Mesh already has element ID numbers, keeping same numbers" << endl;
        }

        // create an extract geometry file instance in case the user enables the cutting plane
        // this would be used on the input dataset before testing for quality.   This filter
        // is not used in the pipeline unless the "EnableCuttingPlane" method is called.
        // We are putting the filter early in the pipelne  because it allows the
        // output of the quality filter to be specific to only
        // the selected cells.  This could be helpful when a subset is enabled -- then the summary
        // information (avg. quality, etc.) would pertain only to the selected element subset.
        // The vtkPlane instance must be set by calling the "SetCuttingPlane" method before enabling
        // the use of this filter.

        // the first time through, create the filter.  Afterwards, use the same filter

//        if (!this->IsInitialized) {
//            this->ExtractSelectedGeometryFilter = vtkExtractGeometry::New();
//            this->ExtractSelectedGeometryFilter->SetInput(this->InitialMesh);
//            this->ExtractSelectedGeometryFilter->SetImplicitFunction(this->SavedCuttingPlaneImplicitFunction);
//            vtkMeshQualityExtended* qual = vtkMeshQualityExtended::New();
//                this->VTKQualityFilterPtr = qual;
//        }
//        else {
//            this->ExtractSelectedGeometryFilter->SetInput(this->InitialMesh);
//            this->ExtractSelectedGeometryFilter->SetImplicitFunction(this->SavedCuttingPlaneImplicitFunction);
//        }
//

        if (!this->IsInitialized) {
               // this->ExtractSelectedGeometryFilter = vtkExtractGeometry::New();
                this->ExtractSelectedGeometryFilter->SetInput(this->InitialMesh);
                this->ExtractSelectedGeometryFilter->SetImplicitFunction(this->SavedCuttingPlaneImplicitFunction);
                vtkMeshQualityExtended* qual = vtkMeshQualityExtended::New();
                    this->VTKQualityFilterPtr = qual;
            }
            else {
                this->ExtractSelectedGeometryFilter->SetInput(this->InitialMesh);
                this->ExtractSelectedGeometryFilter->SetImplicitFunction(this->SavedCuttingPlaneImplicitFunction);
            }

            this->VTKQualityFilterPtr->SetInput((vtkDataSet*)this->InitialMesh);



        //vtkUnstructuredGridWriter *writer = vtkUnstructuredGridWriter::New();
        //writer->SetInput(qual->GetOutput());
        //writer->SetFileName("before.vtk");
        //writer->Write();

        this->InvokeQualityMeasure();
        this->VTKQualityFilterPtr->Modified();
        this->VTKQualityFilterPtr->Update();
        //cout << "mesh quality algorithm completed" << endl;

        // now loop through the quality values and find the minimum and maximum
        // values for this particular dataset

        vtkUnstructuredGrid* mesh = (vtkUnstructuredGrid*)this->VTKQualityFilterPtr->GetOutput();

        //vtkUnstructuredGridWriter *writer2 = vtkUnstructuredGridWriter::New();
        //writer2->SetInput(mesh);
        //writer2->SetFileName("after.vtk");
        //writer2->Write();

        // set a pointer to look at the mesh after it has had quality added to it
        this->SavedMesh = mesh;

        // point the filters at the mesh quality attributes so thresholding will
        // not get confused by other icoming attributes attached to the datasets

        if (this->SavedMesh->GetCellData()->HasArray("Quality"))
            this->SavedMesh->GetCellData()->SetActiveAttribute("Quality",vtkDataSetAttributes::SCALARS);
        if (this->SavedMesh->GetPointData()->HasArray("Quality"))
        this->SavedMesh->GetPointData()->SetActiveAttribute("Quality",vtkDataSetAttributes::SCALARS);

}

void vtkMimxMeshQualityRendering::SetupMeshFilterPipeline(void)
{

  double minQualityFound;
  double maxQualityFound;

  this->CellSelectionFilter = vtkExtractCells::New();
  this->CreateProcessedMesh();

  long numCells = ((this->SavedMesh->GetCellData())->GetArray("Quality"))->GetNumberOfTuples();
  cout << "found " << numCells << " cells in the dataset" << endl;
  //numCells = ((mesh->GetCellData())->GetArray("ELEMENT_ID"))->GetNumberOfTuples();
  //cout << "found " << numCells << " id entries in the dataset" << endl;

  this->FindMinimumAndMaximumQualityForMesh(this->SavedMesh,&minQualityFound,&maxQualityFound);
  cout << "Min quality found: " << minQualityFound << " max quality found: " ;
  cout << maxQualityFound << endl;

  // look at the threshold value and pass only the elements which have quality
  // measure below the threshold value

  vtkThreshold* thresh = vtkThreshold::New();
    thresh->SetInputConnection(this->VTKQualityFilterPtr->GetOutputPort());
    thresh->ThresholdByLower(this->thresholdValue);
    // this caused all the elements to dissappear when uncommented.  Oh no...
    //thresh->SetAttributeModeToUseCellData();
    this->thresholdFilter = thresh;



  // shrink the elements so we can see each one.  We are using a shrink filter
  // instantiated earlier because we needed a pointer to the filter to invoke its
  // modified method as part of a GUI callback to update the pipeline

  // store a handle to this filter instance so we can restrobe the pipeline
  vtkShrinkFilter* shrinker = (vtkShrinkFilter*) vtkShrinkFilter::New();
  //shrinker->SetInputConnection(this->VTKQualityFilterPtr->GetOutputPort());
  shrinker->SetInputConnection(thresh->GetOutputPort());
  shrinker->SetShrinkFactor(this->ElementShrinkFactor);
  this->shrinkFilter = shrinker;
  //shrinker->Update();

  //vtkLogLookupTable* lut = (vtkLogLookupTable*) vtkLogLookupTable::New();
  vtkLookupTable* lut = (vtkLookupTable*) vtkLookupTable::New();
  //lut->SetHueRange (0.0, 0.8);
  lut->SetAlphaRange(this->ElementOpacity,this->ElementOpacity);
  lut->SetTableRange(minQualityFound, maxQualityFound);
  this->lutFilter = lut;

   // initialize the lookup table to run through the spectrum where blue (cool)
   // is assigned to the highest values (of good quality) and red is assigned to
   // the lowest values (the elements with bad quality).  We have to go through the
   // LUT because this red-to-blue progression is opposite the usual initialization of a
   // lookup table.

   // CRL *** 2/22/07 - disabled the LUT overwrite on the colors.  Let the default lookup
   // table calculate colors.  We are rescaling the table dynamically now, so this is OK as
   // long as the quality function values increase as "quality" increases.

//   for (int i=0; i<255; i++)
//   {
//        double si = i/256.0;
//        double decreasing_si = 1.0 - si;
//        double alpha = this->ElementOpacity;
//
//        // assign a color according to the place in the spectrum.
//        // Use the index and input parameters to calculate the colors
//
//        double r,g,b;
//        vtkMath::HSVToRGB(decreasing_si,1.0,1.0,&r,&g,&b);
//        //((vtkLookupTable*)this->lutFilter)->SetTableValue(i, r,g,b,alpha);
//      }
//

  // The scalar mode in the mapper needs to be set to cells here so the cells are uniformly
  // colored with the mesh quality metric.  Without this,the values at the points
  // cause color interpolation across the cell.

  vtkDataSetMapper*  elemMap = vtkDataSetMapper::New();
  elemMap->SetInputConnection(shrinker->GetOutputPort());
  elemMap->SetScalarModeToUseCellData();
  elemMap->SetLookupTable(lut);
  elemMap->SetScalarRange (minQualityFound, maxQualityFound);
  this->SavedElementMapper = elemMap;

  vtkActor* elemAct = vtkActor::New();
  elemAct->SetMapper(elemMap);
  this->SavedElementActor = elemAct;

  // add output PolyData creation for Slicer integration
  vtkGeometryFilter* meshPolyExtractor = vtkGeometryFilter::New();
  meshPolyExtractor->SetInputConnection(shrinker->GetOutputPort());
  this->SavedMeshPolyDataForRendering = meshPolyExtractor->GetOutput();

}


//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::SetupColorLegendActor(void)
{
  // this specialty actor is created from the lookuptable and it draws
  // a labeled legend along the left side of the render window showing the range
  // of quality metric values

  if (this->SavedColorLegendActor == NULL) {
  vtkScalarBarActor *scalarBar = vtkScalarBarActor::New();
  scalarBar->SetLookupTable((this->lutFilter));
  scalarBar->SetTitle("Quality");
  scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  scalarBar->GetPositionCoordinate()->SetValue(0.1,0.05);
  scalarBar->SetOrientationToVertical();
  scalarBar->SetWidth(0.1);
  scalarBar->SetHeight(0.9);
  scalarBar->SetPosition(0.01,0.1);
  // This format is tricky because VTK does some autosizing.  In general, we've had
  // good luck commmenting out this line and letting VTK autosize for us.
  scalarBar->SetLabelFormat("%9.3f");
  scalarBar->GetLabelTextProperty()->SetColor(0,1,1);
  scalarBar->GetTitleTextProperty()->SetColor(0,1,1);
  this->SavedColorLegendActor = scalarBar;
  }
}



//----------------------------------------------------------------
void vtkMimxMeshQualityRendering::UpdatePipeline(void)
{

  double minQualityFound;
  double maxQualityFound;

  //cout << "calling update pipeline" << endl;
  if (this->IsInitialized)
  {

     this->VTKQualityFilterPtr->Modified();
     this->VTKQualityFilterPtr->Update();

    // enable/disable actors according to the user parameters.  *** this might
    // force filter re-evaluation unnecessarily, but it is simpler logic than looking
    // if the parameters have changed.  Since we are working with small meshes initially,
    // lets keep the logic simple

    this->SavedElementActor->SetVisibility(this->ShowFilledElements);
    this->SavedOutlineActor->SetVisibility(this->ShowOutline);
    this->SavedClippedOutlineActor->SetVisibility(this->ShowClippedOutline);
    this->SavedInteriorOutlinesActor->SetVisibility(this->ShowInteriorOutlines);
    this->SavedSurfaceHedgehogActor->SetVisibility(this->ShowSurfaceHedgehog);

    // set the VTK supporting class to have the right test and check the range
    // of values to recalibrate the LUT when the test changes
    //this->InvokeQualityMeasure();
    //this->FindMinimumAndMaximumQualityForMesh(this->SavedMesh,&minQualityFound,&maxQualityFound);
    this->FindMinimumAndMaximumQualityForMesh(
                             (vtkUnstructuredGrid*)(this->VTKQualityFilterPtr->GetOutput()),
                             &minQualityFound,
                             &maxQualityFound);

    ((vtkUnstructuredGrid*)(this->VTKQualityFilterPtr->GetOutput()))->GetCellData()->SetActiveAttribute("Quality",vtkDataSetAttributes::SCALARS);

    if((this->VTKQualityFilterPtr->GetOutput())->GetCellData()->GetArray("Quality"))
        ((vtkUnstructuredGrid*)(this->VTKQualityFilterPtr->GetOutput()))->GetCellData()->GetArray("Quality")->SetName("scalars");

    // try to copy the scalars to  a duplicate array named "scalars" so it will be picked up by Slicer
//    vtkDoubleArray *attrCopy = vtkDoubleArray::New();
//    attrCopy->SetNumberOfComponents(1);
//    attrCopy->Allocate(VTK_DOUBLE,100);
//    attrCopy->DeepCopy((this->VTKQualityFilterPtr->GetOutput())->GetCellData()->GetArray("Jacobian"));
//    attrCopy->SetName("scalars");
//    ((vtkUnstructuredGrid*)(this->VTKQualityFilterPtr->GetOutput()))->GetCellData()->AddArray(attrCopy);

    // autoadjust the colors if the user is not manually overriding this feature
    if (!this->OverrideMeshColorRange)
    {
        this->lutFilter->SetTableRange(minQualityFound, maxQualityFound);
        this->lutFilter->ForceBuild();
        this->SavedElementMapper->SetScalarRange (minQualityFound, maxQualityFound);

    }
     ((vtkScalarBarActor*)this->SavedColorLegendActor)->SetLookupTable(this->lutFilter);
    this->SavedColorLegendActor->Modified();

        // set color to blue for binary quality tests where no elements were flagged.
        // Otherwise, let the LUT do the coloring
        if ( minQualityFound == maxQualityFound) {
                this->SavedElementActor->GetProperty()->SetColor(0,0,1);
                this->SavedElementMapper->ScalarVisibilityOff();
                this->SavedElementActor->Modified();
        } else {
                this->SavedElementMapper->ScalarVisibilityOn();
                this->SavedElementActor->Modified();

        }

    this->shrinkFilter->SetShrinkFactor(this->ElementShrinkFactor);
    //this->shrinkFilter->Update();
    //cout << "shrink value is: " << this->shrinkFilter->GetShrinkFactor() << endl;

    // the user could have changed the opacity of the elements
    this->SavedElementActor->GetProperty()->SetOpacity(this->ElementOpacity);

        // update the threshold value in the filter since the user could have tweaked the
        // interface.  The slider on the GUI is for percent of rejection of elements, so we
        // need to project the threshold percent into an actual element value

        double threshActual = this->thresholdValue*(maxQualityFound-minQualityFound)+minQualityFound;
        this->thresholdFilter->ThresholdByLower(threshActual);
  }

}


//----------------------------------------------------------------
vtkPolyData*  vtkMimxMeshQualityRendering::GetMeshPolygons(void)
{
        // update may not be necessary, but it won't hurt us
        this->UpdatePipeline();
    return this->SavedMeshPolyDataForRendering;
}



vtkPolyData*  vtkMimxMeshQualityRendering::GetOutlinePolygons(void)
{
        this->UpdatePipeline();
        return SavedOutlinePolyDataForRendering;
}


vtkPolyData*  vtkMimxMeshQualityRendering::GetHedgehogPolygons(void)
{
        this->UpdatePipeline();
    return SavedHedgehogPolyDataForRendering;
}
