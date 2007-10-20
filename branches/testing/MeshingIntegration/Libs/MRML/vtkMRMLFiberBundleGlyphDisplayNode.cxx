/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLFiberBundleGlyphDisplayNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkDiffusionTensorGlyph.h"

#include "vtkMRMLFiberBundleGlyphDisplayNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLFiberBundleGlyphDisplayNode* vtkMRMLFiberBundleGlyphDisplayNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiberBundleGlyphDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiberBundleGlyphDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiberBundleGlyphDisplayNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLFiberBundleGlyphDisplayNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLFiberBundleGlyphDisplayNode");
  if(ret)
    {
    return (vtkMRMLFiberBundleGlyphDisplayNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLFiberBundleGlyphDisplayNode;
}


//----------------------------------------------------------------------------
vtkMRMLFiberBundleGlyphDisplayNode::vtkMRMLFiberBundleGlyphDisplayNode()
{
  this->DiffusionTensorGlyphFilter = vtkDiffusionTensorGlyph::New();


  this->TwoDimensionalVisibility = 0;
  this->ColorMode = vtkMRMLFiberBundleDisplayNode::colorModeScalar;
}


//----------------------------------------------------------------------------
vtkMRMLFiberBundleGlyphDisplayNode::~vtkMRMLFiberBundleGlyphDisplayNode()
{
  this->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
  this->DiffusionTensorGlyphFilter->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLFiberBundleGlyphDisplayNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);
  of << indent << " twoDimensionalVisibility=\"" << (this->TwoDimensionalVisibility ? "true" : "false") << "\"";
}



//----------------------------------------------------------------------------
void vtkMRMLFiberBundleGlyphDisplayNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "twoDimensionalVisibility")) 
      {
      if (!strcmp(attValue,"true")) 
        {
        this->TwoDimensionalVisibility  = 1;
        }
      else
        {
        this->TwoDimensionalVisibility = 0;
        }
      }
    }  
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLFiberBundleGlyphDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLFiberBundleGlyphDisplayNode *node = (vtkMRMLFiberBundleGlyphDisplayNode *) anode;

  this->SetTwoDimensionalVisibility(node->TwoDimensionalVisibility);
}

//----------------------------------------------------------------------------
void vtkMRMLFiberBundleGlyphDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  //int idx;
  
  Superclass::PrintSelf(os,indent);
  os << indent << "TwoDimensionalVisibility:             " << this->TwoDimensionalVisibility << "\n";
}


//----------------------------------------------------------------------------
void vtkMRMLFiberBundleGlyphDisplayNode::SetPolyData(vtkPolyData *glyphPolyData)
{
  if (this->DiffusionTensorGlyphFilter)
    {
    this->DiffusionTensorGlyphFilter->SetInput(glyphPolyData);
    }
}

//----------------------------------------------------------------------------
vtkPolyData* vtkMRMLFiberBundleGlyphDisplayNode::GetPolyData()
{
  if (this->DiffusionTensorGlyphFilter)
    {
    this->UpdatePolyDataPipeline();
    this->DiffusionTensorGlyphFilter->Update();
    return this->DiffusionTensorGlyphFilter->GetOutput();
    }
  else
    {
    return NULL;
    }
}

//----------------------------------------------------------------------------
void vtkMRMLFiberBundleGlyphDisplayNode::UpdatePolyDataPipeline() 
{
  // set display properties according to the tensor-specific display properties node for glyphs
  vtkMRMLDiffusionTensorDisplayPropertiesNode * DTDisplayNode = this->GetDTDisplayPropertiesNode( );
  
  if (DTDisplayNode != NULL) {
    // TO DO: need filter to calculate FA, average FA, etc. as requested
    
    // get tensors from the fiber bundle node and glyph them
    // TO DO: include superquadrics
    // if glyph type is other than superquadrics, get glyph source
    if (DTDisplayNode->GetGlyphGeometry( ) != vtkMRMLDiffusionTensorDisplayPropertiesNode::Superquadrics) 
      {
      this->DiffusionTensorGlyphFilter->ClampScalingOff();
      
      // TO DO: implement max # ellipsoids, random sampling features
      this->DiffusionTensorGlyphFilter->SetResolution(2);
      
      this->DiffusionTensorGlyphFilter->SetScaleFactor( DTDisplayNode->GetGlyphScaleFactor( ) );
      
      this->DiffusionTensorGlyphFilter->SetSource( DTDisplayNode->GetGlyphSource( ) );
      
      vtkErrorMacro("setting glyph geometry" << DTDisplayNode->GetGlyphGeometry( ) );
      
      // set glyph coloring
      if (this->GetColorMode ( ) == vtkMRMLFiberBundleDisplayNode::colorModeSolid)
        {
        this->ScalarVisibilityOff( );
        }
      else  
        {
        if (this->GetColorMode ( ) == vtkMRMLFiberBundleDisplayNode::colorModeScalar)
          {
          this->ScalarVisibilityOn( );

          switch ( DTDisplayNode->GetColorGlyphBy( ))
            {
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::FractionalAnisotropy:
              {
                vtkErrorMacro("coloring with FA==============================");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByFractionalAnisotropy( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::LinearMeasure:
              {
                vtkErrorMacro("coloring with Cl=============================");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByLinearMeasure( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::Trace:
              {
                vtkErrorMacro("coloring with trace =================");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByTrace( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::ColorOrientation:
              {
                vtkErrorMacro("coloring with direction (re-implement)");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByOrientation( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::PlanarMeasure:
              {
                vtkErrorMacro("coloring with planar");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByPlanarMeasure( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::MaxEigenvalue:
              {
                vtkErrorMacro("coloring with max eigenval");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByMaxEigenvalue( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::MidEigenvalue:
              {
                vtkErrorMacro("coloring with mid eigenval");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByMidEigenvalue( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::MinEigenvalue:
              {
                vtkErrorMacro("coloring with min eigenval");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByMinEigenvalue( );
              }
              break;
            case vtkMRMLDiffusionTensorDisplayPropertiesNode::RelativeAnisotropy:
              {
                vtkErrorMacro("coloring with relative anisotropy");
                this->DiffusionTensorGlyphFilter->ColorGlyphsByRelativeAnisotropy( );
              }
              break;
              
            }
          }  // if color scalar
      }   // end else
      double *range = this->DiffusionTensorGlyphFilter->GetOutput()->GetScalarRange();
      this->ScalarRange[0] = range[0];
      this->ScalarRange[1] = range[1];
      // avoid Set not to cause event loops
      //this->SetScalarRange( this->DiffusionTensorGlyphFilter->GetOutput()->GetScalarRange() );

    }
  }
}
 

