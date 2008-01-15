/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementBoundingBoxDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

  =========================================================================auto=*/
// .NAME vtkMRMLUnstructuredGridDisplayNode - MRML node to represent display properties for tractography.
// .SECTION Description
// vtkMRMLUnstructuredGridDisplayNode nodes store display properties of trajectories 
// from tractography in diffusion MRI data, including color type (by bundle, by fiber, 
// or by scalar invariants), display on/off for tensor glyphs and display of 
// trajectory as a line or tube.
//

#ifndef __vtkMRMLFiniteElementBoundingBoxDisplayNode_h
#define __vtkMRMLFiniteElementBoundingBoxDisplayNode_h

#include <string>

#include "vtkUnstructuredGrid.h"
#include "vtkShrinkPolyData.h"
#include "vtkGeometryFilter.h"

#include "vtkMRML.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLUnstructuredGridNode.h"


class VTK_MRML_EXPORT vtkMRMLFiniteElementBoundingBoxDisplayNode : public vtkMRMLUnstructuredGridDisplayNode
{
 public:
  static vtkMRMLFiniteElementBoundingBoxDisplayNode *New (  );
  vtkTypeMacro ( vtkMRMLFiniteElementBoundingBoxDisplayNode,vtkMRMLUnstructuredGridDisplayNode );
  void PrintSelf ( ostream& os, vtkIndent indent );
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  virtual vtkMRMLNode* CreateNodeInstance (  );

  // Description:
  // Read node attributes from XML (MRML) file
  virtual void ReadXMLAttributes ( const char** atts );

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML ( ostream& of, int indent );


  // Description:
  // Copy the node's attributes to this object
  virtual void Copy ( vtkMRMLNode *node );
  
  // Description:
  // Get node XML tag name (like Volume, UnstructuredGrid)
  virtual const char* GetNodeTagName ( ) {return "UnstructuredGridDisplay";};

  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                   unsigned long /*event*/, 
                                   void * /*callData*/ );
 

 
   
  // Description:
  // Update the pipeline based on this node attributes
  virtual void UpdatePolyDataPipeline() 
    {
    this->ShrinkPolyData->SetShrinkFactor(this->ShrinkFactor);
    };
 
  //--------------------------------------------------------------------------
  // Display Information: Geometry to display (not mutually exclusive)
  //--------------------------------------------------------------------------

  // Description:
 
    
 protected:
     vtkMRMLFiniteElementBoundingBoxDisplayNode ( );
  ~vtkMRMLFiniteElementBoundingBoxDisplayNode ( );
  vtkMRMLFiniteElementBoundingBoxDisplayNode ( const vtkMRMLFiniteElementBoundingBoxDisplayNode& );
  void operator= ( const vtkMRMLFiniteElementBoundingBoxDisplayNode& );

 

  // dispaly pipeline components declared here

};

#endif
