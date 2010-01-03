/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementMeshQualityDisplayNode.h,v $
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

#ifndef __vtkMRMLFiniteElementMeshQualityDisplayNode_h
#define __vtkMRMLFiniteElementMeshQualityDisplayNode_h

#include <string>

#include "vtkUnstructuredGrid.h"
#include "vtkShrinkPolyData.h"
#include "vtkGeometryFilter.h"

#include "vtkMRML.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLUnstructuredGridNode.h"
#include "vtkMRMLUnstructuredGridDisplayNode.h"
#include "vtkMimxCommonWin32Header.h"

//class vtkMeshQualityExtended; 
class vtkMeshQuality; 
class vtkShrinkFilter;

class VTK_MIMXCOMMON_EXPORT vtkMRMLFiniteElementMeshQualityDisplayNode : public vtkMRMLUnstructuredGridDisplayNode
{
 public:
  static vtkMRMLFiniteElementMeshQualityDisplayNode *New (  );
  vtkTypeMacro ( vtkMRMLFiniteElementMeshQualityDisplayNode,vtkMRMLUnstructuredGridDisplayNode );
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
  virtual const char* GetNodeTagName ( ) {return "FiniteElementBoundingBoxDisplay";};

  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                   unsigned long /*event*/, 
                                   void * /*callData*/ );
 
   // overload the virtual placeholder in the parent class.  This one will setup
   // the beginning of the actual pipeline for rendering an FE Mesh instead
   virtual void SetUnstructuredGrid(vtkUnstructuredGrid *grid);
 
   // declare a rendering pipeline for bblock data in this class
   virtual vtkPolyData* GetPolyData();
     
    
  // Description:
  // Update the pipeline based on this node attributes
  virtual void UpdatePolyDataPipeline();
 
  //--------------------------------------------------------------------------
  // Display Information: Geometry to display (not mutually exclusive)
  //--------------------------------------------------------------------------

  // Description:
 
   //vtkMeshQualityExtended *SavedMeshQualityFilter; 
   vtkMeshQuality *SavedMeshQualityFilter; 
   vtkShrinkFilter *SavedShrinkFilter;
  
 protected:
     vtkMRMLFiniteElementMeshQualityDisplayNode ( );
  ~vtkMRMLFiniteElementMeshQualityDisplayNode ( );
  vtkMRMLFiniteElementMeshQualityDisplayNode ( const vtkMRMLFiniteElementMeshQualityDisplayNode& );
  void operator= ( const vtkMRMLFiniteElementMeshQualityDisplayNode& );

 

  // dispaly pipeline components declared here

};

#endif
