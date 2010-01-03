/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiniteElementMeshDisplayNode.h,v $
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

#ifndef __vtkMRMLFiniteElementMeshDisplayNode_h
#define __vtkMRMLFiniteElementMeshDisplayNode_h

#include <string>

#include "vtkUnstructuredGrid.h"
#include "vtkShrinkPolyData.h"
#include "vtkGeometryFilter.h"
#include "vtkPlane.h"

#include "vtkMRML.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLUnstructuredGridNode.h"
#include "vtkMRMLUnstructuredGridDisplayNode.h"
#include "vtkMimxCommonWin32Header.h"

#include "vtkMimxMeshQualityRendering.h"


class VTK_MIMXCOMMON_EXPORT vtkMRMLFiniteElementMeshDisplayNode : public vtkMRMLUnstructuredGridDisplayNode
{
 public:
  static vtkMRMLFiniteElementMeshDisplayNode *New (  );
  vtkTypeMacro ( vtkMRMLFiniteElementMeshDisplayNode,vtkMRMLUnstructuredGridDisplayNode );
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
  virtual const char* GetNodeTagName ( ) {return "FiniteElementMeshDisplay";};

  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/,
                                   unsigned long /*event*/,
                                   void * /*callData*/ );

   // overload the virtual placeholder in the parent class.  This one will setup
   // the beginning of the actual pipeline for rendering an FE Mesh instead
   virtual void SetUnstructuredGrid(vtkUnstructuredGrid *grid);

   //--------------------------------------------------------------------------
   // Display Information: Geometry to display (not mutually exclusive)
   //--------------------------------------------------------------------------


   // declare a rendering pipeline for bblock data in this class
   virtual vtkPolyData* GetPolyData();

  // Description:
  // Update the pipeline based on this node attributes
  virtual void UpdatePolyDataPipeline();

  // The mesh can be "cut" using a cutting plane. The instance of an implicit function (i.e. vtkPlane)
  // needs to be passed here to control the rendering.  If the cutting plane is enabled, then the value
  // of this implicit function is checked to determine which nodes are rendered
 void SetCuttingPlane(vtkPlane *plane);

 // The cutting plane can be enabled and dispabled during run-time.  Handle this or pass down to the rendering pipeline
 void EnableCuttingPlane(void){if (this->SavedMeshQualityRendering) this->SavedMeshQualityRendering->EnableCuttingPlane();}
 void DisableCuttingPlane(void){if (this->SavedMeshQualityRendering) this->SavedMeshQualityRendering->DisableCuttingPlane();}

 void SetQualityToJacobian(void) {if (this->SavedMeshQualityRendering) this->SavedMeshQualityRendering->SetQualityMeasure(2);}

 // The elements inside the mesh can be sized 0.0 to 1.0, which causes them to be rendered at from 0% to
 // 100% of their native size.  The default is 1.0.
 void SetElementSize(double shrink);

 // specify normalized value (0 to 1) to control what percentage of elements are rendered, according
 // to metric value
 void SetThreshold(double value) {if (this->SavedMeshQualityRendering) this->SavedMeshQualityRendering->SetThresholdValue(value);}

 protected:
     vtkMRMLFiniteElementMeshDisplayNode ( );
  ~vtkMRMLFiniteElementMeshDisplayNode ( );
  vtkMRMLFiniteElementMeshDisplayNode ( const vtkMRMLFiniteElementMeshDisplayNode& );
  void operator= ( const vtkMRMLFiniteElementMeshDisplayNode& );

  // display pipeline components declared here
  vtkMimxMeshQualityRendering* SavedMeshQualityRendering;
  vtkPlane* SavedCuttingPlane;

};

#endif
