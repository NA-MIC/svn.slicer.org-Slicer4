/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiberBundleGlyphDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

  =========================================================================auto=*/
// .NAME vtkMRMLGlyphVolumeSliceDisplayNode - MRML node to represent display properties for tractography.
// .SECTION Description
// vtkMRMLGlyphVolumeSliceDisplayNode nodes store display properties of trajectories 
// from tractography in diffusion MRI data, including color type (by bundle, by fiber, 
// or by scalar invariants), display on/off for tensor glyphs and display of 
// trajectory as a line or tube.
//

#ifndef __vtkMRMLGlyphVolumeSliceDisplayNode_h
#define __vtkMRMLGlyphVolumeSliceDisplayNode_h

#include <string>
#include "vtkMRMLDisplayNode.h"

class vtkTransform;
class vtkTransformPolyDataFilter;
class vtkMatrix4x4;
class vtkPolyData;

class VTK_MRML_EXPORT vtkMRMLGlyphVolumeSliceDisplayNode : public vtkMRMLDisplayNode
{
 public:
  static vtkMRMLGlyphVolumeSliceDisplayNode *New (  ) { return NULL; }
  vtkTypeMacro ( vtkMRMLGlyphVolumeSliceDisplayNode,vtkMRMLDisplayNode );
  void PrintSelf ( ostream& os, vtkIndent indent );
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

//  virtual vtkMRMLNode* CreateNodeInstance (  );

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
  virtual const char* GetNodeTagName ( ) {return "GlyphVolumeSliceDisplayNode";};

  // Description:
  // Gets resulting glyph PolyData 
  virtual vtkPolyData* GetPolyData(){ return NULL; };
  
  // Description:
  // Gets resulting glyph PolyData transfomed to slice XY
  virtual vtkPolyData* GetPolyDataTransformedToSlice() { return NULL; }
   
  // Description:
  // Update the pipeline based on this node attributes
  virtual void UpdatePolyDataPipeline() = 0;

  // Description:
  // Set ImageData for a volume slice
  virtual void SetSliceImage(vtkImageData *image) =0 ;
 
  // Description:
  // Set slice to RAS transformation
  virtual void SetSlicePositionMatrix(vtkMatrix4x4 *matrix);

  // Description:
  // Set slice to IJK transformation
  virtual void SetSliceGlyphRotationMatrix(vtkMatrix4x4 *matrix) = 0;

  //--------------------------------------------------------------------------
  // Display Information: Geometry to display (not mutually exclusive)
  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------
  // Display Information: Color Mode
  // 0) solid color by group 1) color by scalar invariant 
  // 2) color by avg scalar invariant 3) color by other
  //--------------------------------------------------------------------------

  //BTX
  enum
  {
    colorModeSolid = 0,
    colorModeScalar = 1,
    colorModeFunctionOfScalar = 2,
    colorModeUseCellScalars = 3
  };
  //ETX

  //--------------------------------------------------------------------------
  // Display Information: ColorMode for ALL nodes
  //--------------------------------------------------------------------------

 // Description:
  // Color mode for glyphs. The color modes are mutually exclusive.
  vtkGetMacro ( ColorMode, int );
  vtkSetMacro ( ColorMode, int );
 
  // Description:
  // Color by solid color (for example the whole fiber bundle red. blue, etc.)
  void SetColorModeToSolid ( ) {
    this->SetColorMode ( this->colorModeSolid );
  };

  // Description:
  // Color according to the tensors using various scalar invariants.
  void SetColorModeToScalar ( ) {
    this->SetColorMode ( this->colorModeScalar );
  };

  // Description:
  // Color according to the tensors using a function of scalar invariants along the tract.
  // This enables coloring by average FA, for example.
  void SetColorModeToFunctionOfScalar ( ) {
    this->SetColorMode ( this->colorModeFunctionOfScalar );
  };

  // Description:
  // Use to color by the active cell scalars.  This is intended to support
  // external processing of fibers, for example to label each with the distance
  // of that fiber from an fMRI activation.  Then by making that information
  // the active cell scalar field, this will allow coloring by that information.
  // TO DO: make sure this information can be saved with the tract, save name of
  // active scalar field if needed.
  void SetColorModeToUseCellScalars ( ) {
    this->SetColorMode ( this->colorModeUseCellScalars );
  };


 protected:
  vtkMRMLGlyphVolumeSliceDisplayNode ( );
  ~vtkMRMLGlyphVolumeSliceDisplayNode ( );
  vtkMRMLGlyphVolumeSliceDisplayNode ( const vtkMRMLGlyphVolumeSliceDisplayNode& );
  void operator= ( const vtkMRMLGlyphVolumeSliceDisplayNode& );


  // Enumerated
  int ColorMode;

  vtkTransform                  *SliceToXYTransform;
  vtkTransformPolyDataFilter    *SliceToXYTransformer;
  vtkMatrix4x4                  *SliceToXYMatrix;
};

#endif
