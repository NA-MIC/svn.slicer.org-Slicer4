/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMeshQualityExtended.h,v $
  Language:  C++
  Date:      $Date: 2008/02/11 00:51:54 $ 
  Version:   $Revision: 1.1 $

  Copyright 2003-2006 Sandia Corporation.
  Copyright 2007 KnowledgeVis, LLC

=========================================================================*/
// .NAME vtkMeshQualityExtended - Calculate measures of quality of a mesh
//
// .SECTION Description
// vtkMeshQualityExtended extends the metrics provided by vtkMeshQuality
// to also include an edge collaspe and angle out of bound metrics for 
// Hexahedron cells. 
//
// .SECTION Caveats
// 

#ifndef __vtkMeshQualityExtended_h
#define __vtkMeshQualityExtended_h

#include "vtkMeshQuality.h"

class vtkCell;
class vtkDataArray;

/* Added Mesh Quality Metrics by Curt Lisle */
#define VTK_QUALITY_EDGE_COLLAPSE 30
#define VTK_QUALITY_ANGLE_OUT_OF_BOUNDS 31

class VTK_GRAPHICS_EXPORT vtkMeshQualityExtended : public vtkMeshQuality
{
public:
  void PrintSelf(ostream& os, vtkIndent indent);
  vtkTypeRevisionMacro(vtkMeshQualityExtended,vtkMeshQuality);
  static vtkMeshQualityExtended* New();

  
  // Description:
  // Set/Get the particular estimator used to measure the quality of hexahedra.
  // Two aditional metrics are now provided:
  //   VTK_QUALITY_EDGE_COLLAPSE and VTK_QUALITY_ANGLE_OUT_OF_BOUNDS

  void SetHexQualityMeasureToEdgeCollapse()
    {
    this->SetHexQualityMeasure( VTK_QUALITY_EDGE_COLLAPSE );
    }
  void SetHexQualityMeasureToAngleOutOfBounds()
    {
    this->SetHexQualityMeasure( VTK_QUALITY_ANGLE_OUT_OF_BOUNDS );
    }

   
  // Description:
  // This is a static function used to calculate if any of the edges of the hex have collapsed
  // to zero length.  This returns a 1 if collapse occured and a 0 otherwise -- no type checking is
  // performed because this method is called from the inner loop of the Execute()
  // member function.
  // HISTORY:  Added for Iowa FE project April 2007
  
  static double HexEdgeCollapse( vtkCell* cell );
  static double HexAngleOutOfBounds(vtkCell* cell);


protected:
  vtkMeshQualityExtended();
  ~vtkMeshQualityExtended();

  virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
  vtkMeshQualityExtended( const vtkMeshQualityExtended& ); // Not implemented.
  void operator = ( const vtkMeshQualityExtended& ); // Not implemented.
};

#endif // vtkMeshQualityExtended_h
