/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkCellWallSegmentLogic.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
#ifndef __vtkCellWallSegmentLogic_h
#define __vtkCellWallSegmentLogic_h

#include "vtkSlicerModuleLogic.h"
#include "vtkMRMLScene.h"

#include "vtkCellWallSegment.h"
#include "vtkMRMLCellWallSegmentNode.h"

class vtkCellWallVisSeg;
class vtkImageReader2;


class VTK_CellWallSegment_EXPORT vtkCellWallSegmentLogic : public vtkSlicerModuleLogic
{
  public:
  static vtkCellWallSegmentLogic *New();
  vtkTypeMacro(vtkCellWallSegmentLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents ( vtkObject *caller, unsigned long event,
                                   void *callData ){};

  // Description: Get/Set MRML node storing parameter values
  vtkGetObjectMacro (CellWallSegmentNode, vtkMRMLCellWallSegmentNode);
  void SetAndObserveCellWallSegmentNode(vtkMRMLCellWallSegmentNode *n) 
    {
    vtkSetAndObserveMRMLNodeMacro( this->CellWallSegmentNode, n);
    }

  // The methods that create and run VTK or ITK pipelines.  The MRML volume 
  // is loaded, then 2D or 3D segmentation can be performed after a fiducial
  // list is selected.  The segmentation is written into the SegmentationVolume
  
  void InitializeMRMLVolume(char* filename);
  void InitializeMRMLSegmentationVolume();
  void Perform2DSegmentation();
  void Perform3DSegmentation();
  void PaintIntoMRMLSegmentationVolume(int CellID);
  
  vtkCellWallVisSeg* GetCellWallVisSeg(void) { return this->VisSegInstance; }
  
protected:
  vtkCellWallSegmentLogic();
  virtual ~vtkCellWallSegmentLogic();
  vtkCellWallSegmentLogic(const vtkCellWallSegmentLogic&);
  void operator=(const vtkCellWallSegmentLogic&);

  vtkMRMLCellWallSegmentNode* CellWallSegmentNode;
  vtkCellWallVisSeg* VisSegInstance;
  vtkImageReader2* Reader;


};

#endif

