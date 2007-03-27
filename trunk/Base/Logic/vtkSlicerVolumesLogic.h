/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerVolumesLogic.h,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

=========================================================================auto=*/

// .NAME vtkSlicerVolumesLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerVolumesLogic_h
#define __vtkSlicerVolumesLogic_h

#include <stdlib.h>

#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"

#include "vtkMRML.h"
#include "vtkMRMLVolumeNode.h"

class vtkMRMLScalarVolumeNode;
class vtkMRMLVolumeHeaderlessStorageNode;

class VTK_SLICER_BASE_LOGIC_EXPORT vtkSlicerVolumesLogic : public vtkSlicerLogic 
{
  public:
  
  // The Usual vtk class functions
  static vtkSlicerVolumesLogic *New();
  vtkTypeRevisionMacro(vtkSlicerVolumesLogic,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // The currently active mrml volume node 
  vtkGetObjectMacro (ActiveVolumeNode, vtkMRMLVolumeNode);
  void SetActiveVolumeNode (vtkMRMLVolumeNode *ActiveVolumeNode);

  // Description:
  // Create new mrml node and associated storage node.
  // Read image data from a specified file
  vtkMRMLVolumeNode* AddArchetypeVolume (const char* filename, int centerImage, int labelMap, const char* volname);

  // Description:
  // Create new mrml node and associated storage node.
  // Read image data from a specified file
  vtkMRMLVolumeNode* AddHeaderVolume (char* filename, int centerImage, int labelMap, const char* volname, 
                                      vtkMRMLVolumeHeaderlessStorageNode *headerStorage);

  // Description:
  // Write volume's image data to a specified file
  int SaveArchetypeVolume (const char* filename, vtkMRMLVolumeNode *volumeNode);

  // Description:
  // Create a label map volume to match the given volume node and add it to
  // the scene
  vtkMRMLScalarVolumeNode *CreateLabelVolume (vtkMRMLScene *scene, vtkMRMLVolumeNode *volumeNode, char *name);

  // Description:
  // Update MRML events
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                  unsigned long /*event*/, 
                                  void * /*callData*/ );    
  // Description:
  // Update logic events
  virtual void ProcessLogicEvents ( vtkObject * /*caller*/, 
                                  unsigned long /*event*/, 
                                  void * /*callData*/ );  

protected:
  vtkSlicerVolumesLogic();
  virtual ~vtkSlicerVolumesLogic();
  vtkSlicerVolumesLogic(const vtkSlicerVolumesLogic&);
  void operator=(const vtkSlicerVolumesLogic&);

  // Description:
  // Examine the file name to see if the extension is one of the supported
  // freesurfer volume formats. Used to assign the proper colour node to label
  // maps.
  int IsFreeSurferVolume (const char* filename);
  
  // Description:
  //
  vtkMRMLVolumeNode *ActiveVolumeNode;
};

#endif

