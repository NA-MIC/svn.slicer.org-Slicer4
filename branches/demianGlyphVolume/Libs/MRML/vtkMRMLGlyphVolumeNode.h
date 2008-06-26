/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLVolumeNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.13 $

=========================================================================auto=*/
// .NAME vtkMRMLGlyphVolumeNode - MRML node for representing diffusion weighted MRI volume 
// .SECTION Description
// Diffusion Weigthed Volume nodes describe data sets that encode diffusion weigthed
// images. These images are the basis for computing the diffusion tensor.
// The node is a container for the neccesary information to interpert DW images:
// 1. Gradient information.
// 2. B value for each gradient.
// 3. Measurement frame that relates the coordinate system where the gradients are given 
//  to RAS.

#ifndef __vtkMRMLGlyphVolumeNode_h
#define __vtkMRMLGlyphVolumeNode_h


#include "vtkMRMLScalarVolumeNode.h"

class vtkDoubleArray;
class vtkMRMLGlyphVolumeDisplayNode;
class vtkMRMLGlyphVolumeSliceDisplayNode;


class VTK_MRML_EXPORT vtkMRMLGlyphVolumeNode : public vtkMRMLScalarVolumeNode
{
  public:
  static vtkMRMLGlyphVolumeNode *New();
  vtkTypeMacro(vtkMRMLGlyphVolumeNode,vtkMRMLScalarVolumeNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Set node attributes
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "GlyphVolume";};

  // Description:
  // String ID of the display MRML node
  vtkSetReferenceStringMacro(MaskNodeID);
  vtkGetStringMacro(MaskNodeID);

  // Description:
  // Associated volume MRML node
  vtkMRMLVolumeNode* GetMaskNode();

  // Description:
  // Associated volume MRML node
  //vtkMRMLGlyphVolumeDisplayNode* GetDisplayNode();

  // Description:
  // Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

   // Description:
  // Finds the storage node and read the data
  //void UpdateScene(vtkMRMLScene *scene);

  // Description:
  // Updates this node if it depends on other nodes 
  // when the node is deleted in the scene
  void UpdateReferences();

  
  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                   unsigned long /*event*/, 
                                   void * /*callData*/ );

//BTX
  // Description:
  // get associated slice glyph display node or NULL if not set
  virtual std::vector< vtkMRMLGlyphVolumeSliceDisplayNode* > GetSliceGlyphDisplayNodes();


  // Description:
  // add slice glyph display nodes if not already present and return it
  virtual std::vector< vtkMRMLGlyphVolumeSliceDisplayNode* >  AddSliceGlyphDisplayNodes(){ return std::vector< vtkMRMLGlyphVolumeSliceDisplayNode* >(); }
//ETX


protected:
  vtkMRMLGlyphVolumeNode();
  ~vtkMRMLGlyphVolumeNode();
  vtkMRMLGlyphVolumeNode(const vtkMRMLGlyphVolumeNode&);
  void operator=(const vtkMRMLGlyphVolumeNode&);
  
  char *MaskNodeID;
};

#endif
