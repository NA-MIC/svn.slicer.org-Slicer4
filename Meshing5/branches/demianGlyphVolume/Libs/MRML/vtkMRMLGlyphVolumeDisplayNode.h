/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLGlyphVolumeDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/
// .NAME vtkMRMLGlyphVolumeDisplayNode - MRML node for representing a volume display attributes
// .SECTION Description
// vtkMRMLGlyphVolumeDisplayNode nodes describe how volume is displayed.

#ifndef __vtkMRMLGlyphVolumeDisplayNode_h
#define __vtkMRMLGlyphVolumeDisplayNode_h

#include "vtkMRML.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLScalarVolumeDisplayNode.h"
#include "vtkMRMLColorNode.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkLookupTable.h"

class vtkImageData;
class vtkMRMLVolumeNode;
class vtkMRMLGlyphVolumeSliceDisplayNode;

class VTK_MRML_EXPORT vtkMRMLGlyphVolumeDisplayNode : public vtkMRMLScalarVolumeDisplayNode
{
  public:
  static vtkMRMLGlyphVolumeDisplayNode *New(){ return NULL; }
  vtkTypeMacro(vtkMRMLGlyphVolumeDisplayNode,vtkMRMLScalarVolumeDisplayNode);
  void PrintSelf(ostream& os, vtkIndent indent);

//  virtual vtkMRMLNode* CreateNodeInstance();

  // Description:
  // Read node attributes from XML file
  virtual void ReadXMLAttributes( const char** atts);

  // Description:
  // Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  // Description:
  // Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);

  // Description:
  // Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "GlyphVolumeDisplay";};

  // Description:
  // Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

  //--------------------------------------------------------------------------
  // Display Information
  //--------------------------------------------------------------------------

   // Description:
  // Set/Get visualization Mode
  //BTX
  enum 
    {
    visModeScalar = 0,
    visModeGlyph = 1,
    visModeBoth = 2,
    };
  //ETX 
  vtkGetMacro(VisualizationMode, int);
  vtkSetMacro(VisualizationMode, int);

  virtual vtkPolyData* ExecuteGlyphPipeLineAndGetPolyData( vtkImageData* );

  void SetVisualizationModeToScalarVolume() {
    this->SetVisualizationMode(this->visModeScalar);
  };  
  void SetVisualizationModeToGlyphs() {
    this->SetVisualizationMode(this->visModeGlyph);
  };  
  void SetVisualizationModeToBoth() {
    this->SetVisualizationMode(this->visModeBoth);
  };

  // Description:
  // Updates this node if it depends on other nodes 
  // when the node is deleted in the scene
  virtual void UpdateReferences();

  // Description:
  // Finds the storage node and read the data
  virtual void UpdateScene(vtkMRMLScene *scene);
  
  // Description:
  // String ID of the color MRML node
  void SetAndObserveGlyphColorNodeID(const char *GlyphColorNodeID);
  //BTX
  void SetAndObserveGlyphColorNodeID(std::string GlyphColorNodeID);
  //ETX
  vtkGetStringMacro(GlyphColorNodeID);

  // Description:
  // Get associated color MRML node
  vtkMRMLColorNode* GetGlyphColorNode();

  // Description:
  // alternative method to propagate events generated in Display nodes
  virtual void ProcessMRMLEvents ( vtkObject * /*caller*/, 
                                   unsigned long /*event*/, 
                                   void * /*callData*/ );
  // Description:
  // set gray colormap
  void SetDefaultColorMap(int isLabelMap);
 
  virtual void SetImageData(vtkImageData *imageData)
    {
    Superclass::SetImageData(imageData);
    };

  virtual void UpdateImageDataPipeline()
    {
    Superclass::UpdateImageDataPipeline();
    };

  // Description:
  // String ID of the display MRML node
  vtkSetReferenceStringMacro(MaskNodeID);
  vtkGetStringMacro(MaskNodeID);

  //The new slice glyph display node has to have the same scene as the parent one 
  virtual vtkMRMLGlyphVolumeSliceDisplayNode* GetNewGlyphVolumeSliceDisplayNode() = 0;

protected:
  vtkMRMLGlyphVolumeDisplayNode();
  ~vtkMRMLGlyphVolumeDisplayNode();
  vtkMRMLGlyphVolumeDisplayNode(const vtkMRMLGlyphVolumeDisplayNode&);
  void operator=(const vtkMRMLGlyphVolumeDisplayNode&);

  char *GlyphColorNodeID;

  vtkSetReferenceStringMacro(GlyphColorNodeID);

  char *MaskNodeID;

  vtkMRMLColorNode *GlyphColorNode;

  int VisualizationMode;

};

#endif

