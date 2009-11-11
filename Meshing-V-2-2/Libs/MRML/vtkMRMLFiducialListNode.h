/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLFiducialListNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

=========================================================================auto=*/
// .NAME vtkMRMLFiducialListNode - MRML node to represent a 3D surface model.
// .SECTION Description
// Model nodes describe polygonal data.  They indicate where the model is 
// stored on disk, and how to render it (color, opacity, etc).  Models 
// are assumed to have been constructed with the orientation and voxel 
// dimensions of the original segmented volume.

#ifndef __vtkMRMLFiducialListNode_h
#define __vtkMRMLFiducialListNode_h

#include <string>

#include "vtkMatrix4x4.h"
#include "vtkCollection.h"

#include "vtkMRML.h"
#include "vtkMRMLNode.h"
#include "vtkMRMLFiducial.h"
#include "vtkMRMLStorableNode.h"
#include "vtkMRMLFiducialListStorageNode.h"

class VTK_MRML_EXPORT vtkMRMLFiducialListNode : public vtkMRMLStorableNode
{
public:
  static vtkMRMLFiducialListNode *New();
  vtkTypeMacro(vtkMRMLFiducialListNode,vtkMRMLStorableNode);
  void PrintSelf(ostream& os, vtkIndent indent);
  
  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

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
  virtual const char* GetNodeTagName() {return "FiducialList";};

  // Description:
  // 
  virtual void UpdateScene(vtkMRMLScene *scene);

  // Description:
  // update display node ids
  void UpdateReferences();
  
  // Description:
  // Get/Set for Symbol scale
//  vtkSetMacro(SymbolScale,double);
  void SetSymbolScale(double scale);
  vtkGetMacro(SymbolScale,double);


  // Description:
  // Get/Set for list visibility 
  //vtkSetMacro(Visibility,int);
  void SetVisibility(int visible);
  vtkGetMacro(Visibility,int);

  // Description:
  // Get/Set for Text scale
  //vtkSetMacro(TextScale,double);
  void SetTextScale(double scale);
  vtkGetMacro(TextScale,double);
  
  // Description:
  // Get/Set for Glyph and Text color
  //vtkSetVector3Macro(Color,double);
  void SetColor(double r, double g, double b);
  void SetColor(double c[3]);
  vtkGetVectorMacro(Color,double,3);

  // Description:
  // Get/Set for colour for when a fiducial is selected
  void SetSelectedColor(double r, double g, double b);
  void SetSelectedColor(double c[3]);
  vtkGetVectorMacro(SelectedColor,double,3);


  // Description:
  // Get the number of fiducials in the list
  int GetNumberOfFiducials();
  
  // Description:
  // Restrict access to the fiducial points, pass in a value via the list
  // so that the appropriate events can be invoked. Returns 0 on success
  int SetNthFiducialXYZ(int n, float x, float y, float z);
  int SetNthFiducialOrientation(int n, float w, float x, float y, float z);
  int SetNthFiducialLabelText(int n, const char *text);
  int SetNthFiducialSelected(int n, int flag);
  int SetNthFiducialSelectedNoModified(int n, int flag);
  int SetNthFiducialVisibility(int n, int flag);
  int SetNthFiducialVisibilityNoModified(int n, int flag);
  int SetNthFiducialID(int n, const char *id);

  // Description:
  // Restrict access to the fiducial points, access the fiducial by id (used
  // by the vtkSlicerFiducialListWidget). Returns 0 on success.
  //BTX
  int SetFiducialXYZ(std::string fiducialID, float x, float y, float z);
  // Description:
  // Look through the list of fiducials for fiducialID and return the current
  // zero based index. Useful when delete fiducials from the list, as the
  // index will change. Returns -1 if none found.
  int GetFiducialIndex(std::string fiducialID);
  //ETX
  
  // Description:
  // Set all fiducials selected state to flag
  int SetAllFiducialsSelected(int flag);

  // Description:
  // Set all fiducials visible state to flag
  int SetAllFiducialsVisibility(int flag);
  
  // Description:
  // Get the elements of the fiducial points
  // Return a three element float holding the position
  float *GetNthFiducialXYZ(int n);
  // Description:
  // get the orientation of the nth fiducial
  float *GetNthFiducialOrientation(int n);
  // Description:
  // get the label text of the nth fiducial
  const char *GetNthFiducialLabelText(int n);
  // Description:
  // get the selected state on the nth fiducial
  int GetNthFiducialSelected(int n);
  // Description:
  // get the visible state on the nth fiducial
  int GetNthFiducialVisibility(int n);
  // Description:
  // get the id of the nth fiducial
  const char *GetNthFiducialID(int n);
  
  // Description:
  // Add a fiducial point to the list with default values
  int AddFiducial( );

  // Description:
  // Add a fiducial point to the list x, y, z
  int AddFiducialWithXYZ(float x, float y, float z, int selected);
  // Description:
  // Add a fiducial point to the list with a label, x,y,z, selected flag, visibility
  int AddFiducialWithLabelXYZSelectedVisibility(const char *label, float x, float y, float z, int selected, int visibility);

  // Description:
  // remove the passed in fiducial from the list
  void RemoveFiducial(vtkMRMLFiducial *o);
  // Description:
  // remove the fiducial at index i
  void RemoveFiducial(int i);
  // Description:
  // remove all fiducials from the list
  void RemoveAllFiducials();
  // Description:
  // is this fiducial on the list?
  int  IsFiducialPresent(vtkMRMLFiducial *o);

  // Description:
  // Process events from the MRML scene
  void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

  //BTX
  // Description:
  // DisplayModifiedEvent is generated when display node parameters is changed
  // PolyDataModifiedEvent is generated when something else is changed
  enum
    {
      DisplayModifiedEvent = 19000,
      PolyDataModifiedEvent = 19001,
      FiducialModifiedEvent = 19002,
    };
//ETX

  // Description:
  // Opacity of the fiducial surface expressed as a number from 0 to 1
  void SetOpacity(double opacity);
  vtkGetMacro(Opacity, double);

  // Description:
  // Ambient of the fiducial surface expressed as a number from 0 to 1
  void SetAmbient(double val);
  vtkGetMacro(Ambient, double);
  
  // Description:
  // Diffuse of the fiducial surface expressed as a number from 0 to 1
  void SetDiffuse(double val);
  vtkGetMacro(Diffuse, double);
  
  // Description:
  // Specular of the fiducial surface expressed as a number from 0 to 1
  void SetSpecular(double val);
  vtkGetMacro(Specular, double);

  // Description:
  // Power of the fiducial surface expressed as a number from 0 to 1
  void SetPower(double val);
  vtkGetMacro(Power, double);

  // Description:
  // When fiducial lists are locked, they cannot be manipulated using the interactive widgets
  void SetLocked(int locked);
  vtkGetMacro(Locked, int);

  //BTX
  // Description:
  // Which kind of glyph should be used to display this fiducial?
  enum GlyphShapes
  {
    GlyphMin,
    Vertex2D = GlyphMin,
    Dash2D,
    Cross2D,
    ThickCross2D,
    Triangle2D,
    Square2D,
    Circle2D,
    Diamond2D,
    Arrow2D,
    ThickArrow2D,
    HookedArrow2D,
    StarBurst2D,
    Sphere3D,
    Diamond3D,
    GlyphMax = Diamond3D,
  };
  //ETX
  // Description:
  // The glyph type used to display this fiducial
  void SetGlyphType(int type);
  vtkGetMacro(GlyphType, int);
  // Description:
  // Returns 1 if the type is a 3d one, 0 else
  int GlyphTypeIs3D(int glyphType);
  int GlyphTypeIs3D() { return this->GlyphTypeIs3D(this->GlyphType); };

  // Description:
  // Return a string representing the glyph type, set it from a string
  const char* GetGlyphTypeAsString();
  const char* GetGlyphTypeAsString(int g);
  void SetGlyphTypeFromString(const char *glyphString);

  // Description:
  // transform utility functions
  virtual bool CanApplyNonLinearTransforms() { return true; }
  virtual void ApplyTransform(vtkMatrix4x4* transformMatrix);
  virtual void ApplyTransform(vtkAbstractTransform* transform);
  
  // Description:
  // Create default storage node or NULL if does not have one
  virtual vtkMRMLStorageNode* CreateDefaultStorageNode()
    {
      return vtkMRMLFiducialListStorageNode::New();
    };

  // Description:
  // move a fiducial point in the collection, one up/down
  // returns -1 on failure (current index is out of bounds, the fid is already
  // at the top or bottom of the list), the new fiducial index on success
  int MoveFiducialUp(int fidIndex);
  int MoveFiducialDown(int fidIndex);

  // Description:
  // Renumber all the fiducials in the active list. It first removes any numbers
  // from the ends of the label texts and then appends numbers starting from 0
  // by default.
  void RenumberFiducials(int startFrom = 0);

  // Description:
  // Reanme all the fiducials in the active list. It preserves any numbers
  // already on the ends of the labels.
  void RenameFiducials(const char *newName);

  //BTX
  // Description:
  // flags to determine how the next fiducial added to the list is labelled
  enum NumberingSchemes
  {
      SchemeMin = 0,
      UseID = SchemeMin,
      UseIndex,
      UsePrevious,
      SchemeMax = UsePrevious,
  };
  //ETX

  // Description:
  // Flag determining how to number the next added fiducial
  vtkSetMacro(NumberingScheme, int);
  vtkGetMacro(NumberingScheme, int);

  // Description:
  // Return a string representing the numbering scheme, set it from a string
  const char* GetNumberingSchemeAsString();
  const char* GetNumberingSchemeAsString(int g);
  void SetNumberingSchemeFromString(const char *schemeString);

  // Description:
  // Generate the label text for a fiducial from it's id. If NumberingScheme
  // is UseID, uses the ID (default). If NumberingScheme is UseIndex, strips the ID of any
  // trailing numbers and appends the fiducial's index to it. If
  // NumberingScheme is UsePrevious, checks the previous fiducial
  // in the list for a trailing number and increments it by 1, and appends the
  // new number. If it's the first fiducial, uses 0. If the previous fiducial
  // has no trailing number in it's label text, will use 1.
  void SetFiducialLabelTextFromID(vtkMRMLFiducial *fid);

protected:
  vtkMRMLFiducialListNode();
  ~vtkMRMLFiducialListNode();
  vtkMRMLFiducialListNode(const vtkMRMLFiducialListNode&);
  void operator=(const vtkMRMLFiducialListNode&);

  // Description:
  // disallow access to the fiducial points by outside classes, have them use
  // SetNthFiducial
  vtkMRMLFiducial* GetNthFiducial(int n);
  
  double SymbolScale;
  double TextScale;
  int Visibility;
  double Color[3];
  double SelectedColor[3];

  // Description:
  // The collection of fiducial points that make up this list
  vtkCollection *FiducialList;

  // Description:
  // Numbers relating to the display of the fiducials
  double Opacity;
  double Ambient;
  double Diffuse;
  double Specular;
  double Power;
  int Locked;
  int GlyphType;

  // Description:
  // How the next added fiducial will be numbered in it's LabelText field
  int NumberingScheme;
};

#endif
