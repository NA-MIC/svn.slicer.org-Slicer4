/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:29 $
  Version:   $Revision: 1.3 $

=========================================================================auto=*/

#ifndef __vtkMRMLDisplayNode_h
#define __vtkMRMLDisplayNode_h

// MRML includes
#include "vtkMRMLNode.h"
class vtkMRMLColorNode;
class vtkMRMLDisplayableNode;

// VTK includes
class vtkImageData;
class vtkPolyData;

// STD includes
#include <vector>

/// \brief A superclass that contain display properties for displayable nodes.
///
/// \sa vtkMRMLDisplayableNode
class VTK_MRML_EXPORT vtkMRMLDisplayNode : public vtkMRMLNode
{
public:
  vtkTypeMacro(vtkMRMLDisplayNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual vtkMRMLNode* CreateNodeInstance() = 0;

  ///
  /// Returns the first displayable node that is associated to this display node
  /// Warning: This function is slow as it browses the entire scene to find the
  /// displayable node.
  virtual vtkMRMLDisplayableNode* GetDisplayableNode();

  /// 
  /// Read node attributes from XML file
  virtual void ReadXMLAttributes( const char** atts);

  /// 
  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  /// 
  /// Copy the node's attributes to this object
  virtual void Copy(vtkMRMLNode *node);


  /// 
  /// Get node XML tag name (like Storage, Model)
  virtual const char* GetNodeTagName() = 0;

  /// 
  /// Propagate Progress Event generated in ReadData
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

  /// Mark the color and views nodes as references.
  virtual void SetSceneReferences();

  /// 
  /// Updates this node if it depends on other nodes 
  /// when the node is deleted in the scene
  virtual void UpdateReferences();

  /// 
  /// Finds the storage node and read the data
  virtual void UpdateScene(vtkMRMLScene *scene);

  /// 
  /// Update the stored reference to another node in the scene
  virtual void UpdateReferenceID(const char *oldID, const char *newID);

  /// Set the color of the display node.
  /// \sa Color, GetColor()
  vtkSetVector3Macro(Color, double);
  /// Get the color of the display node.
  /// \sa Color, SetColor()
  vtkGetVector3Macro(Color, double);

  /// Set the selected color of the display node.
  /// \sa SelectedColor, GetSelectedColor()
  vtkSetVector3Macro(SelectedColor, double);
  /// Get the selected color of the display node.
  /// \sa SelectedColor, SetSelectedColor()
  vtkGetVector3Macro(SelectedColor, double);

  /// Set the selected ambient coef of the display node.
  /// \sa SelectedAmbient, GetSelectedAmbient()
  vtkSetMacro(SelectedAmbient, double);
  /// Get the selected ambient coef of the display node.
  /// \sa SelectedAmbient, SetSelectedAmbient()
  vtkGetMacro(SelectedAmbient, double);

  /// Set the selected specular coef of the display node.
  /// \sa SelectedSpecular, GetSelectedSpecular()
  vtkSetMacro(SelectedSpecular, double);
  /// Get the selected specular coef of the display node.
  /// \sa SelectedSpecular, SetSelectedSpecular()
  vtkGetMacro(SelectedSpecular, double);

  /// Set the opacity coef of the display node.
  /// \sa Opacity, GetOpacity()
  vtkSetMacro(Opacity, double);
  /// Get the opacity coef of the display node.
  /// \sa Opacity, SetOpacity()
  vtkGetMacro(Opacity, double);

  /// Set the ambient coef of the display node.
  /// \sa Ambient, GetAmbient()
  vtkSetMacro(Ambient, double);
  /// Get the ambient coef of the display node.
  /// \sa Ambient, SetAmbient()
  vtkGetMacro(Ambient, double);

  /// Set the diffuse coef of the display node.
  /// \sa Diffuse, GetDiffuse()
  vtkSetMacro(Diffuse, double);
  /// Get the diffuse coef of the display node.
  /// \sa Diffuse, SetDiffuse()
  vtkGetMacro(Diffuse, double);

  /// Set the specular coef of the display node.
  /// \sa Specular, GetSpecular()
  vtkSetMacro(Specular, double);
  /// Get the specular coef of the display node.
  /// \sa Specular, SetSpecular()
  vtkGetMacro(Specular, double);

  /// Set the specular power coef of the display node.
  /// \sa Power, GetPower()
  vtkSetMacro(Power, double);
  /// Get the specular power coef of the display node.
  /// \sa Power, SetPower()
  vtkGetMacro(Power, double);

  /// Set the visibility of the display node.
  /// \sa Visibility, GetVisibility(), VisibilityOn(), VisibilityOff()
  vtkSetMacro(Visibility, int);
  /// Get the visibility of the display node.
  /// \sa Visibility, SetVisibility(), VisibilityOn(), VisibilityOff()
  vtkGetMacro(Visibility, int);
  /// Set the visibility of the display node.
  /// \sa Visibility, SetVisibility(), GetVisibility(),
  vtkBooleanMacro(Visibility, int);

  /// Set the clipping of the display node.
  /// \sa Clipping, GetClipping(), ClippingOn(), ClippingOff()
  vtkSetMacro(Clipping, int);
  /// Get the clipping of the display node.
  /// \sa Clipping, SetClipping(), ClippingOn(), ClippingOff()
  vtkGetMacro(Clipping, int);
  /// Set the clipping of the display node.
  /// \sa Clipping, SetClipping(), GetClipping()
  vtkBooleanMacro(Clipping, int);

  /// Set the slicer intersection visibility of the display node.
  /// \sa SliceIntersectionVisibility, GetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff()
  vtkSetMacro(SliceIntersectionVisibility, int);
  /// Get the slicer intersection visibility of the display node.
  /// \sa SliceIntersectionVisibility, SetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff()
  vtkGetMacro(SliceIntersectionVisibility, int);
  /// Set the slicer intersection visibility of the display node.
  /// \sa SliceIntersectionVisibility, SetSliceIntersectionVisibility(),
  /// GetSliceIntersectionVisibility(),
  vtkBooleanMacro(SliceIntersectionVisibility, int);

  /// Set the backface culling of the display node.
  /// \sa BackfaceCulling, GetBackfaceCulling(), BackfaceCullingOn(),
  /// BackfaceCullingOff()
  vtkSetMacro(BackfaceCulling, int);
  /// Get the backface culling of the display node.
  /// \sa BackfaceCulling, SetBackfaceCulling(), BackfaceCullingOn(),
  /// BackfaceCullingOff()
  vtkGetMacro(BackfaceCulling, int);
  /// Set the backface culling of the display node.
  /// \sa BackfaceCulling, SetBackfaceCulling(), GetBackfaceCulling()
  vtkBooleanMacro(BackfaceCulling, int);

  /// Set the scalar visibility of the display node.
  /// \sa ScalarVisibility, GetScalarVisibility(), ScalarVisibilityOn(),
  /// ScalarVisibilityOff()
  vtkSetMacro(ScalarVisibility, int);
  /// Get the scalar visibility of the display node.
  /// \sa ScalarVisibility, SetScalarVisibility(), ScalarVisibilityOn(),
  /// ScalarVisibilityOff()
  vtkGetMacro(ScalarVisibility, int);
  /// Set the scalar visibility of the display node.
  /// \sa ScalarVisibility, SetScalarVisibility(), GetScalarVisibility
  vtkBooleanMacro(ScalarVisibility, int);

  /// Set the vector visibility of the display node.
  /// \sa VectorVisibility, GetVectorVisibility(), VectorVisibilityOn(),
  /// VectorVisibilityOff()
  vtkSetMacro(VectorVisibility, int);
  /// Get the vector visibility of the display node.
  /// \sa VectorVisibility, SetVectorVisibility(), VectorVisibilityOn(),
  /// VectorVisibilityOff()
  vtkGetMacro(VectorVisibility, int);
  /// Set the vector visibility of the display node.
  /// \sa VectorVisibility, SetVectorVisibility(), GetVectorVisibility
  vtkBooleanMacro(VectorVisibility, int);

  /// Set the tensor visibility of the display node.
  /// \sa TensorVisibility, GetTensorVisibility(), TensorVisibilityOn(),
  /// TensorVisibilityOff()
  vtkSetMacro(TensorVisibility, int);
  /// Get the tensor visibility of the display node.
  /// \sa TensorVisibility, SetTensorVisibility(), TensorVisibilityOn(),
  /// TensorVisibilityOff()
  vtkGetMacro(TensorVisibility, int);
  /// Set the tensor visibility of the display node.
  /// \sa TensorVisibility, SetTensorVisibility(), GetTensorVisibility
  vtkBooleanMacro(TensorVisibility, int);

  /// Set the auto scalar range flag of the display node.
  /// \sa AutoScalarRange, GetAutoScalarRange(), AutoScalarRangeOn(),
  /// AutoScalarRangeOff()
  vtkSetMacro(AutoScalarRange, int);
  /// Get the auto scalar range flag of the display node.
  /// \sa AutoScalarRange, SetAutoScalarRange(), AutoScalarRangeOn(),
  /// AutoScalarRangeOff()
  vtkGetMacro(AutoScalarRange, int);
  /// Set the auto scalar range flag of the display node.
  /// \sa AutoScalarRange, SetAutoScalarRange(), GetAutoScalarRange()
  vtkBooleanMacro(AutoScalarRange, int);

  /// Set the scalar range of the display node.
  /// \sa ScalarRange, GetScalarRange()
  vtkSetVector2Macro(ScalarRange, double);
  /// Get the scalar range of the display node.
  /// \sa ScalarRange, SetScalarRange()
  vtkGetVector2Macro(ScalarRange, double);

  /// Set and observe the texture image data.
  /// \sa TextureImageData, GetTextureImageData()
  void SetAndObserveTextureImageData(vtkImageData *ImageData);
  /// Get the texture image data.
  /// \sa TextureImageData, SetAndObserveTextureImageData()
  vtkGetObjectMacro(TextureImageData, vtkImageData);

  /// Set the texture interpolation of the display node.
  /// \sa InterpolateTexture, GetInterpolateTexture(), InterpolateTextureOn(),
  /// InterpolateTextureOff()
  vtkSetMacro(InterpolateTexture, int);
  /// Get the texture interpolation of the display node.
  /// \sa InterpolateTexture, SetInterpolateTexture(), InterpolateTextureOn(),
  /// InterpolateTextureOff()
  vtkGetMacro(InterpolateTexture, int);
  /// Set the texture interpolation of the display node.
  /// \sa InterpolateTexture, SetInterpolateTexture(), GetInterpolateTexture()
  vtkBooleanMacro(InterpolateTexture, int);

  /// Set and observe color node of the displaty node.
  /// \sa ColorNodeID, GetColorNodeID()
  virtual void SetAndObserveColorNodeID(const char *ColorNodeID);
  /// Set and observe color node of the displaty node.
  /// Utility method that conveniently takes a string instead of a char*
  /// \sa ColorNodeID, GetColorNodeID()
  void SetAndObserveColorNodeID(const std::string& ColorNodeID);
  /// Get color node ID of the displaty node.
  /// \sa ColorNodeID, SetAndObserveColorNodeID()
  vtkGetStringMacro(ColorNodeID);

  /// Get associated color MRML node. Search the node into the scene if the node
  /// hasn't been cached yet. This can be a slow call.
  /// \sa ColorNodeID, SetAndObserveColorNodeID, GetColorNodeID()
  virtual vtkMRMLColorNode* GetColorNode();

  /// Set the active scalar name of the display node.
  /// \sa ActiveScalarName, GetActiveScalarName()
  virtual void SetActiveScalarName(const char *scalarName);
  /// Return the name of the currently active scalar field for this model.
  /// \sa ActiveScalarName, SetActiveScalarName()
  vtkGetStringMacro(ActiveScalarName);

  /// Set the active attribute location of the display node.
  /// \sa ActiveAttributeLocation, GetActiveAttributeLocation()
  vtkSetMacro(ActiveAttributeLocation, int);
  /// Get the active attribute location of the display node.
  /// \sa ActiveAttributeLocation, SetActiveAttributeLocation()
  vtkGetMacro(ActiveAttributeLocation, int);

  /// Add View Node ID for the view to display this node in.
  /// \sa ViewNodeIDs, RemoveViewNodeID(), RemoveAllViewNodeIDs()
  void AddViewNodeID(const char* viewNodeID);
  /// Remove View Node ID for the view to display this node in.
  /// \sa ViewNodeIDs, AddViewNodeID(), RemoveAllViewNodeIDs()
  void RemoveViewNodeID(char* viewNodeID);
  /// Remove All View Node IDs for the views to display this node in.
  /// \sa ViewNodeIDs, AddViewNodeID(), RemoveViewNodeID()
  void RemoveAllViewNodeIDs();
  /// Get number of View Node ID's for the view to display this node in.
  /// If 0, display in all views
  /// \sa ViewNodeIDs, GetViewNodeIDs(), AddViewNodeID()
  inline int GetNumberOfViewNodeIDs()const;
  /// Get View Node ID's for the view to display this node in.
  /// If NULL, display in all views
  /// \sa ViewNodeIDs, GetViewNodeIDs(), AddViewNodeID()
  const char* GetNthViewNodeID(unsigned int index);
  /// Get all View Node ID's for the view to display this node in.
  /// If empty, display in all views
  /// \sa ViewNodeIDs, GetNthViewNodeID(), AddViewNodeID()
  inline std::vector< std::string > GetViewNodeIDs()const;
  /// True if the view node id is present in the viewnodeid list
  /// false if not found
  /// \sa ViewNodeIDs, IsDisplayableInView(), AddViewNodeID()
  bool IsViewNodeIDPresent(const char* viewNodeID)const;
  /// Returns true if the viewNodeID is present in the ViewNodeId list
  /// or there is no ViewNodeId in the list (meaning all the views display the
  /// node)
  /// \sa ViewNodeIDs, IsViewNodeIDPresent(), AddViewNodeID()
  bool IsDisplayableInView(const char* viewNodeID)const;

protected:
  vtkMRMLDisplayNode();
  ~vtkMRMLDisplayNode();
  vtkMRMLDisplayNode(const vtkMRMLDisplayNode&);
  void operator=(const vtkMRMLDisplayNode&);

  virtual void SetColorNodeInternal(vtkMRMLColorNode* newColorNode);

  virtual void SetTextureImageData(vtkImageData* imageData);

  /// Associated ImageData to apply as texture.
  /// No texture (NULL) by default.
  /// \sa SetAndObserveTextureImageData(), GetTextureImageData()
  /// InterpolateTexture
  vtkImageData    *TextureImageData;

  /// String ID of the color MRML node. The color node LUT or Color transfer
  /// function is set to the VTK mappers.
  /// Note that anytime the color node is modified, the observing display node
  /// fires a Modified event.
  /// No color node by default.
  /// \sa SetColorNodeID(), GetColorNodeID(),
  /// ColorNode
  char *ColorNodeID;
  /// \sa GetColorNode(),
  /// ColorNode
  vtkMRMLColorNode *ColorNode;

  /// Active vtkDataSetAttributes::Scalars field name for the
  /// \a ActiveAttributeLocation array.
  /// This is typically used to specify what field array is the color array that
  /// needs to be used by the VTK mappers.
  /// No active scalar name by default other than the default polydata.
  /// \sa SetActiveScalarName(), GetActiveScalarName(),
  /// ActiveAttributeLocation
  char *ActiveScalarName;
  /// This property controls on which attribute the \a ActiveScalarName applies.
  /// Typically vtkAssignAttribute::POINT_DATA or vtkAssignAttribute::CELL_DATA.
  /// Default to vtkAssignAttribute::POINT_DATA
  /// \sa SetActiveAttributeLocation(), GetActiveAttributeLocation(),
  /// ActiveScalarName
  int ActiveAttributeLocation;

  /// Opacity of the surface expressed as a number from 0 to 1.
  /// Opaque (1.) by default.
  /// \sa SetOpacity(), GetOpacity(),
  /// Color, Visibility, Clipping
  double Opacity;
  /// Ambient of the surface expressed as a number from 0 to 1.
  /// 0. by default.
  /// \sa SetAmbient(), GetAmbient(),
  /// Color, SelectedAmbient, Diffuse, Specular, Power
  double Ambient;
  /// Diffuse of the surface expressed as a number from 0 to 1.
  /// 1. by default.
  /// \sa SetDiffuse(), GetDiffuse(),
  /// Color, Ambient, Specular, Power
  double Diffuse;
  /// Specular of the surface expressed as a number from 0 to 1.
  /// 0. by default.
  /// \sa SetSpecular(), GetSpecular(),
  /// Color, Ambient, Diffuse, Power
  double Specular;
  /// Power of the surface specularity expressed as a number from 0 to 100.
  /// 1. by default.
  /// \sa SetPower(), GetPower(),
  /// Color, Ambient, Diffuse, Specular
  double Power;
  /// Node's selected ambient.
  /// 0.4 by default.
  /// \sa SetSelectedAmbient(), GetSelectedAmbient(),
  /// SelectedColor, Ambient, SelectedSpecular
  double SelectedAmbient;
  /// Node's selected specular.
  /// 0.5 by default.
  /// \sa SetSelectedSpecular(), GetSelectedSpecular(),
  /// SelectedColor, SelectedAmbient, Specular
  double SelectedSpecular;

  /// Indicates if the surface is visible
  /// True by default.
  /// \sa SetVisibility(), GetVisibility(), VisibilityOn(), VisibilityOff()
  /// Color, Opacity, Clipping
  int Visibility;
  /// Specifies whether to clip the surface with the slice planes.
  /// 0 by default.
  /// \sa SetClipping(), GetClipping(), ClippingOn(), ClippingOff()
  /// Visibility, SliceIntersectionVisibility
  int Clipping;
  /// Specifies whether to show model intersections on slice planes.
  /// 0 by default.
  /// \sa SetSliceIntersectionVisibility(), GetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff(),
  /// Visibility, Clipping
  int SliceIntersectionVisibility;
  /// Indicates whether to cull (not render) the backface of the surface.
  /// 1 by default.
  /// \sa SetBackfaceCulling(), GetBackfaceCulling(), BackfaceCullingOn(),
  /// BackfaceCullingOff(),
  /// Visibility, Clipping
  int BackfaceCulling;
  /// Indicates whether to render the scalar value associated with each polygon
  /// vertex.
  /// 0 by default.
  /// \sa SetScalarVisibility(), GetScalarVisibility(), ScalarVisibilityOn(),
  /// ScalarVisibilityOff(),
  /// Visibility, VectorVisibility, TensorVisibility
  int ScalarVisibility;
  /// Indicates whether to render the vector value associated with each polygon
  /// vertex.
  /// 0 by default.
  /// \sa SetVectorVisibility(), GetVectorVisibility(), VectorVisibilityOn(),
  /// VectorVisibilityOff(),
  /// Visibility, ScalarVisibility, TensorVisibility
  int VectorVisibility;
  /// Indicates whether to render the tensor value associated with each polygon
  /// vertex.
  /// 0 by default.
  /// \sa SetTensorVisibility(), GetTensorVisibility(), TensorVisibilityOn(),
  /// TensorVisibilityOff(),
  /// Visibility, ScalarVisibility, VectorVisibility
  int TensorVisibility;
  /// Indicates whether to use scalar range from polydata or the one specidied
  /// by ScalarRange.
  /// 1 by default.
  /// \sa SetAutoScalarRange(), GetAutoScalarRange(), AutoScalarRangeOn(),
  /// AutoScalarRangeOff(),
  /// ScalarRange
  int AutoScalarRange;
  /// Indicates whether to use interpolate texture.
  /// 0 by default.
  /// \sa SetInterpolateTexture(), GetInterpolateTexture(), InterpolateTextureOn(),
  /// InterpolateTextureOff(),
  /// TextureImageData
  int InterpolateTexture;

  /// Range of scalar values to render rather than the single color designated
  /// by colorName.
  /// (0, 100) by default.
  /// \sa SetScalarRange(), GetScalarRange(),
  /// AutoScalarRange
  double ScalarRange[2];

  /// Model's color in the format [r,g,b].
  /// Gray (0.5, 0.5, 0.5) by default.
  /// \sa SetColor(), GetColor(),
  /// SelectedColor, Ambient, Diffuse, Specular, Power
  double Color[3];
  /// Node's selected ambient color (r,g,b).
  /// Red (1., 0., 0.) by default.
  /// \sa SetSelectedColor(), GetSelectedColor(),
  /// Color, SelectedAmbient, SelectedSpecular
  double SelectedColor[3];

  /// List of view node ID's for which the display node should be visible into.
  /// If the list is empty, it means the display node should be visible in all
  /// the view nodes.
  /// Empty by default.
  /// \sa AddViewNodeID(), RemoveViewNodeID(), RemoveAllViewNodeIDs(),
  /// GetNumberOfViewNodeIDs(), GetViewNodeIDs(), IsViewNodeIDPresent(),
  /// IsDisplayableInView()
  std::vector< std::string > ViewNodeIDs;

private:
  void SetColorNodeID(const char* id);
};

//----------------------------------------------------------------------------
int vtkMRMLDisplayNode::GetNumberOfViewNodeIDs()const
{
  return static_cast<int>(this->ViewNodeIDs.size());
}

//----------------------------------------------------------------------------
std::vector< std::string > vtkMRMLDisplayNode::GetViewNodeIDs()const
{
  return this->ViewNodeIDs;
}

#endif
