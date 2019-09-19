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
class vtkAlgorithmOutput;
class vtkImageData;
class vtkPolyData;

// STD includes
#include <vector>

/// \brief Abstract class that contains graphical display properties for
/// displayable nodes.
///
/// vtkMRMLDisplayNode fires a ModifiedEvent event when the texture image data
/// or the color node is modified.
/// \sa vtkMRMLDisplayableNode, TextureImageDataConnection, ColorNode
class VTK_MRML_EXPORT vtkMRMLDisplayNode : public vtkMRMLNode
{
public:
  vtkTypeMacro(vtkMRMLDisplayNode,vtkMRMLNode);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// Representation models
  /// \sa GetRepresentation(), SetRepresentation()
  typedef enum {
    PointsRepresentation = 0,
    WireframeRepresentation,
    SurfaceRepresentation
  } RepresentationType;

  /// Interpolation models
  /// \sa GetInterpolation(), SetInterpolation()
  typedef enum {
    FlatInterpolation = 0,
    GouraudInterpolation,
    PhongInterpolation
  } InterpolationType;

  /// Scalar range options for displaying data associated with this display
  /// node, this setting determines if the display node, color node, or ?
  /// determine the mapping range between the data and the colors used to
  /// display it. Not all are currently supported.
  /// UseDataScalarRange - use the current min/max of the active data (former auto)
  /// UseColorNodeScalarRange - use the range from the associated color node
  /// UseDataTypeScalarRange - use the min/max of the numerical type of the
  /// UseDirectMapping - map scalar values directly to RGB values
  /// data, for example minimum integer to maximum integer
  /// UseManualScalarRange - use user defined values
  /// \sa ScalarRangeFlag, GetScalarRangeFlag(), SetScalarRangeFlag(),
  /// SetScalarRange(), GetScalarRange(), GetScalarRangeFlagTypeAsString()
  typedef enum {
    UseManualScalarRange = 0,
    UseDataScalarRange,
    UseColorNodeScalarRange,
    UseDataTypeScalarRange,
    UseDirectMapping,
    // insert types above this line
    NUM_SCALAR_RANGE_FLAGS
  } ScalarRangeFlagType;

  /// Convert between scalar range flag type id and string
  /// \sa ScalarRangeFlag
  static const char* GetScalarRangeFlagTypeAsString(int flag);
  /// Gets scalar range flag type from string
  static int GetScalarRangeFlagTypeFromString(const char* name);

  /// Returns the first displayable node that is associated to this display node
  /// \sa vtkMRMLDisplayableNode
  virtual vtkMRMLDisplayableNode* GetDisplayableNode();

  /// Read node attributes from XML file.
  /// \sa vtkMRMLParser
  void ReadXMLAttributes( const char** atts) override;

  /// Write this node's information to a MRML file in XML format.
  /// \sa vtkMRMLScene::Commit()
  void WriteXML(ostream& of, int indent) override;

  /// Copy the node's attributes to this object.
  void Copy(vtkMRMLNode *node) override;

  /// Propagate ModifiedEvent generated by the texture image data or the color
  /// node.
  /// \sa TextureImageDataConnection, ColorNode
  void ProcessMRMLEvents(vtkObject *caller, unsigned long event,
                                 void *callData) override;

  /// Mark the color and views nodes as references.
  void SetSceneReferences() override;

  /// Updates this node if it depends on other nodes
  /// when the node is deleted in the scene.
  void UpdateReferences() override;

  /// Finds the storage node and read the data.
  void UpdateScene(vtkMRMLScene *scene) override;

  /// Update the stored reference to another node in the scene.
  void UpdateReferenceID(const char *oldID, const char *newID) override;

  /// Set the color of the display node.
  /// \sa Color, GetColor()
  vtkSetVector3Macro(Color, double);
  /// Get the color of the display node.
  /// \sa Color, SetColor()
  vtkGetVector3Macro(Color, double);

  /// Set the edge color of the display node.
  /// \sa EdgeColor, GetEdgeColor()
  vtkSetVector3Macro(EdgeColor, double);
  /// Get the edge color of the display node.
  /// \sa EdgeColor, SetEdgeColor()
  vtkGetVector3Macro(EdgeColor, double);

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

  /// Set the diameter of points.
  /// \sa PointSize, GetPointSize()
  vtkSetMacro(PointSize, double);
  /// Get the diameter of points.
  /// \sa PointSize, SetPointSize()
  vtkGetMacro(PointSize, double);

  /// Set the width of lines.
  /// \sa LineWidth, GetLineWidth()
  vtkSetMacro(LineWidth, double);
  /// Get the widget of lines.
  /// \sa LineWidth, SetLineWidth()
  vtkGetMacro(LineWidth, double);

  /// Set the representation of the surface.
  /// \sa Representation, GetRepresentation()
  vtkSetMacro(Representation, int);
  /// Get the representation of the surface.
  /// \sa Representation, SetRepresentation()
  vtkGetMacro(Representation, int);

  /// Set the opacity coef of the display node.
  /// \sa Opacity, GetOpacity()
  vtkSetMacro(Opacity, double);
  /// Get the opacity coef of the display node.
  /// \sa Opacity, SetOpacity()
  vtkGetMacro(Opacity, double);

  /// Set the slice intersection opacity coef of the display node.
  /// \sa SliceIntersectionOpacity, GetSliceIntersectionOpacity()
  vtkSetMacro(SliceIntersectionOpacity, double);
  /// Get the slice intersection opacity coef of the display node.
  /// \sa SliceIntersectionOpacity, SetSliceIntersectionOpacity()
  vtkGetMacro(SliceIntersectionOpacity, double);

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
  /// Return true if the display node should be visible in the view node.
  /// To be visible in the view, the node needs to be visible
  /// (\a Visibility == 1) and the view ID must be in the ViewNodeIDs list
  /// or the list must be empty (visible in all views).
  /// \sa Visibility, ViewNodeIDs
  virtual bool GetVisibility(const char* viewNodeID);

  /// Set the 2D visibility of the display node.
  /// \sa Visibility2D, GetVisibility2D(),
  /// Visibility2DOn(), Visibility2DOff()
  vtkSetMacro(Visibility2D, int);
  /// Get the 2D visibility of the display node.
  /// \sa Visibility2D, SetVisibility2D(),
  /// Visibility2DOn(), Visibility2DOff()
  vtkGetMacro(Visibility2D, int);
  /// Set the 2D visibility of the display node.
  /// \sa Visibility2D, SetVisibility2D(),
  /// GetVisibility2D(),
  vtkBooleanMacro(Visibility2D, int);

  /// Set the 3D visibility of the display node.
  /// \sa Visibility3D, GetVisibility3D(),
  /// Visibility3DOn(), Visibility3DOff()
  vtkSetMacro(Visibility3D, int);
  /// Get the 3D visibility of the display node.
  /// \sa Visibility3D, SetVisibility3D(),
  /// Visibility3DOn(), Visibility3DOff()
  vtkGetMacro(Visibility3D, int);
  /// Set the 3D visibility of the display node.
  /// \sa Visibility3D, SetVisibility3D(),
  /// GetVisibility3D(),
  vtkBooleanMacro(Visibility3D, int);

  /// Set the visibility of the edges.
  /// \sa EdgeVisibility, GetEdgeVisibility()
  vtkSetMacro(EdgeVisibility, int);
  vtkBooleanMacro(EdgeVisibility, int);
  /// Get the visibility of the edges.
  /// \sa EdgeVisibility, SetEdgeVisibility()
  vtkGetMacro(EdgeVisibility, int);

  /// Set the clipping of the display node.
  /// \sa Clipping, GetClipping(), ClippingOn(), ClippingOff()
  vtkSetMacro(Clipping, int);
  /// Get the clipping of the display node.
  /// \sa Clipping, SetClipping(), ClippingOn(), ClippingOff()
  vtkGetMacro(Clipping, int);
  /// Set the clipping of the display node.
  /// \sa Clipping, SetClipping(), GetClipping()
  vtkBooleanMacro(Clipping, int);

  /// Set the slice intersection visibility of the display node.
  /// Function to manage \sa Visibility2D for backwards compatibility
  /// \sa Visibility2D, GetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff()
  /// \deprecated SetSliceIntersectionVisibility
  void SetSliceIntersectionVisibility(int on);
  /// Get the slice intersection visibility of the display node.
  /// Function to manage \sa Visibility2D for backwards compatibility
  /// \sa Visibility2D, SetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff()
  /// \deprecated GetSliceIntersectionVisibility
  int GetSliceIntersectionVisibility();
  /// Set the slice intersection visibility of the display node.
  /// Function to manage \sa Visibility2D for backwards compatibility
  /// \sa Visibility2D, SetSliceIntersectionVisibility(),
  /// GetSliceIntersectionVisibility(),
  /// \deprecated SliceIntersectionVisibilityOn
  void SliceIntersectionVisibilityOn();
  /// Set the slice intersection visibility of the display node.
  /// Function to manage \sa Visibility2D for backwards compatibility
  /// \sa Visibility2D, SetSliceIntersectionVisibility(),
  /// GetSliceIntersectionVisibility(),
  /// \deprecated SliceIntersectionVisibilityOff
  void SliceIntersectionVisibilityOff();

  /// Set the slice intersection thickness of the display node. In voxels.
  /// \sa SliceIntersectionThickness, GetSliceIntersectionThickness()
  vtkSetMacro(SliceIntersectionThickness, int);
  /// Get the slice intersection thickness of the display node. In voxels.
  /// \sa SliceIntersectionThickness, SetSliceIntersectionThickness()
  vtkGetMacro(SliceIntersectionThickness, int);

  /// Set the backface culling of the display node.
  /// \sa FrontfaceCulling, GetFrontfaceCulling(), FrontfaceCullingOn(),
  /// FrontfaceCullingOff()
  vtkSetMacro(FrontfaceCulling, int);
  /// Get the backface culling of the display node.
  /// \sa FrontfaceCulling, SetFrontfaceCulling(), FrontfaceCullingOn(),
  /// FrontfaceCullingOff()
  vtkGetMacro(FrontfaceCulling, int);
  /// Set the backface culling of the display node.
  /// \sa FrontfaceCulling, SetFrontfaceCulling(), GetFrontfaceCulling()
  vtkBooleanMacro(FrontfaceCulling, int);
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

  /// Enable/Disable lighting of the display node.
  /// \sa Lighting, GetLighting(), LightingOn(),
  /// LightingOff()
  vtkSetMacro(Lighting, int);
  /// Get the lighting of the display node.
  /// \sa Lighting, SetLighting(), LightingOn(),
  /// LightingOff()
  vtkGetMacro(Lighting, int);
  /// Enable/Disable the lighting of the display node.
  /// \sa Lighting, SetLighting(), GetLighting()
  vtkBooleanMacro(Lighting, int);

  /// Set the interpolation of the surface.
  /// \sa Interpolation, GetInterpolation()
  vtkSetMacro(Interpolation, int);
  /// Get the interpolation of the surface.
  /// \sa Interpolation, SetInterpolation()
  vtkGetMacro(Interpolation, int);

  /// Set the shading mode (None, Gouraud, Flat) of the display node.
  /// \sa Shading, GetShading()
  vtkSetMacro(Shading, int);
  /// Get the shading of the display node.
  /// \sa Shading, SetShading()
  vtkGetMacro(Shading, int);

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
  /// \deprecated
  /// \sa SetScalarRangeFlag(), GetAutoScalarRange(), AutoScalarRangeOn(),
  /// AutoScalarRangeOff()
  void SetAutoScalarRange(int b);
  /// Get the auto scalar range flag of the display node.
  /// \deprecated
  /// \sa GetScalarRangeFlag(), SetAutoScalarRange(), AutoScalarRangeOn(),
  /// AutoScalarRangeOff()
  int GetAutoScalarRange();
  /// Set the auto scalar range flag of the display node.
  /// \deprecated
  /// \sa SetScalarRangeFlag(), SetAutoScalarRange(), GetAutoScalarRange()
  void AutoScalarRangeOn();
  void AutoScalarRangeOff();

  /// Set the scalar range of the display node.
  /// \sa ScalarRange, GetScalarRange()
  vtkSetVector2Macro(ScalarRange, double);
  /// Get the scalar range of the display node.
  /// \sa ScalarRange, SetScalarRange()
  vtkGetVector2Macro(ScalarRange, double);

  /// Set the scalar range to use with color mapping
  /// \sa ScalarRangeFlag, GetScalarRangeFlag(), SetScalarRangeFlagFromString()
  vtkSetMacro(ScalarRangeFlag, int);
  /// Get the interpolation of the surface.
  /// \sa ScalarRangeFlag, SetScalarRangeFlag(), GetScalarRangeFlagAsString()
  vtkGetMacro(ScalarRangeFlag, int);
  /// Get scalar range flag as string
  /// \sa ScalarRangeFlag, GetScalarRangeFlag()
  const char* GetScalarRangeFlagAsString();
  /// Set scalar range flag from string
  /// \sa ScalarRangeFlag, SetScalarRangeFlag()
  void SetScalarRangeFlagFromString(const char* str);

  /// Set flag determining whether folders are allowed to override display properties.
  /// \sa FolderDisplayOverrideAllowed, GetFolderDisplayOverrideAllowed()
  vtkSetMacro(FolderDisplayOverrideAllowed, bool);
  /// Get flag determining whether folders are allowed to override display properties.
  /// \sa FolderDisplayOverrideAllowed, SetFolderDisplayOverrideAllowed()
  vtkGetMacro(FolderDisplayOverrideAllowed, bool);
  /// Set flag determining whether folders are allowed to override display properties.
  /// \sa FolderDisplayOverrideAllowed, SetFolderDisplayOverrideAllowed(), GetFolderDisplayOverrideAllowed()
  vtkBooleanMacro(FolderDisplayOverrideAllowed, bool);

  /// Set and observe the texture image data port.
  /// \sa TextureImageDataConnection, GetTextureImageDataConnection()
  virtual void SetTextureImageDataConnection(vtkAlgorithmOutput *ImageDataConnection);
  /// Get the texture image data port.
  /// \sa TextureImageDataConnection, SetTextureImageDataConnection()
  vtkGetObjectMacro(TextureImageDataConnection, vtkAlgorithmOutput);

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

  /// Set and observe color node of the display node.
  /// \sa ColorNodeID, GetColorNodeID()
  virtual void SetAndObserveColorNodeID(const char *ColorNodeID);
  /// Set and observe color node of the display node.
  /// Utility method that conveniently takes a string instead of a char*
  /// \sa ColorNodeID, GetColorNodeID()
  void SetAndObserveColorNodeID(const std::string& ColorNodeID);
  /// Get color node ID of the display node.
  /// \sa ColorNodeID, SetAndObserveColorNodeID()
  vtkGetStringMacro(ColorNodeID);

  /// Get associated color MRML node. Search the node into the scene if the node
  /// hasn't been cached yet. This can be a slow call.
  /// \sa ColorNodeID, SetAndObserveColorNodeID, GetColorNodeID()
  virtual vtkMRMLColorNode* GetColorNode();

  /// Set the active scalar name of the display node.
  /// \sa ActiveScalarName, GetActiveScalarName()
  vtkSetStringMacro(ActiveScalarName);
  /// Return the name of the currently active scalar field for this model.
  /// \sa ActiveScalarName, SetActiveScalarName()
  vtkGetStringMacro(ActiveScalarName);

  /// Set the active attribute location of the display node.
  /// vtkAssignAttribute::POINT_DATA by default.
  /// \sa ActiveAttributeLocation, GetActiveAttributeLocation(), SetActiveAttributeLocationFromString()
  vtkSetMacro(ActiveAttributeLocation, int);
  /// Get the active attribute location of the display node.
  /// \sa ActiveAttributeLocation, SetActiveAttributeLocation(), GetActiveAttributeLocationAsString()
  vtkGetMacro(ActiveAttributeLocation, int);
  /// Get the active attribute location of the display node as string
  /// \sa ActiveAttributeLocation, GetActiveAttributeLocation()
  const char* GetActiveAttributeLocationAsString();
  /// Set the active attribute location of the display node from string
  /// \sa ActiveAttributeLocation, SetActiveAttributeLocation()
  void SetActiveAttributeLocationFromString(const char* str);

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
  /// If nullptr, display in all views
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
  /// Set the View Node ID as the only view to display this node in.
  /// If the view node id does not exist, the node will show in all views.
  /// Uses a disable/enable modified event block to avoid flicker.
  /// \sa RemoveAllViewNodeIDs(), AddViewNodeID()
  void SetDisplayableOnlyInView(const char *viewNodeID);
  /// Set all the view node IDs at once. Only trigger Modified() if the
  /// new vector is different from the existing vector.
  /// \sa GetViewNodeIDs(), AddViewNodeID()
  void SetViewNodeIDs(const std::vector< std::string >& viewNodeIDs);

  /// Converts attribute location (point or cell data) to string
  static const char* GetAttributeLocationAsString(int id);
  /// Gets attribute location (point or cell data) from string
  static int GetAttributeLocationFromString(const char* name);

protected:
  vtkMRMLDisplayNode();
  ~vtkMRMLDisplayNode() override;
  vtkMRMLDisplayNode(const vtkMRMLDisplayNode&);
  void operator=(const vtkMRMLDisplayNode&);

  /// Internal function to set the color node. Called by
  /// \a SetColorNodeID().
  /// \sa SetColorNodeID(),
  /// ColorNode, ColorNodeID
  virtual void SetColorNodeInternal(vtkMRMLColorNode* newColorNode);

  /// Associated ImageDataConnection to apply as texture. The image data port is
  /// observed and when modified, vtkMRMLDisplayNode fires a ModifiedEvent too.
  /// No texture (nullptr) by default.
  /// \sa SetTextureImageDataConnection(), GetTextureImageDataConnection()
  /// InterpolateTexture
  vtkAlgorithmOutput *TextureImageDataConnection;

  /// String ID of the color MRML node. The color node LUT or Color transfer
  /// function is set to the VTK mappers.
  /// Note that anytime the color node is modified, the observing display node
  /// fires a Modified event.
  /// No color node by default.
  /// \sa SetColorNodeID(), GetColorNodeID(),
  /// ColorNode
  char *ColorNodeID;
  /// Cached instance of the color node ID found in the scene. The node is
  /// observed and when modified, vtkMRMLDisplayNode fires a ModifiedEvent too.
  /// \sa GetColorNode(), ProcessMRMLEvents()
  /// ColorNodeID
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
  /// Opacity of the slice intersections expressed as a number from 0 to 1.
  /// Opaque (1.) by default.
  /// \sa SetSliceIntersectionOpacity(), GetSliceIntersectionOpacity(),
  /// Color, Visibility, Clipping
  double SliceIntersectionOpacity;
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

  /// Diameter of a point. The size is expressed in screen units.
  /// The default is 1.0.
  /// \sa SetPointSize(), GetPointSize(), LineWidth
  double PointSize;

  /// Width of a line. The width is expressed in screen units.
  /// The default is 1.0.
  /// \sa SetLineWidth(), GetLineWidth(), PointSize
  double LineWidth;

  /// Control the surface geometry representation for the object.
  /// SurfaceRepresentation by default.
  /// \sa SetRepresentation(), GetRepresentation(), Interpolation
  int Representation;

  /// This property controls the lighting.
  /// 1 by default.
  /// \sa SetLighting(), GetLighting(),
  /// Interpolation, Shading
  int Lighting;

  /// Set the shading interpolation method for an object. Note that
  /// to use an interpolation other than FlatInterpolation, normals
  /// must be associated to the polydata (Gouraud and Phong are usually the
  /// same).
  /// GouraudInterpolation by default.
  /// \sa SetInterpolation(), GetInterpolation(),
  /// Lighting
  int Interpolation;

  /// This property controls whether the shading is enabled/disabled.
  /// 1 by default.
  /// \sa SetShading(), GetShading(),
  /// Lighting
  int Shading;

  /// Indicates if the surface is visible.
  /// True by default.
  /// \sa SetVisibility(), GetVisibility(), VisibilityOn(), VisibilityOff()
  /// Color, Opacity, Clipping, EdgeVisibility, SliceIntersectionVisibility
  int Visibility;
  /// Indicates whether the object is visible in the slice views. True by default.
  /// In order to show 2D, both this and \sa Visibility needs to be enabled.
  /// \sa SetVisibility2D(), GetVisibility2D(), Visibility2DOn(), Visibility2DOff()
  int Visibility2D;
  /// Indicates whether the object is visible in the 3D views. True by default.
  /// In order to show 3D, both this and \sa Visibility needs to be enabled.
  /// \sa SetVisibility3D(), GetVisibility3D(), Visibility3DOn(), Visibility3DOff()
  int Visibility3D;
  /// This property controls the visibility of edges. On some renderers it is
  /// possible to render the edges of geometric primitives separately
  /// from the interior.
  /// 0 by default.
  /// \sa SetEdgeVisibility(), GetEdgeVisibility(),
  /// EdgeColor, Visibility, SliceIntersectionVisibility
  int EdgeVisibility;
  /// Specifies whether to clip the surface with the slice planes.
  /// 0 by default.
  /// \sa SetClipping(), GetClipping(), ClippingOn(), ClippingOff()
  /// Visibility, EdgeVisibility, SliceIntersectionVisibility
  int Clipping;
  /// Specifies how thick to show the intersections with slice planes if slice
  /// intersection visibility is on
  /// 1 voxel by default.
  /// \sa SetSliceIntersectionVisibility(), GetSliceIntersectionVisibility(),
  /// SliceIntersectionVisibilityOn(), SliceIntersectionVisibilityOff(),
  /// Visibility, SliceIntersectionVisibility,
  int SliceIntersectionThickness;
  /// Indicates whether to cull (not render) the frontface of the surface.
  /// 0 by default.
  /// \sa SetFrontfaceCulling(), GetFrontfaceCulling(), FrontfaceCullingOn(),
  /// FrontfaceCullingOff(),
  /// BackfaceCulling, Visibility, Clipping
  int FrontfaceCulling;
  /// Indicates whether to cull (not render) the backface of the surface.
  /// 1 by default.
  /// \sa SetBackfaceCulling(), GetBackfaceCulling(), BackfaceCullingOn(),
  /// BackfaceCullingOff(),
  /// FrontfaceCulling, Visibility, Clipping
  int BackfaceCulling;
  /// Indicates whether to render the scalar value associated with each polygon
  /// vertex.
  /// Hidden (0) by default.
  /// \sa SetScalarVisibility(), GetScalarVisibility(), ScalarVisibilityOn(),
  /// ScalarVisibilityOff(),
  /// Visibility, VectorVisibility, TensorVisibility
  int ScalarVisibility;
  /// Indicates whether to render the vector value associated with each polygon
  /// vertex.
  /// Hidden (0) by default.
  /// \sa SetVectorVisibility(), GetVectorVisibility(), VectorVisibilityOn(),
  /// VectorVisibilityOff(),
  /// Visibility, ScalarVisibility, TensorVisibility
  int VectorVisibility;
  /// Indicates whether to render the tensor value associated with each polygon
  /// vertex.
  /// Hidden (0) by default.
  /// \sa SetTensorVisibility(), GetTensorVisibility(), TensorVisibilityOn(),
  /// TensorVisibilityOff(),
  /// Visibility, ScalarVisibility, VectorVisibility
  int TensorVisibility;
  /// Indicates whether to use interpolate texture.
  /// Don't interpolate (0) by default.
  /// \sa SetInterpolateTexture(), GetInterpolateTexture(), InterpolateTextureOn(),
  /// InterpolateTextureOff(),
  /// TextureImageDataConnection
  int InterpolateTexture;

  /// Range of scalar values to render rather than the single color designated
  /// by colorName.
  /// [0, 100] by default.
  /// \sa SetScalarRange(), GetScalarRange(),
  /// ScalarRangeFlag
  double ScalarRange[2];

  /// Model's color in the format [r,g,b].
  /// Gray (0.5, 0.5, 0.5) by default.
  /// \sa SetColor(), GetColor(),
  /// SelectedColor, EdgeColor, Ambient, Diffuse, Specular, Power, ColorNodeID
  double Color[3];
  /// The property controls the color of primitive edges (if edge visibility is
  /// enabled).
  /// Black (0,0,0) by default.
  /// \sa SetEdgeColor(), GetEdgeColor(),
  /// EdgeVisibility, Color, SelectedColor
  double EdgeColor[3];
  /// Node's selected ambient color (r,g,b).
  /// Red (1., 0., 0.) by default.
  /// \sa SetSelectedColor(), GetSelectedColor(),
  /// Color, EdgeColor, SelectedAmbient, SelectedSpecular
  double SelectedColor[3];

  /// List of view node ID's for which the display node should be visible into.
  /// If the list is empty, it means the display node should be visible in all
  /// the view nodes.
  /// The displayable managers are responsible for reading this property.
  /// Visible in all views (empty) by default.
  /// \sa AddViewNodeID(), RemoveViewNodeID(), RemoveAllViewNodeIDs(),
  /// GetNumberOfViewNodeIDs(), GetViewNodeIDs(), IsViewNodeIDPresent(),
  /// IsDisplayableInView(),
  /// vtkMRMLAbstractDisplayableManager
  std::vector< std::string > ViewNodeIDs;

  /// A flag to determine which scalar range will be used when mapping
  /// scalars to colors.
  /// UseColorNodeScalarRange by default.
  /// \sa ScalarRangeFlagType,GetScalarRangeFlag(), SetScalarRangeFlag(),
  /// ScalarRange, SetScalarRange(), GetScalarRange()
  int ScalarRangeFlag;

  /// Flag to determine whether folders are allowed to override display properties.
  /// On by default.
  /// \sa GetFolderDisplayOverrideAllowed(), SetFolderDisplayOverrideAllowed()
  bool FolderDisplayOverrideAllowed;

  /// Cached value of last found displayable node (it is expensive to determine it)
  vtkWeakPointer<vtkMRMLDisplayableNode> LastFoundDisplayableNode;
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
