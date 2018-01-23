/*==============================================================================

  Copyright (c) Laboratory for Percutaneous Surgery (PerkLab)
  Queen's University, Kingston, ON, Canada. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Csaba Pinter, PerkLab, Queen's University
  and was supported through the Applied Cancer Research Unit program of Cancer Care
  Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care

==============================================================================*/

// .NAME vtkSlicerSegmentationsModuleLogic - Logic class for segmentation handling
// .SECTION Description
// This class manages the logic associated with converting and handling
// segmentation node objects.

#ifndef __vtkSlicerSegmentationsModuleLogic_h
#define __vtkSlicerSegmentationsModuleLogic_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerSegmentationsModuleLogicExport.h"

// Segmentations includes
#include "vtkMRMLSegmentationNode.h"

class vtkCallbackCommand;
class vtkOrientedImageData;
class vtkPolyData;
class vtkDataObject;
class vtkGeneralTransform;

class vtkMRMLSegmentationStorageNode;
class vtkMRMLScalarVolumeNode;
class vtkMRMLLabelMapVolumeNode;
class vtkMRMLVolumeNode;
class vtkMRMLModelNode;
class vtkMRMLModelHierarchyNode;
class vtkSlicerTerminologiesModuleLogic;

/// \ingroup SlicerRt_QtModules_Segmentations
class VTK_SLICER_SEGMENTATIONS_LOGIC_EXPORT vtkSlicerSegmentationsModuleLogic :
  public vtkSlicerModuleLogic
{
public:
  static vtkSlicerSegmentationsModuleLogic *New();
  vtkTypeMacro(vtkSlicerSegmentationsModuleLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  /// Get segmentation node containing a segmentation object. As segmentation objects are out-of-MRML
  /// VTK objects, there is no direct link from it to its parent node, so must be found from the MRML scene.
  /// \param scene MRML scene
  /// \param segmentation Segmentation to find
  /// \return Segmentation node containing the given segmentation if any, NULL otherwise
  static vtkMRMLSegmentationNode* GetSegmentationNodeForSegmentation(vtkMRMLScene* scene, vtkSegmentation* segmentation);

  /// Get segmentation node containing a given segment. As segments are out-of-MRML
  /// VTK objects, there is no direct link from it to its parent node, so must be found from the MRML scene.
  /// \param scene MRML scene
  /// \param segment Segment to find
  /// \param segmentId Output argument for the ID of the found segment
  /// \return Segmentation node containing the given segment if any, NULL otherwise
  static vtkMRMLSegmentationNode* GetSegmentationNodeForSegment(vtkMRMLScene* scene, vtkSegment* segment, std::string& segmentId);

  /// Load segmentation from file
  /// \param filename Path and name of file containing segmentation (nrrd, vtm, etc.)
  /// \param autoOpacities Optional flag determining whether segment opacities are calculated automatically based on containment. True by default
  /// \return Loaded segmentation node
  vtkMRMLSegmentationNode* LoadSegmentationFromFile(const char* filename, bool autoOpacities=true);

  /// Create labelmap volume MRML node from oriented image data.
  /// Creates a display node if a display node does not exist. Shifts image extent to start from zero.
  /// Image is shallow-copied (voxel array is not duplicated).
  /// \param orientedImageData Oriented image data to create labelmap from
  /// \param labelmapVolumeNode Labelmap volume to be populated with the oriented image data. The volume node must exist and be added to the MRML scene.
  /// \return Success flag
  static bool CreateLabelmapVolumeFromOrientedImageData(vtkOrientedImageData* orientedImageData, vtkMRMLLabelMapVolumeNode* labelmapVolumeNode);

  /// Create volume MRML node from oriented image data. Display node is not created.
  /// \param orientedImageData Oriented image data to create volume node from
  /// \param scalarVolumeNode Volume to be populated with the oriented image data.
  /// \param shallowCopy If true then voxel array is not duplicated.
  /// \param shiftImageDataExtentToZeroStart: Adjust image origin to make image extents start from zero. May be necessary for compatibility with some algorithms
  ///        that assumes image extent start from 0.
  /// \return Success flag
  static bool CopyOrientedImageDataToVolumeNode(vtkOrientedImageData* orientedImageData,
    vtkMRMLVolumeNode* volumeNode, bool shallowCopy = true, bool shiftImageDataExtentToZeroStart = true);

  /// Create oriented image data from a volume node
  /// \param outputParentTransformNode Specifies the parent transform node where the created image data can be placed.
  /// NOTE: Need to take ownership of the created object! For example using vtkSmartPointer<vtkOrientedImageData>::Take
  static vtkOrientedImageData* CreateOrientedImageDataFromVolumeNode(vtkMRMLScalarVolumeNode* volumeNode, vtkMRMLTransformNode* outputParentTransformNode = NULL);

  /// Utility function to determine if a labelmap contains a single label
  /// \return 0 if contains no label or multiple labels, the label if it contains a single one
  static int DoesLabelmapContainSingleLabel(vtkMRMLLabelMapVolumeNode* labelmapVolumeNode);

  /// Utility function that returns all non-empty label values in a labelmap
  static void GetAllLabelValues(vtkIntArray* labels, vtkImageData* labelmap);

  /// Create segment from labelmap volume MRML node. The contents are set as binary labelmap representation in the segment.
  /// Returns NULL if labelmap contains more than one label. In that case \sa ImportLabelmapToSegmentationNode needs to be used.
  /// NOTE: Need to take ownership of the created object! For example using vtkSmartPointer<vtkSegment>::Take
  /// \param labelmapVolumeNode Model node containing image data that will be the binary labelmap representation in the created segment
  /// \param segmentationNode Segmentation node that will be the container of the segment. It is used to get parent transform to
  ///   make sure the created segment will be located the same place the image was, considering all transforms involved. NULL value
  ///   means that this consideration is not needed. Default value is NULL.
  /// \return Created segment that then can be added to the segmentation if needed. Need to take ownership of the created
  ///   object! For example using vtkSmartPointer<vtkSegment>::Take
  static vtkSegment* CreateSegmentFromLabelmapVolumeNode(vtkMRMLLabelMapVolumeNode* labelmapVolumeNode, vtkMRMLSegmentationNode* segmentationNode=NULL);

  /// Create segment from model MRML node.
  /// The contents are set as closed surface model representation in the segment.
  /// NOTE: Need to take ownership of the created object! For example using vtkSmartPointer<vtkSegment>::Take
  /// \param modelNode Model node containing poly data that will be the closed surface representation in the created segment
  /// \param segmentationNode Segmentation node that will be the container of the segment. It is used to get parent transform to
  ///   make sure the created segment will be located the same place the model was, considering all transforms involved. NULL value
  ///   means that this consideration is not needed. Default value is NULL.
  /// \return Created segment that then can be added to the segmentation if needed. Need to take ownership of the created
  ///   object! For example using vtkSmartPointer<vtkSegment>::Take
  static vtkSegment* CreateSegmentFromModelNode(vtkMRMLModelNode* modelNode, vtkMRMLSegmentationNode* segmentationNode=NULL);

  /// Utility function for getting the segmentation node for a segment subject hierarchy item
  static vtkMRMLSegmentationNode* GetSegmentationNodeForSegmentSubjectHierarchyItem(vtkIdType segmentShItemID, vtkMRMLScene* scene);

  /// Utility function for getting the segment object for a segment subject hierarchy item
  static vtkSegment* GetSegmentForSegmentSubjectHierarchyItem(vtkIdType segmentShItemID, vtkMRMLScene* scene);

  /// Export segment to representation MRML node.
  /// 1. If representation node is a labelmap node, then the binary labelmap representation of the
  ///    segment is copied
  /// 2. If representation node is a model node, then the closed surface representation is copied
  /// Otherwise return with failure.
  static bool ExportSegmentToRepresentationNode(vtkSegment* segment, vtkMRMLNode* representationNode);

  /// Export multiple segments into a model hierarchy, a model node from each segment
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param segmentIds List of segment IDs to export
  /// \param modelHierarchyNode Model hierarchy to export the segments to
  static bool ExportSegmentsToModelHierarchy(vtkMRMLSegmentationNode* segmentationNode,
    std::vector<std::string>& segmentIDs, vtkMRMLModelHierarchyNode* modelHierarchyNode);

  /// Export multiple segments into a model hierarchy, a model node from each segment
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param segmentIds List of segment IDs to export
  /// \param modelHierarchyNode Model hierarchy to export the segments to
  static bool ExportSegmentsToModelHierarchy(vtkMRMLSegmentationNode* segmentationNode,
    vtkStringArray* segmentIds, vtkMRMLModelHierarchyNode* modelHierarchyNode);

  /// Export visible segments into a model hierarchy, a model node from each segment
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param modelHierarchyNode Model hierarchy to export the visible segments to
  static bool ExportVisibleSegmentsToModelHierarchy(vtkMRMLSegmentationNode* segmentationNode, vtkMRMLModelHierarchyNode* modelHierarchyNode);

  /// Export all segments into a model hierarchy, a model node from each segment
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param modelHierarchyNode Model hierarchy to export the segments to
  static bool ExportAllSegmentsToModelHierarchy(vtkMRMLSegmentationNode* segmentationNode, vtkMRMLModelHierarchyNode* modelHierarchyNode);

  /// Export multiple segments into a multi-label labelmap volume node
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param segmentIds List of segment IDs to export
  /// \param labelmapNode Labelmap node to export the segments to
  /// \param referenceVolumeNode If specified, then the merged labelmap node will match the geometry of referenceVolumeNode
  static bool ExportSegmentsToLabelmapNode(vtkMRMLSegmentationNode* segmentationNode, std::vector<std::string>& segmentIDs,
    vtkMRMLLabelMapVolumeNode* labelmapNode, vtkMRMLVolumeNode* referenceVolumeNode = NULL);

  /// Export multiple segments into a multi-label labelmap volume node
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param segmentIds List of segment IDs to export
  /// \param labelmapNode Labelmap node to export the segments to
  /// \param referenceVolumeNode If specified, then the merged labelmap node will match the geometry of referenceVolumeNode
  static bool ExportSegmentsToLabelmapNode(vtkMRMLSegmentationNode* segmentationNode, vtkStringArray* segmentIDs,
    vtkMRMLLabelMapVolumeNode* labelmapNode, vtkMRMLVolumeNode* referenceVolumeNode = NULL);

  /// Export visible segments into a multi-label labelmap volume node
  /// \param segmentationNode Segmentation node from which the the visible segments are exported
  /// \param labelmapNode Labelmap node to export the segments to
  /// \param referenceVolumeNode If specified, then the merged labelmap node will match the geometry of referenceVolumeNode
  static bool ExportVisibleSegmentsToLabelmapNode(vtkMRMLSegmentationNode* segmentationNode,
    vtkMRMLLabelMapVolumeNode* labelmapNode, vtkMRMLVolumeNode* referenceVolumeNode = NULL);

  /// Export all segments into a multi-label labelmap volume node
  /// \param segmentationNode Segmentation node from which the the segments are exported
  /// \param labelmapNode Labelmap node to export the segments to
  static bool ExportAllSegmentsToLabelmapNode(vtkMRMLSegmentationNode* segmentationNode, vtkMRMLLabelMapVolumeNode* labelmapNode);

  /// Import all labels from a labelmap node to a segmentation node, each label to a separate segment.
  /// The colors of the new segments are set from the color table corresponding to the labelmap volume.
  /// \param insertBeforeSegmentId New segments will be inserted before this segment.
  static bool ImportLabelmapToSegmentationNode(vtkMRMLLabelMapVolumeNode* labelmapNode,
    vtkMRMLSegmentationNode* segmentationNode, std::string insertBeforeSegmentId="");

  /// Import all labels from a labelmap image to a segmentation node, each label to a separate segment
  /// The colors of the new segments are randomly generated, unless terminology context is specified, in which case the terminology
  ///   entries are attempted to be mapped to the imported labels
  /// LabelmapImage is defined in the segmentation node's coordinate system
  /// (parent transform of the segmentation node is not used during import).
  /// \param baseSegmentName Prefix for the names of the new segments. Empty by default, in which case the prefix will be "Label"
  static bool ImportLabelmapToSegmentationNode(vtkOrientedImageData* labelmapImage,
    vtkMRMLSegmentationNode* segmentationNode, std::string baseSegmentName="", std::string insertBeforeSegmentId="") ;

  /// Update segmentation from segments in a labelmap node.
  /// \param updatedSegmentIDs Defines how label values 1..N are mapped to segment IDs (0..N-1).
  static bool ImportLabelmapToSegmentationNode(vtkMRMLLabelMapVolumeNode* labelmapNode,
    vtkMRMLSegmentationNode* segmentationNode, vtkStringArray* updatedSegmentIDs);

  /// Update segmentation from segments in a labelmap node.
  /// \param updatedSegmentIDs Defines how label values 1..N are mapped to segment IDs (0..N-1).
  static bool ImportLabelmapToSegmentationNode(vtkOrientedImageData* labelmapImage,
    vtkMRMLSegmentationNode* segmentationNode, vtkStringArray* updatedSegmentIDs,
    vtkGeneralTransform* labelmapToSegmentationTransform=NULL );

  /// Import all labels from a labelmap node to a segmentation node, each label to a separate segment.
  /// Terminology and color is set to the segments based on the color table corresponding to the labelmap volume node.
  /// \param terminologyContextName Terminology context the entries of which are mapped to the labels imported from the labelmap node
  /// \param insertBeforeSegmentId New segments will be inserted before this segment.
  bool ImportLabelmapToSegmentationNodeWithTerminology(vtkMRMLLabelMapVolumeNode* labelmapNode,
    vtkMRMLSegmentationNode* segmentationNode, std::string terminologyContextName, std::string insertBeforeSegmentId="");

  /// Import model into the segmentation as a segment.
  static bool ImportModelToSegmentationNode(vtkMRMLModelNode* modelNode, vtkMRMLSegmentationNode* segmentationNode, std::string insertBeforeSegmentId = "");

  /// Import model hierarchy into the segmentation as segments.
  static bool ImportModelHierarchyToSegmentationNode(
    vtkMRMLModelHierarchyNode* modelHierarchyNode, vtkMRMLSegmentationNode* segmentationNode, std::string insertBeforeSegmentId = "" );

  /// Create representation of only one segment in a segmentation.
  /// Useful if only one segment is processed, and we do not want to convert all segments to a certain
  /// segmentation to save time.
  /// NOTE: Need to take ownership of the created object! For example using vtkSmartPointer<vtkDataObject>::Take
  /// \return Representation of the specified segment if found or can be created, NULL otherwise
  static vtkDataObject* CreateRepresentationForOneSegment(vtkSegmentation* segmentation, std::string segmentID, std::string representationName);

  /// Apply the parent transform of a node to an oriented image data.
  /// Useful if we want to get a labelmap representation of a segmentation in the proper geometry for processing.
  /// \return Success flag
  static bool ApplyParentTransformToOrientedImageData(
    vtkMRMLTransformableNode* transformableNode, vtkOrientedImageData* orientedImageData, bool linearInterpolation=false, double backgroundColor[4]=NULL );

  /// Apply the parent transform of a node to a poly data.
  /// Useful if we want to get a surface or contours representation of a segmentation in the proper geometry for processing.
  /// \return Success flag
  static bool ApplyParentTransformToPolyData(vtkMRMLTransformableNode* transformableNode, vtkPolyData* polyData);

  /// Get transform between a representation node (e.g. labelmap or model) and a segmentation node.
  /// Useful if we want to add a representation to a segment, and we want to make sure that the segment will be located the same place
  /// the representation node was. The output transform is the representation node's parent transform concatenated with the inverse
  /// of the segmentation's parent transform. It needs to be applied on the representation.
  /// \param representationNode Transformable node which contains the representation we want to add to the segment
  /// \param segmentationNode Segmentation node that will contain the segment to which the representation is added. It is the
  ///   representation node's parent transform concatenated with the inverse of the segmentation's parent transform.
  /// \param representationToSegmentationTransform General transform between the representation node and the segmentation node.
  /// \return Success flag
  static bool GetTransformBetweenRepresentationAndSegmentation(
    vtkMRMLTransformableNode* representationNode, vtkMRMLSegmentationNode* segmentationNode, vtkGeneralTransform* representationToSegmentationTransform );

  /// Convenience function to get a specified representation of a segment in a segmentation.
  /// A duplicate of the representation data object is copied into the argument output object, with the segmentation's parent transform
  /// applied if requested (on by default).
  /// \param segmentationNode Input segmentation node containing the segment to extract
  /// \param segmentID Segment identifier of the segment to extract
  /// \param representationName Name of the requested representation
  /// \param segmentRepresentation Output representation data object into which the given representation in the segment is copied
  /// \param applyParentTransform Flag determining whether to apply parent transform of the segmentation node. On by default
  /// \return Success flag
  static bool GetSegmentRepresentation(vtkMRMLSegmentationNode* segmentationNode, std::string segmentID, std::string representationName, vtkDataObject* segmentRepresentation, bool applyParentTransform=true);

  /// Convenience function to get binary labelmap representation of a segment in a segmentation. Uses \sa GetSegmentRepresentation
  /// A duplicate of the oriented image data is copied into the argument image data, with the segmentation's parent transform
  /// applied if requested (on by default).
  /// The oriented image data can be used directly for processing, or to create a labelmap volume using \sa CreateLabelmapVolumeFromOrientedImageData.
  /// \param segmentationNode Input segmentation node containing the segment to extract
  /// \param segmentID Segment identifier of the segment to extract
  /// \param imageData Output oriented image data into which the segment binary labelmap is copied
  /// \param applyParentTransform Flag determining whether to apply parent transform of the segmentation node.
  ///   If on, then the oriented image data is in RAS, otherwise in the segmentation node's coordinate frame. On by default
  /// \return Success flag
  static bool GetSegmentBinaryLabelmapRepresentation(vtkMRMLSegmentationNode* segmentationNode, std::string segmentID, vtkOrientedImageData* imageData, bool applyParentTransform=true);

  /// Convenience function to get closed surface representation of a segment in a segmentation. Uses \sa GetSegmentRepresentation
  /// A duplicate of the closed surface data is copied into the argument image data, with the segmentation's parent transform
  /// applied if requested (on by default).
  /// \param segmentationNode Input segmentation node containing the segment to extract
  /// \param segmentID Segment identifier of the segment to extract
  /// \param polyData Output polydata into which the segment polydata is copied
  /// \param applyParentTransform Flag determining whether to apply parent transform of the segmentation node.
  ///   If on, then the oriented image data is in RAS, otherwise in the segmentation node's coordinate frame. On by default
  /// \return Success flag
  static bool GetSegmentClosedSurfaceRepresentation(vtkMRMLSegmentationNode* segmentationNode,
    std::string segmentID, vtkPolyData* polyData, bool applyParentTransform = true);

  /// Set a labelmap image as binary labelmap representation into the segment defined by the segmentation node and segment ID.
  /// Master representation must be binary labelmap! Master representation changed event is disabled to prevent deletion of all
  /// other representation in all segments. The other representations in the given segment are re-converted. The extent of the
  /// segment binary labelmap is shrunk to the effective extent. Display update is triggered.
  /// \param mergeMode Determines if the labelmap should replace the segment, or combined with a maximum or minimum operation.
  /// \param extent If extent is specified then only that extent of the labelmap is used.
  enum
    {
    MODE_REPLACE = 0,
    MODE_MERGE_MAX,
    MODE_MERGE_MIN
    };
  static bool SetBinaryLabelmapToSegment(vtkOrientedImageData* labelmap, vtkMRMLSegmentationNode* segmentationNode, std::string segmentID, int mergeMode=MODE_REPLACE, const int extent[6]=0);

  /// Assign terminology to segments in a segmentation node based on the labels of a labelmap node. Match is made based on the
  /// 3dSlicerLabel terminology type attribute. If the terminology context does not contain that attribute, match cannot be made.
  /// \param terminologyContextName Terminology context the entries of which are mapped to the labels imported from the labelmap node
  bool SetTerminologyToSegmentationFromLabelmapNode(vtkMRMLSegmentationNode* segmentationNode,
    vtkMRMLLabelMapVolumeNode* labelmapNode, std::string terminologyContextName);

public:
  /// Set Terminologies module logic
  void SetTerminologiesLogic(vtkSlicerTerminologiesModuleLogic* terminologiesLogic);

protected:
  virtual void SetMRMLSceneInternal(vtkMRMLScene * newScene) VTK_OVERRIDE;

  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  virtual void RegisterNodes() VTK_OVERRIDE;

  /// Callback function observing UID added events for subject hierarchy nodes.
  /// In case the newly added UID is a volume node referenced from a segmentation,
  /// its geometry will be set as image geometry conversion parameter.
  /// The "other order", i.e. when the volume is loaded first and the segmentation second,
  /// should be handled at loading time of the segmentation (because then we already know about the volume)
  static void OnSubjectHierarchyUIDAdded(vtkObject* caller, unsigned long eid, void* clientData, void* callData);

  /// Handle MRML node added events
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node) VTK_OVERRIDE;

protected:
  vtkSlicerSegmentationsModuleLogic();
  virtual ~vtkSlicerSegmentationsModuleLogic();

  /// Command handling subject hierarchy UID added events
  vtkCallbackCommand* SubjectHierarchyUIDCallbackCommand;

  /// Terminologies module logic
  vtkSlicerTerminologiesModuleLogic* TerminologiesLogic;

private:
  vtkSlicerSegmentationsModuleLogic(const vtkSlicerSegmentationsModuleLogic&); // Not implemented
  void operator=(const vtkSlicerSegmentationsModuleLogic&);               // Not implemented
};

#endif
