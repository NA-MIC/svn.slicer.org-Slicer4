#ifndef __vtkSlicerAnnotationModuleLogic_h
#define __vtkSlicerAnnotationModuleLogic_h

// Annotation QT includes
#include "GUI/qSlicerAnnotationModuleWidget.h"

// Slicer Logic includes
#include "vtkSlicerModuleLogic.h"

// MRML includes
#include "vtkMRMLAnnotationHierarchyNode.h"
#include "vtkMRMLAnnotationSnapshotNode.h"

#include "qSlicerAnnotationModuleExport.h"

/// \ingroup Slicer_QtModules_Annotation
class Q_SLICER_QTMODULES_ANNOTATIONS_EXPORT vtkSlicerAnnotationModuleLogic :
  public vtkSlicerModuleLogic
{
public:

  static vtkSlicerAnnotationModuleLogic *New();
  vtkTypeRevisionMacro(vtkSlicerAnnotationModuleLogic,vtkSlicerModuleLogic);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Initialize listening to MRML events
  virtual void SetMRMLSceneInternal(vtkMRMLScene * newScene);

  // Start the place mode for annotations
  void StartPlaceMode(bool persistent=false);

  // Exit the place mode for annotations
  void StopPlaceMode(bool persistent=false);

  // Start adding a new annotation Node
  void AddAnnotationNode(const char * nodeDescriptor, bool persistent=false);

  // After a node was added, propagate to widget
  void AddNodeCompleted(vtkMRMLAnnotationNode* annotationNode);

  // Cancel the current annotation placement or remove last annotation node
  void CancelCurrentOrRemoveLastAddedAnnotationNode();

  /// Remove an AnnotationNode and also its 1-1 IS-A hierarchyNode, if found.
  void RemoveAnnotationNode(vtkMRMLAnnotationNode* annotationNode);

  // Register the widget
  void SetAndObserveWidget(qSlicerAnnotationModuleWidget* widget);

  // MRML events
  void ProcessMRMLEvents(vtkObject *caller, unsigned long event, void *callData );
  void OnMRMLSceneNodeAddedEvent(vtkMRMLNode* node);
  void OnMRMLAnnotationNodeModifiedEvent(vtkMRMLNode* node);
  void OnMRMLSceneClosedEvent();
  void OnInteractionModeChangedEvent(vtkMRMLInteractionNode *interactionNode);
  void OnInteractionModePersistenceChangedEvent(vtkMRMLInteractionNode *interactionNode);
  //
  // Annotation Properties (interface to MRML)
  //
  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  virtual void RegisterNodes();

  /// Check if node id corresponds to an annotaton node
  bool IsAnnotationNode(const char* id);

  /// Return the text display node for the annotation mrml node with this id,
  /// null if not a valid node, not an annotation node, or doesn't have a text
  /// display node
  vtkMRMLAnnotationTextDisplayNode *GetTextDisplayNode(const char *id);
  /// Return the point display node for the annotation mrml node with this id,
  /// null if not a valid node, not an annotation node, or doesn't have a point
  /// display node
  vtkMRMLAnnotationPointDisplayNode *GetPointDisplayNode(const char *id);
  /// Return the line display node for the annotation mrml node with this id,
  /// null if not a valid node, not an annotation node, or doesn't have a line
  /// display node
  vtkMRMLAnnotationLineDisplayNode *GetLineDisplayNode(const char *id);
  
  /// Get the name of an Annotation MRML node
  const char * GetAnnotationName(const char * id);

  /// Return the text of an Annotation MRML node
  vtkStdString GetAnnotationText(const char* id);
  /// Set the text of an Annotation MRML node
  void SetAnnotationText(const char* id, const char * newtext);

  /// Get the text scale of an Annotation MRML node
  double GetAnnotationTextScale(const char* id);
  /// Set the text scale of an Annotation MRML node
  void SetAnnotationTextScale(const char* id, double textScale);

  /// Get the selected text color of an Annotation MRML node
  double * GetAnnotationTextSelectedColor(const char* id);
  /// Set the selected text color of an Annotation MRML node
  void SetAnnotationTextSelectedColor(const char* id, double * color);

  /// Get the text color of an Annotation MRML node
  double * GetAnnotationTextUnselectedColor(const char* id);
  /// Set the text color of an Annotation MRML node
  void SetAnnotationTextUnselectedColor(const char* id, double * color);

  /// Get the color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationColor(const char *id);
  /// Set the color of an annotation mrml node
  void SetAnnotationColor(const char *id, double *color);

  /// Get the unselected color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationUnselectedColor(const char *id);
  /// Set the unselected color of an annotation mrml node
  void SetAnnotationUnselectedColor(const char *id, double *color);

  /// Get the point color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationPointColor(const char *id);
  /// Set the point color of an annotation mrml node
  void SetAnnotationPointColor(const char *id, double *color);

  /// Get the unselected point color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationPointUnselectedColor(const char *id);
  /// Set the unselected point color of an annotation mrml node
  void SetAnnotationPointUnselectedColor(const char *id, double *color);

  /// Get the point glyph type of the annotation mrml node as a string,
  /// returns null if can't find it
  const char * GetAnnotationPointGlyphTypeAsString(const char *id);
  /// Get the point glyph type of the annotation mrml node, 
  int GetAnnotationPointGlyphType(const char *id);
  /// Set the point glyph type of the annotation mrml node from a string
  void SetAnnotationPointGlyphTypeFromString(const char *id, const char *glyphType);
  /// Set the point glyph type of the annotation mrml node
  void SetAnnotationPointGlyphType(const char *id, int glyphType);
  
  /// Get the line color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationLineColor(const char *id);
  /// Set the line color of an annotation mrml node
  void SetAnnotationLineColor(const char *id, double *color);

  /// Get the unselected line color of an annotation mrml node, returns null if can't find it
  double * GetAnnotationLineUnselectedColor(const char *id);
  /// Set the unselected line color of an annotation mrml node
  void SetAnnotationLineUnselectedColor(const char *id, double *color);

  
  /// Get the measurement value of an Annotation MRML node
  const char * GetAnnotationMeasurement(const char * id, bool showUnits);

  /// Get the icon name of an Annotation MRML node
  const char * GetAnnotationIcon(const char * id);

  /// Get the lock flag of an Annotation MRML node
  int GetAnnotationLockedUnlocked(const char * id);
  /// Toggle the lock flag of an Annotation MRML node
  void SetAnnotationLockedUnlocked(const char * id);

  /// Get the visibility flag of an Annotation MRML node
  int GetAnnotationVisibility(const char * id);
  /// Toggle the visibility flag of an Annotation MRML node
  void SetAnnotationVisibility(const char * id);

  /// Set the selected flag of an Annotation MRML node
  void SetAnnotationSelected(const char * id, bool selected);

  /// Backup an Annotation MRML node
  void BackupAnnotationNode(const char * id);
  /// Restore a backup of an Annotation MRML node
  void RestoreAnnotationNode(const char * id);
  /// Deletes a backup of an Annotation MRML node
  void DeleteBackupNodes(const char * id);

  /// Restore view of an Annotation MRML node
  void RestoreAnnotationView(const char* id);

  const char * MoveAnnotationUp(const char* id);
  const char * MoveAnnotationDown(const char* id);

  //
  // SnapShot functionality
  //
  /// Create a snapShot. This includes a screenshot of a specific view (see \ref GrabScreenShot(int screenshotWindow)),
  /// a multiline text description and the creation of a Scene SnapShot.
  void CreateSnapShot(const char* name, const char* description, int screenshotType, double scaleFactor, vtkImageData* screenshot);

  /// Modify an existing snapShot.
  void ModifySnapShot(vtkStdString id, const char* name, const char* description, int screenshotType, double scaleFactor, vtkImageData* screenshot);

  /// Return the name of an existing annotation snapShot.
  vtkStdString GetSnapShotName(const char* id);

  /// Return the description of an existing annotation snapShot.
  vtkStdString GetSnapShotDescription(const char* id);

  /// Return the screenshotType of an existing annotation snapShot.
  int GetSnapShotScreenshotType(const char* id);

  /// Return the scaleFactor of an existing annotation snapShot.
  double GetSnapShotScaleFactor(const char* id);

  /// Return the screenshot of an existing annotation snapShot.
  vtkImageData* GetSnapShotScreenshot(const char* id);

  /// Check if node id corresponds to a snapShot node
  bool IsSnapshotNode(const char* id);

  //
  // Hierarchy functionality
  //
  /// Add a new visible annotation hierarchy.
  /// The active hierarchy node will be the parent. If there is no
  /// active hierarchy node, use the top-level annotation hierarchy node as the parent.
  /// If there is no top-level annotation hierarchy node, create additionally a top-level hierarchy node which serves as
  /// a parent to the new hierarchy node. The newly added hierarchy node will be the
  /// active hierarchy node. Return true on success, false on failure.
  bool AddHierarchy();

  /// Return the toplevel Annotation hierarchy node or create one if there is none:
  /// If an optional annotationNode is given, insert the new toplevel hierarchy before it. If not,
  /// just add the new toplevel hierarchy node.
  vtkMRMLAnnotationHierarchyNode* GetTopLevelHierarchyNode(vtkMRMLNode* node=0);

  /// Set the active hierarchy node which will be used as a parent for new annotations
  void SetActiveHierarchyNode(vtkMRMLAnnotationHierarchyNode* hierarchyNode);

  /// Set the active hierarchy node which will be used as a parent for new annotations
  void SetActiveHierarchyNodeByID(const char* id);

  /// return the id of the active hierarchy node, null if none
  const char *GetActiveHierarchyNodeID();

  //
  // Place Annotations programmatically
  //

  //
  // Report functionality
  //
  /// Return HTML markup for a specific annotation node
  const char* GetHTMLRepresentation(vtkMRMLAnnotationNode* annotationNode, int level);
  /// Return HTML markup for a specific hierarchy node
  const char* GetHTMLRepresentation(vtkMRMLAnnotationHierarchyNode* hierarchyNode, int level);

protected:

  vtkSlicerAnnotationModuleLogic();

  virtual ~vtkSlicerAnnotationModuleLogic();

private:

  qSlicerAnnotationModuleWidget* m_Widget;

  vtkMRMLAnnotationNode* m_LastAddedAnnotationNode;

  vtkMRMLAnnotationHierarchyNode* m_ActiveHierarchy;

  vtksys_stl::string m_StringHolder;

  char* m_MeasurementFormat;
  char* m_CoordinateFormat;

  //
  // Private hierarchy functionality.
  //
  /// Add a new annotation hierarchy node for a given annotationNode.
  /// If there is an optional annotationNode, insert the new hierarchy node before it else just add it.
  /// The active hierarchy node will be the parent. If there is no
  /// active hierarchy node, use the top-level annotation hierarchy node as the parent.
  /// If there is no top-level annotation hierarchy node, create additionally a top-level hierarchy node which serves as
  /// a parent to the new hierarchy node. Return true on success, false on failure.
  bool AddHierarchyNodeForAnnotation(vtkMRMLAnnotationNode* annotationNode=0);


};

#endif
