/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkMRMLModelDisplayNode.h,v $
  Date:      $Date: 2006/03/19 17:12:28 $
  Version:   $Revision: 1.6 $

=========================================================================auto=*/

#ifndef __vtkMRMLModelDisplayNode_h
#define __vtkMRMLModelDisplayNode_h

// MRML includes
#include "vtkMRMLDisplayNode.h"

// VTK includes
class vtkAlgorithm;
class vtkAlgorithmOutput;
class vtkAssignAttribute;
class vtkPassThrough;
class vtkPointSet;
class vtkPolyData;
class vtkUnstructuredGrid;

/// \brief MRML node to represent a display property of 3D surface model.
///
/// vtkMRMLModelDisplayNode nodes stores display property of a 3D surface model
/// including reference to ColorNode, texture, opacity, etc.
class VTK_MRML_EXPORT vtkMRMLModelDisplayNode : public vtkMRMLDisplayNode
{
public:
  static vtkMRMLModelDisplayNode *New();
  vtkTypeMacro(vtkMRMLModelDisplayNode,vtkMRMLDisplayNode);

  virtual vtkMRMLNode* CreateNodeInstance();

  /// Get node XML tag name (like Volume, Model)
  virtual const char* GetNodeTagName() {return "ModelDisplay";};

  /// Set and observe mesh for this model. It should be the output
  /// mesh connection of the model node.
  /// \sa GetInputMeshConnection(), GetInputMesh()
  virtual void SetInputMeshConnection(vtkAlgorithmOutput* meshConnection);
  virtual void SetInputPolyDataConnection(vtkAlgorithmOutput* polyDataConnection);

  /// Get the input mesh connection. It should be the output
  /// mesh connection of the model node
  /// \sa SetInputMeshConnection(), GetInputMesh()
  virtual vtkAlgorithmOutput* GetInputMeshConnection();
  virtual vtkAlgorithmOutput* GetInputPolyDataConnection();

  /// Return the mesh that was set by SetInputMeshConnection()
  /// \sa SetInputMeshConnection(), GetInputMeshConnection()
  virtual vtkPointSet* GetInputMesh();
  virtual vtkPolyData* GetInputPolyData();
  virtual vtkUnstructuredGrid* GetInputUnstructuredGrid();

  /// Return the mesh that is processed by the display node.
  /// This is the mesh that needs to be connected with the mappers.
  /// Return 0 if there is no input mesh but it is required.
  /// GetOutputMesh() should be reimplemented only if the model display
  /// node doesn't take a mesh as input but produce an oustput mesh.
  /// In all other cases, GetOutputMeshConnection() should be reimplemented.
  /// \sa GetOutputMeshConnection()
  virtual vtkPointSet* GetOutputMesh();
  virtual vtkPolyData* GetOutputPolyData();
  virtual vtkUnstructuredGrid* GetOutputUnstructuredGrid();

  /// Return the mesh that is processed by the display node.
  /// This is the mesh that needs to be connected with the mappers.
  /// \sa GetOutputMesh()
  virtual vtkAlgorithmOutput* GetOutputMeshConnection();
  virtual vtkAlgorithmOutput* GetOutputPolyDataConnection(); ///\deprecated

  /// Reimplemented to update pipeline with new value
  /// Note: If model is generated by a filter pipeline then any modification in the pipeline
  /// resets the output mesh and then recomputes the mesh. During reset the scalars in the mesh
  /// are removed, therefore if a GUI or other component observes the mesh, then it will detect that
  /// the scalar is deleted and so it may deactivate the selected scalar.
  /// \sa SetActiveAttributeLocation()
  virtual void SetActiveScalarName(const char *scalarName);

  /// Reimplemented to update pipeline with new value
  /// \sa SetActiveScalarName()
  virtual void SetActiveAttributeLocation(int location);

  /// Reimplemented to update scalar range accordingly
  /// \sa SetActiveScalarName()
  virtual void SetScalarRangeFlag(int flag);

protected:
  vtkMRMLModelDisplayNode();
  ~vtkMRMLModelDisplayNode();
  vtkMRMLModelDisplayNode(const vtkMRMLModelDisplayNode&);
  void operator=(const vtkMRMLModelDisplayNode&);

  virtual void ProcessMRMLEvents(vtkObject *caller,
                                 unsigned long event,
                                 void *callData);

  /// To be reimplemented in subclasses if the input of the pipeline changes
  virtual void SetInputToMeshPipeline(vtkAlgorithmOutput* meshConnection);

  /// Update the AssignAttribute filter based on
  /// its ActiveScalarName and its ActiveAttributeLocation
  virtual void UpdateAssignedAttribute();

  /// Update the ScalarRange based on the ScalarRangeFlag
  virtual void UpdateScalarRange();

  /// Filter that changes the active scalar of the input mesh
  /// using the ActiveScalarName and ActiveAttributeLocation properties.
  /// This can be useful to specify what field array is the color array that
  /// needs to be used by the VTK mappers.
  vtkAssignAttribute* AssignAttribute;

  /// Default filter when assign attribute is not used, e.g ActiveScalarName is
  /// null.
  /// \sa AssignAttribute
  vtkPassThrough* PassThrough;
};

#endif
