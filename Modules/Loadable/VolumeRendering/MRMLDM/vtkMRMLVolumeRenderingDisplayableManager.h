/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __vtkMRMLVolumeRenderingDisplayableManager_h
#define __vtkMRMLVolumeRenderingDisplayableManager_h

// VolumeRendering includes
#include "vtkSlicerVolumeRenderingModuleMRMLDisplayableManagerExport.h"

// MRML DisplayableManager includes
#include <vtkMRMLAbstractThreeDViewDisplayableManager.h>

class vtkMRMLCPURayCastVolumeRenderingDisplayNode;
class vtkMRMLGPURayCastVolumeRenderingDisplayNode;
class vtkMRMLVolumeNode;
class vtkMRMLVolumeRenderingDisplayNode;
class vtkSlicerVolumeRenderingLogic;
class vtkGPUVolumeRayCastMapper;
class vtkFixedPointVolumeRayCastMapper;
class vtkIntArray;
class vtkMatrix4x4;
class vtkVolume;
class vtkVolumeMapper;

#define VTKIS_VOLUME_PROPS 100

/// \ingroup Slicer_QtModules_VolumeRendering
class VTK_SLICER_VOLUMERENDERING_MODULE_MRMLDISPLAYABLEMANAGER_EXPORT vtkMRMLVolumeRenderingDisplayableManager
  : public vtkMRMLAbstractThreeDViewDisplayableManager
{
public:
  static vtkMRMLVolumeRenderingDisplayableManager *New();
  vtkTypeMacro(vtkMRMLVolumeRenderingDisplayableManager, vtkMRMLAbstractThreeDViewDisplayableManager);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  virtual void Reset();

  void SetGUICallbackCommand(vtkCommand* callback);

  ///
  /// Return the volume mapper of the volume rendering display node
  virtual vtkVolumeMapper* GetVolumeMapper(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  ///
  /// Configure mapper with volume nodes
  void SetupMapperFromVolumeNode(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  ///
  /// Configure mapper with volume node
  void SetupMapperFromVolumeNode(vtkMRMLVolumeNode* volumeNode,
                                 vtkVolumeMapper* volumeMapper,
                                 int inputIndex = 0);

  // Description:
  // setup mapper based on current parameters node
  // return values:
  // -1: requested mapper not supported
  //  0: invalid input parameter
  //  1: success
  virtual int SetupMapperFromParametersNode(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  // Description:
  // Get Volume Actor
  vtkVolume* GetVolumeActor(){return this->Volume;}

  virtual bool UpdateMapper(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  void UpdateMapper(vtkVolumeMapper* mapper,
                    vtkMRMLVolumeRenderingDisplayNode* vspNode);
  void UpdateCPURaycastMapper(vtkFixedPointVolumeRayCastMapper* mapper,
                              vtkMRMLCPURayCastVolumeRenderingDisplayNode* vspNode);
  void UpdateGPURaycastMapper(vtkGPUVolumeRayCastMapper* mapper,
                              vtkMRMLGPURayCastVolumeRenderingDisplayNode* vspNode);
  void UpdateDesiredUpdateRate(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  void UpdateClipping(vtkVolumeMapper* mapper, vtkMRMLVolumeRenderingDisplayNode* vspNode);

  void TransformModified(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  void SetVolumeVisibility(int isVisible);

  /// \return 0: mapper not supported, 1: mapper supported
  int IsMapperSupported(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  virtual int IsMapperSupported(vtkVolumeMapper* volumeMapper,
                                vtkMRMLVolumeRenderingDisplayNode* vspNode);

  virtual void OnMRMLSceneStartClose() VTK_OVERRIDE;
  virtual void OnMRMLSceneEndClose() VTK_OVERRIDE;
  virtual void OnMRMLSceneEndImport() VTK_OVERRIDE;
  virtual void OnMRMLSceneEndRestore() VTK_OVERRIDE;
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node) VTK_OVERRIDE;
  virtual void OnMRMLSceneNodeRemoved(vtkMRMLNode* node) VTK_OVERRIDE;

  void UpdateDisplayNodeList();

  void AddDisplayNode(vtkMRMLVolumeRenderingDisplayNode* dnode);
  void RemoveDisplayNode(vtkMRMLVolumeRenderingDisplayNode* dnode);
  void RemoveDisplayNodes();

public:
  static int DefaultGPUMemorySize;

protected:
  vtkMRMLVolumeRenderingDisplayableManager();
  ~vtkMRMLVolumeRenderingDisplayableManager();

  /// Initialize the displayable manager
  virtual void Create() VTK_OVERRIDE;
  void OnCreate();

  virtual int ActiveInteractionModes() VTK_OVERRIDE;

  // Description:
  // Don't support nested event processing
  // TODO: Probably a bad idea to not support nested calls
  virtual bool EnterMRMLNodesCallback()const VTK_OVERRIDE;

  // Description:
  // Update MRML events
  virtual void ProcessMRMLNodesEvents(vtkObject * caller, unsigned long event, void * callData) VTK_OVERRIDE;

  virtual void OnInteractorStyleEvent(int eventId) VTK_OVERRIDE;

  void OnVolumeRenderingDisplayNodeModified(vtkMRMLVolumeRenderingDisplayNode* dnode);

  void CalculateMatrix(vtkMRMLVolumeRenderingDisplayNode *vspNode, vtkMatrix4x4 *output);
  void EstimateSampleDistance(vtkMRMLVolumeRenderingDisplayNode* vspNode);

  /// Return true if the volume wasn't in the view.
  bool AddVolumeToView();
  void RemoveVolumeFromView();
  void RemoveVolumeFromView(vtkVolume* volume);
  bool IsVolumeInView();
  bool IsVolumeInView(vtkVolume* volume);
  void UpdatePipelineFromDisplayNode(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  int ValidateDisplayNode(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  double GetSampleDistance(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  double GetFramerate(vtkMRMLVolumeRenderingDisplayNode* vspNode);
  virtual vtkIdType GetMaxMemoryInBytes(vtkVolumeMapper* mapper, vtkMRMLVolumeRenderingDisplayNode* vspNode);

protected:
  vtkSlicerVolumeRenderingLogic *VolumeRenderingLogic;

  // Description:
  // The software accelerated software mapper
  vtkFixedPointVolumeRayCastMapper *MapperRaycast;

  // Description:
  // The gpu ray cast mapper.
  vtkGPUVolumeRayCastMapper *MapperGPURaycast3;

  // Description:
  // Actor used for Volume Rendering
  vtkVolume *Volume;

  vtkMRMLVolumeRenderingDisplayNode*    DisplayedNode;

  typedef std::map<std::string, vtkMRMLVolumeRenderingDisplayNode *> DisplayNodesType;
  DisplayNodesType      DisplayNodes;

  vtkIntArray* DisplayObservedEvents;
  // When interaction is >0, we are in interactive mode (low LOD)
  int Interaction;
  double OriginalDesiredUpdateRate;

protected:
  vtkMRMLVolumeRenderingDisplayableManager(const vtkMRMLVolumeRenderingDisplayableManager&); // Not implemented
  void operator=(const vtkMRMLVolumeRenderingDisplayableManager&); // Not implemented

};

#endif
