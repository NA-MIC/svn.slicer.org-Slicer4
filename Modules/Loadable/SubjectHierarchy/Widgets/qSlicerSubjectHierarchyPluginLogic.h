/*==============================================================================

  Program: 3D Slicer

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

#ifndef __qSlicerSubjectHierarchyPluginLogic_h
#define __qSlicerSubjectHierarchyPluginLogic_h

// SlicerQt includes
#include "qSlicerObject.h"

// CTK includes
#include <ctkPimpl.h>
#include <ctkVTKObject.h>

// Qt includes
#include <QObject>

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLSubjectHierarchyNode.h>

#include "qSlicerSubjectHierarchyModuleWidgetsExport.h"

class qSlicerSubjectHierarchyPluginLogicPrivate; 
class qSlicerSubjectHierarchyAbstractPlugin;

/// \ingroup Slicer_QtModules_SubjectHierarchy
/// \brief Qt-based logic class to exercise Qt-related logic functions. The Subject
///   hierarchy plugin mechanism and those are Qt classes
class Q_SLICER_MODULE_SUBJECTHIERARCHY_WIDGETS_EXPORT qSlicerSubjectHierarchyPluginLogic :
  public QObject, public virtual qSlicerObject
{
  Q_OBJECT
  QVTK_OBJECT

  /// Flag determining whether children of subject hierarchy nodes are automatically
  /// deleted upon deleting a parent subject hierarchy node.
  /// By default, a pop-up question asking the user to confirm the deletion of
  /// children nodes will be shown.
  Q_PROPERTY (bool autoDeleteSubjectHierarchyChildren READ autoDeleteSubjectHierarchyChildren WRITE setAutoDeleteSubjectHierarchyChildren)

public:
  typedef QObject Superclass;
  qSlicerSubjectHierarchyPluginLogic(QWidget *parent=0);
  virtual ~qSlicerSubjectHierarchyPluginLogic();

  typedef vtkMRMLSubjectHierarchyNode::SubjectHierarchyItemID SubjectHierarchyItemID;

public:
  /// Set the current MRML scene to the logic
  virtual void setMRMLScene(vtkMRMLScene* scene);

  /// Assessor function for getting subject hierarchy plugin by name (for python)
  Q_INVOKABLE qSlicerSubjectHierarchyAbstractPlugin* subjectHierarchyPluginByName(QString name)const;

  /// Assessor function for getting currently selected subject hierarchy item (for python)
  Q_INVOKABLE SubjectHierarchyItemID currentSubjectHierarchyItem()const;

  /// Utility test function to be able to set currently selected subject hierarchy node from python
  /// Only used in python tests!
  Q_INVOKABLE void setCurrentSubjectHierarchyItem(SubjectHierarchyItemID itemID);

  /// Register subject hierarchy core plugins
  /// Note: Registering plugins provided by other modules is the responsibility
  ///   of the module!
  void registerCorePlugins();

  bool autoDeleteSubjectHierarchyChildren()const;
  void setAutoDeleteSubjectHierarchyChildren(bool flag);

protected:
  /// Add supported nodes to subject hierarchy.
  /// This method is called if auto-creation is enabled and a supported node is added to the
  /// scene, or if the user answers yes to the question that pops up upon entering subject
  /// hierarchy module if supported nodes are found that are not in the hierarchy.
  void addSupportedNodesToSubjectHierarchy();

protected slots:
  /// Called when a node is added to the scene so that a plugin can claim it
  void onNodeAdded(vtkObject* scene, vtkObject* nodeObject);
  /// Called when a node is removed to the scene so that the associated
  /// subject hierarchy node can be deleted too
  void onNodeAboutToBeRemoved(vtkObject* scene, vtkObject* nodeObject);
  /// Called when scene import is finished.
  /// Subject hierarchy nodes are created for supported data nodes if they have not
  /// been imported with the scene (backwards compatibility for older scenes)
  void onSceneImportEnded(vtkObject* sceneObject);
  /// Called when scene end is finished. Hierarchy is cleared in that case.
  void onSceneCloseEnded(vtkObject* sceneObject);

  /// Called when hierarchy modified event is invoked for a data node
  /// This method ensures that model hierarchy changes are reflected in the subject hierarchy as well
  void onMRMLNodeHierarchyModified(vtkObject* nodeObject);

protected:
  QScopedPointer<qSlicerSubjectHierarchyPluginLogicPrivate> d_ptr; 
  
private:
  Q_DECLARE_PRIVATE(qSlicerSubjectHierarchyPluginLogic);
  Q_DISABLE_COPY(qSlicerSubjectHierarchyPluginLogic);
};

#endif
