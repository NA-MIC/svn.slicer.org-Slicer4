/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See COPYRIGHT.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/

#ifndef __vtkSlicerCLIModuleLogic_h
#define __vtkSlicerCLIModuleLogic_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

// MRMLCLI includes
#include "vtkMRMLCommandLineModuleNode.h"
class ModuleDescription;
class ModuleParameter;

// MRML include
#include "vtkMRMLScene.h"

// STL includes
#include <string>

#include "qSlicerBaseQTCLIExport.h"

typedef enum { CommandLineModule, SharedObjectModule, PythonModule } CommandLineModuleType;

/// \brief Logic for running CLI
///
/// vtkSlicerCLIModuleLogic logic allows to run a either synchronously or asynchronously CLI
/// using parameters of a \a vtkMRMLCommandLineModuleNode.
/// While a CLI module logic can run any CLI node, the logic can uniquely be
/// assigned a specific CLI by setting a DefaultModuleDescription.
class Q_SLICER_BASE_QTCLI_EXPORT vtkSlicerCLIModuleLogic :
  public vtkSlicerModuleLogic
{
public:
  static vtkSlicerCLIModuleLogic *New();
  vtkTypeMacro(vtkSlicerCLIModuleLogic,vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// The default module description is used when creating new nodes.
  /// \sa CreateNode()
  void SetDefaultModuleDescription(const ModuleDescription& moduleDescription);
  const ModuleDescription& GetDefaultModuleDescription()const;

  /// Instantiate a default command line module node.
  /// If a default module description has been given, the node is initialized
  /// with the description.
  /// Warning: The caller is responsible for deleting it.
  /// \sa CreateNodeInScene(), SetDefaultModuleDescription()
  vtkMRMLCommandLineModuleNode* CreateNode();

  /// Instantiate a default command line module node and add it into the
  /// scene.
  /// The caller is responsible for remove the node from the scene.
  /// \sa CreateNode(), SetDefaultModuleDescription()
  vtkMRMLCommandLineModuleNode* CreateNodeInScene();

  // TODO: do we need to observe MRML here?
  virtual void ProcessMrmlEvents(vtkObject * vtkNotUsed(caller),
                                 unsigned long vtkNotUsed(event),
                                 void * vtkNotUsed(callData)){}

  /// For debugging, control deletion of temp files
  virtual void DeleteTemporaryFilesOn();
  virtual void DeleteTemporaryFilesOff();
  void SetDeleteTemporaryFiles(int value);
  int GetDeleteTemporaryFiles() const;

  /// For debugging, control redirection of cout and cerr
  virtual void RedirectModuleStreamsOn();
  virtual void RedirectModuleStreamsOff();
  void SetRedirectModuleStreams(int value);
  int GetRedirectModuleStreams() const;

  /// Schedules the command line module to run.
  /// The CLI is scheduled to be run in a separate thread. This methods
  /// is non blocking and returns immediately.
  /// If \a updateDisplay is 'true' the selection node will be updated with the
  /// the created nodes, which would automatically select the created nodes
  /// in the node selectors.
  void Apply( vtkMRMLCommandLineModuleNode* node, bool updateDisplay = true );

  /// Don't start the CLI in a separate thread, but run it in the main thread.
  /// This methods is blocking until the CLI finishes to execute, the UI being
  /// frozen until that time.
  /// If \a updateDisplay is 'true' the selection node will be updated with the
  /// the created nodes, which would automatically select the created nodes
  /// in the node selectors.
  void ApplyAndWait ( vtkMRMLCommandLineModuleNode* node, bool updateDisplay = true);

  /// Set/Get the directory to use for temporary files
  void SetTemporaryDirectory(const char *tempdir);

//   void LazyEvaluateModuleTarget(ModuleDescription& moduleDescriptionObject);
//   void LazyEvaluateModuleTarget(vtkMRMLCommandLineModuleNode* node) 
//     { this->LazyEvaluateModuleTarget(node->GetModuleDescription()); }

protected:
  /// Reimplemented to observe NodeAddedEvent.
  virtual void SetMRMLSceneInternal(vtkMRMLScene * newScene);
  /// Reimplemented for AutoRun mode.
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node);
  /// Reimplemented to observe CLI node.
  virtual void ProcessMRMLNodesEvents(vtkObject *caller, unsigned long event,
                                      void *callData);
  /// Reimplemented to observe vtkSlicerApplicationLogic.
  void ProcessMRMLLogicsEvents(vtkObject*, long unsigned int, void*);


  std::string ConstructTemporaryFileName(const std::string& tag,
                                         const std::string& type,
                                         const std::string& name,
                                     const std::vector<std::string>& extensions,
                                     CommandLineModuleType commandType);
  std::string ConstructTemporarySceneFileName(vtkMRMLScene *scene);
  std::string FindHiddenNodeID(const ModuleDescription& d,
                               const ModuleParameter& p);

  // The method that runs the command line module
  void ApplyTask(void *clientdata);

  // Communicate progress back to the node
  static void ProgressCallback(void *);

  /// Return true if the commandlinemodule node can update the
  /// selection node with the outputs of the CLI
  bool IsCommandLineModuleNodeUpdatingDisplay(
    vtkMRMLCommandLineModuleNode* commandLineModuleNode)const;

  /// Call apply because the node requests it.
  void AutoRun(vtkMRMLCommandLineModuleNode* cliNode);

private:
  vtkSlicerCLIModuleLogic();
  virtual ~vtkSlicerCLIModuleLogic();
  vtkSlicerCLIModuleLogic(const vtkSlicerCLIModuleLogic&);
  void operator=(const vtkSlicerCLIModuleLogic&);

  class vtkInternal;
  vtkInternal * Internal;
};

#endif

