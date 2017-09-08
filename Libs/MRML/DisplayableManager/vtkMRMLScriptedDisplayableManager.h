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

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __vtkMRMLScriptedDisplayableManager_h
#define __vtkMRMLScriptedDisplayableManager_h

// MRMLDisplayableManager includes
#include "vtkMRMLAbstractDisplayableManager.h"

#include "vtkMRMLDisplayableManagerWin32Header.h"

class VTK_MRML_DISPLAYABLEMANAGER_EXPORT vtkMRMLScriptedDisplayableManager :
  public vtkMRMLAbstractDisplayableManager
{

public:
  static vtkMRMLScriptedDisplayableManager* New();
  vtkTypeMacro(vtkMRMLScriptedDisplayableManager,vtkMRMLAbstractDisplayableManager);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  void SetPythonSource(const std::string& pythonSource);

protected:
  vtkMRMLScriptedDisplayableManager();
  virtual ~vtkMRMLScriptedDisplayableManager();

  virtual void SetMRMLSceneInternal(vtkMRMLScene* newScene) VTK_OVERRIDE;

  virtual void ProcessMRMLSceneEvents(vtkObject *caller, unsigned long event, void *callData) VTK_OVERRIDE;
  virtual void ProcessMRMLNodesEvents(vtkObject *caller, unsigned long event, void *callData) VTK_OVERRIDE;

  virtual void Create() VTK_OVERRIDE;

  virtual void RemoveMRMLObservers() VTK_OVERRIDE;

  virtual void UpdateFromMRML() VTK_OVERRIDE;

  virtual void OnInteractorStyleEvent(int eventid) VTK_OVERRIDE;

  virtual void OnMRMLDisplayableNodeModifiedEvent(vtkObject* caller) VTK_OVERRIDE;

private:
  vtkMRMLScriptedDisplayableManager(const vtkMRMLScriptedDisplayableManager&);// Not implemented
  void operator=(const vtkMRMLScriptedDisplayableManager&);                   // Not Implemented

  class vtkInternal;
  vtkInternal * Internal;
};

#endif
