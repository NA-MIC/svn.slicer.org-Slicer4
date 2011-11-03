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

#ifndef __qSlicerAbstractModuleWidget_h
#define __qSlicerAbstractModuleWidget_h

#if defined(_MSC_VER)
#pragma warning( disable:4250 )
#endif

// CTK includes
#include <ctkPimpl.h>

// SlicerQt includes
#include "qSlicerAbstractModuleRepresentation.h"
#include "qSlicerWidget.h"

#include "qSlicerBaseQTGUIExport.h"

///
/// Base class of all the Slicer module widgets. The widget is added in the module panels.
/// Deriving from qSlicerWidget, it inherits the mrmlScene()/setMRMLScene() methods.
class Q_SLICER_BASE_QTGUI_EXPORT qSlicerAbstractModuleWidget
  :public qSlicerWidget, public qSlicerAbstractModuleRepresentation
{
  Q_OBJECT
public:
  /// Constructor
  /// \sa QWidget
  qSlicerAbstractModuleWidget(QWidget *parent=0);

  /// The enter and exit methods are called when the module panel changes.
  /// These give the module a chance to do any setup or shutdown operations
  /// as it becomes active and inactive.
  virtual void enter() {};
  virtual void exit() {};

protected:
  virtual void setup();
};

#endif
