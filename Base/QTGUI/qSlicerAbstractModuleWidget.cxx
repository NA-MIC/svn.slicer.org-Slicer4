/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

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
// Qt includes
#include <QAction>
#include <QIcon>

// SlicerQt includes
#include "qSlicerAbstractModule.h"
#include "qSlicerAbstractModuleWidget.h"

//-----------------------------------------------------------------------------
qSlicerAbstractModuleWidget::qSlicerAbstractModuleWidget(QWidget* parentWidget)
  :qSlicerWidget(parentWidget)
{
}

//-----------------------------------------------------------------------------
void qSlicerAbstractModuleWidget::setup()
{
  const qSlicerAbstractModule* m =
    qobject_cast<const qSlicerAbstractModule*>(this->module());
  this->setWindowTitle(m->title());
  this->setWindowIcon(m->icon());
}
