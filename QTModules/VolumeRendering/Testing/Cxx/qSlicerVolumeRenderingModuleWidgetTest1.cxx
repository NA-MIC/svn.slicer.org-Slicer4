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

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// Qt includes
#include <QTimer>

// Slicer includes
#include <qSlicerApplication.h>

// Volumes includes
#include "qSlicerVolumeRenderingModule.h"
#include "qSlicerVolumeRenderingModuleWidget.h"

// MRML includes

// VTK includes

//-----------------------------------------------------------------------------
int qSlicerVolumeRenderingModuleWidgetTest1( int argc, char * argv[] )
{
  qSlicerApplication app(argc, argv);

  qSlicerVolumeRenderingModule module;
  module.setMRMLScene(app.mrmlScene());
  module.initialize(0);

  qSlicerVolumeRenderingModuleWidget* moduleWidget =
    dynamic_cast<qSlicerVolumeRenderingModuleWidget*>(
      module.widgetRepresentation());

  moduleWidget->show();

  if (argc < 2 || QString(argv[1]) != "-I")
    {
    QTimer::singleShot(100, qApp, SLOT(quit()));
    }

  return app.exec();
}
