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

  This file was originally developed by Johan Andruejol, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// Qt includes
#include <QTimer>

// Slicer includes
#include <qSlicerApplication.h>

// Volumes includes
#include "qSlicerUnitsModule.h"
#include "qSlicerUnitsModuleWidget.h"

// MRML includes

// VTK includes

//-----------------------------------------------------------------------------
int qSlicerUnitsModuleWidgetTest1( int argc, char * argv[] )
{
  qSlicerApplication app(argc, argv);

  qSlicerUnitsModule module;
  module.setMRMLScene(app.mrmlScene());
  module.initialize(0);

  qSlicerUnitsModuleWidget* moduleWidget =
    dynamic_cast<qSlicerUnitsModuleWidget*>(
      module.widgetRepresentation());

  moduleWidget->show();

  if (argc < 2 || QString(argv[1]) != "-I")
    {
    QTimer::singleShot(100, qApp, SLOT(quit()));
    }

  return app.exec();
}
