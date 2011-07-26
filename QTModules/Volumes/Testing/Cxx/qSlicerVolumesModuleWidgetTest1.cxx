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
#include <QWidget>

// SlicerQt includes
#include <qSlicerAbstractModuleRepresentation.h>
#include <qSlicerApplication.h>

// Volumes includes
#include "qSlicerVolumesModule.h"
#include "vtkSlicerVolumesLogic.h"

// MRML includes

// VTK includes
#include <vtkNew.h>

//-----------------------------------------------------------------------------
int qSlicerVolumesModuleWidgetTest1( int argc, char * argv[] )
{
  qSlicerApplication app(argc, argv);

  if (argc < 2)
    {
    std::cerr << "Usage: qSlicerVolumesModuleWidgetTest1 volumeName [-I]" << std::endl;
    return EXIT_FAILURE;
    }

  qSlicerVolumesModule module;
  module.initialize(0);

  vtkNew<vtkMRMLScene> scene;
  vtkNew<vtkSlicerVolumesLogic> volumesLogic;
  volumesLogic->SetMRMLScene(scene.GetPointer());

  vtkMRMLVolumeNode* volumeNode = volumesLogic->AddArchetypeVolume(argv[1], "volume");
  if (!volumeNode)
    {
    std::cerr << "Bad volume file:" << argv[1] << std::endl;
    return EXIT_FAILURE;
    }
  module.setMRMLScene(scene.GetPointer());
  
  dynamic_cast<QWidget*>(module.widgetRepresentation())->show();

  if (argc < 3 || QString(argv[2]) != "-I")
    {
    QTimer::singleShot(200, &app, SLOT(quit()));
    }
  return app.exec();
}
