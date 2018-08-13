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

// Qt includes
#include <QApplication>

// Slicer includes
#include "vtkSlicerConfigure.h"

// SlicerQt includes
#include "qSlicerWidget.h"

// MRML includes
#include <vtkMRMLScene.h>

// VTK includes
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
#include <QSurfaceFormat>
#include <QVTKOpenGLWidget.h>
#endif

// STD includes

int qSlicerWidgetTest1(int argc, char * argv[] )
{
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
  // Set default surface format for QVTKOpenGLWidget
  QSurfaceFormat format = QVTKOpenGLWidget::defaultFormat();
  format.setSamples(0);
  QSurfaceFormat::setDefaultFormat(format);
#endif

  QApplication app(argc, argv);
  qSlicerWidget widget;
  if (widget.mrmlScene() != 0)
    {
    std::cerr << "scene incorrectly initialized." << std::endl;
    return EXIT_FAILURE;
    }
  // check for infinite loop
  QObject::connect(&widget, SIGNAL(mrmlSceneChanged(vtkMRMLScene*)),
                   &widget, SLOT(setMRMLScene(vtkMRMLScene*)));
  vtkMRMLScene* scene = vtkMRMLScene::New();
  widget.setMRMLScene(scene);
  if (widget.mrmlScene() != scene)
    {
    std::cerr << "scene incorrectly set." << std::endl;
    return EXIT_FAILURE;
    }
  scene->Delete();
  if (widget.mrmlScene() != scene)
    {
    std::cerr << "scene has been deleted, qSlicerWidget is supposed to keep a ref on it." << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

