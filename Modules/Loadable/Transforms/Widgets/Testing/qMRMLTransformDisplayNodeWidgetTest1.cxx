/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Andras Lasso and Franklin King at
  PerkLab, Queen's University and was supported through the Applied Cancer
  Research Unit program of Cancer Care Ontario with funds provided by the
  Ontario Ministry of Health and Long-Term Care.

==============================================================================*/


// QT includes
#include <QApplication>
#include <QTimer>

// Slicer includes
#include "vtkSlicerConfigure.h"

// qMRML includes
#include "qMRMLTransformDisplayNodeWidget.h"

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLTransformNode.h>

// VTK includes
#include <vtkNew.h>
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
#include <QSurfaceFormat>
#include <QVTKOpenGLWidget.h>
#endif

// STD includes

int qMRMLTransformDisplayNodeWidgetTest1(int argc, char * argv [] )
{
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
  // Set default surface format for QVTKOpenGLWidget
  QSurfaceFormat format = QVTKOpenGLWidget::defaultFormat();
  format.setSamples(0);
  QSurfaceFormat::setDefaultFormat(format);
#endif

  QApplication app(argc, argv);

  vtkNew<vtkMRMLScene> scene;
  vtkNew< vtkMRMLTransformNode > transformNode;
  scene->AddNode(transformNode.GetPointer());
  transformNode->CreateDefaultDisplayNodes();

  qMRMLTransformDisplayNodeWidget transformDisplay;
  transformDisplay.setMRMLTransformNode(transformNode.GetPointer());
  transformDisplay.setMRMLScene(scene.GetPointer());
  transformDisplay.show();

  if (argc < 2 || QString(argv[1]) != "-I" )
    {
    QTimer::singleShot(200, &app, SLOT(quit()));
    }
  return app.exec();
}
