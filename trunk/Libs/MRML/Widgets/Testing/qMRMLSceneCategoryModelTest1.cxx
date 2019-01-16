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

// QT includes
#include <QApplication>
#include <QTimer>
#include <QTreeView>

// Slicer includes
#include "vtkSlicerConfigure.h"

// CTK includes

// qMRML includes
#include "qMRMLNodeFactory.h"
#include "qMRMLSceneCategoryModel.h"

// MRML includes
#include <vtkMRMLScene.h>

// VTK includes
#include <vtkNew.h>
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
#include <QSurfaceFormat>
#include <QVTKOpenGLWidget.h>
#endif

// STD includes

int qMRMLSceneCategoryModelTest1(int argc, char * argv [])
{
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
  // Set default surface format for QVTKOpenGLWidget
  QSurfaceFormat format = QVTKOpenGLWidget::defaultFormat();
  format.setSamples(0);
  QSurfaceFormat::setDefaultFormat(format);
#endif

  QApplication app(argc, argv);

  qMRMLSceneCategoryModel model;

  vtkNew<vtkMRMLScene> scene;
  qMRMLNodeFactory nodeFactory(0);
  nodeFactory.setMRMLScene(scene.GetPointer());
  nodeFactory.createNode("vtkMRMLROINode");
  nodeFactory.addAttribute("Category", "First Category");
  nodeFactory.createNode("vtkMRMLCameraNode");
  nodeFactory.createNode("vtkMRMLViewNode");
  nodeFactory.createNode("vtkMRMLLinearTransformNode");
  nodeFactory.removeAttribute("Category");
  nodeFactory.createNode("vtkMRMLDoubleArrayNode");
  model.setMRMLScene(scene.GetPointer());
  nodeFactory.createNode("vtkMRMLScalarVolumeNode");
  nodeFactory.addAttribute("Category", "Second Category");
  nodeFactory.createNode("vtkMRMLSliceNode");
  nodeFactory.createNode("vtkMRMLSliceNode");
  nodeFactory.addAttribute("Category", "Third Category");
  nodeFactory.createNode("vtkMRMLViewNode");
  nodeFactory.createNode("vtkMRMLViewNode");

  QStringList scenePreItems =
    QStringList() << "pre 1" << "pre 2" << "separator";
  model.setPreItems(scenePreItems, 0);
  model.setPreItems(scenePreItems, model.mrmlSceneItem());

  if (model.itemFromCategory("Second Category") == 0 ||
      model.itemFromCategory("Second Category") == model.mrmlSceneItem())
    {
    std::cerr << "Wrong category. Item: "
              << model.itemFromCategory("Second Category")
              << " scene item: " << model.mrmlSceneItem() << std::endl;
    return EXIT_FAILURE;
    }
  model.setPreItems(scenePreItems, model.itemFromCategory("Second Category"));

  QTreeView* view = new QTreeView(0);
  view->setModel(&model);
  view->show();
  view->resize(500, 800);

  if (argc < 2 || QString(argv[1]) != "-I" )
    {
    QTimer::singleShot(200, &app, SLOT(quit()));
    }

  return app.exec();
}

