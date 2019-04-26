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
#include <QTimer>

// Slicer includes
#include "vtkSlicerConfigure.h" // For Slicer_USE_PYTHONQT

// CTK includes
#ifdef Slicer_USE_PYTHONQT
# include <ctkPythonConsole.h>
#endif

// VTK includes
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
#include <QSurfaceFormat>
#include <QVTKOpenGLWidget.h>
#endif

// SlicerApp includes
#include "qSlicerApplication.h"
#include "qSlicerAppMainWindow.h"
#ifdef Slicer_USE_PYTHONQT
# include "qSlicerPythonManager.h"
#endif

// STD includes

int qSlicerAppMainWindowTest1(int argc, char * argv[] )
{
#ifdef Slicer_VTK_USE_QVTKOPENGLWIDGET
  // Set default surface format for QVTKOpenGLWidget
  QSurfaceFormat format = QVTKOpenGLWidget::defaultFormat();
  format.setSamples(0);
  QSurfaceFormat::setDefaultFormat(format);
#endif

  // Enables resource sharing between the OpenGL contexts used by classes like QOpenGLWidget and QQuickWidget
  QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

  qSlicerApplication app(argc, argv);

  qSlicerAppMainWindow mainWindow;
  mainWindow.show();

#ifdef Slicer_USE_PYTHONQT
  // Create python console
  Q_ASSERT(qSlicerApplication::application()->pythonManager());
  ctkPythonConsole pythonConsole;
  pythonConsole.initialize(qSlicerApplication::application()->pythonManager());
  pythonConsole.resize(600, 280);
#endif

  if (argc < 2 || QString(argv[1]) != "-I")
    {
    QTimer::singleShot(100, qApp, SLOT(quit()));
    }

  return app.exec();
}

