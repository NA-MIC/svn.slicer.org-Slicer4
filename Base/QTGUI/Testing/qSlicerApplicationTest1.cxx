/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

  See Doc/copyright/copyright.txt
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
#include <QTimer>

// SlicerQt includes
#include "qSlicerApplication.h"
#include "qSlicerCommandOptions.h"

// STD includes
#include <cstdlib>

int qSlicerApplicationTest1(int argc, char * argv[] )
{
  qSlicerApplication app(argc, argv);
  app.setCoreCommandOptions(new qSlicerCommandOptions(app.settings()));
  bool exitWhenDone = false;
  app.parseArguments(exitWhenDone);
  if (exitWhenDone)
    {
    std::cerr << "Line " << __LINE__ << " - Problem with parseArguments()" << std::endl;
    return EXIT_FAILURE;
    }

  if (argc < 2 || QString(argv[1]) != "-I")
    {
    QTimer::singleShot(100, qApp, SLOT(quit()));
    }

  return app.exec();
}

