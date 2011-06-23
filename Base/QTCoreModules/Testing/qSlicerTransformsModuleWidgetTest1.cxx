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

  This file was originally developed by Luis Ibanez, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#include "qSlicerCoreApplication.h"
#include "qSlicerTransformsModuleWidget.h"

#include <stdlib.h>

#include "TestingMacros.h"

int qSlicerTransformsModuleWidgetTest1(int argc, char * argv [] )
{
  qSlicerCoreApplication app(argc, argv);

  qSlicerCoreApplication * aptr = app.application();

  if( aptr != (&app) )
    {
    std::cerr << "Problem with the application() singleton" << std::endl;
    return EXIT_FAILURE;
    }

  qSlicerTransformsModuleWidget * transformsModuleWidget = new qSlicerTransformsModuleWidget;

  delete transformsModuleWidget;

  return EXIT_SUCCESS;
}

