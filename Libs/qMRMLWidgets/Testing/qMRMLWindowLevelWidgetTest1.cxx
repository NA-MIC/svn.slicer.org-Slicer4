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

// QT includes
#include <QApplication>
#include <QHBoxLayout>
#include <QTimer>

// qMRML includes
#include "qMRMLWindowLevelWidget.h"

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLVolumeNode.h>

// VTK includes
#include <vtkSmartPointer.h>

// STD includes
#include <cstdlib>
#include <iostream>

int qMRMLWindowLevelWidgetTest1(int argc, char * argv [] )
{
  QApplication app(argc, argv);
  
  if( argc < 2 )
    {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  inputURL_scene.mrml " << std::endl;
    return EXIT_FAILURE;
    }

  vtkSmartPointer<vtkMRMLScene> scene = vtkSmartPointer<vtkMRMLScene>::New();
  scene->SetURL(argv[1]);
  scene->Connect();
  if (scene->GetNumberOfNodes() == 0)
    {
    std::cerr << "Can't load scene:" << argv[1] << " error: " <<scene->GetErrorMessage() << std::endl;
    return EXIT_FAILURE;
    }
  scene->InitTraversal();
  vtkMRMLNode* node = scene->GetNextNodeByClass("vtkMRMLScalarVolumeNode");
  vtkMRMLVolumeNode* volumeNode = vtkMRMLVolumeNode::SafeDownCast(node);
  if (!volumeNode)
    {
    std::cerr << "Scene must contain a valid vtkMRMLVolumeNode:" << node << std::endl;
    return EXIT_FAILURE;
    }
  
  qMRMLWindowLevelWidget windowLevel;
  windowLevel.setMRMLVolumeNode(volumeNode);
  
  windowLevel.show();

  QWidget topLevel;
  qMRMLWindowLevelWidget windowLevel2(&topLevel);
  windowLevel2.setMRMLVolumeNode(volumeNode);
  QHBoxLayout* l = new QHBoxLayout(&topLevel);
  l->addWidget(&windowLevel2);
  topLevel.show();
  
  if (argc < 3 || QString(argv[2]) != "-I" )
    {
    QTimer::singleShot(200, &app, SLOT(quit()));
    }
  return app.exec();
}

