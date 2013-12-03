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

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// QT includes
#include <QApplication>
#include <QTimer>

// qMRML includes
#include "qMRMLSliceControllerWidget.h"
#include "qMRMLSliceView.h"
#include "qMRMLSliceWidget.h"
#include "qMRMLNodeObject.h"

// MRML includes
#include <vtkMRMLAbstractSliceViewDisplayableManager.h>
#include <vtkMRMLColorTableNode.h>
#include <vtkMRMLScene.h>
#include <vtkMRMLSliceLogic.h>
#include <vtkMRMLSliceCompositeNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkMRMLScalarVolumeDisplayNode.h>
#include <vtkMRMLVolumeArchetypeStorageNode.h>

// VTK includes
#include <vtkCollection.h>
#include <vtkNew.h>

vtkMRMLScalarVolumeNode* loadVolume(const char* volume, vtkMRMLScene* scene)
{
  vtkNew<vtkMRMLScalarVolumeDisplayNode> displayNode;
  vtkNew<vtkMRMLScalarVolumeNode> scalarNode;
  vtkNew<vtkMRMLVolumeArchetypeStorageNode> storageNode;

  displayNode->SetAutoWindowLevel(false);
  displayNode->SetInterpolate(false);

  storageNode->SetFileName(volume);
  if (storageNode->SupportedFileType(volume) == 0)
    {
    return 0;
    }
  scalarNode->SetName("foo");
  scalarNode->SetScene(scene);
  displayNode->SetScene(scene);
  //vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
  //displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
  //colorLogic->Delete();
  scene->AddNode(storageNode.GetPointer());
  scene->AddNode(displayNode.GetPointer());
  scalarNode->SetAndObserveStorageNodeID(storageNode->GetID());
  scalarNode->SetAndObserveDisplayNodeID(displayNode->GetID());
  scene->AddNode(scalarNode.GetPointer());
  storageNode->ReadData(scalarNode.GetPointer());

  vtkMRMLColorTableNode* colorNode = vtkMRMLColorTableNode::New();
  colorNode->SetTypeToGrey();
  scene->AddNode(colorNode);
  colorNode->Delete();
  displayNode->SetAndObserveColorNodeID(colorNode->GetID());

  return scalarNode.GetPointer();
}

int qMRMLSliceWidgetTest2(int argc, char * argv [] )
{
  QApplication app(argc, argv);
  if( argc < 2 )
    {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  input_image.nrrd " << std::endl;
    return EXIT_FAILURE;
    }

  vtkNew<vtkMRMLScene> scene;
  vtkMRMLScalarVolumeNode* scalarNode = loadVolume(argv[1], scene.GetPointer());
  if (scalarNode == 0)
    {
    std::cerr << "Not a valid volume: " << argv[1] << std::endl;
    return EXIT_FAILURE;
    }

  QSize viewSize(256, 256);
  qMRMLSliceWidget sliceWidget;
  sliceWidget.setMRMLScene(scene.GetPointer());
  sliceWidget.resize(viewSize.width(), sliceWidget.sliceController()->height() + viewSize.height() );

  vtkMRMLSliceCompositeNode* sliceCompositeNode = sliceWidget.sliceLogic()->GetSliceCompositeNode();
  sliceCompositeNode->SetBackgroundVolumeID(scalarNode->GetID());
  sliceWidget.show();

  qMRMLNodeObject nodeObject(scalarNode->GetDisplayNode(), &sliceWidget);
  nodeObject.setProcessEvents(false);
  nodeObject.setMessage("vtkMRMLDisplayNode");
  for (int i = 0; i < 30; ++i)
    {
    nodeObject.modify();
    }
  nodeObject.setProcessEvents(true);
  nodeObject.setMessage("vtkMRMLDisplayNode + render");
  for (int i = 0; i < 30; ++i)
    {
    nodeObject.modify();
    }

  // test the list of displayable managers
  QStringList expectedDisplayableManagerClassNames =
    QStringList() << "vtkMRMLVolumeGlyphSliceDisplayableManager"
                  << "vtkMRMLModelSliceDisplayableManager"
                  << "vtkMRMLCrosshairDisplayableManager";
  qMRMLSliceView *sliceView = const_cast<qMRMLSliceView*>(sliceWidget.sliceView());
  vtkNew<vtkCollection> collection;
  sliceView->getDisplayableManagers(collection.GetPointer());
  int numManagers = collection->GetNumberOfItems();
  std::cout << "Slice widget slice view has " << numManagers
            << " displayable managers." << std::endl;
  if (numManagers != expectedDisplayableManagerClassNames.size())
    {
    std::cerr << "Incorrect number of displayable managers, expected "
              << expectedDisplayableManagerClassNames.size()
              << " but got " << numManagers << std::endl;
    return EXIT_FAILURE;
    }
  for (int i = 0; i < numManagers; ++i)
    {
    vtkMRMLAbstractSliceViewDisplayableManager *sliceViewDM =
      vtkMRMLAbstractSliceViewDisplayableManager::SafeDownCast(collection->GetItemAsObject(i));
    if (sliceViewDM)
      {
      std::cout << "\tDisplayable manager " << i << " class name = " << sliceViewDM->GetClassName() << std::endl;
      if (!expectedDisplayableManagerClassNames.contains(sliceViewDM->GetClassName()))
        {
        std::cerr << "\t\tnot in expected list!" << std::endl;
        return EXIT_FAILURE;
        }
      }
    else
      {
      std::cerr << "\tDisplayable manager " << i << " is null." << std::endl;
      return EXIT_FAILURE;
      }
    }
  collection->RemoveAllItems();

/*
  QTimer modifyTimer;
  modifyTimer.setInterval(0);
  QObject::connect(&modifyTimer, SIGNAL(timeout()),
                   &nodeObject, SLOT(modify()));
  modifyTimer.start();
*/
  if (argc < 3 || QString(argv[2]) != "-I" )
    {
    QTimer::singleShot(1000, &app, SLOT(quit()));
    }
  return app.exec();
}
