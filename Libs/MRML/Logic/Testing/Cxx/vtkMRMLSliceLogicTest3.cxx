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
// MRMLLogic includes
#include <vtkMRMLSliceLogic.h>
#include <vtkMRMLSliceLayerLogic.h>

// MRML includes
#include <vtkMRMLColorTableNode.h>
#include <vtkMRMLScalarVolumeDisplayNode.h>
#include <vtkMRMLSliceCompositeNode.h>
#include <vtkMRMLVolumeArchetypeStorageNode.h>

// VTK includes
#include <vtkImageData.h>
#include <vtkImageViewer2.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkTimerLog.h>

// ITK includes
#include <itkConfigure.h>
#if ITK_VERSION_MAJOR > 3
#  include <itkFactoryRegistration.h>
#endif

//-----------------------------------------------------------------------------
vtkMRMLScalarVolumeNode* loadVolume(const char* volume, vtkMRMLScene* scene)
{
  vtkSmartPointer<vtkMRMLScalarVolumeDisplayNode> displayNode = vtkSmartPointer<vtkMRMLScalarVolumeDisplayNode>::New();
  vtkSmartPointer<vtkMRMLScalarVolumeNode> scalarNode = vtkSmartPointer<vtkMRMLScalarVolumeNode>::New();
  vtkSmartPointer<vtkMRMLVolumeArchetypeStorageNode> storageNode = vtkSmartPointer<vtkMRMLVolumeArchetypeStorageNode>::New();

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
  scene->AddNode(storageNode);
  scene->AddNode(displayNode);
  scalarNode->SetAndObserveStorageNodeID(storageNode->GetID());
  scalarNode->SetAndObserveDisplayNodeID(displayNode->GetID());
  scene->AddNode(scalarNode);
  storageNode->ReadData(scalarNode);

  vtkMRMLColorTableNode* colorNode = vtkMRMLColorTableNode::New();
  colorNode->SetTypeToGrey();
  scene->AddNode(colorNode);
  colorNode->Delete();
  displayNode->SetAndObserveColorNodeID(colorNode->GetID());

  return scalarNode.GetPointer();
}

//-----------------------------------------------------------------------------
int vtkMRMLSliceLogicTest3(int argc, char * argv [] )
{
#if ITK_VERSION_MAJOR > 3
  itk::itkFactoryRegistration();
#endif

  if( argc < 2 )
    {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  input_image " << std::endl;
    return EXIT_FAILURE;
    }

  vtkSmartPointer<vtkMRMLScene> scene = vtkSmartPointer<vtkMRMLScene>::New();
  vtkSmartPointer< vtkMRMLSliceLogic > sliceLogic = vtkSmartPointer< vtkMRMLSliceLogic >::New();
  sliceLogic->SetName("Green");
  sliceLogic->SetMRMLScene(scene);
  sliceLogic->ResizeSliceNode(256, 256);
  
  //vtkMRMLSliceNode* sliceNode =sliceLogic->GetSliceNode();
  vtkMRMLSliceCompositeNode* sliceCompositeNode = sliceLogic->GetSliceCompositeNode(); 
  
  vtkSmartPointer<vtkMRMLSliceLayerLogic> sliceLayerLogic = 
    vtkSmartPointer<vtkMRMLSliceLayerLogic>::New();
  
  sliceLogic->SetBackgroundLayer(sliceLayerLogic);

  vtkMRMLScalarVolumeNode* scalarNode = loadVolume(argv[1], scene);
  if (scalarNode == 0 || scalarNode->GetImageData() == 0)
    {
    std::cerr << "Not a valid volume: " << argv[1] << std::endl;
    return EXIT_FAILURE;
    }

  vtkMRMLDisplayNode* displayNode = scalarNode->GetDisplayNode();
  //sliceLayerLogic->SetVolumeNode(scalarNode);
  sliceCompositeNode->SetBackgroundVolumeID(scalarNode->GetID());
  vtkImageData* img = sliceLogic->GetImageData();
  int* dims = img->GetDimensions();
  std::cout << "Logic dimension"  << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  // Not sure why sliceLayerLogic->GetVolumeDisplayNode() is different from displayNode
  //vtkMRMLScalarVolumeDisplayNode* displayNode2 = vtkMRMLScalarVolumeDisplayNode::SafeDownCast(sliceLayerLogic->GetVolumeDisplayNode());

  for (int i = 0; i < 30; ++i)
    {
    vtkSmartPointer<vtkTimerLog> timerLog = vtkSmartPointer<vtkTimerLog>::New();
    timerLog->StartTimer();
    displayNode->Modified();
    timerLog->StopTimer();
    std::cout << "vtkMRMLDisplayNode::Modified(): " << timerLog->GetElapsedTime()
              << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    }
  vtkSmartPointer<vtkImageViewer2> viewer = vtkSmartPointer<vtkImageViewer2>::New();
  viewer->SetInput(sliceLogic->GetImageData());
  //viewer->SetInput(appendComponents->GetOutput());
  
  // Renderer, RenderWindow and Interactor
  vtkRenderWindow* rw = viewer->GetRenderWindow();
  rw->SetSize(dims[0], dims[1]);
  rw->SetMultiSamples(0); // Ensure to have the same test image everywhere
  
  vtkRenderWindowInteractor* ri = vtkRenderWindowInteractor::New();
  viewer->SetupInteractor(ri);
  
  rw->Render();

  for (int i = 0; i < 30; ++i)
    {
    vtkSmartPointer<vtkTimerLog> timerLog = vtkSmartPointer<vtkTimerLog>::New();
    timerLog->StartTimer();
    displayNode->Modified();
    rw->Render();
    timerLog->StopTimer();
    std::cout << "vtkMRMLDisplayNode::Modified() + render: " << timerLog->GetElapsedTime()
              << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    }

  if (argc > 2 && std::string(argv[2]) == "-I")
    {
    ri->Start();
    }

  ri->Delete();

  return EXIT_SUCCESS;
}

