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
#include <vtkMRMLScene.h>
#include <vtkMRMLScalarVolumeDisplayNode.h>
#include <vtkMRMLSliceCompositeNode.h>
#include <vtkMRMLVolumeArchetypeStorageNode.h>

// VTK includes
#include <vtkAlgorithmOutput.h>
#include <vtkImageAppendComponents.h>
#include <vtkImageData.h>
#include <vtkImageCast.h>
#include <vtkImageMapToColors.h>
#include <vtkImageMapToWindowLevelColors.h>
#include <vtkImageReslice.h>
#include <vtkImageThreshold.h>
#include <vtkImageViewer2.h>
#include <vtkLookupTable.h>
#include <vtkNew.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkScalarsToColors.h>
#include <vtkTimerLog.h>
#include <vtkVersion.h>

// ITK includes
#include <itkConfigure.h>
#include <itkFactoryRegistration.h>

//-----------------------------------------------------------------------------
int vtkMRMLSliceLogicTest2(int argc, char * argv [] )
{
  itk::itkFactoryRegistration();
//  vtkMultiThreader::SetGlobalMaximumNumberOfThreads(1);

  if( argc < 2 )
    {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  inputURL_scene.mrml " << std::endl;
    return EXIT_FAILURE;
    }

  vtkNew<vtkMRMLScene> scene;
  vtkNew<vtkMRMLSliceLogic> sliceLogic;
  sliceLogic->SetName("Green");
  sliceLogic->SetMRMLScene(scene.GetPointer());

  vtkMRMLSliceNode* sliceNode =sliceLogic->GetSliceNode();
  vtkMRMLSliceCompositeNode* sliceCompositeNode = sliceLogic->GetSliceCompositeNode();

  vtkNew<vtkMRMLSliceLayerLogic> sliceLayerLogic;

  sliceLogic->SetBackgroundLayer(sliceLayerLogic.GetPointer());

  vtkNew<vtkMRMLScalarVolumeDisplayNode> displayNode;
  vtkNew<vtkMRMLScalarVolumeNode> scalarNode;
  vtkNew<vtkMRMLVolumeArchetypeStorageNode> storageNode;

  displayNode->SetAutoWindowLevel(true);
  displayNode->SetInterpolate(false);

  storageNode->SetFileName(argv[1]);
  if (storageNode->SupportedFileType(argv[1]) == 0)
    {
    return EXIT_FAILURE;
    }
  scalarNode->SetName("foo");
  scalarNode->SetScene(scene.GetPointer());
  displayNode->SetScene(scene.GetPointer());
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

  //sliceLayerLogic->SetVolumeNode(scalarNode);
  sliceCompositeNode->SetBackgroundVolumeID(scalarNode->GetID());
  // Not sure why sliceLayerLogic->GetVolumeDisplayNode() is different from displayNode
  vtkMRMLScalarVolumeDisplayNode* displayNode2 = vtkMRMLScalarVolumeDisplayNode::SafeDownCast(sliceLayerLogic->GetVolumeDisplayNode());

  for (int i = 0; i < 10; ++i)
    {
    vtkNew<vtkTimerLog> timerLog;
    timerLog->StartTimer();
    displayNode->Modified();
    timerLog->StopTimer();
    std::cout << "vtkMRMLDisplayNode::Modified(): " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    timerLog->StartTimer();
    sliceLayerLogic->UpdateTransforms();
    timerLog->StopTimer();
    std::cout << "vtkMRMLSliceLayerLogic::UpdateTransforms(): " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    timerLog->StartTimer();
    sliceLogic->UpdatePipeline();
    timerLog->StopTimer();
    std::cout << "vtkMRMLSliceLogic::UpdatePipeline(): " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    timerLog->StartTimer();
    sliceLayerLogic->UpdateImageDisplay();
#if (VTK_MAJOR_VERSION <= 5)
    sliceLayerLogic->GetImageData();
#else
    sliceLayerLogic->GetImageDataConnection();
#endif
    timerLog->StopTimer();
    std::cout << "vtkMRMLSliceLayerLogic::UpdateImageData(): " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    }

  // Duplicate the pipeline of vtkMRMLScalarVolumeDisplayNode
  vtkNew<vtkImageData> imageData;
  imageData->DeepCopy(displayNode2->GetInputImageData());
  vtkNew<vtkImageReslice> reslice;
  reslice->SetBackgroundColor(0, 0, 0, 0); // only first two are used
  reslice->AutoCropOutputOff();
  reslice->SetOptimization(1);
  reslice->SetOutputOrigin( 0, 0, 0 );
  reslice->SetOutputSpacing( 1, 1, 1 );
  reslice->SetOutputDimensionality( 3 );
  reslice->GenerateStencilOutputOn();
  int dimensions[3];
  sliceNode->GetDimensions(dimensions);

  reslice->SetOutputExtent( 0, dimensions[0]-1,
                            0, dimensions[1]-1,
                            0, dimensions[2]-1);
#if (VTK_MAJOR_VERSION <= 5)
  reslice->SetInput(imageData.GetPointer());
#else
  reslice->SetInputData(imageData.GetPointer());
#endif
  //reslice->SetResliceTransform(sliceLayerLogic->GetXYToIJKTransform());
  vtkNew<vtkImageMapToWindowLevelColors> mapToWindow;
#if (VTK_MAJOR_VERSION <= 5)
  mapToWindow->SetInput(reslice->GetOutput());
#else
  mapToWindow->SetInputConnection(reslice->GetOutputPort());
#endif

  vtkNew<vtkImageMapToColors> mapToColors;
  mapToColors->SetOutputFormatToRGB();
  if (colorNode->GetLookupTable() == 0)
    {
    return EXIT_FAILURE;
    }
  mapToColors->SetLookupTable(colorNode->GetLookupTable());
#if (VTK_MAJOR_VERSION <= 5)
  mapToColors->SetInput(mapToWindow->GetOutput());
#else
  mapToColors->SetInputConnection(mapToWindow->GetOutputPort());
#endif
  //mapToWindow->Update();

  vtkNew<vtkImageThreshold> threshold;
  threshold->SetOutputScalarTypeToUnsignedChar();
#if (VTK_MAJOR_VERSION <= 5)
  threshold->SetInput(reslice->GetOutput());
#else
  threshold->SetInputConnection(reslice->GetOutputPort());
#endif
  threshold->ThresholdBetween( 1, 0 );
  threshold->ReplaceInOn();
  threshold->SetInValue(255);
  threshold->ReplaceOutOn();
  threshold->SetOutValue(255);

  vtkNew<vtkImageCast> resliceAlphaCast;
#if (VTK_MAJOR_VERSION <= 5)
  resliceAlphaCast->SetInput(reslice->GetOutput(1));
#else
  resliceAlphaCast->SetInputConnection(reslice->GetOutputPort(1));
#endif
  resliceAlphaCast->SetOutputScalarTypeToUnsignedChar();

  vtkNew<vtkImageLogic> alphaLogic;
  alphaLogic->SetOperationToAnd();
  alphaLogic->SetOutputTrueValue(255);
#if (VTK_MAJOR_VERSION <= 5)
  alphaLogic->SetInput1(threshold->GetOutput());
  alphaLogic->SetInput2(resliceAlphaCast->GetOutput());
#else
  alphaLogic->SetInputConnection(0, threshold->GetOutputPort());
  alphaLogic->SetInputConnection(1, resliceAlphaCast->GetOutputPort());
#endif

  vtkNew<vtkImageAppendComponents> appendComponents;
  appendComponents->RemoveAllInputs();
#if (VTK_MAJOR_VERSION <= 5)
  appendComponents->AddInputConnection(0, mapToColors->GetOutput()->GetProducerPort() );
  appendComponents->AddInputConnection(0, alphaLogic->GetOutput()->GetProducerPort() );
#else
  appendComponents->AddInputConnection(0, mapToColors->GetOutputPort() );
  appendComponents->AddInputConnection(0, alphaLogic->GetOutputPort() );
#endif

  //displayNode2->GetInput()->SetScalarComponentFromFloat(0, 0, 0, 0, 10.);
  vtkNew<vtkTimerLog> timerLog;
  timerLog->StartTimer();
  appendComponents->Update();
  timerLog->StopTimer();
  std::cout << "vtkMRMLScalarVolumeDisplayNode::pipeline: " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
  for (int i = 0; i < 4; ++i)
    {
    imageData->Modified();
    timerLog->StartTimer();
    appendComponents->Update();
    timerLog->StopTimer();
    std::cout << "vtkMRMLScalarVolumeDisplayNode::pipeline updated: " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;

    timerLog->StartTimer();
    mapToWindow->SetWindow(mapToWindow->GetWindow() + 1);
    appendComponents->Update();
    timerLog->StopTimer();
    std::cout << "vtkMRMLScalarVolumeDisplayNode::window updated: " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;

    timerLog->StartTimer();
    threshold->SetOutValue(threshold->GetOutValue()-1);
    appendComponents->Update();
    timerLog->StopTimer();
    std::cout << "vtkMRMLScalarVolumeDisplayNode::threshold updated: " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;

    timerLog->StartTimer();
    alphaLogic->SetOutputTrueValue(alphaLogic->GetOutputTrueValue()-1);
    appendComponents->Update();
    timerLog->StopTimer();
    std::cout << "vtkMRMLScalarVolumeDisplayNode::alpha updated: " << timerLog->GetElapsedTime() << " fps: " << 1. / timerLog->GetElapsedTime() << std::endl;
    }
  vtkNew<vtkImageViewer2> viewer;
#if (VTK_MAJOR_VERSION <= 5)
  viewer->SetInput(sliceLogic->GetImageData());
#else
  viewer->SetInputConnection(sliceLogic->GetImageDataConnection());
#endif
  //viewer->SetInput(appendComponents->GetOutput());

  // Renderer, RenderWindow and Interactor
  vtkRenderWindow* rw = viewer->GetRenderWindow();
  rw->SetSize(600, 600);
  rw->SetMultiSamples(0); // Ensure to have the same test image everywhere

  vtkRenderWindowInteractor* ri = vtkRenderWindowInteractor::New();
  viewer->SetupInteractor(ri);

  rw->Render();
  if (argc > 2 && std::string(argv[2]) == "-I")
    {
    ri->Start();
    }

  ri->Delete();

  return EXIT_SUCCESS;
}

