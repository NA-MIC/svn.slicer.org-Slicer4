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

// VolumeRendering includes
#include <vtkMRMLGPURayCastVolumeRenderingDisplayNode.h>
#include <vtkMRMLVolumeRenderingDisplayNode.h>
#include <vtkMRMLVolumeRenderingDisplayableManager.h>
#include <vtkSlicerVolumeRenderingLogic.h>

// MRMLDisplayableManager includes
#include <vtkMRMLDisplayableManagerGroup.h>
#include <vtkThreeDViewInteractorStyle.h>

// MRMLLogic includes
#include <vtkMRMLApplicationLogic.h>

// MRML includes
#include <vtkMRMLScene.h>
#include <vtkMRMLScalarVolumeDisplayNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkMRMLViewNode.h>
#include <vtkMRMLVolumePropertyNode.h>

// VTK includes
#include <vtkCamera.h>
#include <vtkErrorCode.h>
#include <vtkImageData.h>
#include <vtkInteractorEventRecorder.h>
#include <vtkNew.h>
#include <vtkPNGWriter.h>
#include <vtkRegressionTestImage.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>

// STD includes

const char vtkMRMLVolumeRenderingDisplayableManagerTest1EventLog[] =
"# StreamVersion 1\n";

//----------------------------------------------------------------------------
int vtkMRMLVolumeRenderingDisplayableManagerTest1(int argc, char* argv[])
{
  // Renderer, RenderWindow and Interactor
  vtkNew<vtkRenderer> renderer;
  vtkNew<vtkRenderWindow> renderWindow;
  vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
  renderWindow->SetSize(600, 600);
  renderWindow->SetMultiSamples(0); // Ensure to have the same test image everywhere

  renderWindow->AddRenderer(renderer.GetPointer());
  renderWindow->SetInteractor(renderWindowInteractor.GetPointer());

  // Set Interactor Style
  //vtkNew<vtkThreeDViewInteractorStyle> iStyle;
  //renderWindowInteractor->SetInteractorStyle(iStyle.GetPointer());

  // move back far enough to see the reformat widgets
  //renderer->GetActiveCamera()->SetPosition(0,0,-500.);

  // MRML scene
  vtkMRMLScene* scene = vtkMRMLScene::New();

  // Application logic - Handle creation of vtkMRMLSelectionNode and vtkMRMLInteractionNode
  vtkMRMLApplicationLogic* applicationLogic = vtkMRMLApplicationLogic::New();
  applicationLogic->SetMRMLScene(scene);

  // Add ViewNode
  vtkNew<vtkMRMLViewNode> viewNode;
  scene->AddNode(viewNode.GetPointer());

  vtkNew<vtkMRMLDisplayableManagerGroup> displayableManagerGroup;
  displayableManagerGroup->SetRenderer(renderer.GetPointer());
  displayableManagerGroup->SetMRMLDisplayableNode(viewNode.GetPointer());

  vtkNew<vtkMRMLVolumeRenderingDisplayableManager> vrDisplayableManager;
  vrDisplayableManager->SetMRMLApplicationLogic(applicationLogic);
  displayableManagerGroup->AddDisplayableManager(vrDisplayableManager.GetPointer());

  vtkNew<vtkMRMLScalarVolumeNode> volumeNode;
  vtkNew<vtkImageData> imageData;
  imageData->SetDimensions(3, 3, 3);
  imageData->SetScalarTypeToUnsignedChar();
  imageData->SetNumberOfScalarComponents(1);
  imageData->AllocateScalars();
  unsigned char* ptr = reinterpret_cast<unsigned char*>(
    imageData->GetScalarPointer(0,0,0));
  for (int z = 0; z < 3; ++z)
    {
    for (int y = 0; y < 3; ++y)
      {
      for (int x = 0; x < 3; ++x)
        {
        double normalizedIntensity = (static_cast<double>(x+(y*3)+(z*3*3)) / static_cast<double>(3*3*3 - 1));
        std::cout << x << " " << y << " " << z << ": " << normalizedIntensity << std::endl;
        *(ptr++) = 255 - static_cast<unsigned char>(255. * normalizedIntensity);
        }
      }
    }
  volumeNode->SetAndObserveImageData(imageData.GetPointer());
  scene->AddNode(volumeNode.GetPointer());

  vtkNew<vtkMRMLVolumePropertyNode> volumePropertyNode;
  scene->AddNode(volumePropertyNode.GetPointer());

  vtkNew<vtkMRMLGPURayCastVolumeRenderingDisplayNode> vrDisplayNode;
  vrDisplayNode->SetAndObserveVolumeNodeID(volumeNode->GetID());
  vrDisplayNode->SetAndObserveVolumePropertyNodeID(volumePropertyNode->GetID());

  displayableManagerGroup->GetInteractor()->Initialize();

  scene->AddNode(vrDisplayNode.GetPointer());

  vtkNew<vtkMRMLScalarVolumeDisplayNode> volumeDisplayNode;
  volumeDisplayNode->SetThreshold(10, 245);
  volumeDisplayNode->SetApplyThreshold(1);
  volumeDisplayNode->SetScalarRange(0, 255);
  volumeDisplayNode->SetAutoWindowLevel(0);
  scene->AddNode(volumeDisplayNode.GetPointer());

  volumeNode->AddAndObserveDisplayNodeID(volumeDisplayNode->GetID());

  vtkNew<vtkSlicerVolumeRenderingLogic> vrLogic;
  vrLogic->SetMRMLScene(scene);

  vrLogic->CopyScalarDisplayToVolumeRenderingDisplayNode(
    vrDisplayNode.GetPointer());

  vrDisplayNode->SetFollowVolumeDisplayNode(1);
  volumeDisplayNode->SetThreshold(128, 245);
  volumeDisplayNode->SetWindowLevelMinMax(128, 245);

  // TODO: Automatically move the camera (simulating movements)
  // to have a good screenshot.
  renderer->SetBackground(0, 169. / 255, 79. /255);
  renderer->SetBackground2(0, 83. / 255, 155. /255);
  renderer->SetGradientBackground(true);
  renderer->ResetCamera();
  
  // Event recorder
  bool disableReplay = false, record = false, screenshot = false;
  for (int i = 0; i < argc; i++)
    {
    disableReplay |= (strcmp("--DisableReplay", argv[i]) == 0);
    record        |= (strcmp("--Record", argv[i]) == 0);
    screenshot    |= (strcmp("--Screenshot", argv[i]) == 0);
    }
  vtkNew<vtkInteractorEventRecorder> recorder;
  recorder->SetInteractor(displayableManagerGroup->GetInteractor());
  if (!disableReplay)
    {
    if (record)
      {
      std::cout << "Recording ..." << std::endl;
      recorder->SetFileName("vtkInteractorEventRecorder.log");
      recorder->On();
      recorder->Record();
      }
    else
      {
      // Play
      recorder->ReadFromInputStringOn();
      recorder->SetInputString(vtkMRMLVolumeRenderingDisplayableManagerTest1EventLog);
      recorder->Play();
      }
    }

  int retval = vtkRegressionTestImageThreshold(renderWindow.GetPointer(), 85.0);
  if ( record || retval == vtkRegressionTester::DO_INTERACTOR)
    {
    displayableManagerGroup->GetInteractor()->Initialize();
    displayableManagerGroup->GetInteractor()->Start();
    }

  if (record || screenshot)
    {
    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(renderWindow.GetPointer());
    windowToImageFilter->SetMagnification(1); //set the resolution of the output image
    windowToImageFilter->Update();

    vtkNew<vtkTesting> testHelper;
    testHelper->AddArguments(argc, const_cast<const char **>(argv));

    vtkStdString screenshootFilename = testHelper->GetDataRoot();
    screenshootFilename += "/Baseline/vtkMRMLCameraDisplayableManagerTest1.png";
    vtkNew<vtkPNGWriter> writer;
    writer->SetFileName(screenshootFilename.c_str());
    writer->SetInput(windowToImageFilter->GetOutput());
    writer->Write();
    std::cout << "Saved screenshot: " << screenshootFilename << std::endl;
    }

  vrDisplayableManager->SetMRMLApplicationLogic(0);
  applicationLogic->Delete();
  scene->Delete();

  return !retval;
}

