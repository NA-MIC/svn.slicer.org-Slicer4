#include <iostream>
#include <vector>
#include <string>

#include "vtkMRMLScene.h"
#include "vtkMRMLVolumeArchetypeStorageNode.h"
#include "vtkEMSegmentLogic.h"
#include "vtkMRMLEMSSegmenterNode.h"
#include "EMSegmentCommandLineCLP.h"
#include <vtksys/SystemTools.hxx>
#include <stdexcept>

#include "vtkImageMathematics.h"
#include "vtkImageAccumulate.h"
#include "vtkITKArchetypeImageSeriesReader.h"
#include "vtkITKArchetypeImageSeriesScalarReader.h"
#include "vtkImageData.h"

vtkMRMLVolumeNode* 
AddScalarArchetypeVolume(vtkMRMLScene* mrmlScene,
                         const char* filename, 
                         int centerImage, 
                         int labelMap, 
                         const char* volname)
{
  vtkMRMLVolumeNode        *volumeNode   = NULL;
  vtkMRMLVolumeDisplayNode *displayNode  = NULL;
  vtkMRMLScalarVolumeNode  *scalarNode   = vtkMRMLScalarVolumeNode::New();

  // i/o mechanism
  vtkMRMLVolumeArchetypeStorageNode *storageNode = 
    vtkMRMLVolumeArchetypeStorageNode::New();
  storageNode->SetFileName(filename);
  storageNode->SetCenterImage(centerImage);

  // try to read the image
  if (storageNode->ReadData(scalarNode))
    {
    displayNode = vtkMRMLVolumeDisplayNode::New();
    scalarNode->SetLabelMap(labelMap);
    volumeNode  = scalarNode;
    }

  if (volumeNode != NULL)
    {
    // set the volume's name
    if (volname == NULL)
      {
      const vtksys_stl::string fname(filename);
      vtksys_stl::string name = vtksys::SystemTools::GetFilenameName(fname);
      volumeNode->SetName(name.c_str());
      }
    else
      {
      volumeNode->SetName(volname);
      }

    volumeNode->SetScene(mrmlScene);  
    storageNode->SetScene(mrmlScene); 
    displayNode->SetScene(mrmlScene); 

    // set basic display info
    double range[2];
    volumeNode->GetImageData()->GetScalarRange(range);
    displayNode->SetLowerThreshold(range[0]);
    displayNode->SetUpperThreshold(range[1]);
    displayNode->SetWindow(range[1] - range[0]);
    displayNode->SetLevel(0.5 * (range[1] + range[0]) );

    // add nodes to scene
    mrmlScene->AddNode(storageNode);  
    mrmlScene->AddNode(displayNode);  

    volumeNode->SetStorageNodeID(storageNode->GetID());
    volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());

    mrmlScene->AddNode(volumeNode);

    }

  scalarNode->Delete();
  storageNode->Delete();
  if (displayNode)
    {
    displayNode->Delete();
    }
  return volumeNode;  
}

//
// This function checks to see if the image stored in standardFilename
// differs from resultData.  True is returned if the images differ,
// false is returned if they are identical.
bool ImageDiff(vtkImageData* resultData, std::string standardFilename)
{
  bool imagesDiffer = false;

  //
  // read segmentation result standard
  vtkITKArchetypeImageSeriesReader* standardReader = 
    vtkITKArchetypeImageSeriesScalarReader::New();
  standardReader->SetArchetype(standardFilename.c_str());
  standardReader->SetOutputScalarTypeToNative();
  standardReader->SetDesiredCoordinateOrientationToNative();
  standardReader->SetUseNativeOriginOn();
  try
  {
    standardReader->Update();
  }
  catch (...)
  {
    std::cerr << "Error reading standard image: " << std::endl;
    standardReader->Delete();
    return true;
  }

  //
  // compare image origin and spacing
  for (unsigned int i = 0; i < 3; ++i)
  {
    if (resultData->GetSpacing()[i] != 
        standardReader->GetOutput()->GetSpacing()[i] ||
        resultData->GetOrigin()[i] != 
        standardReader->GetOutput()->GetOrigin()[i])
    {
      //
      // display spacing and origin info for resultData
      std::cerr << "Image spacing and/or origin does not match standard!" 
                << std::endl;
      std::cerr << "result origin: " 
                << resultData->GetOrigin()[0] << " "
                << resultData->GetOrigin()[1] << " "
                << resultData->GetOrigin()[2] << std::endl;
      std::cerr << "result spacing: " 
                << resultData->GetSpacing()[0] << " "
                << resultData->GetSpacing()[1] << " "
                << resultData->GetSpacing()[2] << std::endl;

      std::cerr << "Standard origin: " 
                << standardReader->GetOutput()->GetOrigin()[0] << " "
                << standardReader->GetOutput()->GetOrigin()[1] << " "
                << standardReader->GetOutput()->GetOrigin()[2] << std::endl;
      std::cerr << "Standard spacing: " 
                << standardReader->GetOutput()->GetSpacing()[0] << " "
                << standardReader->GetOutput()->GetSpacing()[1] << " "
                << standardReader->GetOutput()->GetSpacing()[2] << std::endl;
      imagesDiffer = true;
    }
  }
  if (!imagesDiffer)
  {
    std::cerr << "Result image origin and spacing match." << std::endl;
  }

  //
  // compare image voxels
  vtkImageMathematics* imageDifference = vtkImageMathematics::New();
  imageDifference->SetOperationToSubtract();
  imageDifference->SetInput1(resultData);
  imageDifference->SetInput2(standardReader->GetOutput());

  vtkImageAccumulate* differenceAccumulator = vtkImageAccumulate::New();
  differenceAccumulator->SetInputConnection(imageDifference->GetOutputPort());
  //differenceAccumulator->IgnoreZeroOn();
  differenceAccumulator->Update();
  
  //imagesDiffer = differenceAccumulator->GetVoxelCount() > 0;
  imagesDiffer = 
    differenceAccumulator->GetMin()[0] != 0.0 ||
    differenceAccumulator->GetMax()[0] != 0.0;
    
  if (imagesDiffer)
  {
    std::cerr << "((temporarily not) ignoring zero) Num / Min / Max / Mean difference = " 
              << differenceAccumulator->GetVoxelCount()  << " / "
              << differenceAccumulator->GetMin()[0]      << " / "
              << differenceAccumulator->GetMax()[0]      << " / "
              << differenceAccumulator->GetMean()[0]     << std::endl;
  }
  else
  {
    std::cerr << "Result image voxels match." << std::endl;
  }

  standardReader->Delete();
  imageDifference->Delete();
  differenceAccumulator->Delete();

  return imagesDiffer;
}

int main(int argc, char** argv)
{
  //
  // parse arguments using the CLP system; this creates variables.
  PARSE_ARGS;

  bool useDefaultParametersNode = parametersMRMLNodeName.empty();
  bool useDefaultTarget         = targetVolumeFileNames.empty();
  bool useDefaultOutput         = resultVolumeFileName.empty();
  bool segmentationSucceeded    = true;

  //
  // make sure arguments are sufficient and unique
  bool argsOK = true;
  if (mrmlSceneFileName.empty())
    {
    std::cerr << "Error: mrmlSceneFileName must be specified." << std::endl;
    argsOK = false;
    }
  if (!argsOK)
    {
    std::cerr << "Try --help for usage..." << std::endl;
    exit(EXIT_FAILURE);
    }

  //
  // make sure files exist
  if (!vtksys::SystemTools::FileExists(mrmlSceneFileName.c_str()))
    {
    std::cerr << "Error: MRML scene file does not exist." << std::endl;
      std::cerr << mrmlSceneFileName << std::endl;      
    exit(EXIT_FAILURE);
    }

  if (!resultStandardVolumeFileName.empty() && 
      !vtksys::SystemTools::FileExists(resultStandardVolumeFileName.c_str()))
  {
    std::cerr << "Error: result standard volume file does not exist." 
              << std::endl;
      std::cerr << resultStandardVolumeFileName << std::endl;      
    exit(EXIT_FAILURE);
  }

  for (unsigned int i = 0; i < targetVolumeFileNames.size(); ++i)
  {
    if (!vtksys::SystemTools::
        FileExists(targetVolumeFileNames[i].c_str()))
    {
      std::cerr << "Error: target volume file " << i << " does not exist." 
                << std::endl;
      std::cerr << targetVolumeFileNames[i] << std::endl;      
      exit(EXIT_FAILURE);
    }
  }

  //
  // create a mrml scene that will hold the parameters and data
  vtkMRMLScene* mrmlScene = vtkMRMLScene::New();
  vtkMRMLScene::SetActiveScene(mrmlScene);
  mrmlScene->SetURL(mrmlSceneFileName.c_str());

  //
  // create an instance of vtkEMSegmentLogic and connect it with the
  // MRML scene
  vtkEMSegmentLogic* emLogic = vtkEMSegmentLogic::New();
  emLogic->SetAndObserveMRMLScene(mrmlScene);
  vtkIntArray *emsEvents = vtkIntArray::New();
  emsEvents->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  emsEvents->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  emLogic->SetAndObserveMRMLSceneEvents(mrmlScene, emsEvents);
  emsEvents->Delete();
  emLogic->RegisterMRMLNodesWithScene();

  //
  // global try block makes sure data is cleaned up if anything goes
  // wrong
  try 
    {
    //
    // read the mrml scene
    try 
      {
      if (verbose) std::cerr << "Reading MRML scene...";
      mrmlScene->Connect();
      if (verbose) std::cerr << "DONE" << std::endl;
      }
    catch (...)
      {
      throw std::runtime_error("ERROR: failed to import mrml scene.");
      }

    int numParameterSets = emLogic->GetNumberOfParameterSets();
    if (verbose) std::cerr << "Imported: " << mrmlScene->GetNumberOfNodes()
                           << (mrmlScene->GetNumberOfNodes() == 1 
                               ? " node" : " nodes")
                           << ", including " << numParameterSets 
                           << " EM parameter "
                           << (numParameterSets == 1 ? "node." : "nodes.")
                           << std::endl;
  
    //
    // make sure there is at least one parameter set
    if (numParameterSets < 1)
      {
      throw std::
        runtime_error("ERROR: no EMSegment parameter nodes in scene.");
      }

    //
    // find the parameter set in the MRML scene
    int parameterNodeIndex = 0;
    if (useDefaultParametersNode)
      {
      if (verbose) std::cerr << "Using default parameter set named: " 
                             << emLogic->GetNthParameterSetName(0) 
                             << std::endl;    
      }
    else
      {
      // search for the named parameter set
      bool foundParameters = false;
      if (verbose) std::cerr << "Searching for an EM parameter node named: " 
                             << parametersMRMLNodeName << std::endl;
      
      for (int i = 0; i < numParameterSets; ++i)
        {
        std::string currentNodeName(emLogic->GetNthParameterSetName(i)); 
        if (verbose) std::cerr << "Node " << i 
                               << " name: " << currentNodeName << std::endl;
        if (parametersMRMLNodeName == currentNodeName)
          {
          parameterNodeIndex = i;
          foundParameters = true;
          break;
          }
        else
          {
          if (verbose) std::cerr << "Found non-matching EM parameters node: " 
                                 << currentNodeName << std::endl;
          }
        }
      
      // make sure the parameters were found
      if (!foundParameters)
        {
        std::stringstream ss;
        ss << "ERROR: no EMSegment parameters found in scene with name "
           << parametersMRMLNodeName;
        throw std::
          runtime_error(ss.str());
        }
      }

    //
    // populate the logic with the parameters
    try
      {
      emLogic->SetLoadedParameterSetIndex(parameterNodeIndex);
      }
    catch (...)
      {
      throw std::runtime_error("ERROR: failed to set EMSegment parameters.");
      }

    //
    // make sure the basic parameters are available
    // !!!
    // segmenter node
    // global parameters

    //
    // set the target images
    if (useDefaultTarget)
      {
      if (!emLogic->GetSegmenterNode()->GetTargetNode())
        {
        throw std::runtime_error("ERROR: no default target node available.");
        }
      if (verbose) 
        std::cerr << "Using default target node named: " 
                  << emLogic->GetSegmenterNode()->GetTargetNode()->GetName()
                  << std::endl;
      }
    else
      {
      try 
        {
        if (verbose) 
          std::cerr << "Adding a target node..."; 

        // create target node
        vtkMRMLEMSTargetNode* targetNode = vtkMRMLEMSTargetNode::New();
        mrmlScene->AddNode(targetNode);        

        // connect target node to segmenter
        emLogic->GetSegmenterNode()->SetTargetNodeID(targetNode->GetID());

        if (verbose) 
          std:cerr << targetNode->GetID() << " DONE" << std::endl;

        targetNode->Delete();

        if (verbose)
          std::cerr << "Segmenter's target node is now: " 
                    << emLogic->GetSegmenterNode()->GetTargetNode()->GetID()
                    << std::endl;
        }
      catch (...)
        {
        throw std::runtime_error("ERROR: failed to add target node.");
        }

      if (verbose)
        std::cerr << "Adding " << targetVolumeFileNames.size() 
                  << " target images..." << std::endl;
      for (unsigned int imageIndex = 0; 
           imageIndex < targetVolumeFileNames.size(); ++imageIndex)
        {
        if (verbose) std::cerr << "Loading target image " << imageIndex
                               << "..." << std::endl;
        try
          {
          // load image into scene
          vtkMRMLVolumeNode* volumeNode = 
            AddScalarArchetypeVolume(mrmlScene, 
                                     targetVolumeFileNames[imageIndex].c_str(),
                                     false,
                                     false,
                                     NULL);
          
          if (!volumeNode)
            {
            throw std::runtime_error("failed to load image.");
            }
       
          // set volume name and ID in map
          emLogic->GetSegmenterNode()->GetTargetNode()->
            AddVolume(volumeNode->GetID(), volumeNode->GetID());
          }
        catch(...)
          {
          vtkstd::stringstream ss;
          ss << "ERROR: failed to load target image "
             << targetVolumeFileNames[imageIndex];
          throw std::runtime_error(ss.str());
          }
        }
      }

    //
    // make sure the number of input channels matches the expected
    // value in the parameters
    if (emLogic->GetSegmenterNode()->GetTemplateNode()->
        GetGlobalParametersNode()->GetNumberOfTargetInputChannels() !=
        emLogic->GetSegmenterNode()->GetTargetNode()->GetNumberOfVolumes())
      {
          vtkstd::stringstream ss;
          ss << "ERROR: Number of input channels (" << 
            emLogic->GetSegmenterNode()->GetTargetNode()->GetNumberOfVolumes()
             << ") does not match expected value from parameters (" << 
            emLogic->GetSegmenterNode()->GetTemplateNode()->
            GetGlobalParametersNode()->GetNumberOfTargetInputChannels() 
             << ")";
          throw std::runtime_error(ss.str());
      }
    else
      {
      if (verbose)
        std::cerr << "Number of input channels (" <<
          emLogic->GetSegmenterNode()->GetTargetNode()->GetNumberOfVolumes()
                  << ") matches expected value from parameters (" <<
          emLogic->GetSegmenterNode()->GetTemplateNode()->
          GetGlobalParametersNode()->GetNumberOfTargetInputChannels() 
                  << ")" << std::endl;
      }

    //
    // set the result labelmap image
    if (useDefaultOutput)
      {
      if (!emLogic->GetSegmenterNode()->GetOutputVolumeNode())
        {
        throw std::
          runtime_error("ERROR: no default output volume node available.");
        }
      if (verbose) 
        std::cerr << "Using default output volume node named: " 
                  << 
          emLogic->GetSegmenterNode()->GetOutputVolumeNode()->GetName()
                  << std::endl;
      }
    else
      {
      try 
        {
        // create volume node
        if (verbose) std::cerr << "Creating output volume node...";
        vtkMRMLVolumeNode* outputNode = 
          AddScalarArchetypeVolume(mrmlScene,
                                   resultVolumeFileName.c_str(),
                                   true,
                                   true,
                                   NULL);

        if (!outputNode)
          {
          throw std::runtime_error("failed to load image.");
          }

        // connect output volume node to segmenter
        emLogic->GetSegmenterNode()->
          SetOutputVolumeNodeID(outputNode->GetID());

        if (verbose) 
          std::cerr << "DONE" << std::endl;
        }
      catch (...)
        {
        throw std::runtime_error("ERROR: failed to add result volume node.");
        }
      }

    //
    // update logic parameters from command line
    emLogic->SetEnableMultithreading(!disableMultithreading);
    if (verbose) 
      std::cerr << "Multithreading is " 
                << (disableMultithreading ? "disabled." : "enabled.")
                << std::endl;

    int segmentationBoundaryMin[3];
    int segmentationBoundaryMax[3];
    emLogic->GetSegmentationBoundaryMin(segmentationBoundaryMin);
    emLogic->GetSegmentationBoundaryMax(segmentationBoundaryMax);
    if (verbose) std::cerr 
      << "ROI is [" 
      << segmentationBoundaryMin[0] << ", "
      << segmentationBoundaryMin[1] << ", "
      << segmentationBoundaryMin[2] << "] -->> ["
      << segmentationBoundaryMax[0] << ", "
      << segmentationBoundaryMax[1] << ", "
      << segmentationBoundaryMax[2] << "]" << std::endl;

    //
    // check parameters' node structure
    if (!emLogic->CheckMRMLNodeStructure())
      {
      throw std::
        runtime_error("ERROR: EMSegment invalid parameter node structure");
      }

    //
    // run the segmentation
    try
      {
      if (verbose) std::cerr << "Starting segmentation..." << std::endl;
      emLogic->StartSegmentation();
      if (verbose) std::cerr << "Segmentation complete." << std::endl;
      }
    catch (...)
      {
      throw std::runtime_error("ERROR: failed to run segmentation.");
      }

    //
    // save the results
    // !!!
    }
  catch (std::runtime_error& e)
    {
    std::cerr << e.what() << std::endl;
    std::cerr << "Errors detetected.  Segmentation failed." << std::endl;
    segmentationSucceeded = false;
    }
  catch (...)
    {
    std::cerr << "Unknown error detected.  Segmentation failed." << std::endl;
    segmentationSucceeded = false;
    }

  //
  // compare results to standard image
  if (!resultStandardVolumeFileName.empty())
  {
    if (verbose) 
      cerr << "Comparing results with standard..." << std::endl;

    try
    {
      //
      // get a pointer to the results
      std::string resultMRMLID = emLogic->GetOutputVolumeMRMLID();
      std::cerr << "Extracting results from mrml node: " 
                << resultMRMLID << std::endl;
      vtkImageData* resultImage = NULL;
      vtkMRMLVolumeNode* node   = vtkMRMLVolumeNode::
        SafeDownCast(mrmlScene->GetNodeByID(resultMRMLID.c_str()));
      resultImage = node->GetImageData();
      resultImage->SetSpacing(node->GetSpacing());
      resultImage->SetOrigin(node->GetOrigin());

      //
      // compare result with standard image
      bool imagesDiffer = ImageDiff(resultImage, resultStandardVolumeFileName);
      if (imagesDiffer)
      {
        if (verbose) 
          if (verbose) cerr << "Result DOES NOT match standard!" << std::endl;
        segmentationSucceeded = false;
      }
      else
      {
        if (verbose) cerr << "Result matches standard!" << std::endl;
        segmentationSucceeded = true;
      }
    }
    catch (std::runtime_error& e)
    {
      std::cerr << e.what() << std::endl;
      std::cerr << "Errors detetected.  Comparison failed." << std::endl;
      segmentationSucceeded = false;
    }
    catch (...)
    {
      std::cerr << "Unknown error detected.  Comparison failed." << std::endl;
      segmentationSucceeded = false;
    }
  }

  //
  // clean up
  if (verbose) std::cerr << "Cleaning up...";
  mrmlScene->Clear(true);
  mrmlScene->Delete();
  emLogic->SetAndObserveMRMLScene(NULL);
  emLogic->Delete();
  if (verbose) std::cerr << "DONE" << std::endl;

  return segmentationSucceeded ? EXIT_SUCCESS : EXIT_FAILURE;  
}
