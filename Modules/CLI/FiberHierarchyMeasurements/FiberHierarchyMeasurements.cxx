// vtkTeem includes
#include <vtkDiffusionTensorMathematics.h>
#include <vtkNRRDReader.h>
#include <vtkNRRDWriter.h>

// VTK includes
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyDataTensorToColor.h>
#include <vtkPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkGlobFileNames.h>
#include <vtkStringArray.h>

// MRML
#include <vtkMRMLModelHierarchyNode.h>
#include <vtkMRMLModelNode.h>
#include <vtkMRMLModelStorageNode.h>
#include <vtkMRMLScene.h>
#include <vtkMRMLFiberBundleNode.h>
#include <vtkMRMLFiberBundleLineDisplayNode.h>
#include <vtkMRMLFiberBundleTubeDisplayNode.h>
#include <vtkMRMLFiberBundleGlyphDisplayNode.h>
#include <vtkMRMLFiberBundleStorageNode.h>
#include <vtkMRMLSceneViewNode.h>
#include <vtkMRMLSceneViewStorageNode.h>
#include <vtkMRMLCommandLineModuleNode.h>

// VTKsys includes
#include <vtksys/SystemTools.hxx>

// ITK includes
#include <itkFloatingPointExceptions.h>

#include "FiberHierarchyMeasurementsCLP.h"

void computeScalarMeasurements(vtkPolyData *poly,
                               std::string &id,
                               std::string &operation);

int computeTensorMeasurement(vtkPolyDataTensorToColor *math,
                             vtkAlgorithmOutput *input,
                             std::string &id,
                             std::string &operation);

bool setTensors(vtkPolyData *poly);

void printTable(std::ofstream &ofs);

std::map< std::string, std::map<std::string, double> > OutTable;

int main( int argc, char * argv[] )
{
  itk::FloatingPointExceptions::Disable();

  PARSE_ARGS;

  std::ofstream ofs(outputFile.c_str());
  if (ofs.fail())
    {
    std::cerr << "Output file doesn't exist: " <<  outputFile << std::endl;
    return EXIT_FAILURE;
    }

  vtkPolyDataTensorToColor *math = vtkPolyDataTensorToColor::New();
  std::vector<std::string> operations;
  operations.push_back(std::string("Trace"));
  operations.push_back(std::string("RelativeAnisotropy"));
  operations.push_back(std::string("FractionalAnisotropy"));
  operations.push_back(std::string("LinearMeasurement"));
  operations.push_back(std::string("PlanarMeasurement"));
  operations.push_back(std::string("SphericalMeasurement"));
  operations.push_back(std::string("MinEigenvalue"));
  operations.push_back(std::string("MidEigenvalue"));
  operations.push_back(std::string("MaxEigenvalue"));

  if (inputType == std::string("Hierarchy") )
    {
    // get the model hierarchy id from the scene file
    std::string::size_type loc;
    std::string            sceneFilename;
    std::string            modelHierarchyID;

    if (FiberSceneFile.size() == 0)
      {
      // make one up from the input volume's name
      std::cerr << "********\nERROR: no model scene defined!" << endl;
      }
    else
      {
      loc = FiberSceneFile[0].find_last_of("#");
      if (loc != std::string::npos)
        {
        sceneFilename = std::string(FiberSceneFile[0].begin(),
                                    FiberSceneFile[0].begin() + loc);
        loc++;

        modelHierarchyID = std::string(FiberSceneFile[0].begin() + loc, FiberSceneFile[0].end());
        }
      else
        {
        // the passed in file is missing a model hierarchy node, work around it
        sceneFilename = FiberSceneFile[0];
        }
      }

    // check for the model mrml file
    if (sceneFilename == "")
      {
      std::cout << "No file to store models!" << std::endl;
      return EXIT_FAILURE;
      }

    // get the directory of the scene file
    std::string rootDir
      = vtksys::SystemTools::GetParentDirectory(sceneFilename.c_str());

    vtkNew<vtkMRMLScene> modelScene;

    // load the scene that Slicer will re-read
    modelScene->SetURL(sceneFilename.c_str());

    modelScene->RegisterNodeClass(vtkNew<vtkMRMLSceneViewNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLSceneViewStorageNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLCommandLineModuleNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLFiberBundleNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLFiberBundleLineDisplayNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLFiberBundleTubeDisplayNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLFiberBundleGlyphDisplayNode>().GetPointer());
    modelScene->RegisterNodeClass(vtkNew<vtkMRMLFiberBundleStorageNode>().GetPointer());

    // only try importing if the scene file exists
    if (vtksys::SystemTools::FileExists(sceneFilename.c_str()))
      {
      modelScene->Import();
      }
    else
      {
      std::cerr << "Model scene file doesn't exist: " <<  sceneFilename.c_str() << std::endl;
      }

    // make sure we have a model hierarchy node
    vtkMRMLNode *node = modelScene->GetNodeByID(modelHierarchyID);
    vtkSmartPointer<vtkMRMLModelHierarchyNode> topHierNode =
       vtkMRMLModelHierarchyNode::SafeDownCast(node);
    if (!topHierNode)
      {
      std::cerr << "Model hierachy node doesn't exist: " <<  modelHierarchyID.c_str() << std::endl;
      return EXIT_FAILURE;
      }

    // get all the children nodes
    std::vector< vtkMRMLHierarchyNode *> allChildren;
    topHierNode->GetAllChildrenNodes(allChildren);

    // and loop over them
    for (unsigned int i = 0; i < allChildren.size(); ++i)
      {
      vtkMRMLDisplayableHierarchyNode *dispHierarchyNode = vtkMRMLDisplayableHierarchyNode::SafeDownCast(allChildren[i]);
      if (dispHierarchyNode)
        {
        // get any associated node
          vtkMRMLFiberBundleNode *fiberNode = vtkMRMLFiberBundleNode::SafeDownCast(
            dispHierarchyNode->GetAssociatedNode());
        if (fiberNode)
          {
          std::string id = std::string(fiberNode->GetName());
          for (int o=0; o<operations.size(); o++)
            {
            computeTensorMeasurement(math,
                                     fiberNode->GetPolyDataConnection(),
                                     id,
                                     operations[o]);
            }
          }
        }
      }
    } //if (inputType == std::string("Fibers Hierarchy Schene") )
  else if (inputType == std::string("Folder") )
    {
    // File based
    if (InputDirectory.size() == 0)
      {
      std::cerr << "Input directory doesn't exist: " << std::endl;
      return EXIT_FAILURE;
      }

    vtkGlobFileNames* gfnVTK = vtkGlobFileNames::New();
    gfnVTK->SetDirectory(InputDirectory.c_str());
    gfnVTK->AddFileNames("*.vtk");
    vtkStringArray *fileNamesVTK = gfnVTK->GetFileNames();

    vtkGlobFileNames* gfnVTP = vtkGlobFileNames::New();
    gfnVTP->SetDirectory(InputDirectory.c_str());
    gfnVTP->AddFileNames("*.vtp");
    vtkStringArray *fileNamesVTP = gfnVTP->GetFileNames();

    vtkXMLPolyDataReader *readerVTP = vtkXMLPolyDataReader::New();
    vtkPolyDataReader *readerVTK = vtkPolyDataReader::New();

    // Loop over polydatas
    for (int i=0; i<fileNamesVTP->GetNumberOfValues(); i++)
      {
      vtkStdString fileName = fileNamesVTP->GetValue(i);
      readerVTP->SetFileName(fileName.c_str());
      readerVTP->Update();

      computeScalarMeasurements(readerVTP->GetOutput(), std::string(fileName.c_str()), std::string(""));

      if( !setTensors(readerVTP->GetOutput()) )
        {
        std::cerr << argv[0] << ": No tensor data for file " << fileName << std::endl;
        continue;
        }

      for (int o=0; o<operations.size(); o++)
        {
        computeTensorMeasurement(math,
                                 readerVTP->GetOutputPort(),
                                 std::string(fileName.c_str()),
                                 operations[o]);
        }
      }
    for (int i=0; i<fileNamesVTK->GetNumberOfValues(); i++)
      {
      vtkStdString fileName = fileNamesVTK->GetValue(i);
      readerVTK->SetFileName(fileName.c_str());
      readerVTK->Update();

      computeScalarMeasurements(readerVTK->GetOutput(), std::string(fileName.c_str()), std::string(""));

      if( !setTensors(readerVTK->GetOutput()) )
        {
        std::cerr << argv[0] << ": No tensor data for file " << fileName << std::endl;
        continue;
        }

      for (int o=0; o<operations.size(); o++)
        {
        computeTensorMeasurement(math,
                                 readerVTK->GetOutputPort(),
                                 std::string(fileName.c_str()),
                                 operations[o]);
        }
      }
    gfnVTK->Delete();
    gfnVTP->Delete();
    readerVTK->Delete();
    readerVTP->Delete();
    } //if (inputType == std::string("Fibers File Folder") )

  math->Delete();

  printTable(ofs);

  return EXIT_SUCCESS;
}

void computeScalarMeasurements(vtkPolyData *poly,
                               std::string &id,
                               std::string &operation)
{
  // averagre measurement for each scalar array
  for (int i=0; i<poly->GetPointData()->GetNumberOfArrays(); i++)
    {
    vtkDataArray *arr = poly->GetPointData()->GetArray(i);
    if (arr->GetNumberOfComponents() > 1)
      {
      continue;
      }

    std::string name = operation;
    if (arr->GetName())
      {
      name = std::string(arr->GetName());
      }

    double val;
    double sum = 0;
    for (int n=0; n<poly->GetNumberOfPoints(); n++)
      {
      arr->GetTuple(n, &val);
      sum += val;
      }
    sum /= poly->GetNumberOfPoints();

    std::cout << " : " << name << " = " << sum << std::endl;
    //ofs << id << " : " << name << " = " << sum << std::endl;

    std::map< std::string, std::map<std::string, double> >::iterator it = OutTable.find(id);
    if (it == OutTable.end())
      {
      OutTable[id] = std::map<std::string, double>();
      it = OutTable.find(id);
      }
      it->second[name] = sum;
    } //for (int i=0; i<poly->GetPointData()->GetNumberOfArrays(); i++)
}

int computeTensorMeasurement(vtkPolyDataTensorToColor *math,
                             vtkAlgorithmOutput *input,
                             std::string &id,
                             std::string &operation)
{
#if (VTK_MAJOR_VERSION <= 5)
  math->SetInput(0, input );
#else
  math->SetInputConnection(0, input );
#endif

  if( operation == std::string("Trace") )
    {
    math->ColorGlyphsByTrace();
    }
  else if( operation == std::string("RelativeAnisotropy") )
    {
    math->ColorGlyphsByRelativeAnisotropy();
    }
  else if( operation == std::string("FractionalAnisotropy") )
    {
    math->ColorGlyphsByFractionalAnisotropy();
    }
  else if( operation == std::string("LinearMeasurement") || operation == std::string("LinearMeasure") )
    {
    math->ColorGlyphsByLinearMeasure();
    }
  else if( operation == std::string("PlanarMeasurement") || operation == std::string("PlanarMeasure") )
    {
    math->ColorGlyphsByPlanarMeasure();
    }
  else if( operation == std::string("SphericalMeasurement") || operation == std::string("SphericalMeasure") )
    {
    math->ColorGlyphsBySphericalMeasure();
    }
  else if( operation == std::string("MinEigenvalue") )
    {
    math->ColorGlyphsByMinEigenvalue();
    }
  else if( operation == std::string("MidEigenvalue") )
    {
    math->ColorGlyphsByMidEigenvalue();
    }
  else if( operation == std::string("MaxEigenvalue") )
    {
    math->ColorGlyphsByMaxEigenvalue();
    }
  else
    {
    std::cerr << operation << ": Operation " << operation << "not supported" << std::endl;
    return EXIT_FAILURE;
    }

  math->Update();

  if (!math->GetOutput()->GetPointData() || !math->GetOutput()->GetPointData()->GetScalars())
    {
    std::cerr << "no scalars computed" << std::endl;
    }

  computeScalarMeasurements(math->GetOutput(), id, operation);

  return EXIT_SUCCESS;
}

bool setTensors(vtkPolyData *poly)
{
  bool hasTensors = false;
  if (poly)
    {
    if (poly->GetPointData()->GetTensors())
      {
      hasTensors = true;
      }
    else
      {
      for (int i=0; i<poly->GetPointData()->GetNumberOfArrays(); i++)
        {
        vtkDataArray *arr = poly->GetPointData()->GetArray(i);
        if (arr->GetNumberOfComponents() == 9)
          {
          poly->GetPointData()->SetTensors(arr);
          hasTensors = true;
          }
        }
      }
    }
  return hasTensors;
}

void printTable(std::ofstream &ofs)
{
  std::map<std::string, std::string> names;
  std::map< std::string, std::map<std::string, double> >::iterator it;
  std::map<std::string, double>::iterator it1;
  std::map<std::string, std::string>::iterator it2;

  for(it = OutTable.begin(); it != OutTable.end(); it++)
    {
    for (it1 = it->second.begin(); it1 != it->second.end(); it1++)
      {
      names[it1->first] = it1->first;
      }
    }

  std::cout << "Name";
  ofs << "Name";

  for (it2 = names.begin(); it2 != names.end(); it2++)
    {
    std::cout << " , " << it2->second;
    ofs << " , " << it2->second;
    }
  std::cout << std::endl;
  ofs << std::endl;

  for(it = OutTable.begin(); it != OutTable.end(); it++)
    {
    std::cout << it->first;
    ofs << it->first;

    for (it2 = names.begin(); it2 != names.end(); it2++)
      {
      std::cout << " , ";
      ofs << " , ";
      it1 = it->second.find(it2->second);
      if (it1 != it->second.end())
        {
        std::cout << it1->second;
        ofs << it1->second;
        }
      }
    std::cout << std::endl;
    ofs << std::endl;
    }
}
