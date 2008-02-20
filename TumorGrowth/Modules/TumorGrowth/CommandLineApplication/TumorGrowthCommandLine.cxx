#include <iostream>
#include <vector>
#include <string>

#include "vtkTumorGrowthLogic.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "TumorGrowthCommandLineCLP.h"

int main(int argc, char** argv)
{
  //
  // parse arguments using the CLP system; this creates variables.
  PARSE_ARGS;

  //bool useDefaultOutput         = resultVolumeFileName.empty();
  // if (verbose) std::cerr << "Starting EMSegment Command Line." << std::endl;

  // return segmentationSucceeded ? EXIT_SUCCESS : EXIT_FAILURE;  
  return EXIT_SUCCESS;  
}
