#undef OPENTRACKER_EXPORTS
#include <OpenTracker/OpenTracker.h>
#include <OpenTracker/input/SPLModules.h>
#include <OpenTracker/input/BRPImageIOModule.h>
#include <OpenTracker/types/Image.h>

#include <iostream>
#include <exception>
#include <stdlib.h>

using namespace ot;

// for image read

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkGDCMImageIO.h"
#include "itkSpatialOrientationAdapter.h"

#include "pgmimage.h"

#define DEFAULT_LOOP_RATE    1000

Image* dicomRead(char* filename, int* width, int* height,
                 std::vector<float>& position, std::vector<float>& orientation);

int main(int argc, char *argv[])
{
  char* xmlfile;
  char* dicomfile;
  long  rate;

  // set parameters based on arguments
  if (argc <= 2) {
    std::cerr << "Usage: " << argv[0] << " <OT XML> <DICOM file>"  << std::endl;
    exit(-1);
  }
  xmlfile = argv[1];
  dicomfile = argv[2];

  rate = DEFAULT_LOOP_RATE * 1000;
  std::cerr << "XML File  : " << xmlfile   << std::endl;
  std::cerr << "DICOM File: " << dicomfile << std::endl;

  int w, h;
  std::vector<float> position;
  std::vector<float> orientation;

  position.resize(3, 0.0);
  orientation.resize(4, 0.0);

  Image *img = dicomRead(dicomfile, &w, &h, position, orientation);
  
  PgmImage pi(w, h, (short*)img->image_ptr);
  std::ofstream fout("test.pgm");
  fout << pi;


  // context loop
  addSPLModules();

  Context context(1);
  context.parseConfiguration(xmlfile);
  context.start();

  BRPImageIOModule* module = (BRPImageIOModule*) context.getModule("BRPImageIOConfig");

  if (!module) {
    std::cerr << "Failed to get BRPImageIOModule." << std::endl;
    exit(-1);
  }

  int stopflag = 0;
  while (stopflag == 0) {
    std::cerr << "ORIGIN: " << position[0] << ", "
              << position[1] << ", "
              << position[2] << std::endl;
    module->setImage(*img, w, h, position, orientation);
    /*
    orientation[0] = 0;
    orientation[1] = 0;
    orientation[2] = 0;
    orientation[3] = 1.0;
    module->setTracker(position, orientation);
    */
    stopflag = context.loopOnce();
    usleep(rate);
  }

}


Image* dicomRead(char* filename, int* width, int* height,
                 std::vector<float>& position, std::vector<float>& orientation)
{
  position.resize(3, 0.0);
  orientation.resize(4, 0.0);

  const   unsigned int   Dimension = 2;
  typedef unsigned short InputPixelType;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);

  typedef itk::GDCMImageIO           ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();
  reader->SetImageIO( gdcmImageIO );

  try {
    reader->Update();
  } catch (itk::ExceptionObject & e) {
    std::cerr << "exception in file reader " << std::endl;
    std::cerr << e.GetDescription() << std::endl;
    std::cerr << e.GetLocation() << std::endl;
    return NULL;
  }

  char name[100];
  gdcmImageIO->GetPatientName(name);
  std::cerr << name << std::endl;

  double origin[3];
  double center[3];
  int    size[3];
  double spacing[3];

  for (int i = 0; i < 3;i ++) {
    origin[i]  = gdcmImageIO->GetOrigin(i);
    size[i]    = gdcmImageIO->GetDimensions(i);
    spacing[i] = gdcmImageIO->GetSpacing(i);
  }

  float imageDir[3][3];
  for (int i = 0; i < 3; i ++) {
    std::vector<double> v;
    v = gdcmImageIO->GetDirection(i);
    imageDir[i][0] = v[0];
    imageDir[i][1] = v[1];
    imageDir[i][2] = v[2];
  }

  // LPS to RAS
  origin[0] *= -1.0;
  origin[1] *= -1.0;
  imageDir[0][0] *= -1.0;
  imageDir[0][1] *= -1.0;
  imageDir[0][2] *= -1.0;
  imageDir[1][0] *= -1.0;
  imageDir[1][1] *= -1.0;
  imageDir[1][2] *= -1.0;

  std::cerr << "DICOM IMAGE:" << std::endl;
  std::cerr << " Dimension = ( "
            << size[0] << ", " << size[1] << ", " << size[2] << " )" << std::endl;
  std::cerr << " Origin    = ( "
            << origin[0] << ", " << origin[1] << ", " << origin[2] << " )" << std::endl;
  std::cerr << " Spacing   = ( "
            << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << " )" << std::endl;

  std::cerr << " Orientation: " << std::endl;
  std::cerr << "   " << imageDir[0][0] << ", " << imageDir[0][1] << ", " 
            << imageDir[0][2] << std::endl;
  std::cerr << "   " << imageDir[1][0] << ", " << imageDir[1][1] << ", "
            << imageDir[1][2] << std::endl;
  std::cerr << "   " << imageDir[2][0] << ", " << imageDir[2][1] << ", "
            << imageDir[2][2] << std::endl;

  InputImageType::Pointer    inputImage = reader->GetOutput();
  InputImageType::RegionType region   = inputImage->GetLargestPossibleRegion();


  // position is the center of the image
  double coffset[3];
  for (int i = 0; i < 3; i ++) {
    coffset[i] = ((size[i]-1)*spacing[i])/2.0;
  }

  for (int i = 0; i < 3; i ++) {
    position[i] = origin[i] + (coffset[0]*imageDir[i][0] + coffset[1]*imageDir[i][1]
                               + coffset[2]*imageDir[i][2]);
  }
  std::cerr << " Center   =  ( "
            << position[0] << ", " << position[1] << ", " << position[2] << " )" << std::endl;


  float matrix[3][3];
  float quat[4];
  MathUtils::matrixToQuaternion(imageDir, quat);
  for (int i = 0; i < 4; i ++) {
    orientation[i] = quat[i];
  }


  int w = size[0];
  int h = size[1];

  short* data = new short[w*h];
  InputImageType::IndexType index;

  for (int j = 0; j < h; j ++) {
    index[1] = j;
    for (int i = 0; i < w; i ++) {
      index[0] = w-i;
      data[j*w+i] = (short) inputImage->GetPixel(index);
    }
  }

  *width = w;
  *height = h;
  Image* img = new Image(size[0], size[1], sizeof(short), (void*)data);



  return img;

}






