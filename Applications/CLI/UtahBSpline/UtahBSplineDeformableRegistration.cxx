/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $HeadURL: http://www.na-mic.org/svn/Slicer3/branches/TransformIO/Applications/CLI/BSplineDeformableRegistration.cxx $
  Language:  C++
  Date:      $Date: 2008-02-07 09:51:14 -0500 (Thu, 07 Feb 2008) $
  Version:   $Revision: 5746 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "UtahBSplineDeformableRegistrationCLP.h"


#include "itkCommand.h"

#include "itkOrientedImage.h"
#include "itkOrientImageFilter.h"
#include "itkResampleImageFilter.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkPluginUtilities.h"

#include "itkTimeProbesCollectorBase.h"

#include "GradientDescent.h"

#include "NormalizedCorrelationCostFunction3D.h"

#include "itkRescaleIntensityImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkResampleImageFilter.h"

#include <fstream>
#include <sstream>

#define Precision float

typedef itk::OrientedImage<Precision, 3> Image;
typedef Image::Pointer ImagePointer;
typedef Image::RegionType ImageRegion;
typedef Image::IndexType ImageIndex;
typedef Image::PointType ImagePoint;
typedef Image::SpacingType ImageSpacing;
typedef Image::SizeType ImageSize;
typedef ImageIndex::IndexValueType IndexValue;

typedef NormalizedCorrelationCostFunction3D<Precision, Image> TCostFunction;


typedef TCostFunction::TParametric TParametric;
typedef TCostFunction::TControlPoint TControlPoint;
typedef TCostFunction::TControlMesh TControlMesh;

typedef itk::ImageFileReader<Image> ImageReader;
typedef ImageReader::Pointer ImageReaderPointer;

typedef itk::ImageFileWriter<Image> ImageWriter;


typedef itk::RescaleIntensityImageFilter<Image> RescaleFilter;
typedef RescaleFilter::Pointer RescaleFilterPointer;

typedef itk::ImageFileReader<Image> ImageReader;
typedef ImageReader::Pointer ImageReaderPointer;

typedef itk::ImageFileWriter<Image> ImageWriter;
typedef ImageWriter::Pointer ImageWriterPointer;
typedef itk::SmoothingRecursiveGaussianImageFilter<Image, Image> GaussianImageFilter;
typedef GaussianImageFilter::Pointer GaussianImageFilterPointer;

//typedef itk::DiscreteGaussianImageFilter<Image, Image> GaussianImageFilter;
//typedef GaussianImageFilter::Pointer GaussianImageFilterPointer;

typedef itk::ResampleImageFilter<Image, Image> ResampleFilter;
typedef ResampleFilter::Pointer ResampleFilterPointer;
    
typedef Image::PointType ImagePoint;



//Create image pyramid
ImagePointer *downsample(ImagePointer im, int nres, Precision sigma){
  ImagePointer *pyramid = new ImagePointer[nres];
  pyramid[0] = im; 
  //Downsample
  for(int i=1; i<nres; i++){
    GaussianImageFilterPointer smooth = GaussianImageFilter::New();
    smooth->SetSigma(sigma);
    smooth->SetInput(pyramid[i-1]);
    smooth->Update();

    ResampleFilterPointer downsample = ResampleFilter::New();
    downsample->SetInput(smooth->GetOutput());
                
    ImageSize size = pyramid[i-1]->GetLargestPossibleRegion().GetSize();
    ImageSpacing spacing = pyramid[i-1]->GetSpacing();
    for(unsigned int n = 0; n < size.GetSizeDimension(); n++){
      size[n] = size[n]/2;
      spacing[n] = spacing[n]*2;
    }
    downsample->UseReferenceImageOff();
    downsample->SetOutputOrigin(pyramid[i-1]->GetOrigin()); 
    downsample->SetOutputDirection(pyramid[i-1]->GetDirection());
    downsample->SetOutputSpacing(spacing); 
    downsample->SetSize(size);

    downsample->Update();
    pyramid[i] = downsample->GetOutput();    
  }
  return pyramid;
};

ImageRegion findBoundingBox(ImagePointer maskImage){

  ImageIndex minIndex;
  minIndex.Fill( vnl_numeric_traits< IndexValue >::maxval );
  ImageIndex maxIndex;
  maxIndex.Fill(0);

  typedef itk::ImageRegionConstIteratorWithIndex<Image> MaskImageIterator;
  MaskImageIterator maskIt(maskImage, maskImage->GetLargestPossibleRegion());
  for(maskIt.GoToBegin(); !maskIt.IsAtEnd(); ++maskIt){
    if(maskIt.Get() != 0){
      ImageIndex tmp = maskIt.GetIndex();
      for(unsigned int i = 0; i < Image::GetImageDimension(); i++){
        if(maxIndex[i] < tmp[i]){
          maxIndex[i] = tmp[i];
        }
        if(minIndex[i] > tmp[i]){
          minIndex[i] = tmp[i];
        }
      }
    }
  }
    


  ImageIndex boundsIndex;
  ImageSize boundsSize;
  for(unsigned int i = 0; i < Image::GetImageDimension(); i++){
    boundsSize[i] = (maxIndex[i]  - minIndex[i]);
    boundsIndex[i] = minIndex[i] - boundsSize[i]/2;
    boundsSize[i] *= 2;
  }

  ImageRegion maxRegion = maskImage->GetLargestPossibleRegion();
  ImageSize maxSize = maxRegion.GetSize();
  for(unsigned int i = 0; i < Image::GetImageDimension();  i++){
    if(boundsIndex[i] < 0){
      boundsIndex[i] = 0;
    }
    if(boundsIndex[i] + boundsSize[i] > maxSize[i]){
      boundsSize[i] = maxSize[i] - boundsIndex[i];
    }
  }

  ImageRegion bounds(boundsIndex, boundsSize);
  return bounds;
}

//Compute region for registration
ImageRegion computeRegion(ImagePointer image){  
  
  //Find bounding box
  ImageRegion region = findBoundingBox(image); 

  //Add 10 pixels to region
  ImageRegion largest = image->GetLargestPossibleRegion();
  ImageIndex lIndex = largest.GetIndex();
  ImageSize lSize = largest.GetSize();

  ImageIndex rIndex = region.GetIndex();
  ImageSize rSize = region.GetSize();
  for(int i=0; i<3; i++){
    rIndex[i] -= 10;
    if(lIndex[i] > rIndex[i]){
      rIndex[i] = lIndex[i];
    }
    rSize[i] += 10;
    if(rSize[i] > lSize[i]){
      rSize[i] = lSize[i];
    }
  }
  return region;
}

template<class T> int DoIt( int argc, char * argv[], T )
{
  PARSE_ARGS;




  //Read Input Image
  ImageReaderPointer fixedImageReader = ImageReader::New();
  fixedImageReader->SetFileName( FixedImageFileName.c_str() );
  RescaleFilterPointer fixedRescaleFilter = RescaleFilter::New();
  fixedRescaleFilter->SetInput(fixedImageReader->GetOutput());
  fixedRescaleFilter->SetOutputMaximum(1);
  fixedRescaleFilter->SetOutputMinimum(0);
  fixedRescaleFilter->Update();
  ImagePointer fixedImage = fixedRescaleFilter->GetOutput();



  ImageReaderPointer movingImageReader = ImageReader::New();
  movingImageReader->SetFileName( MovingImageFileName.c_str() );
  movingImageReader->Update();
  ImagePointer movingOrig = movingImageReader->GetOutput();

  RescaleFilterPointer movingRescaleFilter = RescaleFilter::New();
  movingRescaleFilter->SetInput(movingOrig);
  movingRescaleFilter->SetOutputMaximum(1);
  movingRescaleFilter->SetOutputMinimum(0);
  movingRescaleFilter->Update();
  ImagePointer movingImage = movingRescaleFilter->GetOutput();


  //TODO mask support
  ImageReaderPointer maskImageReader = ImageReader::New();
  maskImageReader->SetFileName( MovingImageFileName.c_str() );
  maskImageReader->Update();
  ImagePointer maskImage = maskImageReader->GetOutput();


  itk::TimeProbesCollectorBase collector;




  //Setup normalized correlation cost function
  ImageRegion region = findBoundingBox(movingImage);
  ImageIndex index = region.GetIndex();
  ImageSize size = region.GetSize();
  
  unsigned int maxSize = size[0];
  unsigned int minSize = size[0];
  for(int i=1; i<3; i++){
    if(maxSize < size[i]){
      maxSize = size[i];
    }
    if(minSize > size[i]){
      minSize = size[i];
    }
  }

  //Set params according to level of detail
  int gridSize = 0;
  double alpha = 0;
  double gdtol = 0;
  double sigma = 1;
  double step = 1000;

  //Determine number of resolutions
  int nResolutions = 0;
  for(unsigned int tmpSize = minSize; tmpSize > 15; tmpSize/=2){
    nResolutions++;
  }
  if(nResolutions == 0){
    nResolutions = 1;
  }

  //early abort for coarse registrations
  int nAbort = 0;

  switch(Detail){
    case 1:
    alpha = 0.2;
    gridSize = std::floor(maxSize/16.f);
    gdtol = 0.0001;
    nAbort = 3;
    break;
    case 2:
    alpha = 0.2;
    gridSize = std::floor(maxSize/16.f);
    gdtol = 0.0001;
    nAbort = 2;
    break;
    case 3:
    gridSize = std::floor(maxSize/16.f);
    alpha = 0.2;
    gdtol = 0.0001;
    nAbort = 1;
    break;
    case 4:
    gridSize = std::floor(maxSize/16.f);
    alpha = 0.1;
    gdtol = 0.0001;
    nAbort = 1;
    break;
    case 5:
    gridSize = std::floor(maxSize/16.f);
    alpha = 0.1;
    gdtol =0.0001;
    break;
    case 6:
    gridSize = std::floor(maxSize/8.f);
    alpha = 0.1;
    gdtol = 0.0001;
    break;
    case 7:
    gridSize = std::floor(maxSize/8.f);
    alpha = 0.05;
    gdtol = 0.0001;
    break;
    case 8:
    gridSize = std::floor(maxSize/4.f);
    alpha = 0.05;
    gdtol =0.0001;
    break;
    case 9:
    gridSize = std::floor(maxSize/4.f);
    alpha = 0;
    gdtol = 0.0001;
    break;
    case 10:
    gridSize = std::floor(maxSize/2.f);
    alpha = 0;
    gdtol = 0.0001;
    break;
  }

  if(nAbort >= nResolutions){
    nAbort = nResolutions-1;
  }


  //Setup cost function
  bool useMask = false;
  int nElements = 3 * gridSize * gridSize * gridSize;
  TCostFunction costFunction(nElements);
  costFunction.SetFixedImage(fixedImage);
  costFunction.SetMovingImage(movingImage);
  costFunction.SetMaskImage(maskImage);
  costFunction.SetRange( region );
  costFunction.SetUseMask(useMask);
  costFunction.SetFixedBoundary(true);
  costFunction.ComputePhysicalRange(region);
  
  //Create Identity transfrom - setup BSpline deformation
  //Physical bspline location
  ImagePoint rangeStart = costFunction.GetStart();
  ImagePoint rangeSize = costFunction.GetSize();
  Precision rStart[3];
  Precision rSize[3];
  double maxPhysicalSize = 0;
  for(int i=0; i<3; i++){
    rStart[i] = rangeStart[i];
    rSize[i] = rangeSize[i];
    if(maxPhysicalSize < rSize[i]){
      maxPhysicalSize =rSize[i];
    }
  }   
  //Default degree 2 --- maybe change for coarser registration to 3 for a smoother warp
  int ncps[3]; 
  int degree[3]; 
  for(int i=0; i < 3; i++){
    ncps[i] = gridSize;
    degree[i] = 2;
  }
  TParametric surface = TParametric::createIdentity( rStart, rSize, ncps, degree);
  
  //Set Idenety transform as intialization
  costFunction.SetParametric(surface);
  
  //regularization according to phycical size  
  costFunction.SetAlpha(alpha/maxPhysicalSize);
  
  //Initalize cost function
  costFunction.Init();
  costFunction.SetVerbose(10);


  //----- Do multiresolution optimization
  collector.Start("Computing multiresolution images");
  ImagePointer *fixedPyramid = downsample(fixedImage, nResolutions, sigma);
  ImagePointer *maskPyramid = NULL;
  if(useMask){
    maskPyramid = downsample(maskImage, nResolutions, sigma);
  }
  ImagePointer *movingPyramid = downsample(movingImage, nResolutions, sigma);
  collector.Stop("Computing multiresolution images");




 
 ImagePointer output = NULL;
  
  //run registration
  try { 

    for(int i=nResolutions-1; i>=nAbort; i--){
      std::stringstream ss;
      ss << "Registration Reoslution " << i; 
      collector.Start( ss.str().c_str() );
     
      //Setup optimizer 
      GradientDescent optimizer;  
      optimizer.SetStepSize(step);

      //Set lowerresolution images
      costFunction.SetFixedImage(fixedPyramid[i]);
      if(useMask){
        costFunction.SetMaskImage(maskPyramid[i]); 
        region = computeRegion(maskPyramid[i]);
      }else{
        costFunction.SetMaskImage(fixedPyramid[i]);
        region = computeRegion(fixedPyramid[i]);
      }
      costFunction.SetMovingImage(movingPyramid[i]);
      costFunction.SetRange(region);
    
      //Set up optimizer
      optimizer.SetTolerance(gdtol / pow(10.f, i) );

      optimizer.SetVerbose(10);

      //--- minimize cost function 

      //intial parameters
      vnl_vector<double> params(nElements);
      costFunction.GetVNLParametersFromParametric(params);

      optimizer.Minimize(costFunction, params);    
      
      collector.Stop( ss.str().c_str() );    

    }


    //Resample original image  
    
    TCostFunction::TImageTransformation &transform = costFunction.GetTransformation();
    region = computeRegion(fixedImage);
    transform.SetRange(region);
    transform.SetImage(movingOrig);
    output = transform.Transform();

  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ExceptionObject caught !" << std::endl; 
    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  } 
  

  if (ResampledImageFileName != ""){
    
    typedef itk::ImageFileWriter< Image >  WriterType;
    typename WriterType::Pointer      writer =  WriterType::New();
    writer->SetFileName( ResampledImageFileName.c_str() );
    writer->SetInput( output );

    try
      {
      collector.Start( "Write resampled volume" );
      writer->Update();
      collector.Stop( "Write resampled volume" );
      }
    catch( itk::ExceptionObject & err ) 
      { 
      std::cerr << "ExceptionObject caught !" << std::endl; 
      std::cerr << err << std::endl; 
      return EXIT_FAILURE;
      }
  }

  // Report the time taken by the registration
  collector.Report();

  return EXIT_SUCCESS;
}

int main( int argc, char * argv[] )
{
  
  // Print out the arguments (need to add --echo to the argument list 
  // 
  std::vector<char *> vargs1;
  for (int vi=0; vi < argc; ++vi) vargs1.push_back(argv[vi]);
  vargs1.push_back(const_cast<char *>("--echo"));
  
  argc = vargs1.size();
  argv = &(vargs1[0]);

  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType (FixedImageFileName, pixelType, componentType);

    // This filter handles all types
    
    switch (componentType)
      {
      case itk::ImageIOBase::CHAR:
      case itk::ImageIOBase::UCHAR:
      case itk::ImageIOBase::USHORT:
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0));
        break;
      case itk::ImageIOBase::ULONG:
      case itk::ImageIOBase::LONG:
      case itk::ImageIOBase::UINT:
      case itk::ImageIOBase::INT:
      case itk::ImageIOBase::DOUBLE:
      case itk::ImageIOBase::FLOAT:
        return DoIt( argc, argv, static_cast<float>(0));
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cout << "unknown component type" << std::endl;
        break;
      }
    }
  catch( itk::ExceptionObject &excep)
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
