#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "RegisterImagesSlicerCLP.h"

#include "itkTimeProbesCollectorBase.h"

#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkMultiThreader.h"

#include "itkImageToImageRegistrationHelper.h"

// Description:
// Get the PixelType and ComponentType from fileName
void GetImageType (std::string fileName,
                   itk::ImageIOBase::IOPixelType &pixelType,
                   itk::ImageIOBase::IOComponentType &componentType,
                   unsigned int & dimensions )
{
  typedef itk::OrientedImage<short, 3> ImageType;
  itk::ImageFileReader<ImageType>::Pointer imageReader =
        itk::ImageFileReader<ImageType>::New();
  imageReader->SetFileName(fileName.c_str());
  imageReader->UpdateOutputInformation();

  pixelType = imageReader->GetImageIO()->GetPixelType();
  componentType = imageReader->GetImageIO()->GetComponentType();
  dimensions = imageReader->GetImageIO()->GetNumberOfDimensions();
}

template <unsigned int DimensionT, class T>
int DoIt( int argc, char *argv[] )
{

  PARSE_ARGS;

  enum VerboseLevelEnum {SILENT, STANDARD, VERBOSE};
  VerboseLevelEnum verbosity = SILENT;
  if( verbosityLevel == "Standard" )
    {
    verbosity = STANDARD;
    }
  else if( verbosityLevel == "Verbose" )
    {
    verbosity = VERBOSE;
    }

  typedef typename itk::OrientedImage<T, DimensionT> ImageType;

  typedef typename itk::ImageToImageRegistrationHelper< ImageType >  RegerType;

  typename RegerType::Pointer reger = RegerType::New();

  reger->SetReportProgress( true );

  if (verbosity >= STANDARD)
    {
    std::cout << "### Loading fixed image...";
    }
  reger->LoadFixedImage( fixedImage );
  if (verbosity >= STANDARD)
    {
    std::cout << "### DONE" << std::endl;
    }

  if (verbosity >= STANDARD)
    {
    std::cout << "### Loading moving image...";
    }
  reger->LoadMovingImage( movingImage );
  if (verbosity >= STANDARD)
    {
    std::cout << "### DONE" << std::endl;
    }

  /*
  if( loadParameters.size() > 1)
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Loading parameters...";
      }
    reger->LoadParameters( loadParameters );
    if (verbosity >= STANDARD)
      {
      std::cout << "### DONE" << std::endl;
      }
    }
  */

  if( loadTransform.size() > 1 )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Loading transform...";
      }
    reger->LoadTransform( loadTransform );
    if (verbosity >= STANDARD)
      {
      std::cout << "### DONE" << std::endl;
      }
    }

  if( initialization == "None" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Initialization: None" << std::endl;
      }
    reger->SetInitialMethodEnum( RegerType::INIT_WITH_NONE );
    }
  else if( initialization == "ImageCenters")
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Initialization: ImageCenters" << std::endl;
      }
    reger->SetInitialMethodEnum( RegerType::INIT_WITH_IMAGE_CENTERS );
    }
  else if( initialization == "SecondMoments")
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Initialization: SecondMoments" << std::endl;
      }
    reger->SetInitialMethodEnum( RegerType::INIT_WITH_SECOND_MOMENTS );
    }
  else if( initialization == "CentersOfMass")
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Initialization: CentersOfMass" << std::endl;
      }
    reger->SetInitialMethodEnum( RegerType::INIT_WITH_CENTERS_OF_MASS );
    }
  else // LANDMARKS
    {
    reger->SetInitialMethodEnum( RegerType::INIT_WITH_LANDMARKS );
    reger->SetFixedLandmarks( fixedLandmarks );
    reger->SetMovingLandmarks( movingLandmarks );
    }


  if( registration == "None" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: None" << std::endl;
      }
    reger->SetEnableInitialRegistration( false );
    reger->SetEnableRigidRegistration( false );
    reger->SetEnableAffineRegistration( false );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "Initial" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: Initial" << std::endl;
      }
    reger->SetEnableInitialRegistration( true );
    reger->SetEnableRigidRegistration( false );
    reger->SetEnableAffineRegistration( false );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "Rigid" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: Rigid" << std::endl;
      }
    reger->SetEnableInitialRegistration( false );
    reger->SetEnableRigidRegistration( true );
    reger->SetEnableAffineRegistration( false );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "Affine" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: Affine" << std::endl;
      }
    reger->SetEnableInitialRegistration( false );
    reger->SetEnableRigidRegistration( false );
    reger->SetEnableAffineRegistration( true );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "BSpline" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: BSpline" << std::endl;
      }
    reger->SetEnableInitialRegistration( false );
    reger->SetEnableRigidRegistration( false );
    reger->SetEnableAffineRegistration( false );
    reger->SetEnableBSplineRegistration( true );
    }
  else if( registration == "PipelineRigid" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: PipelineRigid" << std::endl;
      }
    reger->SetEnableInitialRegistration( true );
    reger->SetEnableRigidRegistration( true );
    reger->SetEnableAffineRegistration( false );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "PipelineAffine" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: PipelineAffine" << std::endl;
      }
    reger->SetEnableInitialRegistration( true );
    reger->SetEnableRigidRegistration( true );
    reger->SetEnableAffineRegistration( true );
    reger->SetEnableBSplineRegistration( false );
    }
  else if( registration == "PipelineBSpline" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Registration: PipelineBSpline" << std::endl;
      }
    reger->SetEnableInitialRegistration( true );
    reger->SetEnableRigidRegistration( true );
    reger->SetEnableAffineRegistration( true );
    reger->SetEnableBSplineRegistration( true );
    }

  if( metric == "NormCorr" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Metric: NormalizedCorrelation" << std::endl;
      }
    reger->SetRigidMetricMethodEnum( RegerType
                                       ::OptimizedRegistrationMethodType
                                       ::NORMALIZED_CORRELATION_METRIC );
    }
  else if( metric == "MeanSqrd" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Metric: MeanSquared" << std::endl;
      }
    reger->SetRigidMetricMethodEnum( RegerType
                                       ::OptimizedRegistrationMethodType
                                       ::MEAN_SQUARED_ERROR_METRIC );
    }
  else // if( metric == "MattesMI" )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Metric: MattesMutualInformation" << std::endl;
      }
    reger->SetRigidMetricMethodEnum( RegerType
                                       ::OptimizedRegistrationMethodType
                                       ::MATTES_MI_METRIC );
    }

  reger->SetUseOverlapAsROI( useOverlapAsROI );
  if (verbosity >= STANDARD)
    {
    std::cout << "### UseOverlapsAsROI: " << useOverlapAsROI << std::endl;
    }

  reger->SetMinimizeMemory( minimizeMemory );
  if (verbosity >= STANDARD)
    {
    std::cout << "### MinimizeMemory: " << minimizeMemory << std::endl;
    }

  reger->SetRandomNumberSeed( randomNumberSeed );

  reger->SetRigidMaxIterations( rigidMaxIterations );
  if (verbosity >= STANDARD)
    {
    std::cout << "### RigidMaxIterations: " << rigidMaxIterations 
              << std::endl;
    }

  reger->SetAffineMaxIterations( affineMaxIterations );
  if (verbosity >= STANDARD)
    {
    std::cout << "### AffineMaxIterations: " << affineMaxIterations 
              << std::endl;
    }

  reger->SetBSplineMaxIterations( bsplineMaxIterations );
  if (verbosity >= STANDARD)
    {
    std::cout << "### BSplineMaxIterations: " << bsplineMaxIterations 
              << std::endl;
    }

  reger->SetRigidSamplingRatio( rigidSamplingRatio );
  if (verbosity >= STANDARD)
    {
    std::cout << "### RigidSamplingRatio: " << rigidSamplingRatio 
              << std::endl;
    }
  reger->SetAffineSamplingRatio( affineSamplingRatio );
  if (verbosity >= STANDARD)
    {
    std::cout << "### AffineSamplingRatio: " << affineSamplingRatio 
              << std::endl;
    }
  reger->SetBSplineSamplingRatio( bsplineSamplingRatio );
  if (verbosity >= STANDARD)
    {
    std::cout << "### BSplineSamplingRatio: " << bsplineSamplingRatio 
              << std::endl;
    }

  reger->SetExpectedOffsetPixelMagnitude( expectedOffset );
  if (verbosity >= STANDARD)
    {
    std::cout << "### ExpectedOffsetPixelMagnitude: " << expectedOffset 
              << std::endl;
    }

  reger->SetExpectedRotationMagnitude( expectedRotation );
  if (verbosity >= STANDARD)
    {
    std::cout << "### ExpectedRotationMagnitude: " << expectedRotation 
              << std::endl;
    }

  reger->SetExpectedScaleMagnitude( expectedScale );
  if (verbosity >= STANDARD)
    {
    std::cout << "### ExpectedScaleMagnitude: " << expectedScale 
              << std::endl;
    }

  reger->SetExpectedSkewMagnitude( expectedSkew );
  if (verbosity >= STANDARD)
    {
    std::cout << "### ExpectedSkewMagnitude: " << expectedSkew 
              << std::endl;
    }

  reger->SetBSplineControlPointPixelSpacing( controlPointSpacing );
  if (verbosity >= STANDARD)
    {
    std::cout << "### ExpectedBSplineControlPointPixelSpacing: " 
              << controlPointSpacing 
              << std::endl;
    }

  try
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Starting registration..." << std::endl;
      }
    reger->Update();
    }
  catch( itk::ExceptionObject &excep)
    {
    std::cerr << "Exception caught during helper class registration." 
              << excep << std::endl;
    std::cerr << "Current Matrix Transform = " << std::endl; 
    reger->GetCurrentMatrixTransform()->Print(std::cerr , 2);
    return EXIT_FAILURE;
    }
  catch( ... )
    {
    std::cerr << "Uncaught exception during helper class registration." 
              << std::endl;
    return EXIT_FAILURE;
    }

  if( resampledImage.size() > 1 )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### Resampling..." << std::endl;
      }
    typename ImageType::ConstPointer resultImage;
    if( useWindowedSinc )
      {
      try
        {
        resultImage = reger->ResampleImage( RegerType
                                           ::OptimizedRegistrationMethodType
                                           ::SINC_INTERPOLATION );
        }
      catch( itk::ExceptionObject &excep)
        {
        std::cerr << "Exception caught during helper class resampling." 
                  << excep << std::endl;
        std::cerr << "Current Matrix Transform = " << std::endl; 
        reger->GetCurrentMatrixTransform()->Print(std::cerr , 2);
        return EXIT_FAILURE;
        }
      catch( ... )
        {
        std::cerr << "Uncaught exception during helper class resampling." 
                  << std::endl;
        return EXIT_FAILURE;
        }
      }
    else
      {
      try
        {
        resultImage = reger->ResampleImage( RegerType
                                           ::OptimizedRegistrationMethodType
                                           ::BSPLINE_INTERPOLATION );
        }
      catch( itk::ExceptionObject &excep)
        {
        std::cerr << "Exception caught during helper class resampling." 
                  << excep << std::endl;
        std::cerr << "Current Matrix Transform = " << std::endl; 
        reger->GetCurrentMatrixTransform()->Print(std::cerr , 2);
        return EXIT_FAILURE;
        }
      catch( ... )
        {
        std::cerr << "Uncaught exception during helper class resampling." 
                  << std::endl;
        return EXIT_FAILURE;
        }
      }
    
    try
      {
      reger->SaveImage( resampledImage, resultImage );
      }
    catch( itk::ExceptionObject &excep)
      {
      std::cerr << "Exception caught during helper class resampled image saving." 
                << excep << std::endl;
      return EXIT_FAILURE;
      }
    catch( ... )
      {
      std::cerr << "Uncaught exception during helper class resampled image saving." << std::endl;
      return EXIT_FAILURE;
      }
    }
    
  if( saveTransform.size() > 1 )
    {
    try
      {
      reger->SaveTransform( saveTransform );
      }
    catch( itk::ExceptionObject &excep)
      {
      std::cerr << "Exception caught during helper class transform saving." 
                << excep << std::endl;
      return EXIT_FAILURE;
      }
    catch( ... )
      {
      std::cerr << "Uncaught exception during helper class saving." << std::endl;
      return EXIT_FAILURE;
      }
    }
  
  /*
  if( saveParameters.size() > 1 )
    {
    try
      {
      reger->SaveParameters( saveParameters );
      }
    catch( itk::ExceptionObject &excep)
      {
      std::cerr << "Exception caught during helper class parameter saving." 
                << excep << std::endl;
      return EXIT_FAILURE;
      }
    catch( ... )
      {
      std::cerr << "Uncaught exception during helper class parameter saving." << std::endl;
      return EXIT_FAILURE;
      }
    }
  */

  if( baselineImage.size() > 1 )
    {
    try
      {
      reger->LoadBaselineImage( baselineImage );
      reger->SetBaselineNumberOfFailedPixelsTolerance( baselineNumberOfFailedPixelsTolerance );
      reger->SetBaselineIntensityTolerance( static_cast< typename ImageType::PixelType >( baselineIntensityTolerance ) );
      reger->SetBaselineRadiusTolerance( baselineRadiusTolerance );
      reger->ComputeBaselineDifference();
      if( baselineDifferenceImage.size() > 1 )
        {
        reger->SaveImage( baselineDifferenceImage,
                          reger->GetBaselineDifferenceImage() );
        }
      if( baselineResampledMovingImage.size() > 1 )
        {
        reger->SaveImage( baselineResampledMovingImage,
                          reger->GetBaselineResampledMovingImage() );
        }
      if( reger->GetBaselineTestPassed() )
        {
        std::cout << "Baseline test passed with "
                  << reger->GetBaselineNumberOfFailedPixels()
                  << " failed pixels." << std::endl;
        }
      else
        {
        std::cerr << "Baseline test failed with "
                  << reger->GetBaselineNumberOfFailedPixels()
                  << " failed pixels." << std::endl;
        return EXIT_FAILURE;
        }
      }
    catch( itk::ExceptionObject &excep)
      {
      std::cerr << "Exception caught during helper class baseline computations." 
                << excep << std::endl;
      return EXIT_FAILURE;
      }
    catch( ... )
      {
      std::cerr << "Uncaught exception during helper class baseline computations." << std::endl;
      return EXIT_FAILURE;
      }
    }
  return EXIT_SUCCESS;
}

int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  enum VerboseLevelEnum {SILENT, STANDARD, VERBOSE};
  VerboseLevelEnum verbosity = SILENT;
  if( verbosityLevel == "Standard" )
    {
    verbosity = STANDARD;
    }
  else if( verbosityLevel == "Verbose" )
    {
    verbosity = VERBOSE;
    }


  if( numberOfThreads != 0 )
    {
    if (verbosity >= STANDARD)
      {
      std::cout << "### numberOfThreads: " << numberOfThreads << std::endl;
      }
    itk::MultiThreader::SetGlobalDefaultNumberOfThreads(numberOfThreads);
    }

  unsigned int dimensions = 0;
  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;
 
  try
    {
    GetImageType( fixedImage, pixelType, componentType, dimensions ); 
    if( dimensions < 2 || dimensions > 3 )
      {
      std::cerr << "ERROR: Only 2 and 3 dimensional images supported."
                << std::endl;
      return EXIT_FAILURE;
      }

    switch( componentType )
      {
      case itk::ImageIOBase::UCHAR:
      case itk::ImageIOBase::CHAR:
      case itk::ImageIOBase::SHORT:
        if(dimensions == 2)
          {
          return DoIt<2, short>( argc, argv );
          }
        else
          {
          return DoIt<3, short>( argc, argv );
          }
        break;
      case itk::ImageIOBase::USHORT:
      case itk::ImageIOBase::UINT:
      case itk::ImageIOBase::INT:
      case itk::ImageIOBase::ULONG:
      case itk::ImageIOBase::LONG:
      case itk::ImageIOBase::FLOAT:
      case itk::ImageIOBase::DOUBLE:
        if(dimensions == 2)
          {
          return DoIt<2, float>( argc, argv );
          }
        else
          {
          return DoIt<3, float>( argc, argv );
          }
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cerr << "ERROR: unknown component type" << std::endl;
        return EXIT_FAILURE;
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

