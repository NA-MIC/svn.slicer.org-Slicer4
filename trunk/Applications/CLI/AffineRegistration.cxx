/*=========================================================================

  Program:   Registration stand-alone
  Module:    $RCSfile: $
  Language:  C++
  Date:      $Date: 2006-07-21 10:13:01 -0400 (Fri, 21 Jul 2006) $
  Version:   $Revision: 916 $

=========================================================================*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>

#include "AffineRegistrationCLP.h"

// ITK Stuff
// Registration
#include "itkOrientedImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkOrientImageFilter.h"

#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkAffineTransform.h"
#include "itkResampleImageFilter.h"

#include "itkTimeProbesCollectorBase.h"

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"
class CommandIterationUpdate : public itk::Command 
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizer  OptimizerType;
  typedef   const OptimizerType   *    OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
      Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
      OptimizerPointer optimizer = 
        dynamic_cast< OptimizerPointer >( object );
      if( !(itk::IterationEvent().CheckEvent( &event )) )
        {
        return;
        }
      std::cout << optimizer->GetCurrentIteration() << "   ";
      std::cout << optimizer->GetCurrentStepLength() << "   ";
      std::cout << optimizer->GetValue() << std::endl;
    }
};

const    unsigned int  ImageDimension = 3;
typedef  signed short  PixelType;
typedef itk::OrientedImage<PixelType, ImageDimension> ImageType;

int main ( int argc, char* argv[] ) 
{  
  //
  // Command line processing
  //
  PARSE_ARGS;

  typedef itk::ImageFileReader<ImageType> FileReaderType;
  typedef itk::OrientImageFilter<ImageType,ImageType> OrientFilterType;
  typedef itk::MattesMutualInformationImageToImageMetric<ImageType, ImageType>
    MetricType; 
  typedef itk::RegularStepGradientDescentOptimizer
    OptimizerType;    
  typedef itk::LinearInterpolateImageFunction<ImageType, double>
    InterpolatorType;
  typedef itk::ImageRegistrationMethod<ImageType,ImageType>
    RegistrationType;
  typedef itk::AffineTransform<double> TransformType;
  typedef OptimizerType::ScalesType OptimizerScalesType;
  typedef itk::ResampleImageFilter<ImageType,ImageType> ResampleType;
  typedef itk::LinearInterpolateImageFunction<ImageType, double> ResampleInterpolatorType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  bool DoInitializeTransform = false;
  int RandomSeed = 1234567;

  // Add a time probe
  itk::TimeProbesCollectorBase collector;

  /////////////////////////////////////////////////////////////////////////////
  // Read the fixed and moving volumes
  //
  FileReaderType::Pointer fixedReader = FileReaderType::New();
    fixedReader->SetFileName ( fixedImageFileName.c_str() );

  try
    {
    collector.Start( "Read fixed volume" );
    fixedReader->Update();
    collector.Stop( "Read fixed volume" );
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "Error Reading Fixed image: " << std::endl;
    return EXIT_FAILURE;
    }

  FileReaderType::Pointer movingReader = FileReaderType::New();
    movingReader->SetFileName ( movingImageFileName.c_str() );

  try
    {
    collector.Start( "Read moving volume" );
    movingReader->Update();
    collector.Stop( "Read moving volume" );
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "Error Reading Moving image: " << std::endl;
    return EXIT_FAILURE;
    }

  /////////////////////////////////////////////////////////////////////////////
  // Reorient the volumes to a consistent acquisition direction
  //
  OrientFilterType::Pointer orientFixed = OrientFilterType::New();
    orientFixed->UseImageDirectionOn();
    orientFixed->SetDesiredCoordinateOrientationToAxial();
    orientFixed->SetInput (fixedReader->GetOutput());
    collector.Start( "Orient fixed volume" );
    orientFixed->Update();
    collector.Stop( "Orient fixed volume" );

  OrientFilterType::Pointer orientMoving = OrientFilterType::New();
    orientMoving->UseImageDirectionOn();
    orientMoving->SetDesiredCoordinateOrientationToAxial();
    orientMoving->SetInput (movingReader->GetOutput());
    collector.Start( "Orient moving volume" );
    orientMoving->Update();
    collector.Stop( "Orient moving volume" );

  /////////////////////////////////////////////////////////////////////////////
  // Register the volumes
  //

  OptimizerType::Pointer      optimizer     = OptimizerType::New();
    optimizer->SetNumberOfIterations ( Iterations );
    optimizer->SetMinimumStepLength ( .0005 );
    optimizer->SetMaximumStepLength ( 10.0 );
    optimizer->SetMinimize(true);   

  TransformType::Pointer transform = TransformType::New();
  OptimizerScalesType scales( transform->GetNumberOfParameters() );
    scales.Fill ( 1.0 );
  for( unsigned j = 9; j < 12; j++ )
    {
    scales[j] = 1.0 / vnl_math_sqr(TranslationScale);
    }
    optimizer->SetScales( scales );

  // Create the Command observer and register it with the optimizer.
  //
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
    optimizer->AddObserver( itk::IterationEvent(), observer );

  /////////////////////////////////////////////////////////////////////////////
  // Initialize the transform
  //
  TransformType::InputPointType centerFixed;
  ImageType::RegionType::SizeType sizeFixed = orientFixed->GetOutput()->GetLargestPossibleRegion().GetSize();
  // Find the center
  ImageType::IndexType indexFixed;
  for ( unsigned j = 0; j < 3; j++ )
    {
    indexFixed[j] = (long) ( (sizeFixed[j]-1) / 2.0 );
    }
  orientFixed->GetOutput()->TransformIndexToPhysicalPoint ( indexFixed, centerFixed );

  TransformType::InputPointType centerMoving;
  ImageType::RegionType::SizeType sizeMoving = orientMoving->GetOutput()->GetLargestPossibleRegion().GetSize();
  // Find the center
  ImageType::IndexType indexMoving;
  for ( unsigned j = 0; j < 3; j++ )
    {
    indexMoving[j] = (long) ( (sizeMoving[j]-1) / 2.0 );
    }
  orientMoving->GetOutput()->TransformIndexToPhysicalPoint ( indexMoving, centerMoving );

  transform->Translate(centerMoving-centerFixed);
  std::cout << "---------------" << centerMoving-centerFixed << std::endl;
  
  MetricType::Pointer         metric        = MetricType::New();
    metric->SetNumberOfHistogramBins ( HistogramBins );
    metric->SetNumberOfSpatialSamples( SpatialSamples );

  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetTransform ( transform );
    registration->SetInitialTransformParameters ( transform->GetParameters() );
    registration->SetMetric ( metric );
    registration->SetOptimizer ( optimizer );
    registration->SetInterpolator ( interpolator );
    registration->SetFixedImage ( orientFixed->GetOutput() );
    registration->SetMovingImage ( orientMoving->GetOutput() );

  try
    {
    collector.Start( "Register" );
    registration->StartRegistration();     
    collector.Stop( "Register" );
    } 
  catch( itk::ExceptionObject & err )
    {
    std::cout << err << std::endl;
    exit ( EXIT_FAILURE );
    } 
  catch ( ... )
    {
    exit ( EXIT_FAILURE );
    }

  /////////////////////////////////////////////////////////////////////////////
  // Resample using the registration results
  //
  ResampleType::Pointer resample = ResampleType::New();
  ResampleInterpolatorType::Pointer Interpolator = ResampleInterpolatorType::New();

    transform->SetParameters ( registration->GetLastTransformParameters() );

    resample->SetInput ( orientMoving->GetOutput() ); 
    resample->SetTransform ( transform );
    resample->SetInterpolator ( Interpolator );
    resample->SetOutputOrigin ( orientFixed->GetOutput()->GetOrigin() );
    resample->SetOutputSpacing ( orientFixed->GetOutput()->GetSpacing() );
    resample->SetOutputDirection ( orientFixed->GetOutput()->GetDirection() );
    resample->SetSize ( orientFixed->GetOutput()->GetLargestPossibleRegion().GetSize() );
    collector.Start( "Resample" );
    resample->Update();
    collector.Stop( "Resample" );

  /////////////////////////////////////////////////////////////////////////////
  // Write the registerded volume
  //
  WriterType::Pointer resampledWriter = WriterType::New();
    resampledWriter->SetFileName ( resampledImageFileName.c_str() );
    resampledWriter->SetInput ( resample->GetOutput() );
  try
    {
    collector.Start( "Write volume" );
    resampledWriter->Write();
    collector.Stop( "Write volume" );
    }
  catch( itk::ExceptionObject & err )
    { 
    std::cerr << err << std::endl;
    exit ( EXIT_FAILURE );
    }
  
  // Report the time taken by the registration
  collector.Report();

  exit ( EXIT_SUCCESS );
}
  
