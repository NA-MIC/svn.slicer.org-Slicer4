#include "itkIO.h"
#ifdef USE_DEBUG_IMAGE_VIEWER
#include "DebugImageViewerClient.h"
#endif

#include "itkEuler3DTransform.h"
#include "itkCheckerBoardImageFilter.h"
#include "itkOtsuHistogramMatchingImageFilter.h"

#include "BRAINSFitHelper.h"

namespace itk
{

void ValidateTransformRankOrdering(const std::vector<std::string> & transformType)
{
  // Need to review transform types in the transform type vector to ensure that they are ordered appropriately
  // for processing.  I.e. that they go from low dimensional to high dimensional, and throw an error if
  // B-Spline is before Rigid, or any other non-sensical ordering of the transform types.
  //Rigid=1, ScaleVersor3D=2, Affine=4, ScaleSkewVersor3D=3, and BSpline=5
  unsigned int CurrentTransformRank=0;
  for(unsigned int l=0;l<transformType.size();l++)
    {
    if(transformType[l] == "Rigid")
      {
      if(CurrentTransformRank<=1)
        {
        CurrentTransformRank=1;
        }
      else
        {
        std::cerr << "Ordering of transforms does not proceed from\n"
          << "smallest to largest.  Please review settings for transformType.\n"
          << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
        exit (-1);
        }
      }
    else if(transformType[l] == "ScaleVersor3D")
      {
      if(CurrentTransformRank<=2)
        {
        CurrentTransformRank=2;
        }
      else
        {
        std::cerr << "Ordering of transforms does not proceed from\n"
          << "smallest to largest.  Please review settings for transformType.\n"
          << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
        exit (-1);
        }
      }
    else if(transformType[l] == "ScaleSkewVersor3D")
      {
      if(CurrentTransformRank<=3)
        {
        CurrentTransformRank=3;
        }
      else
        {
        std::cerr << "Ordering of transforms does not proceed from\n"
          << "smallest to largest.  Please review settings for transformType.\n"
          << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
        exit (-1);
        }
      }
    else if(transformType[l] == "Affine")
      {
      if(CurrentTransformRank<=4)
        {
        CurrentTransformRank=4;
        }
      else
        {
        std::cerr << "Ordering of transforms does not proceed from\n"
          << "smallest to largest.  Please review settings for transformType.\n"
          << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
        exit (-1);
        }
      }
    else if(transformType[l] == "BSpline")
      {
      if(CurrentTransformRank<=5)
        {
        CurrentTransformRank=5;
        }
      else
        {
        std::cerr << "Ordering of transforms does not proceed from\n"
          << "smallest to largest.  Please review settings for transformType.\n"
          << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
        exit (-1);
        }
      }
    else
      {
      std::cerr << " ERROR:  Invalid transform type specified for element " << l << " of --transformType: " << transformType[l] <<std::endl;
      std::cerr << "Ordering of transforms must proceed from\n"
        << "smallest to largest.  Please review settings for transformType.\n"
        << "Rigid < ScaleVersor3D < ScaleSkewVersor3D < Affine < BSpline" << std::endl;
      exit(-1);
      }
    }
}


template <class FixedVolumeType, class MovingVolumeType, class TransformType,
  class SpecificInitializerType>
typename TransformType::Pointer DoCenteredInitialization(
  typename FixedVolumeType::Pointer & orientedFixedVolume,
  typename MovingVolumeType::Pointer & orientedMovingVolume,
        ImageMaskPointer &fixedMask, //NOTE:  This is both input and output variable,  the Mask is updated by this function
        ImageMaskPointer &movingMask, //NOTE:  This is both input and output variable,  the Mask is updated by this function
  unsigned int NumberOfHistogramBins,
  unsigned int NumberOfSamples,
  std::string & initializeTransformMode)
{
  typename TransformType::Pointer initialITKTransform = TransformType::New();
  initialITKTransform->SetIdentity();

  if ( initializeTransformMode == "GeometryOn" )
    {
    // GeometryOn assumes objects are center in field of view, with different
    // physical extents to the fields of view
    typedef itk::CenteredTransformInitializer<TransformType, FixedVolumeType,
      MovingVolumeType> OrdinaryInitializerType;
    typename OrdinaryInitializerType::Pointer CenteredInitializer
      = OrdinaryInitializerType::New();

    CenteredInitializer->SetFixedImage(orientedFixedVolume);
    CenteredInitializer->SetMovingImage(orientedMovingVolume);
    CenteredInitializer->SetTransform(initialITKTransform);
    CenteredInitializer->GeometryOn(); // Use the image spce center
    CenteredInitializer->InitializeTransform();
    }
  else if ( initializeTransformMode == "CenterOfHead" )
    {
    typedef typename itk::ImageMaskSpatialObject<FixedVolumeType::ImageDimension> ImageMaskSpatialObjectType;
    typedef itk::Image<unsigned char, 3> MaskImageType;
    //     typename MovingVolumeType::PointType movingCenter =
    // GetCenterOfBrain<MovingVolumeType>(orientedMovingVolume);
    //     typename FixedVolumeType::PointType fixedCenter =
    // GetCenterOfBrain<FixedVolumeType>(orientedFixedVolume);
    typedef typename itk::FindCenterOfBrainFilter<MovingVolumeType>
    MovingFindCenterFilter;
    typename MovingFindCenterFilter::Pointer movingFindCenter
      = MovingFindCenterFilter::New();
    movingFindCenter->SetInput(orientedMovingVolume);
    if(movingMask.IsNotNull() )
      {
      typename ImageMaskSpatialObjectType::Pointer movingImageMask(
        dynamic_cast< ImageMaskSpatialObjectType * >( movingMask.GetPointer() ));
      typename MaskImageType::Pointer tempOutputMovingVolumeROI = const_cast<MaskImageType *>( movingImageMask->GetImage() );
      movingFindCenter->SetImageMask(tempOutputMovingVolumeROI);
      }
    movingFindCenter->Update();
    typename MovingVolumeType::PointType movingCenter
      = movingFindCenter->GetCenterOfBrain();
#if 1
    {
    // convert mask image to mask
    typedef typename itk::ImageMaskSpatialObject<Dimension>
      ImageMaskSpatialObjectType;
    typename ImageMaskSpatialObjectType::Pointer mask
      = ImageMaskSpatialObjectType::New();
    mask->SetImage(movingFindCenter->GetClippedImageMask());

    typename MaskImageType::Pointer ClippedMask=movingFindCenter->GetClippedImageMask();
    itkUtil::WriteImage<MaskImageType>( ClippedMask , std::string("MOVING_MASK.nii.gz"));

    mask->ComputeObjectToWorldTransform();
    typename SpatialObjectType::Pointer p= dynamic_cast<SpatialObjectType *>( mask.GetPointer() );
    if ( p.IsNull() )
      { 
      std::cout << "ERROR::" << __FILE__ << " " << __LINE__ << std::endl;
      exit(-1);
      }
    movingMask=p;
    }
#endif



    typedef typename itk::FindCenterOfBrainFilter<FixedVolumeType>
    FixedFindCenterFilter;
    typename FixedFindCenterFilter::Pointer fixedFindCenter
      = FixedFindCenterFilter::New();
    fixedFindCenter->SetInput(orientedFixedVolume);
    if(fixedMask.IsNotNull())
      {
      typename ImageMaskSpatialObjectType::Pointer fixedImageMask(
        dynamic_cast< ImageMaskSpatialObjectType * >( fixedMask.GetPointer() ));
      typename MaskImageType::Pointer tempOutputFixedVolumeROI = const_cast<MaskImageType *>( fixedImageMask->GetImage() );
      fixedFindCenter->SetImageMask(tempOutputFixedVolumeROI);
      }
    fixedFindCenter->Update();
    typename FixedVolumeType::PointType fixedCenter
      = fixedFindCenter->GetCenterOfBrain();
#if 1
    {
    // convert mask image to mask
    typedef typename itk::ImageMaskSpatialObject<Dimension>
      ImageMaskSpatialObjectType;
    typename ImageMaskSpatialObjectType::Pointer mask
      = ImageMaskSpatialObjectType::New();
    mask->SetImage(fixedFindCenter->GetClippedImageMask());

    typename MaskImageType::Pointer ClippedMask=fixedFindCenter->GetClippedImageMask();
    itkUtil::WriteImage<MaskImageType>(ClippedMask,std::string("FIXED_MASK.nii.gz"));

    mask->ComputeObjectToWorldTransform();
    typename SpatialObjectType::Pointer p= dynamic_cast<SpatialObjectType *>( mask.GetPointer() );
    if ( p.IsNull() )
      { 
      std::cout << "ERROR::" << __FILE__ << " " << __LINE__ << std::endl;
      exit(-1);
      }
    fixedMask=p;
    }
#endif

    //HACK:
    std::cout << "MOVING HEAD SIZE: " << movingFindCenter->GetHeadSizeEstimate() << std::endl;
    std::cout << "FIXED  HEAD SIZE: " <<  fixedFindCenter->GetHeadSizeEstimate() << std::endl;

    const double movingHeadScaleGuessRatio= 1;//fixedFindCenter->GetHeadSizeEstimate()/ movingFindCenter->GetHeadSizeEstimate();

    typename TransformType::InputPointType rotationCenter;
    typename TransformType::OutputVectorType translationVector;
    itk::Vector<double, 3> scaleValue;

    for ( unsigned int i = 0; i < Dimension; i++ )
      {
      rotationCenter[i]    = fixedCenter[i];
      translationVector[i] = movingCenter[i] - fixedCenter[i];
      scaleValue[i]=movingHeadScaleGuessRatio;
      }
#if 1
    typedef itk::Euler3DTransform<double> EulerAngle3DTransformType;
    typename EulerAngle3DTransformType::Pointer bestEulerAngles3D=EulerAngle3DTransformType::New();
    bestEulerAngles3D->SetCenter(rotationCenter);
    bestEulerAngles3D->SetTranslation(translationVector);

    typedef itk::Euler3DTransform<double> EulerAngle3DTransformType;
    typename EulerAngle3DTransformType::Pointer currentEulerAngles3D=EulerAngle3DTransformType::New();
    currentEulerAngles3D->SetCenter(rotationCenter);
    currentEulerAngles3D->SetTranslation(translationVector);

    double max_cc=0.0;
    //void QuickSampleParameterSpace(void)
      {

      typedef itk::MattesMutualInformationImageToImageMetric<
        FixedVolumeType,
        MovingVolumeType>    MetricType;
      typedef itk::LinearInterpolateImageFunction<
        MovingVolumeType,
        double>    InterpolatorType;

      typename MetricType::Pointer       metric         = MetricType::New();
      typename InterpolatorType::Pointer interpolator   = InterpolatorType::New();
      metric->SetInterpolator(interpolator);
      metric->SetFixedImage(orientedFixedVolume);
      metric->SetFixedImageRegion(orientedFixedVolume->GetLargestPossibleRegion() ); //This should be the bouding box of the fixedMask
      metric->SetMovingImage(orientedMovingVolume);

      metric->SetNumberOfHistogramBins( NumberOfHistogramBins );
      metric->SetNumberOfSpatialSamples( NumberOfSamples );
      // metric->SetUseAllPixels(true);  //DEBUG -- This was way too slow.
      //
      // set the masks on the metric
      if ( fixedMask.IsNotNull() )
        {
        metric->SetFixedImageMask(fixedMask);
        }
      if ( movingMask.IsNotNull() )
        {
        metric->SetMovingImageMask(movingMask);
        }
      //metric->SetUseExplicitPDFDerivatives( true ); // the default
      metric->SetTransform(currentEulerAngles3D);
      metric->Initialize();

      currentEulerAngles3D->SetRotation(0,0,0);
      //Initialize with current guess;
      max_cc=metric->GetValue( currentEulerAngles3D->GetParameters() );
      const double HARange=12.0;
      const double PARange=12.0;
      //rough search in neighborhood.
      const double one_degree=1.0F*vnl_math::pi/180.0F;
      const double HAStepSize=3.0*one_degree;
      const double PAStepSize=3.0*one_degree;
      //Quick search just needs to get an approximate angle correct.
        {
        for(double HA=-HARange*one_degree; HA<= HARange*one_degree; HA+=HAStepSize)
          {
          for(double PA=-PARange*one_degree; PA<= PARange*one_degree; PA+=PAStepSize)
            {
            currentEulerAngles3D->SetRotation(PA,0,HA);
            const double current_cc=metric->GetValue( currentEulerAngles3D->GetParameters() );
            if(current_cc < max_cc)
              {
              max_cc=current_cc;
              bestEulerAngles3D->SetFixedParameters(currentEulerAngles3D->GetFixedParameters());
              bestEulerAngles3D->SetParameters(currentEulerAngles3D->GetParameters());
              }
//#define DEBUGGING_PRINT_IMAGES
#ifdef DEBUGGING_PRINT_IMAGES
              {
              std::cout << "quick search "
                << " HA= " << (currentEulerAngles3D->GetParameters()[2])*180.0/vnl_math::pi
                << " PA= " << (currentEulerAngles3D->GetParameters()[0])*180.0/vnl_math::pi
                << " cc="  <<  current_cc
                << std::endl;
              }
            if ( 0 )
              {
              typedef itk::ResampleImageFilter<FixedVolumeType,MovingVolumeType,double> ResampleFilterType;
              typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();

              resampler->SetTransform( currentEulerAngles3D );
              resampler->SetInput( orientedMovingVolume );
              // Remember:  the Data is Moving's, the shape is Fixed's.
              resampler->SetOutputParametersFromImage(orientedFixedVolume);
              resampler->Update();       //  Explicit Update() required here.
              typename FixedVolumeType::Pointer ResampledImage = resampler->GetOutput();

              typedef itk::CheckerBoardImageFilter < FixedVolumeType > Checkerfilter;
              typename Checkerfilter::Pointer checker = Checkerfilter::New ();
              unsigned int array[3] = {36,36,36};

              checker->SetInput1 (orientedFixedVolume);
              checker->SetInput2 (ResampledImage);
              checker->SetCheckerPattern (array);
              try
                {
                checker->Update ();
                }
              catch (itk::ExceptionObject & err)
                {
                std::cout << "Caught an ITK exception: " << std::endl;
                std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
                }
              char filename[300];
              sprintf(filename,"%05.2f_%05.2f_%05.2f.nii.gz",(currentEulerAngles3D->GetParameters()[2])*180/vnl_math::pi,(currentEulerAngles3D->GetParameters()[0])*180/vnl_math::pi,current_cc);

                {
                typedef typename itk::ImageFileWriter<FixedVolumeType> WriterType;
                typename WriterType::Pointer writer = WriterType::New();
                writer->SetFileName( filename );
                writer->SetInput(checker->GetOutput());
                try
                  {
                  writer->Update();
                  }
                catch ( itk::ExceptionObject & err )
                  {
                  std::cout << "Exception Object caught: " << std::endl;
                  std::cout << err << std::endl;
                  throw;
                  }
                }
              }
#endif
            }
          }
        }
      //DEBUGGING_PRINT_IMAGES INFORMATION
#ifdef DEBUGGING_PRINT_IMAGES
        {
        std::cout << "FINAL: quick search "
          << " HA= " << (bestEulerAngles3D->GetParameters()[2])*180.0/vnl_math::pi
          << " PA= " << (bestEulerAngles3D->GetParameters()[0])*180.0/vnl_math::pi
          << " cc="  <<  max_cc
          << std::endl;
        }
#endif
      }
     typename VersorRigid3DTransformType::Pointer quickSetVersor=VersorRigid3DTransformType::New();
     quickSetVersor->SetCenter( bestEulerAngles3D->GetCenter() );
     quickSetVersor->SetTranslation( bestEulerAngles3D->GetTranslation() );
        {
        itk::Versor<double> localRotation;
        localRotation.Set(bestEulerAngles3D->GetRotationMatrix());
        quickSetVersor->SetRotation(localRotation);
        }
#ifdef DEBUGGING_PRINT_IMAGES
              {
              typedef itk::ResampleImageFilter<FixedVolumeType,MovingVolumeType,double> ResampleFilterType;
              typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();

              resampler->SetTransform( quickSetVersor );
              resampler->SetInput( orientedMovingVolume );
              // Remember:  the Data is Moving's, the shape is Fixed's.
              resampler->SetOutputParametersFromImage(orientedFixedVolume);
              resampler->Update();       //  Explicit Update() required here.
              typename FixedVolumeType::Pointer ResampledImage = resampler->GetOutput();

              typedef itk::CheckerBoardImageFilter < FixedVolumeType > Checkerfilter;
              typename Checkerfilter::Pointer checker = Checkerfilter::New ();
              unsigned int array[3] = {18,18,18};

              checker->SetInput1 (orientedFixedVolume);
              checker->SetInput2 (ResampledImage);
              checker->SetCheckerPattern (array);
              try
                {
                checker->Update ();
                }
              catch (itk::ExceptionObject & err)
                {
                std::cout << "Caught an ITK exception: " << std::endl;
                std::cout << err << " " << __FILE__ << " " << __LINE__ << std::endl;
                }
              char filename[300];
              sprintf(filename,"FINAL_%05.2f_%05.2f_%05.2f.nii.gz",(bestEulerAngles3D->GetParameters()[2])*180/vnl_math::pi,(bestEulerAngles3D->GetParameters()[0])*180/vnl_math::pi,max_cc);

                {
                typedef typename itk::ImageFileWriter<FixedVolumeType> WriterType;
                typename WriterType::Pointer writer = WriterType::New();
                writer->SetFileName( filename );
                //writer->SetInput(checker->GetOutput());
                writer->SetInput(ResampledImage);
                try
                  {
                  writer->Update();
                  }
                catch ( itk::ExceptionObject & err )
                  {
                  std::cout << "Exception Object caught: " << std::endl;
                  std::cout << err << std::endl;
                  throw;
                  }
                }
              }
#endif
#endif
            AssignRigid::AssignConvertedTransform(initialITKTransform,quickSetVersor);
    }
  else if ( initializeTransformMode == "MomentsOn" )
    {
    // MomentsOn assumes that the structures being registered have same amount
    // of mass approximately uniformly distributed.
    typename SpecificInitializerType::Pointer CenteredInitializer
      = SpecificInitializerType::New();

    CenteredInitializer->SetFixedImage(orientedFixedVolume);
    CenteredInitializer->SetMovingImage(orientedMovingVolume);
    CenteredInitializer->SetTransform(initialITKTransform);
    CenteredInitializer->MomentsOn(); // Use intensity center of mass

    CenteredInitializer->InitializeTransform();
    }
  else  // can't happen unless an unimplemented CLP option was added:
    {
    std::cout << "FAILURE:  Improper mode for initializeTransformMode: "
              << initializeTransformMode << std::endl;
    exit(-1);
    }
  std::cout << "Initializing transform with "  << initializeTransformMode
            << " to " << std::endl;
  std::cout << initialITKTransform << std::endl;
  std::cout << "===============================================" << std::endl;
  return initialITKTransform;
}





BRAINSFitHelper::BRAINSFitHelper()
  {
  m_TransformType.resize(1);
  m_TransformType[0]="Rigid";
  m_FixedVolume=NULL;
  m_MovingVolume=NULL;
  m_PreprocessedMovingVolume=NULL;
  m_FixedBinaryVolume=NULL;
  m_MovingBinaryVolume=NULL;
  m_PermitParameterVariation.resize(0);
  m_NumberOfSamples=500000;
  m_NumberOfHistogramBins=50;
  m_HistogramMatch=false;
  m_NumberOfMatchPoints=10;
  m_NumberOfIterations.resize(1);
  m_NumberOfIterations[0]=1500;
  m_MaximumStepLength=0.2;
  m_MinimumStepLength.resize(1);
  m_MinimumStepLength[0]=0.005;
  m_RelaxationFactor=0.5;
  m_TranslationScale=1000.0;
  m_ReproportionScale=1.0;
  m_SkewScale=1.0;
  m_UseExplicitPDFDerivativesMode="AUTO";
  m_UseCachingOfBSplineWeightsMode="ON";
  m_BackgroundFillValue=0.0;
  m_InitializeTransformMode="Off";
  m_MaskInferiorCutOffFromCenter=1000;
  m_CurrentGenericTransform=NULL;
  m_UseWindowedSinc=false;
  m_SplineGridSize.resize(3);
  m_SplineGridSize[0]=14;
  m_SplineGridSize[1]=10;
  m_SplineGridSize[2]=12;
  m_MaxBSplineDisplacement=0.0,
  m_CostFunctionConvergenceFactor = 1e+9,
  m_ProjectedGradientTolerance = 1e-4,
  m_ActualNumberOfIterations=0;
  m_PermittedNumberOfIterations=0;
  //m_AccumulatedNumberOfIterationsForAllLevels=0;
  m_DebugLevel=0;
  m_DisplayDeformedImage=false;
  m_PromptUserAfterDisplay=false;
  }

void
BRAINSFitHelper::
  StartRegistration (void)
{
  if ( m_HistogramMatch )
    {
    typedef itk::OtsuHistogramMatchingImageFilter<itk::Image<float, 3>, itk::Image<float, 3> > HistogramMatchingFilterType;
    HistogramMatchingFilterType::Pointer histogramfilter
      = HistogramMatchingFilterType::New();

    histogramfilter->SetInput( this->m_MovingVolume  );
    histogramfilter->SetReferenceImage( this->m_FixedVolume );

    histogramfilter->SetNumberOfHistogramLevels( m_NumberOfHistogramBins );
    histogramfilter->SetNumberOfMatchPoints( m_NumberOfMatchPoints );
    histogramfilter->ThresholdAtMeanIntensityOff();
    //histogramfilter->ThresholdAtMeanIntensityOn();
    histogramfilter->Update();
    this->m_PreprocessedMovingVolume=histogramfilter->GetOutput() ;
    }
  else
    {
    this->m_PreprocessedMovingVolume=this->m_MovingVolume;
    }

  if( this->m_DebugLevel > 3 )
    {
    this->PrintSelf(std::cout, 3 );
    }
  std::vector<double> localMinimumStepLength(m_TransformType.size() );
  if(m_MinimumStepLength.size() != m_TransformType.size() )
    {
    if(m_MinimumStepLength.size() != 1 )
      {
      std::cout << "ERROR:  Wrong number of parameters for MinimumStepLength. It either needs to be 1 or the same size as TransformType." << std::endl;
      exit(-1);
      }
    for(unsigned int q=0;q<m_TransformType.size();q++)
      {
      localMinimumStepLength[q]=m_MinimumStepLength[0];
      }
    }
  else
    {
    localMinimumStepLength=m_MinimumStepLength;
    }
  std::vector<int> localNumberOfIterations(m_TransformType.size() );
  if(m_NumberOfIterations.size() != m_TransformType.size() )
    {
    if(m_NumberOfIterations.size() != 1 )
      {
      std::cout << "ERROR:  Wrong number of parameters for NumberOfIterations. It either needs to be 1 or the same size as TransformType." << std::endl;
      exit(-1);
      }
    for(unsigned int q=0;q<m_TransformType.size();q++)
      {
      localNumberOfIterations[q]=m_NumberOfIterations[0];
      }
    }
  else
    {
    localNumberOfIterations=m_NumberOfIterations;
    }
  std::string localInitializeTransformMode(this->m_InitializeTransformMode);
  for ( unsigned int currentTransformIndex = 0; currentTransformIndex < m_TransformType.size(); currentTransformIndex++ )
    {
    const std::string currentTransformType(m_TransformType[currentTransformIndex]);
    std::cout << "TranformTypes: " <<
      currentTransformType << "(" << currentTransformIndex+1 << " of " << m_TransformType.size() << ")."
     << std::endl;
     std::cout << std::flush << std::endl;
    }

  for ( unsigned int currentTransformIndex = 0; currentTransformIndex < m_TransformType.size(); currentTransformIndex++ )
    {
    //m_AccumulatedNumberOfIterationsForAllLevels += localNumberOfIterations[currentTransformIndex];
    const std::string currentTransformType(m_TransformType[currentTransformIndex]);
    std::cout <<
    "\n\n\n=============================== Starting Transform Estimations for "
     << currentTransformType << "(" << currentTransformIndex+1 << " of " << m_TransformType.size() << ")."
     << "==============================="
     << std::endl;
     std::cout << std::flush << std::endl;
    //
    // Break into cases on TransformType:
    //
    if ( currentTransformType == "Rigid" )
      {
      //  Choose TransformType for the itk registration class template:
      typedef VersorRigid3DTransformType           TransformType;
      typedef itk::VersorRigid3DTransformOptimizer OptimizerType;
      //const int NumberOfEstimatedParameter = 6;

      //
      // Process the initialITKTransform as VersorRigid3DTransform:
      //
      TransformType::Pointer initialITKTransform = TransformType::New();
      initialITKTransform->SetIdentity();

      if ( localInitializeTransformMode != "Off" )
      // Use CenteredVersorTranformInitializer
        {
        std::cout << "Initializing transform " << currentTransformType << " with " << localInitializeTransformMode << std::endl;
        typedef itk::CenteredVersorTransformInitializer<FixedVolumeType,
          MovingVolumeType> InitializerType;
        initialITKTransform
          = DoCenteredInitialization<FixedVolumeType, MovingVolumeType,
          TransformType, InitializerType>(
          m_FixedVolume,
          m_PreprocessedMovingVolume,
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          this->m_NumberOfHistogramBins,
          this->m_NumberOfSamples,
          localInitializeTransformMode);
        DoCenteredTransformMaskClipping<TransformType,
          FixedVolumeType::ImageDimension>(
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          initialITKTransform,
          m_MaskInferiorCutOffFromCenter);
        }
      else if ( m_CurrentGenericTransform.IsNotNull() )
        {
        try
          {
          const std::string transformFileType = m_CurrentGenericTransform->GetNameOfClass();
          if ( transformFileType == "VersorRigid3DTransform" )
            {
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<VersorRigid3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if( (transformFileType == "ScaleVersor3DTransform" ) ||
          ( transformFileType == "ScaleSkewVersor3DTransform" ) ||
          ( transformFileType == "AffineTransform" ) )
            {
            //CONVERTING TO RIGID TRANSFORM TYPE from other type:
            std::cout << "WARNING:  Extracting Rigid component type from transform." << std::endl;
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform=ComputeRigidTransformFromGeneric(m_CurrentGenericTransform);
            AssignRigid::AssignConvertedTransform(initialITKTransform, tempInitializerITKTransform);
            }
            else
            {
            std::cout
                  <<
            "Unsupported initial transform file -- TransformBase first transform typestring, "
                  << transformFileType
                  << " not equal to required type VersorRigid3DTransform"
                  << std::endl;
            return;
            }
          }
        catch ( itk::ExceptionObject & excp )
          {
          std::cout << "[FAILED]" << std::endl;
          std::cerr
           << "Error while reading the m_CurrentGenericTransform" << std::endl;
          std::cerr << excp << std::endl;
          return;
          }
        }
#include "FitCommonCode.tmpl"
      }
    else if ( currentTransformType == "ScaleVersor3D" )
      {
      //  Choose TransformType for the itk registration class template:
      typedef ScaleVersor3DTransformType    TransformType;
      typedef itk::VersorTransformOptimizer OptimizerType;
      //const int NumberOfEstimatedParameter = 9;

      //
      // Process the initialITKTransform as ScaleVersor3DTransform:
      //
      TransformType::Pointer initialITKTransform = TransformType::New();
      initialITKTransform->SetIdentity();

      if ( localInitializeTransformMode != "Off" )     // Use
      // CenteredVersorTranformInitializer
        {
        std::cout << "Initializing transform " << currentTransformType << " with " << localInitializeTransformMode << std::endl;
        typedef itk::CenteredVersorTransformInitializer<FixedVolumeType,
          MovingVolumeType> InitializerType;
        VersorRigid3DTransformType::Pointer tempVersorITKTransform
          = DoCenteredInitialization<FixedVolumeType, MovingVolumeType,
          VersorRigid3DTransformType, InitializerType>(
          m_FixedVolume,
          m_PreprocessedMovingVolume,
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          this->m_NumberOfHistogramBins,
          this->m_NumberOfSamples,
          localInitializeTransformMode);
        AssignRigid::AssignConvertedTransform(initialITKTransform,
          tempVersorITKTransform);
        DoCenteredTransformMaskClipping<TransformType,
          FixedVolumeType::ImageDimension>(
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          initialITKTransform,
          m_MaskInferiorCutOffFromCenter);
        }
      else if ( m_CurrentGenericTransform.IsNotNull() )
        {
        try
          {
          const std::string transformFileType = m_CurrentGenericTransform->GetNameOfClass();
          if ( transformFileType == "VersorRigid3DTransform" )
            {
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<VersorRigid3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "ScaleVersor3DTransform" )
            {
            ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if( ( transformFileType == "ScaleSkewVersor3DTransform" ) ||
          ( transformFileType == "AffineTransform" ) )
            {
            //CONVERTING TO RIGID TRANSFORM TYPE from other type:  HACK we should preserve the Scale components
            std::cout << "WARNING:  Extracting Rigid component type from transform." << std::endl;
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform=ComputeRigidTransformFromGeneric(m_CurrentGenericTransform);
            AssignRigid::AssignConvertedTransform(initialITKTransform, tempInitializerITKTransform);
            }
          else      // || transformFileType == "ScaleSkewVersor3DTransform" ||
          // transformFileType == "AffineTransform"
            {
            std::cout
                  <<
            "Unsupported initial transform file -- TransformBase first transform typestring, "
                  << transformFileType
                  <<
            " not equal to required type VersorRigid3DTransform OR ScaleVersor3DTransform"
                  << std::endl;
            return;
            }
          }
        catch ( itk::ExceptionObject & excp )
          {
          std::cout << "[FAILED]" << std::endl;
          std::cerr
           << "Error while reading the m_CurrentGenericTransform"
           << std::endl;
          std::cerr << excp << std::endl;
          return;
          }
        }
#include "FitCommonCode.tmpl"
      }
    else if ( currentTransformType == "ScaleSkewVersor3D" )
      {
      //  Choose TransformType for the itk registration class template:
      typedef ScaleSkewVersor3DTransformType TransformType;
      typedef itk::VersorTransformOptimizer  OptimizerType;
      //const int NumberOfEstimatedParameter = 15;

      //
      // Process the initialITKTransform as ScaleSkewVersor3D:
      //
      TransformType::Pointer initialITKTransform = TransformType::New();
      initialITKTransform->SetIdentity();

      if ( localInitializeTransformMode != "Off" )     // Use
      // CenteredVersorTranformInitializer
        {
        std::cout << "Initializing transform " << currentTransformType << " with " << localInitializeTransformMode << std::endl;
        typedef itk::CenteredVersorTransformInitializer<FixedVolumeType,
          MovingVolumeType> InitializerType;
        VersorRigid3DTransformType::Pointer tempVersorITKTransform
          = DoCenteredInitialization<FixedVolumeType, MovingVolumeType,
          VersorRigid3DTransformType, InitializerType>(
          m_FixedVolume,
          m_PreprocessedMovingVolume,
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          this->m_NumberOfHistogramBins,
          this->m_NumberOfSamples,
          localInitializeTransformMode);
        AssignRigid::AssignConvertedTransform(initialITKTransform,
          tempVersorITKTransform);
        DoCenteredTransformMaskClipping<TransformType,
          FixedVolumeType::ImageDimension>(
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          initialITKTransform,
          m_MaskInferiorCutOffFromCenter);
        }
      else if ( m_CurrentGenericTransform.IsNotNull() )
        {
        try
          {
          const std::string transformFileType = m_CurrentGenericTransform->GetNameOfClass();
          if ( transformFileType == "VersorRigid3DTransform" )
            {
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<VersorRigid3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "ScaleVersor3DTransform" )
            {
            ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "ScaleSkewVersor3DTransform" )
            {
            ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleSkewVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if( ( transformFileType == "AffineTransform" ) )
            {
            //CONVERTING TO RIGID TRANSFORM TYPE from other type: HACK:  We should really preserve the Scale and Skew components
            std::cout << "WARNING:  Extracting Rigid component type from transform." << std::endl;
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform=ComputeRigidTransformFromGeneric(m_CurrentGenericTransform);
            AssignRigid::AssignConvertedTransform(initialITKTransform, tempInitializerITKTransform);
            }
          else      // || transformFileType == "AffineTransform" ||
                    // transformFileType
          // == "ScaleVersor3DTransform"
            {
            std::cout
                  <<
            "Unsupported initial transform file -- TransformBase first transform typestring, "
                  << transformFileType
                  <<
            " not equal to required type VersorRigid3DTransform OR ScaleVersor3DTransform OR ScaleSkewVersor3DTransform"
                  << std::endl;
            return;
            }
          }
        catch ( itk::ExceptionObject & excp )
          {
          std::cout << "[FAILED]" << std::endl;
          std::cerr
           << "Error while reading the m_CurrentGenericTransform"
           << std::endl;
          std::cerr << excp << std::endl;
          return;
          }
        }
#include "FitCommonCode.tmpl"
      }
    else if ( currentTransformType == "Affine" )
      {
      //  Choose TransformType for the itk registration class template:
      typedef itk::AffineTransform<double, Dimension>  TransformType;
      typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
      //const int NumberOfEstimatedParameter = 12;

      //
      // Process the initialITKTransform
      //
      TransformType::Pointer initialITKTransform = TransformType::New();
      initialITKTransform->SetIdentity();

      if ( localInitializeTransformMode != "Off" )     // Use
                                                    //
                                                    // CenteredTranformInitializer
        {
        std::cout << "Initializing transform " << currentTransformType << " with " << localInitializeTransformMode << std::endl;
        typedef itk::CenteredTransformInitializer<TransformType,
          FixedVolumeType,
          MovingVolumeType> InitializerType;
        initialITKTransform
          = DoCenteredInitialization<FixedVolumeType, MovingVolumeType,
          TransformType, InitializerType>(
          m_FixedVolume,
          m_PreprocessedMovingVolume,
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          this->m_NumberOfHistogramBins,
          this->m_NumberOfSamples,
          localInitializeTransformMode);
        DoCenteredTransformMaskClipping<TransformType,
          FixedVolumeType::ImageDimension>(
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          initialITKTransform,
          m_MaskInferiorCutOffFromCenter);
        }
      else if ( m_CurrentGenericTransform.IsNotNull() )
        {
        try
          {
          const std::string transformFileType = m_CurrentGenericTransform->GetNameOfClass();
          if ( transformFileType == "VersorRigid3DTransform" )
            {
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<VersorRigid3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "ScaleVersor3DTransform" )
            {
            ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "ScaleSkewVersor3DTransform" )
            {
            ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleSkewVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else if ( transformFileType == "AffineTransform" )
            {
            AffineTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<AffineTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(initialITKTransform,
              tempInitializerITKTransform);
            }
          else      //  NO SUCH CASE!!
            {
            std::cout
                  <<
            "Unsupported initial transform file -- TransformBase first transform typestring, "
                  << transformFileType
                  <<
            " not equal to any recognized type VersorRigid3DTransform OR ScaleVersor3DTransform OR ScaleSkewVersor3DTransform OR AffineTransform"
                  << std::endl;
            return;
            }
          }
        catch ( itk::ExceptionObject & excp )
          {
          std::cout << "[FAILED]" << std::endl;
          std::cerr
           << "Error while reading the m_CurrentGenericTransform"
           << std::endl;
          std::cerr << excp << std::endl;
          return;
          }
        }

#include "FitCommonCode.tmpl"
      }
    else if ( currentTransformType == "BSpline" )
      {
      //
      // Process the bulkAffineTransform for BSpline's BULK
      //
      AffineTransformType::Pointer bulkAffineTransform
        = AffineTransformType::New();
      bulkAffineTransform->SetIdentity();

      typedef itk::Image<float, 3> RegisterImageType;

      BSplineTransformType::Pointer outputBSplineTransform
        = BSplineTransformType::New();
      outputBSplineTransform->SetIdentity();

      BSplineTransformType::Pointer initialBSplineTransform
        = BSplineTransformType::New();
      initialBSplineTransform->SetIdentity();

        {
        typedef BSplineTransformType::RegionType TransformRegionType;
        typedef TransformRegionType::SizeType    TransformSizeType;
        typedef itk::BSplineDeformableTransformInitializer<
          BSplineTransformType,
          RegisterImageType>      InitializerType;
        InitializerType::Pointer transformInitializer = InitializerType::New();
        transformInitializer->SetTransform( initialBSplineTransform );
        transformInitializer->SetImage( m_FixedVolume );
        TransformSizeType tempGridSize;
        tempGridSize[0] = m_SplineGridSize[0];
        tempGridSize[1] = m_SplineGridSize[1];
        tempGridSize[2] = m_SplineGridSize[2];
        transformInitializer->SetGridSizeInsideTheImage( tempGridSize );
        transformInitializer->InitializeTransform();
        }


      if ( localInitializeTransformMode != "Off" )     // Use
        //
        // CenteredTranformInitializer
        {
        std::cout << "Initializing transform " << currentTransformType << " with " << localInitializeTransformMode << std::endl;
        typedef itk::CenteredTransformInitializer<AffineTransformType,
                FixedVolumeType,
                MovingVolumeType> InitializerType;
        bulkAffineTransform
          = DoCenteredInitialization<FixedVolumeType, MovingVolumeType,
          AffineTransformType, InitializerType>(
            m_FixedVolume,
            m_PreprocessedMovingVolume,
          m_FixedBinaryVolume,
          m_MovingBinaryVolume,
          this->m_NumberOfHistogramBins,
          this->m_NumberOfSamples,
            localInitializeTransformMode);
        DoCenteredTransformMaskClipping<AffineTransformType,
          FixedVolumeType::ImageDimension>(
            m_FixedBinaryVolume,
            m_MovingBinaryVolume,
            bulkAffineTransform,
            m_MaskInferiorCutOffFromCenter);
        initialBSplineTransform->SetBulkTransform(   bulkAffineTransform   );
        }
      else if ( m_CurrentGenericTransform.IsNotNull() )
        {
        try
          {
          const std::string transformFileType = m_CurrentGenericTransform->GetNameOfClass();
          if ( transformFileType == "VersorRigid3DTransform" )
            {
            VersorRigid3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<VersorRigid3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(bulkAffineTransform,
              tempInitializerITKTransform);
            initialBSplineTransform->SetBulkTransform(   bulkAffineTransform   );
            }
          else if ( transformFileType == "ScaleVersor3DTransform" )
            {
            ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(bulkAffineTransform,
              tempInitializerITKTransform);
            initialBSplineTransform->SetBulkTransform(   bulkAffineTransform   );
            }
          else if ( transformFileType == "ScaleSkewVersor3DTransform" )
            {
            ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<ScaleSkewVersor3DTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(bulkAffineTransform,
              tempInitializerITKTransform);
            initialBSplineTransform->SetBulkTransform(   bulkAffineTransform   );
            }
          else if ( transformFileType == "AffineTransform" )
            {
            AffineTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<AffineTransformType *>( m_CurrentGenericTransform.GetPointer() );
            AssignRigid::AssignConvertedTransform(bulkAffineTransform,
              tempInitializerITKTransform);
            initialBSplineTransform->SetBulkTransform(   bulkAffineTransform   );
            }
          else if ( transformFileType == "BSplineDeformableTransform" )
            {
            BSplineTransformType::Pointer tempInitializerITKTransform
              = dynamic_cast<BSplineTransformType *>( m_CurrentGenericTransform.GetPointer() );

            initialBSplineTransform->SetBulkTransform(
              tempInitializerITKTransform->GetBulkTransform()   );
            BSplineTransformType::ParametersType tempFixedInitialParameters
              = tempInitializerITKTransform->GetFixedParameters();
            BSplineTransformType::ParametersType initialFixedParameters
              = initialBSplineTransform->GetFixedParameters();

            bool checkMatch = true; //Assume true;
            if( initialFixedParameters.GetSize() != tempFixedInitialParameters.GetSize() )
              {
              checkMatch=false;
              std::cerr << "ERROR INITILIZATION FIXED PARAMETERS DO NOT MATCH: " << initialFixedParameters.GetSize() << " != " << tempFixedInitialParameters.GetSize() << std::endl;
              }
            if ( checkMatch )    //  This ramus covers the hypothesis that the
              // FixedParameters represent the grid locations of the spline nodes.
              {
              for ( unsigned int i = 0; i < initialFixedParameters.GetSize(); i++ )
                {
                if( initialFixedParameters.GetElement(i) != tempFixedInitialParameters.GetElement(i) )
                  {
                  checkMatch = false;
                  std::cerr << "ERROR FIXED PARAMETERS DO NOT MATCH: " << initialFixedParameters.GetElement(i) << " != " << tempFixedInitialParameters.GetElement(i) << std::endl;
                  }
                }
              }
            if ( checkMatch )
              {
              BSplineTransformType::ParametersType tempInitialParameters
                = tempInitializerITKTransform->GetParameters();
              if ( initialBSplineTransform->GetNumberOfParameters() ==
                tempInitialParameters.Size() )
                {
                initialBSplineTransform->SetFixedParameters(
                  tempFixedInitialParameters );
                initialBSplineTransform->SetParametersByValue( tempInitialParameters );
                }
              else
                {
                // Error, initializing from wrong size transform parameters;
                //  Use its bulk transform only?
                std::cerr <<
                  "Trouble using the m_CurrentGenericTransform for initializing a BSPlineDeformableTransform:"
                  << std::endl;
                std::cerr <<
                  "The initializing BSplineDeformableTransform has a different number of Parameters, than what is required for the requested grid."
                  << std::endl;
                std::cerr <<
                  "BRAINSFit was only able to use the bulk transform that was before it."
                  << std::endl;
                exit(-1);
                }
              }
            else
              {
              std::cerr << "ERROR:  initialization BSpline transform does not have the same parameter dimensions as the one currently specified." << std::endl;
              exit(-1);
              }
            }
          else
            {
            std::cerr << "ERROR:  Invalid transform initializer type found:  " << transformFileType << std::endl;
            exit(-1);
            }
          }
        catch ( itk::ExceptionObject & excp )
          {
          std::cout << "[FAILED]" << std::endl;
          std::cerr
            << "Error while reading the m_CurrentGenericTransform"
            << std::endl;
          std::cerr << excp << std::endl;
          return;
          }
        }
      const bool UseCachingOfBSplineWeights
        = ( m_UseCachingOfBSplineWeightsMode == "ON" ) ? true : false;
      // As recommended in documentation in
      // itkMattesMutualInformationImageToImageMetric.h
      // "UseExplicitPDFDerivatives = False ... This method is well suited
      // for Transforms with a large number of parameters, such as,
      // BSplineDeformableTransforms."
      const bool UseExplicitPDFDerivatives
        =  ( m_UseExplicitPDFDerivativesMode == "ON" ) ? true : false;

      outputBSplineTransform
        = DoBSpline<RegisterImageType, SpatialObjectType,
        BSplineTransformType>(
          initialBSplineTransform,
          m_FixedVolume, m_PreprocessedMovingVolume,
          m_FixedBinaryVolume, m_MovingBinaryVolume,
          m_NumberOfSamples,
          UseCachingOfBSplineWeights, UseExplicitPDFDerivatives,
          this->m_MaxBSplineDisplacement,
          this->m_CostFunctionConvergenceFactor,
          this->m_ProjectedGradientTolerance,
          this->m_DisplayDeformedImage,
          this->m_PromptUserAfterDisplay);
      if (outputBSplineTransform.IsNull() )
        {
        std::cout
          << "Error -- the BSpline fit has failed." <<std::endl;
        std::cout
          << "Error -- the BSpline fit has failed." <<std::endl;

        m_ActualNumberOfIterations = 1;
        m_PermittedNumberOfIterations = 1;

        }
      else
        {
        //Initialize next level of transformations with previous transform result
        //TransformList.clear();
        //TransformList.push_back(finalTransform);
        m_CurrentGenericTransform=outputBSplineTransform;
        localInitializeTransformMode="Off"; //Now turn of the initiallize code to off
        }
      }
    else
      {
      std::cout
        << "Error choosing what kind of transform to fit \""
        << currentTransformType << "(" << currentTransformIndex + 1 << " of " << m_TransformType.size() << "). " <<std::endl;
      std::cout << std::flush << std::endl;
      exit(-1);
      return;
      }
    }
  return;
}

void
BRAINSFitHelper::
  PrintSelf(std::ostream & os, Indent indent) const
{
  // Superclass::PrintSelf(os,indent);
  os << indent << "FixedVolume:\n"  <<   this->m_FixedVolume << std::endl;
  os << indent << "MovingVolume:\n" <<   this->m_MovingVolume << std::endl;
  os << indent << "PreprocessedMovingVolume:\n" <<   this->m_PreprocessedMovingVolume << std::endl;
  if (this->m_FixedBinaryVolume.IsNotNull() )
    {
    os << indent << "FixedBinaryVolume:\n" << this->m_FixedBinaryVolume << std::endl;
    }
  else
    {
    os << indent << "FixedBinaryVolume: IS NULL" << std::endl;
    }
  if (this->m_MovingBinaryVolume.IsNotNull() )
    {
    os << indent << "MovingBinaryVolume:\n" << this->m_MovingBinaryVolume << std::endl;
    }
  else
    {
    os << indent << "MovingBinaryVolume: IS NULL" << std::endl;
    }
  os << indent << "NumberOfSamples:      " << this->m_NumberOfSamples << std::endl ;

  os << indent << "NumberOfIterations:    [" ;
  for(unsigned int q=0; q< this->m_NumberOfIterations.size(); q++)
    {
     os << this->m_NumberOfIterations[q] << " ";
    }
  os << "]" << std::endl;
  os << indent << "NumberOfHistogramBins:" << this->m_NumberOfHistogramBins << std::endl;
  os << indent << "MaximumStepLength:    " << this->m_MaximumStepLength << std::endl;
  os << indent << "MinimumStepLength:     [" ;
  for(unsigned int q=0; q< this->m_MinimumStepLength.size(); q++)
    {
    os << this->m_MinimumStepLength[q] << " ";
    }
  os << "]" << std::endl;
  os << indent << "TransformType:     [" ;
  for(unsigned int q=0; q< this->m_TransformType.size(); q++)
    {
    os << this->m_TransformType[q] << " ";
    }
  os << "]" << std::endl;

  os << indent << "RelaxationFactor:    " << this->m_RelaxationFactor << std::endl;
  os << indent << "TranslationScale:    " << this->m_TranslationScale << std::endl;
  os << indent << "ReproportionScale:   " << this->m_ReproportionScale << std::endl;
  os << indent << "SkewScale:           " << this->m_SkewScale << std::endl;
  os << indent << "UseExplicitPDFDerivativesMode:  " << this->m_UseExplicitPDFDerivativesMode << std::endl;
  os << indent << "UseCachingOfBSplineWeightsMode: " << this->m_UseCachingOfBSplineWeightsMode << std::endl;
  os << indent << "BackgroundFillValue:            " << this->m_BackgroundFillValue << std::endl;
  os << indent << "InitializeTransformMode:        " << this->m_InitializeTransformMode << std::endl;
  os << indent << "MaskInferiorCutOffFromCenter:   " << this->m_MaskInferiorCutOffFromCenter << std::endl;
  os << indent << "UseWindowedSinc:                " << this->m_UseWindowedSinc << std::endl;
  os << indent << "ActualNumberOfIterations:       " << this->m_ActualNumberOfIterations << std::endl;
  os << indent << "PermittedNumberOfIterations:       " << this->m_PermittedNumberOfIterations << std::endl;
  //os << indent << "AccumulatedNumberOfIterationsForAllLevels: " << this->m_AccumulatedNumberOfIterationsForAllLevels << std::endl;

  os << indent << "SplineGridSize:     [" ;
  for(unsigned int q=0; q< this->m_SplineGridSize.size(); q++)
    {
    os << this->m_SplineGridSize[q] << " ";
    }
  os << "]" << std::endl;


  os << indent << "PermitParameterVariation:     [" ;
  for(unsigned int q=0; q< this->m_PermitParameterVariation.size(); q++)
    {
    os << this->m_PermitParameterVariation[q] << " ";
    }
  os << "]" << std::endl;

  if(m_CurrentGenericTransform.IsNotNull())
    {
    os << indent << "CurrentGenericTransform:\n" << this->m_CurrentGenericTransform << std::endl;
    }
  else
    {
    os << indent << "CurrentGenericTransform: IS NULL" << std::endl;
    }
}

void
BRAINSFitHelper::
  PrintCommandLine(const bool dumpTempVolumes, const std::string suffix) const
{
  std::cout << "The equivalent command line to the current run would be:" <<std::endl;

  std::cout << "BRAINSFit \\" << std::endl;
  const std::string fixedVolumeString("DEBUGFixedVolume_"+ suffix+".nii.gz");
  const std::string movingVolumeString("DEBUGMovingVolume_"+suffix+".nii.gz");
  const std::string fixedBinaryVolumeString("DEBUGFixedBinaryVolume_"+suffix+".nii.gz");
  const std::string movingBinaryVolumeString("DEBUGMovingBinaryVolume_"+suffix+".nii.gz");
  if(dumpTempVolumes == true)
    {
      {
      typedef itk::ImageFileWriter<FixedVolumeType> WriterType;
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName( fixedVolumeString );
      writer->SetInput(this->m_FixedVolume);
      try
        {
        writer->Update();
        }
      catch ( itk::ExceptionObject & err )
        {
        std::cout << "Exception Object caught: " << std::endl;
        std::cout << err << std::endl;
        throw;
        }
      }
      {
      typedef itk::ImageFileWriter<MovingVolumeType> WriterType;
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName( movingVolumeString );
      writer->SetInput(this->m_MovingVolume);
      try
        {
        writer->Update();
        }
      catch ( itk::ExceptionObject & err )
        {
        std::cout << "Exception Object caught: " << std::endl;
        std::cout << err << std::endl;
        throw;
        }
      }
    }
  std::cout  << "--fixedVolume "  <<  fixedVolumeString   << "  \\" << std::endl;
  std::cout  << "--movingVolume " <<  movingVolumeString  << "  \\" << std::endl;
  std::cout  << "--histogramMatch " <<  "  \\" << std::endl;

    {
    if (this->m_FixedBinaryVolume.IsNotNull() )
      {
      std::cout  << "--fixedBinaryVolume " << fixedBinaryVolumeString  << "  \\" << std::endl;
        {
        typedef itk::Image<unsigned char, 3> MaskImageType;
        typedef itk::ImageMaskSpatialObject<MaskImageType::ImageDimension> ImageMaskSpatialObjectType;
          {
          ImageMaskSpatialObjectType::Pointer fixedImageMask(
            dynamic_cast< ImageMaskSpatialObjectType * >( m_FixedBinaryVolume.GetPointer() ));
          MaskImageType::Pointer tempOutputFixedVolumeROI = const_cast<MaskImageType *>( fixedImageMask->GetImage() );
          itkUtil::WriteImage<MaskImageType>(tempOutputFixedVolumeROI,fixedBinaryVolumeString);
          }
        }
      }
    if (this->m_MovingBinaryVolume.IsNotNull() )
      {
      std::cout  << "--movingBinaryVolume " << movingBinaryVolumeString  << "  \\" << std::endl;
        {
        typedef itk::Image<unsigned char, 3> MaskImageType;
        typedef itk::ImageMaskSpatialObject<MaskImageType::ImageDimension> ImageMaskSpatialObjectType;
          {
          ImageMaskSpatialObjectType::Pointer movingImageMask(
            dynamic_cast< ImageMaskSpatialObjectType * >( m_MovingBinaryVolume.GetPointer() ));
          MaskImageType::Pointer tempOutputMovingVolumeROI = const_cast<MaskImageType *>( movingImageMask->GetImage() );
          itkUtil::WriteImage<MaskImageType>(tempOutputMovingVolumeROI,movingBinaryVolumeString);
          }
        }
      }
    if( this->m_FixedBinaryVolume.IsNotNull()  || this->m_MovingBinaryVolume.IsNotNull() )
      {
      std::cout  << "--maskProcessingMode ROI "   << "  \\" << std::endl ;
      }
    }
  std::cout  << "--numberOfSamples " << this->m_NumberOfSamples  << "  \\" << std::endl ;

  std::cout  << "--numberOfIterations " ;
  for(unsigned int q=0; q< this->m_NumberOfIterations.size(); q++)
    {
     std::cout << this->m_NumberOfIterations[q] ;
     if(q < this->m_NumberOfIterations.size() -1 )
       {
       std::cout << ",";
       }
    }
  std::cout << " \\" << std::endl;
  std::cout  << "--numberOfHistogramBins " << this->m_NumberOfHistogramBins  << "  \\" << std::endl;
  std::cout  << "--maximumStepSize " << this->m_MaximumStepLength  << "  \\" << std::endl;
  std::cout  << "--minimumStepSize " ;
  for(unsigned int q=0; q< this->m_MinimumStepLength.size(); q++)
    {
    std::cout << this->m_MinimumStepLength[q];
    if(q < this->m_MinimumStepLength.size() -1)
      {
      std::cout << ",";
      }
    }
  std::cout << " \\" << std::endl;
  std::cout  << "--transformType " ;
  for(unsigned int q=0; q< this->m_TransformType.size(); q++)
    {
    std::cout << this->m_TransformType[q] ;
    if(q < this->m_TransformType.size() -1)
      {
      std::cout << ",";
      }
    }
  std::cout << " \\" << std::endl;

  std::cout  << "--relaxationFactor " << this->m_RelaxationFactor  << "  \\" << std::endl;
  std::cout  << "--translationScale " << this->m_TranslationScale  << "  \\" << std::endl;
  std::cout  << "--reproportionScale " << this->m_ReproportionScale  << "  \\" << std::endl;
  std::cout  << "--skewScale " << this->m_SkewScale  << "  \\" << std::endl;
  std::cout  << "--useExplicitPDFDerivativesMode " << this->m_UseExplicitPDFDerivativesMode  << "  \\" << std::endl;
  std::cout  << "--useCachingOfBSplineWeightsMode " << this->m_UseCachingOfBSplineWeightsMode  << "  \\" << std::endl;
  std::cout  << "--backgroundFillValue " << this->m_BackgroundFillValue  << "  \\" << std::endl;
  std::cout  << "--initializeTransformMode " << this->m_InitializeTransformMode  << "  \\" << std::endl;
  std::cout  << "--maskInferiorCutOffFromCenter " << this->m_MaskInferiorCutOffFromCenter  << "  \\" << std::endl;
   if( this->m_UseWindowedSinc == true )
     {
     std::cout  << "--useWindowedSinc " ;
     }

  std::cout  << "--splineGridSize " ;
  for(unsigned int q=0; q< this->m_SplineGridSize.size(); q++)
    {
    std::cout << this->m_SplineGridSize[q] ;
    if (q < this->m_SplineGridSize.size() -1)
      {
      std::cout << ",";
      }
    }
  std::cout << " \\" << std::endl;

  if(this->m_PermitParameterVariation.size() > 0)
    {
    std::cout  << "--permitParameterVariation " ;
    for(unsigned int q=0; q< this->m_PermitParameterVariation.size(); q++)
      {
      std::cout << this->m_PermitParameterVariation[q] ;
      if ( q< this->m_PermitParameterVariation.size() -1)
        {
        std::cout << ",";
        }
      }
    std::cout << " \\" << std::endl;
    }

  if(m_CurrentGenericTransform.IsNotNull())
    {
    std::cout  << "--currentGenericTransform " << "<INITIAL TRANSFORM>"  << "  \\" << std::endl;
    }
}

void
BRAINSFitHelper::
  GenerateData ()
{
  this->StartRegistration();
}

VersorRigid3DTransformType::Pointer ComputeRigidTransformFromGeneric(GenericTransformType::Pointer & genericTransformToWrite)
{
  typedef VersorRigid3DTransformType VersorRigidTransformType;
  VersorRigidTransformType::Pointer versorRigid = VersorRigidTransformType::New();
  versorRigid->SetIdentity();
  ////////////////////////////////////////////////////////////////////////////
  // ConvertTransforms
  //
  if ( genericTransformToWrite.IsNotNull() )
    {
    try
      {
      const std::string transformFileType = genericTransformToWrite->GetNameOfClass();
      if ( transformFileType == "VersorRigid3DTransform" )
        {
        VersorRigid3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<VersorRigid3DTransformType *>( genericTransformToWrite.GetPointer() );
        AssignRigid::ExtractVersorRigid3DTransform(versorRigid, tempInitializerITKTransform);
        }
      else if ( transformFileType == "ScaleVersor3DTransform" )
        {
        ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<ScaleVersor3DTransformType *>( genericTransformToWrite.GetPointer() );
        AssignRigid::ExtractVersorRigid3DTransform(versorRigid, tempInitializerITKTransform);
        }
      else if ( transformFileType == "ScaleSkewVersor3DTransform" )
        {
        ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<ScaleSkewVersor3DTransformType *>( genericTransformToWrite.GetPointer() );
        AssignRigid::ExtractVersorRigid3DTransform(versorRigid, tempInitializerITKTransform);
        }
      else if ( transformFileType == "AffineTransform" )
        {
        AffineTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<AffineTransformType *>( genericTransformToWrite.GetPointer() );
        AssignRigid::ExtractVersorRigid3DTransform(versorRigid, tempInitializerITKTransform);
        }
#if 0
      else if(transformFileType == "BSplineDeformableTransform")
        {
        BSplineTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<BSplineTransformType *>( genericTransformToWrite.GetPointer() );
        //AssignRigid::ExtractVersorRigid3DTransform(versorRigid, tempInitializerITKTransform->GetBulkTransform());
        versorRigid=NULL; //NOT: Perhaps it makes sense to extract the rigid part of the bulk transform.  But that is pretty obscure case.
        return NULL;
        }
#endif
      else      //  NO SUCH CASE!!
        {
        std::cout
          << "Unsupported initial transform file -- TransformBase first transform typestring, "
          << transformFileType
          << " not equal to any recognized type VersorRigid3DTransform OR "
          << " ScaleVersor3DTransform OR ScaleSkewVersor3DTransform OR AffineTransform"
          << std::endl;
        return NULL;
        }
      }
    catch ( itk::ExceptionObject & excp )
      {
      std::cout << "[FAILED]" << std::endl;
      std::cerr
        << "Error while converting the genericTransformToWrite to Rigid"
        << std::endl;
      std::cerr << excp << std::endl;
      return NULL;
      }
    }
  return versorRigid;
}

int WriteBothTransformsToDisk(GenericTransformType::Pointer & genericTransformToWrite, const std::string & outputTransform, const std::string & strippedOutputTransform)
{
  ////////////////////////////////////////////////////////////////////////////
  // Write out tranfoms.
  //
  if ( genericTransformToWrite.IsNotNull() )
    {
    try
      {
      const std::string transformFileType = genericTransformToWrite->GetNameOfClass();
      if ( transformFileType == "VersorRigid3DTransform" )
        {
        VersorRigid3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<VersorRigid3DTransformType *>( genericTransformToWrite.GetPointer() );
        if (outputTransform.size() > 0) // Write out the transform
          {
          WriteTransform<VersorRigid3DTransformType>(tempInitializerITKTransform,outputTransform);
          }
        }
      else if ( transformFileType == "ScaleVersor3DTransform" )
        {
        ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<ScaleVersor3DTransformType *>( genericTransformToWrite.GetPointer() );
        if (outputTransform.size() > 0) // Write out the transform
          {
          WriteTransform<ScaleVersor3DTransformType>(tempInitializerITKTransform,outputTransform);
          }
        }
      else if ( transformFileType == "ScaleSkewVersor3DTransform" )
        {
        ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<ScaleSkewVersor3DTransformType *>( genericTransformToWrite.GetPointer() );
        if (outputTransform.size() > 0) // Write out the transform
          {
          WriteTransform<ScaleSkewVersor3DTransformType>(tempInitializerITKTransform,outputTransform);
          }
        }
      else if ( transformFileType == "AffineTransform" )
        {
        AffineTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<AffineTransformType *>( genericTransformToWrite.GetPointer() );
        if (outputTransform.size() > 0) // Write out the transform
          {
          WriteTransform<AffineTransformType>(tempInitializerITKTransform,outputTransform);
          }
        }
      else if(transformFileType == "BSplineDeformableTransform")
        {
        BSplineTransformType::Pointer tempInitializerITKTransform
          = dynamic_cast<BSplineTransformType *>( genericTransformToWrite.GetPointer() );
        if (strippedOutputTransform.size() > 0)
          {
          std::cout << "ERROR:  The rigid component of a BSpline transform is not supported." << std::endl;
          }
        if ( outputTransform.size() > 0 )
          {
          WriteTransformWithBulk<BSplineTransformType>(tempInitializerITKTransform,
            outputTransform);
          }
        }
      else      //  NO SUCH CASE!!
        {
        std::cout
          <<
          "Unsupported initial transform file -- TransformBase first transform typestring, "
          << transformFileType
          <<
          " not equal to any recognized type VersorRigid3DTransform OR ScaleVersor3DTransform OR ScaleSkewVersor3DTransform OR AffineTransform"
          << std::endl;
        return -1;
        }
      }
    catch ( itk::ExceptionObject & excp )
      {
      std::cout << "[FAILED]" << std::endl;
      std::cerr
        << "Error while reading the genericTransformToWrite"
        << std::endl;
      std::cerr << excp << std::endl;
      return -1;
      }
    //Should just write out the rigid transform here.
    if (strippedOutputTransform.size() > 0  )
      {
      typedef VersorRigid3DTransformType VersorRigidTransformType;
      VersorRigidTransformType::Pointer versorRigid = ComputeRigidTransformFromGeneric(genericTransformToWrite);
      if(versorRigid.IsNotNull() )
        {
        WriteTransform<VersorRigidTransformType>(versorRigid,strippedOutputTransform);
        }
      }
    }
  return 0;
}

int WriteStrippedRigidTransformToDisk(GenericTransformType::Pointer & genericTransformToWrite, const std::string & strippedOutputTransform)
{
  return WriteBothTransformsToDisk(genericTransformToWrite, std::string(""), strippedOutputTransform);
}
int WriteTransformToDisk(GenericTransformType::Pointer & genericTransformToWrite, const std::string & outputTransform)
{
  return WriteBothTransformsToDisk(genericTransformToWrite, outputTransform, std::string(""));
}


GenericTransformType::Pointer ReadTransformFromDisk(const std::string initialTransform)
{
  GenericTransformType::Pointer genericTransform=NULL;
  //
  //
  // read in the initial ITKTransform
  //
  //
  TransformReaderType::Pointer transformListReader =  TransformReaderType::New();
  TransformListType currentTransformList;
  TransformListType::const_iterator currentTransformListIterator;
  if ( initialTransform.size() > 0 )
    {
    std::cout
      << "Read ITK transform from text file: "
      << initialTransform << std::endl;

    transformListReader->SetFileName( initialTransform.c_str() );
    try
      {
      transformListReader->Update();
      }
    catch ( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      exit(-1);
      }
    currentTransformList = *( transformListReader->GetTransformList() );
    if ( currentTransformList.size() == 0 )
      {
      std::cout << "Number of currentTransformList = " << currentTransformList.size() << std::endl;
      std::cout << "FATAL ERROR: Failed to read transform" << initialTransform << std::endl;
      exit(-1);
      }
    }
  if (currentTransformList.size() == 1) //Most simple transform types
    {

    //NOTE:  The dynamic casting here circumvents the standard smart pointer behavior.  It is important that
    // by making a new copy and transfering the parameters it is more safe.  Now we only need to ensure
    // that currentTransformList.begin() smart pointer is stable (i.e. not deleted) while the variable
    // temp has a reference to it's internal structure.
    const std::string transformFileType = (*(currentTransformList.begin()))->GetNameOfClass();
    if ( transformFileType == "VersorRigid3DTransform" )
      {
      VersorRigid3DTransformType::Pointer tempInitializerITKTransform
        = dynamic_cast<VersorRigid3DTransformType *>( (*(currentTransformList.begin())).GetPointer() );
      VersorRigid3DTransformType::Pointer tempCopy=VersorRigid3DTransformType::New();
      AssignRigid::AssignConvertedTransform(tempCopy,
        tempInitializerITKTransform);
      genericTransform=tempCopy.GetPointer();
      }
    else if ( transformFileType == "ScaleVersor3DTransform" )
      {
      ScaleVersor3DTransformType::Pointer tempInitializerITKTransform
        = dynamic_cast<ScaleVersor3DTransformType *>( (*(currentTransformList.begin())).GetPointer() );
      ScaleVersor3DTransformType::Pointer tempCopy=ScaleVersor3DTransformType::New();
      AssignRigid::AssignConvertedTransform(tempCopy,
        tempInitializerITKTransform);
      genericTransform=tempCopy.GetPointer();
      }
    else if ( transformFileType == "ScaleSkewVersor3DTransform" )
      {
      ScaleSkewVersor3DTransformType::Pointer tempInitializerITKTransform
        = dynamic_cast<ScaleSkewVersor3DTransformType *>( (*(currentTransformList.begin())).GetPointer() );
      ScaleSkewVersor3DTransformType::Pointer tempCopy=ScaleSkewVersor3DTransformType::New();
      AssignRigid::AssignConvertedTransform(tempCopy,
        tempInitializerITKTransform);
      genericTransform=tempCopy.GetPointer();
      }
    else if ( transformFileType == "AffineTransform" )
      {
      AffineTransformType::Pointer tempInitializerITKTransform
        = dynamic_cast<AffineTransformType *>( (*(currentTransformList.begin())).GetPointer() );
      AffineTransformType::Pointer tempCopy=AffineTransformType::New();
      AssignRigid::AssignConvertedTransform(tempCopy,
        tempInitializerITKTransform);
      genericTransform=tempCopy.GetPointer();
      }
    }
  else if (currentTransformList.size() == 2) //A special case for BSplineTransforms
    //To recombine the bulk and the bSpline transforms.
    {
    // transformListReader->GetTransformList();
    TransformListType::const_iterator initializeTransformsListIterator
      = currentTransformList.begin();

    GenericTransformType::Pointer BulkTransform=dynamic_cast<GenericTransformType *>
      ((*(initializeTransformsListIterator)).GetPointer() );

    BSplineTransformType::Pointer outputBSplineTransform
      = BSplineTransformType::New();
    outputBSplineTransform->SetIdentity();

    // initialITKTransform may usefully be FOLLOWED BY an initial BSpline in
    // currentTransformList.
    outputBSplineTransform->SetBulkTransform( BulkTransform );

    //Now get the BSpline information
    initializeTransformsListIterator++;
    const std::string transformFileType =
      (*(initializeTransformsListIterator))->GetNameOfClass();
    if ( transformFileType == "BSplineDeformableTransform" )
      {
      BSplineTransformType::Pointer tempInitializerITKTransform
        = dynamic_cast<BSplineTransformType *>
        ( ( * initializeTransformsListIterator ) .GetPointer() );
      outputBSplineTransform->SetFixedParameters( tempInitializerITKTransform->GetFixedParameters() );
      outputBSplineTransform->SetParametersByValue( tempInitializerITKTransform->GetParameters() );
      }
    genericTransform=outputBSplineTransform.GetPointer();
    }
  else if ( currentTransformList.size() > 2 )
    {
    // Error, too many transforms on transform list.
    std::cout << "[FAILED]" << std::endl;
    std::cerr
      <<
      "Error using the currentTransformList for initializing a BSPlineDeformableTransform:"
      << std::endl;
    std::cerr
      <<
      "There should not be more than two transforms in the transform list."
      << std::endl;
    return NULL;
    }
  return genericTransform;
}

} // end namespace itk
