#ifndef _itkGrowCutSegmentationImageFilter_txx_
#define _itkGrowCutSegmentationImageFilter_txx_   


#include "itkGrowCutSegmentationImageFilter.h"

#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionConstIterator.h"

#include "itkConstantBoundaryCondition.h"
#include "itkNumericTraits.h"
//#include "vtkITKNumericTraits.h"

#include "itkCastImageFilter.h"

#include "itkFlipImageFilter.h"

#include "itkMaskImageFilter.h"
//#include "itkRelabelComponentImageFilter.h"

#include "itkImageRandomConstIteratorWithIndex.h"
#include "itkExtractImageFilter.h"

#include "itkImageFileWriter.h"

#include "itkLabelMap.h"
#include "itkShapeLabelObject.h"
#include "itkLabelImageToShapeLabelMapFilter.h"

//#include <map>
#include <vcl_map.h>
#include <vcl_string.h>

#include <iostream.h>

#include <ctime>

#include <vcl_cmath.h>
#include <vcl_algorithm.h>
#include <vcl_utility.h>
#include <iostream>


// #define LOG_OUTPUT
// #define DEBUG_LEVEL_0

#define USE_LABELSHAPEFILTER


namespace itk
{

template< class TInputImage, class TOutputImage, class TWeightPixelType>
 GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
 ::GrowCutSegmentationImageFilter() {
  
    m_ForegroundPoints = NULL;
    m_BackgroundPoints = NULL;
    m_PriorForegroundPoints = NULL;


    m_ConfThresh = 0.2;

    m_MaxIterations = 10000;
    m_ObjectRadius = 10;
   
    m_T1 = 17;
    m_T2 = 17;
   
    m_UseSlow = true;
    
    m_SeedStrength = 1.0;
    m_PriorSegmentStrength = 0.0;  
    
    m_LabelImage = OutputImageType::New();
    
    m_WeightImage = WeightImageType::New();
    
    m_PrevWeightImage = WeightImageType::New();
    

    m_ObjectLabel = static_cast<OutputImagePixelType>( NumericTraits<OutputImagePixelType>::max(OutputImagePixelType() ));
    m_BackgroundLabel = static_cast<OutputImagePixelType>( 1 );
    // m_UnknownLabel = static_cast<OutputImagePixelType>( 0 );
    m_UnknownLabel = static_cast<OutputImagePixelType>( NumericTraits<OutputImagePixelType>::ZeroValue() );
    
    m_Radius.Fill(1);   
}


/**
 * Standard PrintSelf method.
 */
  template <class TInputImage, class TOutputImage, class TWeightPixelType>
  void
  GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::PrintSelf(std::ostream& os, Indent indent) const 
{
    
    this->Superclass::PrintSelf(os, indent);
    //    os << indent << "Object Seed Points : " << m_ForegroundPoints.Print(std::cout)<< std::endl;
    //    os << indent << "Background Seed Points : " << m_BackgroundPoints.Print(std::cout)<< std::endl;
    os << indent << "max iterations : " << m_MaxIterations<< std::endl;  
    os << indent << "max enemies for attack T1 : " << m_T1<< std::endl;  
    os << indent << "min enemies for submit T2 : " << m_T2<< std::endl;  
    os << indent << "starting seed strength :" <<m_SeedStrength<< std::endl;     
    os << indent << "use Algorithm Speed Slow : " << m_UseSlow<< std::endl;    
  
} 



template <class TInputImage, class TOutputImage, class TWeightPixelType>
void 
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::GenerateInputRequestedRegion() 
{

  Superclass::GenerateInputRequestedRegion();
  if ( this->GetInput() )
    {
    InputImagePointer image = const_cast< InputImageType * >( this->GetInput() );
    image->SetRequestedRegionToLargestPossibleRegion();
    }
}


template <class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::EnlargeOutputRequestedRegion(DataObject *output) 
{

  Superclass::EnlargeOutputRequestedRegion(output);
  output->SetRequestedRegionToLargestPossibleRegion();
  
}



template <class TInputImage, class TOutputImage, class TWeightPixelType>
void
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::Initialize( OutputImageType * output ) 
{

  // std::ofstream debugger;  
  //  debugger.open("D:/projects/namic2010/debuglog.txt", std::ios::out|std::ios::app);
  
  // allocate memory for the output buffer
  output->SetBufferedRegion( output->GetRequestedRegion() );
  output->Allocate();
  output->FillBuffer ( m_UnknownLabel );
 
  // allocate memory if it has not been set already
  typename OutputImageType::SizeType lsize = 
    m_LabelImage->GetBufferedRegion().GetSize();
  
  unsigned int dim = static_cast< unsigned int>(output->GetImageDimension());
  bool ans = false;
  for (unsigned d = 0; !ans && d < dim; d++)
    {
      ans = (lsize[d] == 0);
    }
  
  if( ans)
  {
   // debugger<<"Initializing label image ... "<<std::endl;
     m_LabelImage->CopyInformation( output );
     m_LabelImage->SetBufferedRegion( output->GetRequestedRegion() );
     m_LabelImage->Allocate();
     m_LabelImage->FillBuffer( m_UnknownLabel );   
  }


  // allocate memory if the strength image has not been set already
  m_WeightImage->CopyInformation( output );
  m_WeightImage->SetBufferedRegion( output->GetRequestedRegion() );
  m_WeightImage->Allocate();
  m_WeightImage->FillBuffer( 0.0 );   
 
 // allocate memory
  typename WeightImageType::SizeType wsize = 
    m_PrevWeightImage->GetBufferedRegion().GetSize();
  
  ans = false;
  for (unsigned d = 0; !ans && d < dim; d++)
    {
      ans = wsize[d] == 0;
    }

  if( ans) 
  {
     //debugger<<"Initializing weight image ... "<<std::endl;
     m_PrevWeightImage->CopyInformation( output );
     m_PrevWeightImage->SetBufferedRegion( output->GetRequestedRegion() );
     m_PrevWeightImage->Allocate();
     m_PrevWeightImage->FillBuffer( 0.0 );   
  }
  //else
  //{
  //   debugger<<"Weight Image has already been set by the user "<<std::endl;
  //}

  
  // if the object seed points are available, then initialize the 
  // labels and the seed strength  
  m_BufferedRegion = output->GetRequestedRegion();

  ConstNeighborhoodIterator<OutputImageType>
        out(m_Radius, output, output->GetRequestedRegion());
   
  ConstantBoundaryCondition<OutputImageType> cbco;
  cbco.SetConstant( NumericTraits< OutputImagePixelType >::min(OutputImagePixelType() ) );
  out.OverrideBoundaryCondition( &cbco );


  if ( m_ForegroundPoints ) {

     //debugger<<"There are some foreground points to initialize "<<std::endl;

     typename NodeContainer::ConstIterator pointsIter = 
       m_ForegroundPoints->Begin();
     typename NodeContainer::ConstIterator pointsEnd = 
       m_ForegroundPoints->End();

     for (; pointsIter != pointsEnd; ++pointsIter ) {

       IndexType node = pointsIter.Value();
  
      // check if node index is within the output level set
      if ( !m_BufferedRegion.IsInside( node ) ) 
        continue;

      // make this an object point
      m_LabelImage->SetPixel( node, m_ObjectLabel );
      
      output->SetPixel( node, m_ObjectLabel );    
      
      // Set the Weight
      m_WeightImage->SetPixel( node, m_SeedStrength );
      m_PrevWeightImage->SetPixel( node, m_SeedStrength );

     // add the neighbor pixels of the seeds to the alive points
     // we don't add the seeds themselves as they are not going to get updated
      if( !m_UseSlow )
      {
        out.SetLocation( node );
        for (unsigned i = 0; i < out.Size(); i++)
        {
          IndexType nIndex = out.GetIndex(i);
          m_AlivePoints.push_back( nIndex );
        }
      }
     }
  }
  

  if ( m_BackgroundPoints ) {

   //debugger<<"There are some background points to initialize "<<std::endl;
    typename NodeContainer::ConstIterator pointsIter = 
      m_BackgroundPoints->Begin();
     typename NodeContainer::ConstIterator pointsEnd = 
       m_BackgroundPoints->End();

      for (; pointsIter != pointsEnd; ++pointsIter ) 
      {

         IndexType node = pointsIter.Value();
  
         // check if node index is within the output level set
   if ( !m_BufferedRegion.IsInside( node ) ) 
     continue;

   // make this a background point
   m_LabelImage->SetPixel( node, m_BackgroundLabel );
   
   output->SetPixel( node, m_BackgroundLabel );  
   
   m_WeightImage->SetPixel( node, m_SeedStrength );  
   m_PrevWeightImage->SetPixel( node, m_SeedStrength );  
   
   if( !m_UseSlow )
     {
       out.SetLocation( node );
       for (unsigned i = 0; i < out.Size(); i++)
  {
    IndexType nIndex = out.GetIndex(i);
    m_AlivePoints.push_back( nIndex );
  }
     }
 }   
  }

// debugger.close();

}

#if 0
template<class TInputImage, class TOutputImage, class TWeightPixelType> 
void GrowCutSegmentationImageFilter< TInputImage, TOutputImage, TWeightPixelType>::
ComputeSegmentedVolumeROI(TOutputImage *stateImage, vcl_vector< unsigned > &volumes)
{


  typename TOutputImage::Pointer mask = TOutputImage::New();
  mask->CopyInformation(stateImage);
  mask->SetBufferedRegion(stateImage->GetBufferedRegion());
  mask->Allocate();
  mask->FillBuffer( 0 );

  typedef MaskImageFilter<TOutputImage, TOutputImage > MaskFilterType;
  typedef RelabelComponentImageFilter< TOutputImage, TOutputImage > RelabelFilterType;
  typename MaskFilterType::Pointer filterMask = MaskFilterType::New();
  typename RelabelFilterType::Pointer filterRelabel = RelabelFilterType::New();
 
  typedef typename TOutputImage::RegionType InRegionType;
  InRegionType inputRegion;
  typename TOutputImage::IndexType newStart; 
  typename TOutputImage::SizeType newSize; 
  unsigned ndims = static_cast< unsigned int>(stateImage->GetImageDimension()); 
  for (unsigned d = 0; d < ndims; d++) 
    { 
      newStart[d] = m_roiStart[d]; 
      newSize[d]  = m_roiEnd[d]-m_roiStart[d]; 
    } 
  
  inputRegion.SetIndex(newStart); 
  inputRegion.SetSize(newSize); 
  
  ImageRegionIterator< TOutputImage > maskIter(mask, inputRegion );
  for(maskIter.GoToBegin(); !maskIter.IsAtEnd(); ++maskIter)
    {
      maskIter.Set(1);
    }
  
  filterMask->SetInput1(stateImage); 
  filterMask->SetInput2(mask); 
  filterMask->Update(); 


#ifndef USE_LABELSHAPEFILTER
  ImageRegionConstIteratorWithIndex< OutputImageType > state( filterMask->GetOutput(),
             filterMask->GetOutput()->GetBufferedRegion());

  int count = 0;
  volumes[0] = 0;
  volumes[1] = 0;
  volumes[2] = 0;

  for (state.GoToBegin(); !state.IsAtEnd(); ++state)
    {
      ++count;
      if(state.Get() == LABELED)
 {
   ++volumes[0];
 }
      else if(state.Get() == LOCALLY_SATURATED)
 {
   ++volumes[1];
 }
      else if(state.Get() == SATURATED)
 {
   ++volumes[2];
 }
    }
  std::cout<<" Volume ROI analyzed "<<count<<std::endl;
#else

  typedef LabelImageToShapeLabelMapFilter< OutputImageType > LabelFilterType;
  typename LabelFilterType::Pointer labelFilter = LabelFilterType::New();

  labelFilter->SetInput(filterMask->GetOutput());
  labelFilter->SetBackgroundValue(0);
  labelFilter->Update();
  
  unsigned long numObjects = labelFilter->GetNumberOfLabelObjects();
  numObjects = 3;
  for (unsigned n = 0; n < numObjects; n++)
    {
      volumes[n] = labelFilter->GetNthLabelObject(n)->GetSize();
    }


#endif

}
#endif



template<class TInputImage, class TOutputImage, class TWeightPixelType> 
void GrowCutSegmentationImageFilter< TInputImage, TOutputImage, TWeightPixelType>::
ComputeLabelVolumes(TOutputImage *outputImage, vcl_vector< unsigned > &volumes, 
      vcl_vector< unsigned > &physicalVolumes)
{

#ifndef USE_LABELSHAPEFILTER

  vcl_map<unsigned short, unsigned int>labelMap;
  vcl_vector< unsigned short > labels;
  unsigned int index = 0;
  ImageRegionConstIteratorWithIndex< OutputImageType > label( outputImage,
             outputImage->GetBufferedRegion());

  for(label.GoToBegin(); !label.IsAtEnd(); ++label)
    {
      
      unsigned short pix = static_cast<unsigned short>(label.Get());
      vcl_map<unsigned short, unsigned int>::iterator it = labelMap.find(pix);
      if(it == labelMap.end())
 {
   labelMap.insert(vcl_pair<unsigned short, unsigned int>(pix, index));
   volumes.push_back(1);
   labels.push_back(pix);
   ++index;
 }
      else
 {
   int i = it->second;
   ++volumes[i];
 }
    }
  std::cout<<" label \t "<<" volume "<<std::endl;
  for (unsigned int i = 0; i < index; i++)
    {
      vcl_map<unsigned short, unsigned int>::iterator it = labelMap.find(labels[i]); 
      std::cout<<labels[i]<<"\t"<<volumes[it->second]<<std::endl;
    }

#else
  typedef LabelImageToShapeLabelMapFilter< OutputImageType > LabelFilterType;
  typename LabelFilterType::Pointer labelFilter = LabelFilterType::New();

  labelFilter->SetInput(outputImage);
  labelFilter->SetBackgroundValue(0);
  labelFilter->Update();
  
  unsigned long numObjects = labelFilter->GetOutput()->GetNumberOfLabelObjects();

  for (unsigned n = 0; n < numObjects; n++)
    {
      volumes.push_back(labelFilter->GetOutput()->GetNthLabelObject(n)->GetSize());
      
      physicalVolumes.push_back(labelFilter->GetOutput()->GetNthLabelObject(n)->GetSize()/1000.0);
    }  

#endif
  
}



template<class TInputImage, class TOutputImage, class TWeightPixelType> 
bool GrowCutSegmentationImageFilter< TInputImage, TOutputImage, TWeightPixelType>::
IsInsideROI(OutputIndexType &seed)
{

  for (unsigned i = 0; i < seed.GetIndexDimension(); i++)
    {
      if(seed[i] < m_roiStart[i] || seed[i] >= m_roiEnd[i])
 {
   return false;
 }
    }
  return true;
}


template<class TInputImage, class TOutputImage, class TWeightPixelType> 
bool GrowCutSegmentationImageFilter< TInputImage, TOutputImage, TWeightPixelType>::
IsInsideImage(OutputIndexType &seed, OutputSizeType &imSize)
{

  for (unsigned i = 0; i < seed.GetIndexDimension(); i++)
    {
      if(seed[i] < 0 || seed[i] >= imSize[i])
 {
   return false;
 }
    }
  return true;
}




template<class TInputImage, class TOutputImage, class TWeightPixelType>
typename TOutputImage::IndexType GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
GetSeed( OutputImageType * state, 
  vcl_vector< OutputIndexType> &visitedSeeds, 
  OutputImagePixelType pickLabelType)
{

  typedef typename TOutputImage::IndexType OutIndexType;
  // typedef ImageRegionConstIteratorWithIndex< OutputImageType > IteratorType;

  typename OutputImageType::IndexType newIndex;
  unsigned ndim = state->GetImageDimension();
  for (unsigned d = 0; d < ndim; d++)
    {
      newIndex[d] = 0;
    }

  std::cout<<"Getting a new seed : "<<std::endl;
  
  typedef ImageRandomConstIteratorWithIndex< OutputImageType > RandomIteratorType;

  
  unsigned int sampleSize = 10000000;

  RandomIteratorType iter(state, state->GetBufferedRegion());
  iter.SetNumberOfSamples(sampleSize);
  iter.ReinitializeSeed();

  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
    {

      OutputImagePixelType s = iter.Get();
      
      if( s != UNLABELED && s != SATURATED)
 {
   
   OutputIndexType curr = iter.GetIndex();
      
   if(!IsInsideROI(curr))
     {
       continue;
     }
   else
     {
       
       typename vcl_vector< OutputIndexType>::iterator it;
       if(visitedSeeds.empty())
  {
    visitedSeeds.push_back(curr);
    return static_cast< OutIndexType> (curr);
  }
       else
  {
    it = vcl_find(visitedSeeds.begin(), 
    visitedSeeds.end(), 
    curr);
    if(it == visitedSeeds.end())
      {
        visitedSeeds.push_back(curr);
        return static_cast< OutIndexType> (curr);
      }  
  }
     }
 }
    }
  
  return (static_cast< OutIndexType >( newIndex ) );
  
}



template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
InitializeStateImage( OutputImageType *stateImage)
{

  stateImage->CopyInformation(m_LabelImage);
  stateImage->SetBufferedRegion( m_LabelImage->GetBufferedRegion() );
  stateImage->Allocate();
  stateImage->FillBuffer(UNLABELED);
  
  bool setStart = false;
  bool setEnd = false;
  
  typename OutputImageType::SizeType lsize = 
   m_LabelImage->GetBufferedRegion().GetSize();

  ImageRegionConstIteratorWithIndex< OutputImageType > label( m_LabelImage, 
          m_LabelImage->GetBufferedRegion());
  ImageRegionIterator< OutputImageType > state( stateImage, 
            stateImage->GetBufferedRegion());
  for (label.GoToBegin(), state.GoToBegin(); !label.IsAtEnd(); 
       ++label, ++state)
    {
      if(label.Get() != m_UnknownLabel)
 {
   OutputIndexType idx = label.GetIndex();
   
   if(!setStart)
     {
       m_roiStart = idx;
       setStart = true;
     }
   else
     {
       for (unsigned i = 0; i < m_LabelImage->GetImageDimension(); i++)
  {
    if(idx[i] < m_roiStart[i])
      {
        m_roiStart[i] = idx[i];
      }
  }
     }
   
   if(!setEnd)
     {
       m_roiEnd = idx;
       setEnd = true;
     }
   else
     { 
       for (unsigned i = 0; i < m_LabelImage->GetImageDimension(); i++)
  {
    if(idx[i] > m_roiEnd[i])
      {
        m_roiEnd[i] = idx[i]; 
      }
  }
     }
   
   state.Set(static_cast< OutputImagePixelType>(LABELED));
 }
    }

  std::cout<<" ROI before adding the object Radius "<<m_ObjectRadius<<std::endl;
  std::cout<< " ROI start ";
  for (int i = 0; i < m_LabelImage->GetImageDimension(); i++)
    {
      std::cout<<m_roiStart[i]<<" ";
    }
  std::cout<<" ROI end ";
  for (unsigned int i = 0; i < m_LabelImage->GetImageDimension(); i++)
    {
      std::cout<<m_roiEnd[i]<<" ";
    }
  std::cout<<std::endl;
  


  for (unsigned i = 0; i < m_LabelImage->GetImageDimension(); i++)
    {
      m_roiStart[i] = (m_roiStart[i] - m_ObjectRadius >= 0) ? 
 (m_roiStart[i] - m_ObjectRadius) : 0;

      m_roiEnd[i] = (static_cast<unsigned int>(m_roiEnd[i] + m_ObjectRadius) < lsize[i]) ? 
 (m_roiEnd[i] + m_ObjectRadius) : lsize[i]-1;
      
    }
  

  std::cout<< " ROI start ";
  for (unsigned int i = 0; i < m_LabelImage->GetImageDimension(); i++)
    {
      std::cout<<m_roiStart[i]<<" ";
    }
  std::cout<<" ROI end ";
  for (unsigned int i = 0; i < m_LabelImage->GetImageDimension(); i++)
    {
      std::cout<<m_roiEnd[i]<<" ";
    }
  std::cout<<std::endl;
  
}




template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
InitializeDistancesImage( TInputImage *inputImage, 
     WeightImageType *distances)
{

  typedef ConstNeighborhoodIterator<InputImageType> InputNeighborhoodIteratorType;
  typedef typename InputNeighborhoodIteratorType::RadiusType InputRadiusType;
  InputRadiusType radiusIn = static_cast<InputRadiusType> (m_Radius);
  
  InputNeighborhoodIteratorType input(radiusIn, 
          inputImage, 
          inputImage->GetBufferedRegion());


  distances->CopyInformation( inputImage );
  distances->SetBufferedRegion( inputImage->GetBufferedRegion() );  
  distances->Allocate();
  distances->FillBuffer( 0 );     

  ImageRegionIterator< WeightImageType> dist(distances, 
            distances->GetBufferedRegion());

  for(input.GoToBegin(), dist.GoToBegin(); !input.IsAtEnd();
      ++input, ++dist)
     {

       WeightImagePixelType center = static_cast< WeightImagePixelType>(input.GetCenterPixel());
       WeightImagePixelType maxDistance = 0.0;
       WeightImagePixelType distance;

       for (unsigned i = 0; i < input.Size(); i++)
 {
   WeightImagePixelType pix = static_cast< WeightImagePixelType>(input.GetPixel(i)); 
   
   distance = (pix - center)*(pix-center);
   maxDistance = (distance > maxDistance) ? distance : 
     maxDistance;       
 }    
       
       dist.Set(maxDistance);      
     }
}




template<class TInputImage, class TOutputImage, class TWeightPixelType>
float GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
UpdateLabels( unsigned int &newRadius, 
       InputImageType *inputImage,
       OutputImageType *outputImage,
       OutputImageType *stateImage, 
       WeightImageType *distancesImage, 
       WeightImageType *maxSaturationImage,
       bool useMax, bool useROI,
       OutputIndexType &seed, 
       vcl_vector< unsigned > &volumes) 
{

  //unsigned int irad = newRadius;
try{
  OutputIndexType newSeed;
  unsigned int Dimension = inputImage->GetImageDimension();
  for (unsigned d = 0; d < Dimension; d++)
    {
      newSeed[d] = 0;
    }


  typename OutputImageType::SizeType imageSize = 
    m_LabelImage->GetBufferedRegion().GetSize();
  
  // Extract the image regions corresponding to the seed and ROI defined by the radius
  typename InputImageType::IndexType iIndex;
  typename InputImageType::SizeType iSize;

  if(!useROI)
    {
      for (unsigned d = 0; d < Dimension; d++)
 {
   // make sure we don't go over the image boundary
   iIndex[d] = (seed[d]-newRadius > 0) ? (seed[d]-newRadius) : seed[d];
   
   iSize[d] = (static_cast< unsigned int>(seed[d]+newRadius*2+1) < 
        imageSize[d]-m_Radius[0]-1) ? (newRadius*2) : 
     (((seed[d]+newRadius*2+1)-iIndex[d]) - 
      ((seed[d]+newRadius*2+1)-(imageSize[d]-m_Radius[0])));
          iSize[d] = (iSize[d] < imageSize[d]-m_Radius[0]) ? iSize[d] : imageSize[d]-m_Radius[0]; 
   std::cout<<" Index "<<d<<" : "<<iIndex[d]<<" size "<<iSize[d]<<std::endl;
   if(iSize[d] > imageSize[d]){
      std::cout<<"Error in dimensions "<<iSize[d]<<" is greater than image dimensions "<<imageSize[d]<<std::endl;
     std::cout<<" seed+newRadius*2+1 "<<seed[d]+newRadius*2+1<<"imageSize - m_Radius "<<imageSize[d] - m_Radius[0]<<std::endl; 
          }
 }
    }
  else
    {
      for (unsigned d = 0; d < Dimension; d++)
 {
   iIndex[d] = m_roiStart[d];
   iSize[d] = m_roiEnd[d] - m_roiStart[d];
 }
      
    }

  ////////////////////////////////////////////////////////////////////////////////
  typedef typename InputImageType::RegionType InputRegionType; 
  typedef typename OutputImageType::RegionType OutputRegionType; 
  typedef typename WeightImageType::RegionType WeightRegionType; 
  
  InputRegionType input;
  input.SetIndex( iIndex );
  input.SetSize( iSize );
  
  OutputRegionType label;
  label.SetIndex( iIndex);
  label.SetSize( iSize);
  
  WeightRegionType weight;
  weight.SetIndex( iIndex);
  weight.SetSize( iSize);   

  typedef ConstNeighborhoodIterator<InputImageType> InputIteratorType;
  typedef typename InputIteratorType::RadiusType InputRadiusType;

  typedef NeighborhoodIterator<OutputImageType> LabelIteratorType;
  typedef typename LabelIteratorType::RadiusType LabelRadiusType;

  typedef NeighborhoodIterator<WeightImageType> WeightIteratorType;
  typedef typename WeightIteratorType::RadiusType WeightRadiusType;

    
  InputRadiusType radiusIn = static_cast< InputRadiusType >(m_Radius);  
  WeightRadiusType radiusWt = static_cast< WeightRadiusType >(m_Radius);  
  LabelRadiusType radiusLab = static_cast< LabelRadiusType >(m_Radius);  
  
  InputIteratorType inputIt(radiusIn, inputImage,input);
  
  LabelIteratorType oldLabelIt(radiusLab, m_LabelImage, label);
  
  LabelIteratorType newLabelIt(radiusLab, outputImage, label);
  
  WeightIteratorType oldWeightIt(radiusWt, m_PrevWeightImage, weight);
  
  WeightIteratorType newWeightIt(radiusWt, m_WeightImage, weight);
  
  WeightIteratorType distancesIt(radiusWt, distancesImage, weight);

  WeightIteratorType maxSatIt(radiusWt, maxSaturationImage, weight);
  
  LabelIteratorType stateIt(radiusLab, stateImage, label);

  // set boundary conditions for the radius

  InputImagePixelType minI = NumericTraits< InputImagePixelType > ::min(InputImagePixelType());
  //OutputImagePixelType minL = NumericTraits< OutputImagePixelType > ::min(OutputImagePixelType());
  WeightImagePixelType minW = NumericTraits< WeightImagePixelType > ::min(WeightImagePixelType());
  
  ConstantBoundaryCondition<InputImageType> cbci;
  cbci.SetConstant( NumericTraits< InputImagePixelType >::min(InputImagePixelType()) );
  
  inputIt.OverrideBoundaryCondition( &cbci ); 


  ConstantBoundaryCondition<OutputImageType> cbcl;
  cbcl.SetConstant( NumericTraits< OutputImagePixelType >::min(OutputImagePixelType()) );
  
  oldLabelIt.OverrideBoundaryCondition( &cbcl ); 

  newLabelIt.OverrideBoundaryCondition( &cbcl ); 

  stateIt.OverrideBoundaryCondition( &cbcl ); 


  ConstantBoundaryCondition<WeightImageType> cbcw;
  cbcw.SetConstant( NumericTraits< WeightImagePixelType >::min(WeightImagePixelType()) );
  
  oldWeightIt.OverrideBoundaryCondition( &cbcw ); 

  newWeightIt.OverrideBoundaryCondition( &cbcw ); 

  maxSatIt.OverrideBoundaryCondition( &cbcw ); 

  distancesIt.OverrideBoundaryCondition( &cbcw ); 

  /////////////////////////////////////////////////////////////////////////////////////////
  
  inputIt.GoToBegin(); oldLabelIt.GoToBegin(); newLabelIt.GoToBegin();
  oldWeightIt.GoToBegin(); newWeightIt.GoToBegin(); stateIt.GoToBegin();
  distancesIt.GoToBegin(); maxSatIt.GoToBegin();
  
  unsigned int numPixels = 0;
  
  float nextWeight = m_SeedStrength;
  bool chooseMin = useMax;
  
  nextWeight = chooseMin ? m_SeedStrength : 0.0;
  OutputIndexType maxSeed = seed;  

  unsigned int countModified = 0;

  //InputImagePixelType f_seed = inputImage->GetPixel(seed);
  //OutputImagePixelType l_seed = m_LabelImage->GetPixel(seed);
  //WeightImagePixelType w_seed = m_PrevWeightImage->GetPixel(seed);
  
  int countLocallyRelabled = 0;
  unsigned maxCapacity = 0;

  volumes[0] = 0; 
  volumes[1] = 0;
  volumes[2] = 0;  

  for (; !inputIt.IsAtEnd(); ++inputIt, ++oldLabelIt, 
  ++newLabelIt, ++oldWeightIt, ++newWeightIt, 
  ++stateIt, ++distancesIt, ++maxSatIt)
    {

      OutputImagePixelType s_center = stateIt.GetCenterPixel();
      InputImagePixelType f_center = inputIt.GetCenterPixel();            
      OutputImagePixelType l_center = oldLabelIt.GetCenterPixel();
      WeightImagePixelType w_center = oldWeightIt.GetCenterPixel();
      
      OutputIndexType index = stateIt.GetIndex();

      ++numPixels;
    
      if(s_center == LABELED)
 {
   ++volumes[0];
 }
      else if(s_center == LOCALLY_SATURATED)
 {
   ++volumes[1];
 }
      else if(s_center == SATURATED)
 {
   ++volumes[2];
 }   
      
      if(s_center == SATURATED || !IsInsideImage(index, imageSize))
 {
   continue;
 }
      
       OutputImagePixelType winnerLabel = l_center;
       WeightImagePixelType winnerWeight = w_center;

       WeightImagePixelType maxDist = distancesIt.GetCenterPixel();

       unsigned int countSaturatedLinks = 0;
       unsigned int countLocalSaturatedLinks = 0;

       bool modified = false;
       WeightImagePixelType maxWt = 0.0;
       
       int nlinks = 0;
       OutputIndexType currmaxSeed = maxSeed;  
       
       unsigned countLocalCapacity = 0;

       for (unsigned k = 0; k < inputIt.Size(); k++) 
 {
          
   OutputImagePixelType s = stateIt.GetPixel(k);
   InputImagePixelType f = inputIt.GetPixel(k);            
   OutputImagePixelType l = oldLabelIt.GetPixel(k);
   WeightImagePixelType w = oldWeightIt.GetPixel(k);
   
 
   if(f == minI && w == minW)
     {
       continue;
     }
   
   ++nlinks;
   
   WeightImagePixelType attackWeight = (f_center - f)*(f_center - f);
   attackWeight = (maxDist > 0) ? (1.0 - attackWeight/maxDist) : 1.0;
   
   WeightImagePixelType maxAttackWeight = 0.0; // attackWeight*maxSatIt.GetPixel(k);
   maxAttackWeight = (s == UNLABELED) ? 0.0 : 
     ((maxSatIt.GetPixel(k) == 0.0) ? (attackWeight * m_SeedStrength) : 
      ((s == SATURATED) ? attackWeight * w : attackWeight*maxSatIt.GetPixel(k)));
       
   attackWeight *= w;
   
   if(maxWt < maxAttackWeight)
     {
       maxWt = maxAttackWeight;
     }
       
   if(s_center != UNLABELED)
     {
       countSaturatedLinks += (maxAttackWeight <= w_center) ? 1 : 0;
       countLocalSaturatedLinks += (attackWeight <= w_center) ? 1 : 0;
     }
   
   countLocalCapacity += (s == LOCALLY_SATURATED || s == UNLABELED) ? 4 :
     ((s == LABELED ) ? 6 : 0);
        
   if(s != UNLABELED && attackWeight > winnerWeight)
     {
    
       winnerWeight = attackWeight;
       winnerLabel = l;
       modified = true;

       if(stateIt.GetCenterPixel() == LOCALLY_SATURATED)
  ++countLocallyRelabled;
   
       stateIt.SetCenterPixel(LABELED);
       
     }
 }

       if(modified && countLocalCapacity >= maxCapacity)
  {
  
    newSeed = stateIt.GetIndex();
    
    if(newSeed != seed && IsInsideROI(newSeed))
      {
        if(chooseMin && winnerWeight <= nextWeight)
   {
     nextWeight = winnerWeight;
     maxSeed = newSeed;
     maxCapacity = countLocalCapacity;
   }
        else if(!chooseMin && winnerWeight >= nextWeight)
   {
     nextWeight = winnerWeight;
     maxSeed = newSeed;
     maxCapacity = countLocalCapacity;
   }
      }
  }

       if(nlinks > 0 && countSaturatedLinks == nlinks)
  {
    stateIt.SetCenterPixel(SATURATED);
  }
       else if((nlinks > 0 && countLocalSaturatedLinks == nlinks) || 
        (stateIt.GetCenterPixel() != UNLABELED && !modified))
  {
    stateIt.SetCenterPixel(LOCALLY_SATURATED);
  }
       
       maxSatIt.SetCenterPixel(maxWt);
       newLabelIt.SetCenterPixel(winnerLabel);
       newWeightIt.SetCenterPixel(winnerWeight);
       
       countModified += (modified) ? 1 : 0;
    }
  
  seed = maxSeed;

  return (numPixels > 0) ? countModified/(float)numPixels : 0.0;

 } catch(itk::ExceptionObject &err) {
    (&err)->Print(std::cerr);
    exit(-1);
     return 0;   
} 

}



template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
MaskSegmentedImageByWeight( float confThresh)
{

  ImageRegionConstIterator< WeightImageType > weight( m_WeightImage,
            m_WeightImage->GetBufferedRegion());  

  ImageRegionIterator< OutputImageType > label( m_LabelImage, 
      m_LabelImage->GetBufferedRegion());

  weight.GoToBegin(); 
  label.GoToBegin();

  for(;!weight.IsAtEnd(); ++weight, ++label)
    {
      if(weight.Get() < confThresh)
 {
   label.Set(static_cast< OutputImagePixelType > (0));
 }
    }

}





template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
GrowCutFast( TOutputImage * output)
{

#ifdef LOG_OUTPUT
  std::ofstream debugger;
  debugger.open("iterationLog.txt");
  debugger.close();
  std::ofstream debugger1;
  debugger1.open("saturationLog.txt");
#endif

  unsigned int ndims = static_cast< unsigned int>(output->GetImageDimension());
  
  InputImagePointer inputImage = const_cast<InputImageType * >(this->GetInput());

  typename OutputImageType::Pointer pixelStateImage = OutputImageType::New();

  InitializeStateImage(pixelStateImage);

  // Initialize the max distance for each pixel in the image
  typename WeightImageType::Pointer maxDistancesImage = WeightImageType::New();

  InitializeDistancesImage(inputImage, maxDistancesImage);

  typename WeightImageType::Pointer maxSaturationImage = WeightImageType::New();
  
  maxSaturationImage->CopyInformation( inputImage );
  maxSaturationImage->SetBufferedRegion( inputImage->GetBufferedRegion() );  
  maxSaturationImage->Allocate();
  maxSaturationImage->FillBuffer( 0 );       

  
  vcl_vector< OutputIndexType > visitedSeeds;

  OutputIndexType seed = GetSeed( pixelStateImage, 
      visitedSeeds, LABELED);
  
  unsigned int iterRadius = m_ObjectRadius;
  unsigned int iter = 0;
  bool converged = false;

  time_t startTime;
  time_t endTime;
  time_t start;
  time_t end;


  double prevSaturatedPix = 0.0;
  double prevLocallySaturatedPix = 0.0;
  double prevUnlabeledPix = 1.0;

  unsigned int maxRadius = 0;
  unsigned int minRadius = 1;
  
  float totalROIVolume = 1.0;

  for (unsigned n = 0; n < ndims; n++)
    {
      prevUnlabeledPix *= (m_roiEnd[n]-m_roiStart[n]);
      totalROIVolume *= (m_roiEnd[n]-m_roiStart[n]);
      
      if(n == 0)
 {
   maxRadius = m_roiEnd[n]-m_roiStart[n];
 }
      else if(maxRadius < (m_roiEnd[n]-m_roiStart[n]))
 {
   maxRadius = m_roiEnd[n]-m_roiStart[n];
 }
    }
  maxRadius = vcl_floor(maxRadius*0.5);
  minRadius = vcl_ceil(maxRadius*0.15);
  minRadius = (minRadius < 1) ? 1 : minRadius;
  
  //maxRadius = (maxRadius >= 30) ? 30 : maxRadius; //static_cast< unsigned int>(vcl_ceil(maxRadius*0.5));

  
  float convergenceCapacity = 0.4; //0.35;
  float threshSaturation = .96; //.999; // Determine the saturation according to the size of the object
  float threshUnchanged = 0.05;
  float threshCapacity = 0.2;
  float threshDiff = 0.001;
  //float threshConvergenceDiff = 0.00001;

  float prevCapacity = 0.0;
  
  //m_MaxIterations = 50;
  //iterRadius = m_ObjectRadius;
  iterRadius = 1;
  
  OutputImagePointer tmpLabels = OutputImageType::New();
  tmpLabels->CopyInformation( output );
  tmpLabels->SetBufferedRegion( output->GetBufferedRegion() );
  tmpLabels->Allocate();
  
  WeightImagePointer tmpWeights = WeightImageType::New();
  tmpWeights->CopyInformation( m_WeightImage );
  tmpWeights->SetBufferedRegion( m_WeightImage->GetBufferedRegion() );
  tmpWeights->Allocate();
  
  bool chooseMin = false;

  
#ifdef LOG_OUTPUT
  debugger.open("iterationSeeds.txt");
#endif


  time(&start);

  bool useROI = false;

  vcl_vector< unsigned > stateVolumes;
  stateVolumes.resize(3);
  
  while (iter < m_MaxIterations && !converged)
    {
      
      //std::cout<<"Iteration Radius "<<iterRadius<<std::endl;

      time(&startTime);
      
      bool LocallyConverged = false;

      float currCapacity = UpdateLabels(iterRadius, 
     inputImage,
     output,
     pixelStateImage, 
     maxDistancesImage, 
     maxSaturationImage,
     chooseMin,
     useROI,
     seed, 
     stateVolumes);
      
      
      // swap the weights and labels
      tmpWeights = m_WeightImage;
      m_WeightImage = m_PrevWeightImage;
      m_PrevWeightImage = tmpWeights;
      
      tmpLabels = output;
      output = m_LabelImage; 
      m_LabelImage = tmpLabels;
      
      std::cout<<"Current Capacity "<<currCapacity<<" Prev Capacity "<<prevCapacity<<std::endl;
      if(!useROI)
 LocallyConverged = currCapacity < convergenceCapacity;
      else
 LocallyConverged = false;

      if (useROI)
 {

   unsigned changeablePix = stateVolumes[0];
   unsigned currLocallySaturatedPix = stateVolumes[1];
   unsigned currSaturatedPix = stateVolumes[2];
   
   unsigned unlabeled = (changeablePix + currLocallySaturatedPix + currSaturatedPix <= totalROIVolume) ? 
     (static_cast<unsigned > (totalROIVolume) -(currLocallySaturatedPix + 
             currSaturatedPix + changeablePix)) : 0;
   
   std::cout<<" saturated Pixels "<<currSaturatedPix<<"( "<<
     totalROIVolume<<") %"<<currSaturatedPix/totalROIVolume<<std::endl;
   std::cout<<" locally saturated Pixels "<<currLocallySaturatedPix<<" %"<<currLocallySaturatedPix/totalROIVolume<<std::endl;
   std::cout<<"Number of Labeled Pixels "<<changeablePix<<" %"<<changeablePix/totalROIVolume<<std::endl;
   
   double diff = vcl_abs(unlabeled - prevUnlabeledPix) + 
     vcl_abs(currSaturatedPix - prevSaturatedPix) + 
     vcl_abs(currLocallySaturatedPix - prevLocallySaturatedPix);

   float s = currLocallySaturatedPix + currSaturatedPix;
   
         float den = (s > changeablePix) ? ((s > 0.0 ) ? s : 1.0) : 
    (changeablePix > 0.0 ? changeablePix : 1.0);
        
  float num = (s < changeablePix) ? s : 
    changeablePix;
  
  if( unlabeled > (changeablePix + s))
    {
      convergenceCapacity = (s + changeablePix)/(float)unlabeled;
    }else {
      convergenceCapacity = num/den;
    }
  
  convergenceCapacity = (convergenceCapacity < 0.4 ) ? convergenceCapacity : 0.4; 
  convergenceCapacity = (convergenceCapacity > 0.1) ? convergenceCapacity : 0.1; 
  
  std::cout<<" convergence capacity "<<convergenceCapacity<<std::endl;
  
         if(diff == 0)
     {
       seed = GetSeed(pixelStateImage, visitedSeeds, LABELED);
     }
   std::cout<<"Unlabeled pixels "<<unlabeled<<" %"<<unlabeled/totalROIVolume<<std::endl;
   
   converged = (unlabeled/totalROIVolume < threshUnchanged && 
         ((currSaturatedPix+currLocallySaturatedPix)/totalROIVolume > threshSaturation));
   
#ifdef LOG_OUTPUT
   debugger1<<iter<<"\t"<<currSaturatedPix<<"\t"<<currLocallySaturatedPix<<
     "\t"<<changeablePix<<"\t"<<unlabeled<<"\t"<<totalROIVolume<<std::endl;
#endif

   prevSaturatedPix = currSaturatedPix;
   prevUnlabeledPix = unlabeled;
   prevLocallySaturatedPix = currLocallySaturatedPix;
 }

      if(currCapacity < threshCapacity && prevCapacity < threshCapacity)
 {
   --iterRadius;
 }
      else if (vcl_abs(currCapacity - prevCapacity) < threshDiff)
 {
   if(currCapacity < threshCapacity)
     --iterRadius;
   else
     ++iterRadius;
 }
      else{
 
 iterRadius = static_cast<unsigned int>(vcl_ceil(iterRadius + (currCapacity-prevCapacity)*iterRadius*2));
      }
      
      iterRadius = (iterRadius <= minRadius) ? minRadius : ((iterRadius >= maxRadius) ? maxRadius-1 : iterRadius);

      prevCapacity = currCapacity;

      useROI = LocallyConverged ? true : false;
      
/*     
      if(LocallyConverged)
 {
   std::cout<<"Locally Converged.. Acquiring new seed...."
     <<std::endl;
 
   if(!converged)
     {
       float val;
       seed = GetSeed(pixelStateImage, visitedSeeds, LABELED);
       val = m_WeightImage->GetPixel(seed);
       chooseMin = (val < m_SeedStrength*.45);
       
       std::cout<<" Newly acquired seed ";
       unsigned int sum = 0;
       for (unsigned d = 0; d < ndims; d++)
  {
    std::cout<<seed[d]<<" ";
    sum += seed[d];
  }
       std::cout<<std::endl;
       if(sum == 0)
  {
    converged = true;
    std::cout<<" solution converged "<<std::endl;
  }
     }
   
#ifdef LOG_OUTPUT
   debugger<<seed[0]<<"\t"<<seed[1]<<"\t"<<seed[2]<<"\t"<<!chooseMin<<iterRadius<<std::endl;
#endif
 }
 else*/
      if(!IsInsideROI(seed))
 {
   float val;
   bool selectSat = rand()%2 == 0 ? true : false;
   if(selectSat)
     seed = GetSeed(pixelStateImage, visitedSeeds, LOCALLY_SATURATED);
   else
     seed = GetSeed(pixelStateImage, visitedSeeds, LABELED);
   
   val = m_WeightImage->GetPixel(seed);
   chooseMin = (val < m_SeedStrength*.45);
 }
      else
 {
   chooseMin = (chooseMin) ? false : true;
 }
      
      time(&endTime);
      std::cout<<".............................................."<<std::endl;
      std::cout<<" Time taken for iteration "<<iter<<
 " : "<<difftime(endTime, startTime)<<" secs"<<std::endl;
      std::cout<<" New seed is : ";
      for (unsigned d = 0; d < ndims; d++)
 {
   std::cout<<seed[d]<<" ";
 }
      std::cout<<"; Radius "<<iterRadius<<std::endl;
      std::cout<<".............................................."<<std::endl;
      
      ++iter;

    }

  MaskSegmentedImageByWeight(m_ConfThresh);

  this->GraftOutput(m_LabelImage);
  
  vcl_vector< unsigned > labelVolumes;
  vcl_vector< unsigned > physicalVolumes;
  ComputeLabelVolumes(m_LabelImage, labelVolumes, physicalVolumes);
  
  std::cout<<"...................................................."<<std::endl;
  for (unsigned n = 0; n < labelVolumes.size(); n++)
    {
      std::cout<<" Object "<<n<<" volume "<<labelVolumes[n]<<" Physical Volume "<<
 physicalVolumes[n]<<std::endl;
    } 

  typedef ImageFileWriter < OutputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  
  vcl_string outfilename = "segmented_img.mhd";
  writer->SetFileName(outfilename.c_str());
  writer->SetInput(m_LabelImage);
  writer->Update();
  
#ifdef LOG_OUTPUT
  debugger.close();
  debugger.close();
#endif
  
  time(&end);
  
  std::cout<<" Time taken for segmentation "<<iter<<
    " : "<<difftime(end, start)<<" secs"<<std::endl;
    
}  
  


template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
GrowCutSlowROI( TOutputImage * output)
{

  unsigned int ndims = static_cast< unsigned int>(output->GetImageDimension());
  
  InputImagePointer inputImage = const_cast<InputImageType * >(this->GetInput());

  // Initialize the state image
  typename OutputImageType::Pointer pixelStateImage = OutputImageType::New();

  InitializeStateImage(pixelStateImage);

  // Initialize the max distance for each pixel in the image
  typename WeightImageType::Pointer maxDistancesImage = WeightImageType::New();
  InitializeDistancesImage(inputImage, maxDistancesImage);  

  // Initialize the maximum saturation for each pixel
  typename WeightImageType::Pointer maxSaturationImage = WeightImageType::New();
  
  maxSaturationImage->CopyInformation( inputImage );
  maxSaturationImage->SetBufferedRegion( inputImage->GetBufferedRegion() );  
  maxSaturationImage->Allocate();
  maxSaturationImage->FillBuffer( 0 );       

  unsigned int iter = 0;
  bool converged = false;
  
  time_t startTime;
  time_t endTime;
  time_t start;
  time_t end;


  double prevSaturatedPix = 0.0;
  double prevLocallySaturatedPix = 0.0;
  double prevUnlabeledPix = 1.0;

  unsigned int maxRadius = 0;
  
  float totalROIVolume = 1.0;

  for (unsigned n = 0; n < ndims; n++)
    {
      prevUnlabeledPix *= (m_roiEnd[n]-m_roiStart[n]);
      totalROIVolume *= (m_roiEnd[n]-m_roiStart[n]);
      
      if(n == 0)
 {
   maxRadius = m_roiEnd[n]-m_roiStart[n];
 }
      else if(maxRadius < (m_roiEnd[n]-m_roiStart[n]))
 {
   maxRadius = m_roiEnd[n]-m_roiStart[n];
 }
    }
  
  maxRadius = static_cast< unsigned int>(vcl_ceil(maxRadius*0.5));

  
  float convergenceCapacity = 0.2;
  float threshSaturation = .96; //.999; // Determine the saturation according to the size of the object
  float threshUnchanged = 0.05;
  float threshCapacity = 0.2;
  float threshDiff = 0.001;
  
  float prevCapacity = 0.0;
  
  OutputImagePointer tmpLabels = OutputImageType::New();
  tmpLabels->CopyInformation( output );
  tmpLabels->SetBufferedRegion( output->GetBufferedRegion() );
  tmpLabels->Allocate();
  
  WeightImagePointer tmpWeights = WeightImageType::New();
  tmpWeights->CopyInformation( m_WeightImage );
  tmpWeights->SetBufferedRegion( m_WeightImage->GetBufferedRegion() );
  tmpWeights->Allocate();

  bool useROI = true;

  vcl_vector< unsigned > stateVolumes;
  stateVolumes.resize(3);
  
  time(&start);
  unsigned int iterRadius = m_ObjectRadius;


  vcl_vector< OutputIndexType > visitedSeeds;
  OutputIndexType seed = GetSeed( pixelStateImage, 
      visitedSeeds, LABELED);
  
  while (iter < m_MaxIterations && !converged)
    {
      
      time(&startTime);
      
      bool LocallyConverged = false;

      float currCapacity = UpdateLabels(iterRadius, 
     inputImage,
     output,
     pixelStateImage, 
     maxDistancesImage, 
     maxSaturationImage,
     true,
     useROI,
     seed, 
     stateVolumes);
      
      
      // swap the weights and labels
      tmpWeights = m_WeightImage;
      m_WeightImage = m_PrevWeightImage;
      m_PrevWeightImage = tmpWeights;
      
      tmpLabels = output;
      output = m_LabelImage; 
      m_LabelImage = tmpLabels;

      std::cout<<"Current Capacity "<<currCapacity<<" Prev Capacity "<<prevCapacity<<std::endl;
      unsigned changeablePix = stateVolumes[0];
      unsigned currLocallySaturatedPix = stateVolumes[1];
      unsigned currSaturatedPix = stateVolumes[2];
      
      unsigned unlabeled = (changeablePix + currLocallySaturatedPix + currSaturatedPix <= totalROIVolume) ? 
 (static_cast<unsigned > (totalROIVolume) -(currLocallySaturatedPix + 
         currSaturatedPix + changeablePix)) : 0;
      
      std::cout<<" saturated Pixels "<<currSaturatedPix<<"( "<<
 totalROIVolume<<") %"<<currSaturatedPix/totalROIVolume<<std::endl;
      std::cout<<" locally saturated Pixels "<<currLocallySaturatedPix<<" %"<<
 currLocallySaturatedPix/totalROIVolume<<std::endl;
      std::cout<<"Number of Labeled Pixels "<<changeablePix<<" %"<<
 changeablePix/totalROIVolume<<std::endl;
   
      std::cout<<"Unlabeled pixels "<<unlabeled<<" %"<<
 unlabeled/totalROIVolume<<std::endl;
   
      converged = (unlabeled/totalROIVolume < threshUnchanged && 
     ((currSaturatedPix+currLocallySaturatedPix)/totalROIVolume > threshSaturation));

      prevSaturatedPix = currSaturatedPix;
      prevUnlabeledPix = unlabeled;
      prevLocallySaturatedPix = currLocallySaturatedPix;
      
      prevCapacity = currCapacity;

      ++iter;

    }

  MaskSegmentedImageByWeight(m_ConfThresh);

  this->GraftOutput(m_LabelImage);
  
  time(&end);
  
  vcl_vector< unsigned > labelVolumes;
  vcl_vector< unsigned > physicalVolumes;
  ComputeLabelVolumes(m_LabelImage, labelVolumes, physicalVolumes);
  
  std::cout<<"...................................................."<<std::endl;
  for (unsigned n = 0; n < labelVolumes.size(); n++)
    {
      std::cout<<" Object "<<n<<" volume "<<labelVolumes[n]<<" Physical Volume "<<
 physicalVolumes[n]<<std::endl;
    } 
  
  typedef ImageFileWriter < OutputImageType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  
  vcl_string outfilename = "segmented_img.mhd";
  writer->SetFileName(outfilename.c_str());
  writer->SetInput(m_LabelImage);
  writer->Update();

  std::cout<<" Time taken for segmentation "<<iter<<
    " : "<<difftime(end, start)<<" secs"<<std::endl;
  
}








template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
GrowCutSlow( OutputImageType * output)
{

//  std::ofstream debugger;  
//  debugger.open("D:/projects/namic2010/debuglog.txt", std::ios::out|std::ios::app);

//  debugger<<"Applying the grow cut algorithm.. "<<std::endl;

    // pre-process
  WeightImagePointer maxNeighborDistance = WeightImageType::New();
  maxNeighborDistance->CopyInformation( output );
  maxNeighborDistance->SetBufferedRegion( output->GetBufferedRegion() );
  maxNeighborDistance->Allocate();
  maxNeighborDistance->FillBuffer( 0 );   
 
  typedef Image<unsigned int, itkGetStaticConstMacro(InputImageDimension)> CountImageType;
  typedef typename CountImageType::Pointer CountImagePointer; 
  typedef typename CountImageType::PixelType CountPixelType; 

  CountImagePointer enemyCountImage = CountImageType::New();
  enemyCountImage->CopyInformation( output );
  enemyCountImage->SetBufferedRegion( output->GetBufferedRegion() );
  enemyCountImage->Allocate();
  enemyCountImage->FillBuffer( 0 );   


  // allocate tmp buffers for labels and weights
  OutputImagePointer tmpLabels = OutputImageType::New();
  tmpLabels->CopyInformation( output );
  tmpLabels->SetBufferedRegion( output->GetBufferedRegion() );
  tmpLabels->Allocate();

  WeightImagePointer tmpWeights = WeightImageType::New();
  tmpWeights->CopyInformation( m_WeightImage );
  tmpWeights->SetBufferedRegion( m_WeightImage->GetBufferedRegion() );
  tmpWeights->Allocate();

  InputImagePointer inputImage = const_cast<InputImageType * >( this->GetInput() );//this->GetInput();

  ImageRegionIterator< WeightImageType > dist(maxNeighborDistance, 
    maxNeighborDistance->GetBufferedRegion() );
  
  int checkIndx = -1;
  InputImagePixelType pix;

  ConstNeighborhoodIterator<InputImageType>
        input(m_Radius, inputImage, inputImage->GetBufferedRegion());
  ConstantBoundaryCondition<InputImageType> cbc;
  cbc.SetConstant( NumericTraits< InputImagePixelType >::min(InputImagePixelType()) );
  input.OverrideBoundaryCondition( &cbc );

  for (input.GoToBegin(), dist.GoToBegin(); !input.IsAtEnd(); ++input, ++dist)
  { 
     ++checkIndx;
    // get the value of the center pixel
    InputImagePixelType center = input.GetCenterPixel();

    InputImagePixelType maxDistance = (InputImagePixelType)0.0;
    InputImagePixelType distance;
 
    for (unsigned i = 0; i < input.Size(); i++)
    {
      pix = input.GetPixel(i); 

      distance = (pix - center)*(pix-center);
      maxDistance = (distance > maxDistance) ? distance : maxDistance;       
    }    
//    maxDistance += 0.0000001; // to compensate for divide by zero condition
    dist.Set(maxDistance);
  }

  vcl_cout<<"Finished computing the max Distance "<<vcl_endl;
  
  NeighborhoodIterator< CountImageType > count(m_Radius, enemyCountImage, 
    enemyCountImage->GetBufferedRegion() );

  ConstantBoundaryCondition<CountImageType> cbci;
  cbci.SetConstant( NumericTraits< CountPixelType >::min(CountPixelType()) );
  count.OverrideBoundaryCondition( &cbci );  

  bool changedPix = true;
  unsigned int iter = 0;
  while(iter < m_MaxIterations && changedPix)
  {

      changedPix = false;

      NeighborhoodIterator< OutputImageType > label(m_Radius, m_LabelImage, 
       m_LabelImage->GetBufferedRegion() );
  
     ConstantBoundaryCondition<OutputImageType> cbco;
     cbco.SetConstant( NumericTraits< OutputImagePixelType >::min( OutputImagePixelType() ) );
     label.OverrideBoundaryCondition( &cbco );
     
     NeighborhoodIterator< WeightImageType > weight(m_Radius, m_PrevWeightImage, 
       m_PrevWeightImage->GetBufferedRegion() );
  
     ConstantBoundaryCondition<WeightImageType> cbcs;
     cbcs.SetConstant( NumericTraits< WeightImagePixelType >::min(WeightImagePixelType() ) );
     weight.OverrideBoundaryCondition( &cbcs );

     NeighborhoodIterator< WeightImageType > newWeight(m_Radius, m_WeightImage, 
     m_WeightImage->GetBufferedRegion() );
     newWeight.OverrideBoundaryCondition( &cbcs );
  
     ImageRegionIterator< OutputImageType > out(output, output->GetRequestedRegion() );

     std::cout<<" iteration : "<<iter<<std::endl;
//   debugger<<"iteration "<<iter<<std::endl;

     dist.GoToBegin(); 
     input.GoToBegin();
     label.GoToBegin();
     weight.GoToBegin();
     out.GoToBegin();
     newWeight.GoToBegin();
     count.GoToBegin();

     // count the number of enemies
     for (; !label.IsAtEnd() && !count.IsAtEnd(); ++label, ++count)
     {
    
       OutputImagePixelType l_center = label.GetCenterPixel();
       
       CountPixelType c = 0;
       for (unsigned i = 0; i < label.Size(); i++){
         
         OutputImagePixelType l = label.GetPixel(i);
         c += (l != l_center && l != m_UnknownLabel) ? 1 : 0;

       }    
       count.SetCenterPixel(c);
     }

     label.GoToBegin(); count.GoToBegin();


     for(; !input.IsAtEnd(); ++input, ++label, ++weight, 
 ++newWeight, ++dist, ++out, ++count) {

         OutputImagePixelType    l_center = label.GetCenterPixel();
         WeightImagePixelType  w_center = weight.GetCenterPixel();
         InputImagePixelType     f_center = input.GetCenterPixel();
         
         unsigned int i;
         i = 0;
         
  // CountPixelType numEnemies = count.GetCenterPixel();
         
         OutputImagePixelType winnerLabel = l_center;
         WeightImagePixelType winnerWeight = w_center;

         OutputImagePixelType weakestLabel = l_center;
         WeightImagePixelType weakestWeight = w_center;
    
         for (; i < input.Size(); ++i) {

            
            InputImagePixelType     f = input.GetPixel(i);            
            OutputImagePixelType     l = label.GetPixel(i);
            WeightImagePixelType  w = weight.GetPixel(i);

            if( l == m_UnknownLabel || w_center >= w) 
            {
              continue; 
            }
            else
            {
               WeightImagePixelType attackWeight = (f_center - f)*(f_center - f);
 
               attackWeight = (dist.Value() > 0) ? (1.0 - attackWeight/dist.Value())*w : 0.0;
            
        //               CountPixelType nEnemies = count.GetPixel(i);
 
               // keep the maximum weight and label so we saturate faster
//               if(attackWeight > w_center && nEnemies < m_T1) { 
//                  winnerLabel = l;
//                  winnerWeight = attackWeight;
//               }

              if(attackWeight > w_center) { 
           w_center = attackWeight;
           l_center = l;
                  winnerLabel = l;
                  winnerWeight = attackWeight;
               }

               //update the weakest member
               if(attackWeight < weakestWeight ){
              
                 weakestLabel = l;
                 weakestWeight = attackWeight; 
               }
           } 
        } // end for
       
        // update the label and weight after applying the smoothing condition
 if(winnerLabel != m_UnknownLabel)
        {
          out.Set(winnerLabel);
   newWeight.SetCenterPixel(winnerWeight);
   changedPix = true;
        }
//        if(numEnemies >= m_T2 && weakestLabel != m_UnknownLabel){ // force occupy by the weakest member
//          out.Set(weakestLabel);
//          newWeight.SetCenterPixel(weakestWeight);
//          changedPix = true;
 //       }
//        else if(winnerLabel != m_UnknownLabel){
//          out.Set(winnerLabel);
//          newWeight.SetCenterPixel(winnerWeight);
//          changedPix = true;
 //       }  
     }

     // swap the weights and labels
     tmpWeights = m_WeightImage;
     m_WeightImage = m_PrevWeightImage;
     m_PrevWeightImage = tmpWeights;

    tmpLabels = output;
    output = m_LabelImage; 
    m_LabelImage = tmpLabels;

    std::cout<<"Setting new weights and labels to old weights and labels"<<std::endl;
//  debugger<<"Setting new weights and labels to old weights and labels"<<std::endl;
     ++iter;
  }

  m_WeightImage = m_PrevWeightImage;
  
  this->GraftOutput(m_LabelImage);
  
//  debugger.close();
}


template <class TInputImage, class TOutputImage, class TWeightPixelType>
void 
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::GenerateData() {
  
    
  OutputImagePointer output = this->GetOutput();   
 
  this->Initialize(output);

  std::cout<<"Number of Iterations for grow cut : "<<m_MaxIterations<<std::endl;
  if(m_UseSlow)
    {
      this->GrowCutSlowROI(output);
    }
  else
    {
      this->GrowCutFast(output);
    }
   
}



}//namespace itk

#endif
