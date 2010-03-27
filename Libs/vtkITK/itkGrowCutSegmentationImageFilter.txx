#ifndef _itkGrowCutSegmentationImageFilter_txx_
#define _itkGrowCutSegmentationImageFilter_txx_   


#include "itkGrowCutSegmentationImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodIterator.h"
//#include "itkConstNeighborhoodIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkVectorContainer.h"
#include "itkNeighborhoodIterator.h"
#include "itkConstantBoundaryCondition.h"
#include "itkImageFunction.h"
#include "itkNumericTraits.h"
#include "vtkITKNumericTraits.h"


#include <iostream>

#define DEBUG_LEVEL_0

namespace itk
{

template< class TInputImage, class TOutputImage, class TWeightPixelType>
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::GrowCutSegmentationImageFilter() {
  
   m_ForegroundPoints = NULL;
   m_BackgroundPoints = NULL;
   m_PriorForegroundPoints = NULL;

   m_MaxIterations = 100;
   
   m_T1 = 10;
   m_T2 = 10;
   
   m_UseSlow = true;
   
   m_SeedStrength = 1.0;
   m_PriorSegmentStrength = 0.0;  

   m_LabelImage = OutputImageType::New();
  
   m_StrengthImage = StrengthImageType::New();

   m_PrevStrengthImage = StrengthImageType::New();


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
::PrintSelf(std::ostream& os, Indent indent) const {

  this->Superclass::PrintSelf(os, indent);
  os << indent << "Object Seed Points : " << m_ForegroundPoints.Print(std::cout)<< std::endl;
  os << indent << "Background Seed Points : " << m_BackgroundPoints.Print(std::cout)<< std::endl;
  os << indent << "max iterations : " << m_MaxIterations<< std::endl;  
  os << indent << "max enemies for attack T1 : " << m_T1<< std::endl;  
  os << indent << "min enemies for submit T2 : " << m_T2<< std::endl;  
  os << indent << "starting seed strength :" <<m_SeedStrength<< std::endl;     
  os << indent << "use Algorithm Speed Slow : " << m_UseSlow<< std::endl;    
  
#ifdef DEBUG_LEVEL_0
   // print the foreground points 
  os <<indent<<" Detail foreground points :"<<std::endl;
  typename NodeContainer::ConstIterator pointsIter = m_ForegroundPoints->Begin();
  typename NodeContainer::ConstIterator pointsEnd = m_ForegroundPoints->End();
  
  int i = 0;
  for (; pointsIter != pointsEnd; ++pointsIter, ++i){
    os<<indent<<" point "<<i<<" : "<<pointsIter.Value()<<std::endl;
  }
#endif

} 

template <class TInputImage, class TOutputImage, class TWeightPixelType>
void 
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::GenerateInputRequestedRegion() {

  Superclass::GenerateInputRequestedRegion();
  if ( this->GetInput() )
    {
    InputImagePointer image = const_cast< InputImageType * >( this->GetInput() );
    image->SetRequestedRegionToLargestPossibleRegion();
    }
}


template <class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::EnlargeOutputRequestedRegion(DataObject *output) {

  Superclass::EnlargeOutputRequestedRegion(output);
  output->SetRequestedRegionToLargestPossibleRegion();
  
}



template <class TInputImage, class TOutputImage, class TWeightPixelType>
void
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::Initialize( OutputImageType * output ) {

 // std::ofstream debugger;  
//  debugger.open("D:/projects/namic2010/debuglog.txt", std::ios::out|std::ios::app);
  
  // allocate memory for the output buffer
  output->SetBufferedRegion( output->GetRequestedRegion() );
  output->Allocate();
  output->FillBuffer ( m_UnknownLabel );
 
  // allocate memory if it has not been set already
  typename LabelImageType::SizeType lsize = m_LabelImage->GetBufferedRegion().GetSize();
  
  if( lsize[0] == 0 && lsize[1] == 0 && lsize[2] == 0)
  {
   // debugger<<"Initializing label image ... "<<std::endl;
     m_LabelImage->CopyInformation( output );
     m_LabelImage->SetBufferedRegion( output->GetRequestedRegion() );
     m_LabelImage->Allocate();
     m_LabelImage->FillBuffer( m_UnknownLabel );   
  }
  else
  {
    //debugger<<"Label Image has already been set by the user "<<std::endl;
  }

  // allocate memory if the strength image has not been set already
  m_StrengthImage->CopyInformation( output );
  m_StrengthImage->SetBufferedRegion( output->GetRequestedRegion() );
  m_StrengthImage->Allocate();
  m_StrengthImage->FillBuffer( m_SeedStrength*0.0 );   
 
 // allocate memory
  typename StrengthImageType::SizeType wsize = m_PrevStrengthImage->GetBufferedRegion().GetSize();
  if( wsize[0] == 0 && wsize[1] == 0 && wsize[2] == 0) 
  {
     //debugger<<"Initializing weight image ... "<<std::endl;
     m_PrevStrengthImage->CopyInformation( output );
     m_PrevStrengthImage->SetBufferedRegion( output->GetRequestedRegion() );
     m_PrevStrengthImage->Allocate();
     m_PrevStrengthImage->FillBuffer( m_SeedStrength*0.0 );   
  }
  else
  {
//   debugger<<"Weight Image has already been set by the user "<<std::endl;
  }

  
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

     typename NodeContainer::ConstIterator pointsIter = m_ForegroundPoints->Begin();
     typename NodeContainer::ConstIterator pointsEnd = m_ForegroundPoints->End();

      for (; pointsIter != pointsEnd; ++pointsIter ) {

       IndexType node = pointsIter.Value();
  
      // check if node index is within the output level set
      if ( !m_BufferedRegion.IsInside( node ) ) 
        continue;

      // make this an object point
      m_LabelImage->SetPixel( node, m_ObjectLabel );
      
      output->SetPixel( node, m_ObjectLabel );    
      
      // Set the Strength
      m_StrengthImage->SetPixel( node, m_SeedStrength );
      m_PrevStrengthImage->SetPixel( node, m_SeedStrength );

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
     typename NodeContainer::ConstIterator pointsIter = m_BackgroundPoints->Begin();
     typename NodeContainer::ConstIterator pointsEnd = m_BackgroundPoints->End();

      for (; pointsIter != pointsEnd; ++pointsIter ) {

       IndexType node = pointsIter.Value();
  
       // check if node index is within the output level set
       if ( !m_BufferedRegion.IsInside( node ) ) 
        continue;

       // make this a background point
       m_LabelImage->SetPixel( node, m_BackgroundLabel );

       output->SetPixel( node, m_BackgroundLabel );  

       m_StrengthImage->SetPixel( node, m_SeedStrength );  
       m_PrevStrengthImage->SetPixel( node, m_SeedStrength );  

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

/*
   if ( m_PriorForegroundPoints ) {

     typename NodeContainer::ConstIterator pointsIter = m_PriorForegroundPoints->Begin();
     typename NodeContainer::ConstIterator pointsEnd = m_PriorForegroundPoints->End();

      for (; pointsIter != pointsEnd; ++pointsIter ) {

       IndexType node = pointsIter.Value();
  
      // check if node index is within the output level set
      if ( !m_BufferedRegion.IsInside( node ) ) 
        continue;

      // make this an object point
      m_LabelImage->SetPixel( node, m_ObjectLabel );
      
      output->SetPixel( node, m_ObjectLabel );    
      
      // Set the Strength
      m_StrengthImage->SetPixel( node, m_PriorSegmentStrength );
      m_PrevStrengthImage->SetPixel( node, m_PriorSegmentStrength );

     // add the neighbor pixels of the seeds to the alive points
     // we don't add the seeds themselves as they are not going to get updated
     if( !m_UseSlow )
     {
      out.SetLocation( node );
      for (unsigned i = 0; i < out.Size(); i++){
     
           IndexType nIndex = out.GetIndex(i);
           m_AlivePoints.push_back( nIndex );
      }
     }  
    }
  }
*/

// debugger.close();

}




template<class TInputImage, class TOutputImage, class TWeightPixelType>
void GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>::
GrowCutSlow( OutputImageType * output)
{
   
//  std::ofstream debugger;  
//  debugger.open("D:/projects/namic2010/debuglog.txt", std::ios::out|std::ios::app);

//  debugger<<"Applying the grow cut algorithm.. "<<std::endl;

    // pre-process
  StrengthImagePointer maxNeighborDistance = StrengthImageType::New();
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

  StrengthImagePointer tmpWeights = StrengthImageType::New();
  tmpWeights->CopyInformation( m_StrengthImage );
  tmpWeights->SetBufferedRegion( m_StrengthImage->GetBufferedRegion() );
  tmpWeights->Allocate();

  InputImagePointer inputImage = const_cast<InputImageType * >( this->GetInput() );//this->GetInput();

  ImageRegionIterator< StrengthImageType > dist(maxNeighborDistance, 
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

    InputImagePixelType maxDistance = 0.0;
    InputImagePixelType distance;
 
    for (unsigned i = 0; i < input.Size(); i++)
    {
      pix = input.GetPixel(i); 

      distance = (pix - center)*(pix-center);
      maxDistance = (distance > maxDistance) ? distance : maxDistance;       
    }    
    maxDistance += 1; // to compensate for divide by zero condition
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

     NeighborhoodIterator< StrengthImageType > weight(m_Radius, m_PrevStrengthImage, 
       m_PrevStrengthImage->GetBufferedRegion() );
  
     ConstantBoundaryCondition<StrengthImageType> cbcs;
     cbcs.SetConstant( NumericTraits< StrengthImagePixelType >::min(StrengthImagePixelType() ) );
     weight.OverrideBoundaryCondition( &cbcs );

     NeighborhoodIterator< StrengthImageType > newWeight(m_Radius, m_StrengthImage, 
     m_StrengthImage->GetBufferedRegion() );
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
         c += (l != l_center && l_center != m_UnknownLabel && l != m_UnknownLabel) ? 1 : 0;
       }    
       count.SetCenterPixel(c);
     }

     label.GoToBegin(); count.GoToBegin();


     for(; (!input.IsAtEnd() && !label.IsAtEnd() && 
            !weight.IsAtEnd() && !newWeight.IsAtEnd() && 
            !dist.IsAtEnd() && !out.IsAtEnd() && !count.IsAtEnd() ); 
           ++input, ++label, ++weight, ++newWeight, ++dist, ++out, ++count) {

         OutputImagePixelType    l_center = label.GetCenterPixel();
         StrengthImagePixelType  w_center = weight.GetCenterPixel();
         InputImagePixelType     f_center = input.GetCenterPixel();
         
         unsigned int i, j, k, c;
         i = j = k = c = 0;
         
         CountPixelType numEnemies = count.GetCenterPixel();
         
         OutputImagePixelType winnerLabel = l_center;
         StrengthImagePixelType winnerWeight = w_center;

         OutputImagePixelType weakestLabel = l_center;
         StrengthImagePixelType weakestWeight = w_center;
    
         for (; i < input.Size() && j < label.Size() && k < weight.Size() && c < count.Size(); 
                ++i, ++j, ++k, ++c) {

            
            InputImagePixelType     f = input.GetPixel(i);            
            OutputImagePixelType     l = label.GetPixel(j);
            StrengthImagePixelType  w = weight.GetPixel(k);

            if( l == m_UnknownLabel || l == l_center) 
            {
              continue; 
            }
            else
            {
               StrengthImagePixelType attackStrength = (f_center - f)*(f_center - f);
               attackStrength = (1.0 - attackStrength/dist.Value())*w;
            
               CountPixelType nEnemies = count.GetPixel(c);
 
               // keep the maximum weight and label so we saturate faster
               if(attackStrength > winnerWeight && 
   nEnemies < m_T1) { //attackStrength > w_center
                  winnerLabel = l;
                  winnerWeight = attackStrength;
               }

               //update the weakest member
               if(attackStrength < weakestWeight ){
              
                 weakestLabel = l;
                 weakestWeight = attackStrength; 
               }
           } 
        } // end for
       
        // update the label and weight after applying the smoothing condition
        if(numEnemies >= m_T2 && weakestLabel != m_UnknownLabel){ // force occupy by the weakest member
          out.Set(weakestLabel);
          newWeight.SetCenterPixel(weakestWeight);
          changedPix = true;
        }
        else if(winnerLabel != m_UnknownLabel){
          out.Set(winnerLabel);
          newWeight.SetCenterPixel(winnerWeight);
          changedPix = true;
        }  
     }

     // swap the weights and labels
     tmpWeights = m_StrengthImage;
     m_StrengthImage = m_PrevStrengthImage;
     m_PrevStrengthImage = tmpWeights;

    tmpLabels = output;
    output = m_LabelImage; 
    m_LabelImage = tmpLabels;

    std::cout<<"Setting new weights and labels to old weights and labels"<<std::endl;
//  debugger<<"Setting new weights and labels to old weights and labels"<<std::endl;
     ++iter;
  }
  
  this->GraftOutput(m_LabelImage);
  
//  debugger.close();
}


template <class TInputImage, class TOutputImage, class TWeightPixelType>
void 
GrowCutSegmentationImageFilter<TInputImage, TOutputImage, TWeightPixelType>
::GenerateData() {
  
    
  OutputImagePointer output = this->GetOutput();   
 
  this->Initialize(output);

  this->GrowCutSlow(output);  
   
}



}//namespace itk

#endif
