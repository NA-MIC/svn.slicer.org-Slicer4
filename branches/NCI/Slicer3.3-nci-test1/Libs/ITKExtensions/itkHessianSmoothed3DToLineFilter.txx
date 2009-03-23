/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkHessianSmoothed3DToLineFilter.txx,v $
  Language:  C++
  Date:      $ $
  Version:   $ $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkHessianSmoothed3DToLineFilter_txx
#define __itkHessianSmoothed3DToLineFilter_txx

#include <algorithm>

#include "itkHessianSmoothed3DToLineFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "vnl/vnl_math.h"

#define EPSILON  1e-03

namespace itk
{

/**
 * Constructor
 */
template < typename TPixel >
HessianSmoothed3DToLineFilter< TPixel >
::HessianSmoothed3DToLineFilter()
{
  m_Alpha = 0.25;
  m_Gamma12 = 1.0;
  m_Gamma23 = 1.0;

  m_SymmetricEigenValueFilter = EigenAnalysisFilterType::New();
  m_SymmetricEigenValueFilter->SetDimension( ImageDimension );
  m_SymmetricEigenValueFilter->OrderEigenValuesBy( 
      EigenAnalysisFilterType::FunctorType::OrderByValue );
  
}


template < typename TPixel >
void 
HessianSmoothed3DToLineFilter< TPixel >
::GenerateData()
{
  itkDebugMacro(
      << "HessianSmoothed3DToLineFilter generating data ");

  m_SymmetricEigenValueFilter->SetInput( this->GetInput() );
  
  typename OutputImageType::Pointer output = this->GetOutput();

  typedef typename EigenAnalysisFilterType::OutputImageType
                                            EigenValueImageType;

  m_SymmetricEigenValueFilter->Update();
  
  const typename EigenValueImageType::ConstPointer eigenImage = 
                    m_SymmetricEigenValueFilter->GetOutput();
  
  // walk the region of eigen values and get the vesselness measure
  EigenValueArrayType eigenValue;
  ImageRegionConstIterator<EigenValueImageType> it;
  it = ImageRegionConstIterator<EigenValueImageType>(
      eigenImage, eigenImage->GetRequestedRegion());
  ImageRegionIterator<OutputImageType> oit;
  this->AllocateOutputs();
  oit = ImageRegionIterator<OutputImageType>(output,
                                             output->GetRequestedRegion());

  //hessian
//  typedef Image< SymmetricSecondRankTensor< double, 3 >, 3 > MatrixImageType;
//  MatrixImageType::ConstPointer hessian = this->GetInput();
//  ImageRegionConstIterator<MatrixImageType> hessianIter = ImageRegionConstIterator<MatrixImageType>(hessian, hessian->GetRequestedRegion());
//  hessianIter.GoToBegin();

  oit.GoToBegin();
  it.GoToBegin();
  while (!it.IsAtEnd())
    {

//    std::cout << "Hessian" << std::endl;
//    std::cout << hessianIter.Get() << std::endl;

    // Get the eigen value
    eigenValue = it.Get();

    //Should be sorted by value
    double Lambda1 = eigenValue[0];  //smallest
    double Lambda2 = eigenValue[1];
    double Lambda3 = eigenValue[2];  //largest

    //Make sure they are sorted so that Lambda1 >= Lambda2 >= Lambda3
    if ( Lambda1 < Lambda2  ) std::swap(Lambda1, Lambda2);
    if ( Lambda2 < Lambda3  ) std::swap(Lambda2, Lambda3);
    if ( Lambda1 < Lambda2  ) std::swap(Lambda1, Lambda2);

    double Lambda2Abs = vnl_math_abs( Lambda2 );
    double Lambda3Abs = vnl_math_abs( Lambda3 );
    double SigmaSquared = this->m_Sigma * this->m_Sigma;
    
    double lineResponse = 0.0;
    
    if( Lambda3 < Lambda2 && Lambda2 < Lambda1 && Lambda1 <= 0.0 )
      {
        double term1 = vcl_pow (Lambda2/Lambda3, this->m_Gamma23); 
        double term2 = vcl_pow (1.0 + Lambda1/Lambda2Abs, this->m_Gamma12); 

        lineResponse = SigmaSquared * Lambda3Abs * term1 * term2;

        oit.Set( lineResponse );
      }
    else if( Lambda3 < Lambda2 && Lambda2 < 0.0 && 0.0 < Lambda1 && Lambda1 < Lambda2Abs/this->m_Alpha)
      {
        double term1 = vcl_pow (Lambda2/Lambda3, this->m_Gamma23); 
        double term2 = vcl_pow (1.0 - this->m_Alpha*Lambda1/Lambda2Abs, this->m_Gamma12); 

        lineResponse =  SigmaSquared * Lambda3Abs * term1 * term2;

        oit.Set( lineResponse );
      }

    else
      {
        oit.Set( NumericTraits< OutputPixelType >::Zero );
      } 

//     std::cout << "Eigenvalues & Vesselness " << std::endl;
//     std::cout << Lambda1 << " " << Lambda2 << " " << Lambda3 << " " << lineResponse << std::endl;

    ++it;
    ++oit;
//    ++hessianIter;
    }
    
}

template < typename TPixel >
void
HessianSmoothed3DToLineFilter< TPixel >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  
  os << indent << "Alpha: " << m_Alpha << std::endl;
  os << indent << "Gamma12: " << m_Gamma12 << std::endl;
  os << indent << "Gamma23: " << m_Gamma23 << std::endl;
}


} // end namespace itk
  
#endif
