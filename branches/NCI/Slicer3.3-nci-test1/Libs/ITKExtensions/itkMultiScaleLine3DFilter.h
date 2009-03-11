/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMultiScaleHessianLine3DFilter.h,v $
  Language:  C++
  Date:      $Date:   $
  Version:   $Revision:   $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMultiScaleHessianLine3DFilter_h
#define __itkMultiScaleHessianLine3DFilter_h


#include "itkImageToImageFilter.h"
#include "itkImage.h" 
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkHessianSmoothed3DToLineFilter.h"

namespace itk
{
/**\class MultiScaleHessianLine3DFilter
 * \brief A filter to enhance 3D vascular structures using Hessian 
 *         eigensystem in a multiscale framework
 * 
 * This is an implementation of the work described in the reference below
 *
 * \par References
 * Yoshinobu Sato, Shin Nakajima, Nobuyuki Shiraga, Hideki Atsumi,
 * Shigeyuki Yoshida, Thomas Koller, Guido Gerig and Ron Kikinis (1998). 
 * Three-dimensional multi-scale line filter for segmentation and
 * visualization of curvilinear structures in medical images
 * Medical Image Analysis, 2(2), 143-168. 
 * 
 *  
 * \sa HessianSmoothed3DToLineFilte
 * \sa HessianSmoothedRecursiveGaussianImageFilter 
 * \sa SymmetricEigenAnalysisImageFilter
 * \sa SymmetricSecondRankTensor
 * 
 * \ingroup IntensityImageFilters TensorObjects
 *
 */
template <class TInputImage, 
          class TOutputImage = TInputImage >
class ITK_EXPORT MultiScaleLine3DFilter 
: public
ImageToImageFilter< TInputImage,TOutputImage > 
{
public:
  /** Standard class typedefs. */
  typedef MultiScaleLine3DFilter Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>              Superclass;

  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;

  
  typedef TInputImage                                    InputImageType;
  typedef TOutputImage                                   OutputImageType;

  typedef typename TInputImage::PixelType                InputPixelType;
  typedef typename TOutputImage::PixelType               OutputPixelType;

  /** Update image buffer that holds the best vesselness response */ 
  typedef Image< double, 3>                              UpdateBufferType;

  /** Image dimension = 3. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                   ::itk::GetImageDimension<InputImageType>::ImageDimension);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Set/Get macros for Alpha */
  itkSetMacro(SigmaMin, double);
  itkGetMacro(SigmaMin, double);
  
  /** Set/Get macros for Beta */
  itkSetMacro(SigmaMax, double);
  itkGetMacro(SigmaMax, double);

  /** Set/Get macros for Number of Scales */
  itkSetMacro(NumberOfSigmaSteps, int);
  itkGetMacro(NumberOfSigmaSteps, int);

  /**Set/Get for VesselnessFilterType*/
  void SetAlpha(double alpha){ this->m_VesselnessFilter->SetAlpha(alpha); }
  double GetAlpha(){ return( this->m_VesselnessFilter->GetAlpha() );}

  void SetGamma12(double gamma12){ this->m_VesselnessFilter->SetGamma12(gamma12); }
  double GetGamma12(){ return( this->m_VesselnessFilter->GetGamma12() ); }

  void SetGamma23(double gamma23){ this->m_VesselnessFilter->SetGamma23(gamma23); }
  double GetGamma23(){ return( this->m_VesselnessFilter->GetGamma23() ); }


protected:
  MultiScaleLine3DFilter();
  ~MultiScaleLine3DFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  typedef HessianRecursiveGaussianImageFilter< InputImageType >
                                                        HessianFilterType;

  typedef HessianSmoothed3DToLineFilter< double > VesselnessFilterType;

  /** Generate Data */
  void GenerateData( void );

private:
  void UpdateMaximumResponse();

  double ComputeSigmaValue( int scaleLevel );
  
  void   AllocateUpdateBuffer();

  //purposely not implemented
  MultiScaleLine3DFilter(const Self&); 
  void operator=(const Self&); //purposely not implemented

  double                                            m_SigmaMin;
  double                                            m_SigmaMax;

  int                                               m_NumberOfSigmaSteps;

  typename VesselnessFilterType::Pointer            m_VesselnessFilter;
  typename HessianFilterType::Pointer               m_HessianFilter;


  UpdateBufferType::Pointer                         m_UpdateBuffer;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiScaleLine3DFilter.txx"
#endif
  
#endif
