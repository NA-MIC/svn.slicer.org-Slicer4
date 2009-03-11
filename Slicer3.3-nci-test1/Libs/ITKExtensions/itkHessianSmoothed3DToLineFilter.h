/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: HessianSmoothed3DToLineFilter.h,v $
  Language:  C++
  Date:      $ $
  Version:   $ $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkHessianSmoothed3DToLineFilter_h
#define __itkHessianSmoothed3DToLineFilter_h

#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"

namespace itk
{
/** \class HessianSmoothed3DToLineFilter
 * \brief A filter to enhance 3D vascular structures 
 * 
 *  * This is an implementation of the work described in the reference below
 *
 * \par References
 * Yoshinobu Sato, Shin Nakajima, Nobuyuki Shiraga, Hideki Atsumi,
 * Shigeyuki Yoshida, Thomas Koller, Guido Gerig and Ron Kikinis (1998). 
 * Three-dimensional multi-scale line filter for segmentation and
 * visualization of curvilinear structures in medical images
 * Medical Image Analysis, 2(2), 143-168. 
 * 
 *   
 * 
 * \sa HessianSmoothedRecursiveGaussianImageFilter 
 * \sa SymmetricEigenAnalysisImageFilter
 * \sa SymmetricSecondRankTensor
 * 
 * \ingroup IntensityImageFilters TensorObjects
 *
 */
  
template < typename  TPixel >
class ITK_EXPORT HessianSmoothed3DToLineFilter : public
ImageToImageFilter< Image< SymmetricSecondRankTensor< double, 3 >, 3 >, 
                                                  Image< TPixel, 3 > >
{
public:
  /** Standard class typedefs. */
  typedef  HessianSmoothed3DToLineFilter Self;

  typedef ImageToImageFilter< 
          Image< SymmetricSecondRankTensor< double, 3 >, 3 >, 
          Image< TPixel, 3 > >                 Superclass;

  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;
  
  typedef typename Superclass::InputImageType            InputImageType;
  typedef typename Superclass::OutputImageType           OutputImageType;
  typedef typename InputImageType::PixelType             InputPixelType;
  typedef TPixel                                         OutputPixelType;
  
  /** Image dimension = 3. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                    ::itk::GetImageDimension<InputImageType>::ImageDimension);

  itkStaticConstMacro(InputPixelDimension, unsigned int,
                    InputPixelType::Dimension);

  typedef  FixedArray< double, itkGetStaticConstMacro(InputPixelDimension) >
                                                          EigenValueArrayType;
  typedef  Image< EigenValueArrayType, itkGetStaticConstMacro(ImageDimension) >
                                                          EigenValueImageType;
  typedef   SymmetricEigenAnalysisImageFilter< 
            InputImageType, EigenValueImageType >     EigenAnalysisFilterType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Set/Get macros for Alpha */
  itkSetMacro(Alpha, double);
  itkGetMacro(Alpha, double);

  /** Set/Get macros for Gamma12 */
  itkSetMacro(Gamma12, double);
  itkGetMacro(Gamma12, double);

  /** Set/Get macros for Gamma23 */
  itkSetMacro(Gamma23, double);
  itkGetMacro(Gamma23, double);

  /** Set/Get macros for Sigma */
  itkSetMacro(Sigma, double);
  itkGetMacro(Sigma, double);


#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(DoubleConvertibleToOutputCheck,
                  (Concept::Convertible<double, OutputPixelType>));
  /** End concept checking */
#endif

protected:
  HessianSmoothed3DToLineFilter();
  ~HessianSmoothed3DToLineFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;
  
  /** Generate Data */
  void GenerateData( void );

private:

  //purposely not implemented
  HessianSmoothed3DToLineFilter(const Self&); 

  void operator=(const Self&); //purposely not implemented

  typename EigenAnalysisFilterType::Pointer     m_SymmetricEigenValueFilter;

  double                                        m_Alpha;
  double                                        m_Gamma12;
  double                                        m_Gamma23;
  double                                        m_Sigma;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkHessianSmoothed3DToLineFilter.txx"
#endif
  
#endif
