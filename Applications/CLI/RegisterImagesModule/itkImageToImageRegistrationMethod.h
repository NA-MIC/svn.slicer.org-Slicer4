/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: RigidRegistrator.h,v $
  Language:  C++
  Date:      $Date: 2006/11/06 14:39:34 $
  Version:   $Revision: 1.15 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __ImageToImageRegistrationMethod_h
#define __ImageToImageRegistrationMethod_h

#include "itkImage.h"
#include "itkCommand.h"

#include "itkSpatialObject.h"

#include "itkImageRegistrationMethod.h"

namespace itk
{

template< class TImage >
class ImageToImageRegistrationMethod 
: public ProcessObject
  {

  public:

    typedef ImageToImageRegistrationMethod            Self;
    typedef ProcessObject                             Superclass;
    typedef SmartPointer< Self >                      Pointer;
    typedef SmartPointer< const Self >                ConstPointer;

    itkTypeMacro( ImageToImageRegistrationMethod, ProcessObject );

    itkNewMacro( Self );

    //
    // Custom Typedefs
    //
    itkStaticConstMacro( ImageDimension, unsigned int,
                         TImage::ImageDimension );

    typedef Transform< double, 
                       itkGetStaticConstMacro( ImageDimension ),
                       itkGetStaticConstMacro( ImageDimension ) >
                                                      TransformType;

    typedef DataObjectDecorator< TransformType >      TransformOutputType;

    typedef typename DataObject::Pointer              DataObjectPointer;

    typedef TImage                                    ImageType;

    typedef SpatialObject< itkGetStaticConstMacro( ImageDimension )  >
                                                      MaskObjectType;

    //
    // Custom Methods
    //
    itkSetMacro( RegistrationNumberOfThreads, unsigned int );
    itkGetMacro( RegistrationNumberOfThreads, unsigned int );

    itkSetObjectMacro( Observer, Command );
    itkGetObjectMacro( Observer, Command );

    void SetFixedImage( typename ImageType::ConstPointer & fixedImage );
    itkGetConstObjectMacro( FixedImage, ImageType );

    void SetMovingImage( typename ImageType::ConstPointer & movingImage );
    itkGetConstObjectMacro( MovingImage, ImageType );

    void SetFixedImageMaskObject( typename MaskObjectType::ConstPointer & maskObject );
    itkGetConstObjectMacro( FixedImageMaskObject, MaskObjectType );

    itkSetMacro( UseFixedImageMaskObject, bool );
    itkGetMacro( UseFixedImageMaskObject, bool );

    void SetMovingImageMaskObject( typename MaskObjectType::ConstPointer & maskObject );
    itkGetConstObjectMacro( MovingImageMaskObject, MaskObjectType );

    itkSetMacro( UseMovingImageMaskObject, bool );
    itkGetMacro( UseMovingImageMaskObject, bool );

    itkSetObjectMacro( Transform, TransformType );
    itkGetObjectMacro( Transform, TransformType );

    const TransformOutputType *     GetOutput( void ) const;

    virtual DataObjectPointer       MakeOutput( unsigned int idx );

    unsigned long                   GetMTime( void ) const;

    itkSetMacro( ReportProgress, bool );
    itkGetMacro( ReportProgress, bool );

  protected:

    ImageToImageRegistrationMethod( void );
    virtual ~ImageToImageRegistrationMethod( void );

    virtual void    Initialize( void );

    void GenerateData( void );

    void PrintSelf( std::ostream & os, Indent indent ) const;

  protected:

    typename TransformType::Pointer        m_Transform;

  private:

    ImageToImageRegistrationMethod( const Self & ); // Purposely not implemented
    void operator = ( const Self & );               // Purposely not implemented

    unsigned int                           m_RegistrationNumberOfThreads;

    Command::Pointer                       m_Observer;

    typename ImageType::ConstPointer       m_FixedImage;
    typename ImageType::ConstPointer       m_MovingImage;

    bool                                   m_UseFixedImageMaskObject;
    typename MaskObjectType::ConstPointer  m_FixedImageMaskObject;

    bool                                   m_UseMovingImageMaskObject;
    typename MaskObjectType::ConstPointer  m_MovingImageMaskObject;

    bool                                   m_ReportProgress;

  };

} 

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToImageRegistrationMethod.txx"
#endif


#endif 

