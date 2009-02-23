/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/trunk/Applications/CLI/GradientAnisotropicDiffusion.cxx $
  Language:  C++
  Date:      $Date: 2006-12-18 09:29:35 -0500 (Mon, 18 Dec 2006) $
  Version:   $Revision: 1855 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"

#include "itkVesselEnhancingDiffusion3DImageFilter.h"

#include "itkPluginUtilities.h"
#include "VesselEnhancingDiffusionCLP.h"

template<class T> int DoIt( int argc, char * argv[], T )
{

  PARSE_ARGS;

//  typedef    float   InputPixelType;
//  typedef    T       OutputPixelType;
//
//  typedef itk::Image< InputPixelType,  3 >   InputImageType;
//  typedef itk::Image< OutputPixelType, 3 >   OutputImageType;
//
//  typedef itk::ImageFileReader< InputImageType >  ReaderType;
//  typedef itk::ImageFileWriter< OutputImageType > WriterType;
//
//  //typedef itk::VesselEnhancingDiffusion3DImageFilter<InputImageType, InputImageType >  FilterType;
//  typedef itk::VesselEnhancingDiffusion3DImageFilter<short >  FilterType;
//  typedef itk::CastImageFilter<InputImageType, OutputImageType> CastType;
//
//  typename ReaderType::Pointer reader = ReaderType::New();
//
//  typedef itk::VesselEnhancingDiffusion3DImageFilter<InputImageType, InputImageType >  FilterType;
//

  typedef itk::VesselEnhancingDiffusion3DImageFilter<short> VT;
  typedef VT::ImageType                                     IT;
  typedef itk::ImageFileReader<IT>                          RT;
  typedef itk::ImageFileWriter<IT>                          WT;
  typedef itk::CastImageFilter<IT,IT>                          CastType;
  typedef itk::ImageFileWriter< IT > WriterType;

  RT::Pointer reader = RT::New();

  itk::PluginFilterWatcher watchReader(reader, "Read Volume",
                                       CLPProcessInformation);

  reader->SetFileName( inputVolume.c_str() );

  VT::Pointer filter = VT::New();
  itk::PluginFilterWatcher watchFilter(filter, "Vessel Enhancing Diffusion",
    CLPProcessInformation);

  filter->SetInput( reader->GetOutput() );
  // set the default parameters, but override default number of iterations to 
  // keep execution time down during integration.  Default was 30 iterations.   
  filter->SetDefaultPars();
  filter->SetIterations(2);

  CastType::Pointer cast = CastType::New();
  cast->SetInput( filter->GetOutput());

  WriterType::Pointer writer = WriterType::New();
  itk::PluginFilterWatcher watchWriter(writer, "Write Volume",
                                   CLPProcessInformation);
  writer->SetFileName( outputVolume.c_str() );
  writer->SetInput( cast->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}

int main( int argc, char * argv[] )
{

  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType (inputVolume, pixelType, componentType);

    // This filter handles all types

    switch (componentType)
      {
      case itk::ImageIOBase::UCHAR:
        return DoIt( argc, argv, static_cast<unsigned char>(0));
        break;
      case itk::ImageIOBase::CHAR:
        return DoIt( argc, argv, static_cast<char>(0));
        break;
      case itk::ImageIOBase::USHORT:
        return DoIt( argc, argv, static_cast<unsigned short>(0));
        break;
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0));
        break;
      case itk::ImageIOBase::UINT:
        return DoIt( argc, argv, static_cast<unsigned int>(0));
        break;
      case itk::ImageIOBase::INT:
        return DoIt( argc, argv, static_cast<int>(0));
        break;
      case itk::ImageIOBase::ULONG:
        return DoIt( argc, argv, static_cast<unsigned long>(0));
        break;
      case itk::ImageIOBase::LONG:
        return DoIt( argc, argv, static_cast<long>(0));
        break;
      case itk::ImageIOBase::FLOAT:
        return DoIt( argc, argv, static_cast<float>(0));
        break;
      case itk::ImageIOBase::DOUBLE:
        return DoIt( argc, argv, static_cast<double>(0));
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

