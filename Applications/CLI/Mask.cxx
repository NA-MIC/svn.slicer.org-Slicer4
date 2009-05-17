/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/trunk/Applications/CLI/Mask.cxx $
  Language:  C++
  Date:      $Date: 2009-04-10 13:33:12 -0400 (Fri, 10 Apr 2009) $
  Version:   $Revision: 9136 $

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

#include "itkMaskImageFilter.h"
#include "itkThresholdImageFilter.h"

#include "itkPluginUtilities.h"
#include "MaskCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace {

template<class T> int DoIt( int argc, char * argv[], T )
{

  PARSE_ARGS;

  typedef    T       InputPixelType;
  typedef    T       OutputPixelType;

  typedef itk::Image< InputPixelType,  3 >   InputImageType;
  typedef itk::Image< OutputPixelType, 3 >   OutputImageType;

  typedef itk::ImageFileReader< InputImageType >  ReaderType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;

  typedef itk::MaskImageFilter<
    InputImageType, InputImageType, OutputImageType >  FilterType;

  typedef itk::ThresholdImageFilter<
    InputImageType >  ThresholdFilterType;
  
  typename ReaderType::Pointer reader1 = ReaderType::New();
  itk::PluginFilterWatcher watchReader1(reader1, "Read Input Volume",
                                        CLPProcessInformation);
  
  typename ReaderType::Pointer reader2 = ReaderType::New();
  itk::PluginFilterWatcher watchReader2(reader2,
                                        "Read Mask Volume",
                                        CLPProcessInformation);
  reader1->SetFileName( InputVolume.c_str() );
  reader2->SetFileName( MaskVolume.c_str() );

  // have to threshold the mask volume
  typename ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();
  itk::PluginFilterWatcher watchThresholdFilter(thresholdFilter,
                                       "Threshold Image",
                                       CLPProcessInformation);

  thresholdFilter->SetInput(0, reader2->GetOutput());
  thresholdFilter->SetOutsideValue(0);
  thresholdFilter->ThresholdOutside(Label,Label);
  
  typename FilterType::Pointer filter = FilterType::New();
  itk::PluginFilterWatcher watchFilter(filter,
                                       "Mask Image",
                                       CLPProcessInformation);

  filter->SetInput( 0, reader1->GetOutput() );
  filter->SetInput( 1, thresholdFilter->GetOutput() );

  typename WriterType::Pointer writer = WriterType::New();
  itk::PluginFilterWatcher watchWriter(writer,
                                       "Write Volume",
                                       CLPProcessInformation);
  writer->SetFileName( OutputVolume.c_str() );
  writer->SetInput( filter->GetOutput() );
  writer->Update();

  return EXIT_SUCCESS;
}

} // end of anonymous namespace


int main( int argc, char * argv[] )
{
  
  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType (InputVolume, pixelType, componentType);

    // This filter handles all types on input, but only produces
    // signed types
    
    switch (componentType)
      {
      case itk::ImageIOBase::UCHAR:
      case itk::ImageIOBase::CHAR:
        return DoIt( argc, argv, static_cast<char>(0));
        break;
      case itk::ImageIOBase::USHORT:
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0));
        break;
      case itk::ImageIOBase::UINT:
      case itk::ImageIOBase::INT:
        return DoIt( argc, argv, static_cast<int>(0));
        break;
      case itk::ImageIOBase::ULONG:
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
