/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $HeadURL: http://svn.slicer.org/Slicer4/trunk/Modules/CLI/Threshold.cxx $
  Language:  C++
  Date:      $Date: 2006-12-20 16:00:24 -0500 (Wed, 20 Dec 2006) $
  Version:   $Revision: 1892 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

// ITK includes
#include "itkImageFileWriter.h"

#include "itkThresholdImageFilter.h"

#include "itkPluginUtilities.h"
#include "ThresholdScalarVolumeCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{

template <class T>
int DoIt( int argc, char * argv[] )
{

  PARSE_ARGS;

  typedef    T InputPixelType;
  typedef    T OutputPixelType;

  typedef itk::Image<InputPixelType,  3> InputImageType;
  typedef itk::Image<OutputPixelType, 3> OutputImageType;

  typedef itk::ImageFileReader<InputImageType>  ReaderType;
  typedef itk::ImageFileWriter<OutputImageType> WriterType;

  typedef itk::ThresholdImageFilter<
    InputImageType>  FilterType;

  typename ReaderType::Pointer reader1 = ReaderType::New();
  itk::PluginFilterWatcher watchReader1(reader1, "Read Volume",
                                        CLPProcessInformation);

  reader1->SetFileName( InputVolume.c_str() );

  typename FilterType::Pointer filter = FilterType::New();
  itk::PluginFilterWatcher watchFilter(filter,
                                       "Threshold image",
                                       CLPProcessInformation);

  filter->SetInput( 0, reader1->GetOutput() );
  filter->SetOutsideValue(OutsideValue);

  if( ThresholdType == std::string("Outside") )
    {
    filter->ThresholdOutside(Lower, Upper);
    }
  else if( ThresholdType == std::string("Below") )
    {
    filter->ThresholdBelow(ThresholdValue);
    }
  else if( ThresholdType == std::string("Above") )
    {
    filter->ThresholdAbove(ThresholdValue);
    }
  typename WriterType::Pointer writer = WriterType::New();
  itk::PluginFilterWatcher watchWriter(writer,
                                       "Write Volume",
                                       CLPProcessInformation);
  writer->SetFileName( OutputVolume.c_str() );
  writer->SetInput( filter->GetOutput() );
  writer->SetUseCompression(1);
  writer->Update();

  return EXIT_SUCCESS;
}

} // end of anonymous namespace

int main( int argc, char * argv[] )
{

  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType     pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType(InputVolume, pixelType, componentType);

    switch( componentType )
      {
      case itk::ImageIOBase::UCHAR:
        return DoIt<unsigned char>( argc, argv );
        break;
      case itk::ImageIOBase::CHAR:
        return DoIt<char>( argc, argv );
        break;
      case itk::ImageIOBase::USHORT:
        return DoIt<unsigned short>( argc, argv );
        break;
      case itk::ImageIOBase::SHORT:
        return DoIt<short>( argc, argv );
        break;
      case itk::ImageIOBase::UINT:
        return DoIt<unsigned int>( argc, argv );
        break;
      case itk::ImageIOBase::INT:
        return DoIt<int>( argc, argv );
        break;
      case itk::ImageIOBase::ULONG:
        return DoIt<unsigned long>( argc, argv );
        break;
      case itk::ImageIOBase::LONG:
        return DoIt<long>( argc, argv );
        break;
      case itk::ImageIOBase::FLOAT:
        return DoIt<float>( argc, argv );
        break;
      case itk::ImageIOBase::DOUBLE:
        return DoIt<double>( argc, argv );
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cout << "unknown component type" << std::endl;
        break;
      }
    }

  catch( itk::ExceptionObject & excep )
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
