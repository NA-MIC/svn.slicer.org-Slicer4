/**********************************************************************
 * vtkItkLogicImageFilter
 * Implements wrapper for implementing logical operations 
 * Images should be integer types
 * current version supports only 'Or', 'Xor', and 'And' Operations
 **********************************************************************/

#include "vtkITKLogicImageFilter.h"

#include "itkOrImageFilter.h"
#include "itkXorImageFilter.h"
#include "itkAndImageFilter.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"

#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include <vcl_string.h>

#include <iostream>


vtkCxxRevisionMacro(vtkITKLogicImageFilter, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkITKLogicImageFilter);

template<class IT1>//, class IT2=IT1, class OT=IT1>
void vtkITKImageLogicExecute(vtkImageData *inData, 
      IT1 *inPtr1, IT1 *inPtr2, 
      IT1 *output, int outExt[6],
           int &Operation)     
{

  std::ofstream debugger;
  debugger.open("D:/projects/namic2010/debuglog1.txt", std::ios::out| std::ios::app);
  debugger<<"Performing operation "<<Operation<<std::endl;

  typedef itk::Image<IT1, 3> ImageType;
  typename ImageType::Pointer image1 = ImageType::New();
  typename ImageType::Pointer image2 = ImageType::New();

  typename ImageType::Pointer outImage = ImageType::New();
  
  int dims[3];
  int extent[6];
  double spacing[3], origin[3];

  inData->GetDimensions(dims);
  inData->GetExtent(extent);
  inData->GetOrigin(origin);
  inData->GetSpacing(spacing);

  image1->SetOrigin( origin );
  image1->SetSpacing( spacing );
            
  typename ImageType::RegionType region;
  typename ImageType::IndexType index;
  typename ImageType::SizeType size;

  index[0] = extent[0];   
  index[1] = extent[2];
  index[2] = extent[4];
  region.SetIndex( index );
  size[0] = extent[1] - extent[0] + 1;
  size[1] = extent[3] - extent[2] + 1;
  size[2] = extent[5] - extent[4] + 1;
  region.SetSize( size );

  image1->SetRegions(region);

  image1->GetPixelContainer()->SetImportPointer(inPtr1, dims[0]*dims[1]*dims[2], false);

  image2->SetOrigin( origin );
  image2->SetSpacing( spacing );
  image2->SetRegions( region );
  image2->GetPixelContainer()->SetImportPointer(inPtr2, dims[0]*dims[1]*dims[2], false);

  switch (Operation) 
    {
    case VTK_AND:
      {
  typedef itk::AndImageFilter<ImageType, ImageType > FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput1(image1);
  filter->SetInput2(image2);
  filter->Update();
  outImage = filter->GetOutput();
  break;
      }
    case VTK_OR:
      {
  typedef itk::OrImageFilter<ImageType, ImageType > FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput1(image1);
  filter->SetInput2(image2);
  filter->Update();
  outImage = filter->GetOutput();
  break;
      }
    case VTK_XOR:
      {
  typedef itk::XorImageFilter<ImageType, ImageType > FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetInput1(image1);
  filter->SetInput2(image2);
  filter->Update();    
  outImage = filter->GetOutput();
  break;
      }
    default:
    {
    std::cerr<<" vtkITKLogicImageFilter does not support this Logic functionality"<<std::endl;
      break;
    }
    }
  
  if( Operation == VTK_AND || Operation == VTK_OR || Operation == VTK_XOR ) 
    {
      itk::ImageRegionIterator< ImageType > out(outImage, outImage->GetBufferedRegion() );
      itk::ImageRegionConstIterator< ImageType > in1(image1, image1->GetBufferedRegion() );
      itk::ImageRegionConstIterator< ImageType > in2(image2, image2->GetBufferedRegion() );
      
      for (out.GoToBegin(), in1.GoToBegin(), in2.GoToBegin(); !out.IsAtEnd(); ++out, ++in1, ++in2)
  {
    typename ImageType::PixelType o = out.Get();
    if(o)
      {
      if(in1.Get())
      {
        out.Set(in1.Get());
      } 
        else
      {
        out.Set(in2.Get());
      }
      }
  }
      memcpy(output, outImage->GetBufferPointer(),
       outImage->GetBufferedRegion().GetNumberOfPixels()*sizeof(IT1) ); 
    }

  debugger.close();
}



// constructor
vtkITKLogicImageFilter::vtkITKLogicImageFilter()
{
  Operation = VTK_NOP;
}



void vtkITKLogicImageFilter::ExecuteData(
        vtkDataObject *outData)
{

  vtkImageData *input1 = GetInput1();
  vtkImageData *input2 = GetInput2();

  vtkImageData * out = vtkImageData::SafeDownCast(outData);

 /* input1->SetScalarTypeToShort();
  input2->SetScalarTypeToShort();
  out->SetScalarTypeToShort(); */

  out->SetScalarType( input1->GetScalarType() );

  int outExt[6];
  int dims[3];
  double spacing[3], origin[3];
  
  input1->GetDimensions(dims);
  input1->GetOrigin(origin);
  input1->GetSpacing(spacing);
  input1->GetExtent(outExt);
  
  void *inPtr1 = input1->GetScalarPointerForExtent(outExt);
  void *inPtr2 =  input2->GetScalarPointerForExtent(outExt);
  
  input1->GetWholeExtent(outExt);
  out->SetExtent(outExt);
  out->SetOrigin(origin);
  out->SetSpacing(spacing);
  out->SetDimensions(dims);
  out->AllocateScalars();
  
  input1->GetExtent(outExt);

  void *outPtr = out->GetScalarPointerForExtent(outExt);
  
  if(input1->GetScalarType() == VTK_SHORT)
    {
    vtkITKImageLogicExecute(input1, (short*)(inPtr1), (short*)(inPtr2), 
      (short*)(outPtr), outExt,this->Operation);
    }
  else if(input1->GetScalarType() == VTK_UNSIGNED_SHORT)
  {
    vtkITKImageLogicExecute(input1, (unsigned short*)(inPtr1), (unsigned short*)(inPtr2), 
      (unsigned short*)(outPtr), outExt,this->Operation);
  }
  else if(input1->GetScalarType() == VTK_INT)
  {
    vtkITKImageLogicExecute(input1, (int*)(inPtr1), (int*)(inPtr2), 
      (int*)(outPtr), outExt,this->Operation);
  }
  else if(input1->GetScalarType() == VTK_UNSIGNED_INT)
  {
    vtkITKImageLogicExecute(input1, (unsigned int*)(inPtr1), (unsigned int*)(inPtr2), 
      (unsigned int*)(outPtr), outExt,this->Operation);
  }
  else if (input1->GetScalarType() == VTK_CHAR)
  {
    vtkITKImageLogicExecute(input1, (char*)(inPtr1), (char*)(inPtr2), 
      (char*)(outPtr), outExt,this->Operation);
  }
 else if (input1->GetScalarType() == VTK_UNSIGNED_CHAR)
  {
    vtkITKImageLogicExecute(input1, (unsigned char*)(inPtr1), (unsigned char*)(inPtr2), 
      (unsigned char*)(outPtr), outExt,this->Operation);
  }
 else
 {
  out->DeepCopy(input1);  
 }
    
  return;
}

void vtkITKLogicImageFilter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "Operation : " << Operation<< std::endl;
}

