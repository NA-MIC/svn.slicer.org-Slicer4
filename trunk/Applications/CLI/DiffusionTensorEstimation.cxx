#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <iostream>
#include <algorithm>
#include <string>

#include "itkPluginFilterWatcher.h"
#include "itkPluginUtilities.h"
#include "vtkSmartPointer.h"
#include "vtkTeemEstimateDiffusionTensor.h"
#include "vtkMatrix4x4.h"
#include "vtkNRRDReader.h"
#include "vtkNRRDWriter.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkMath.h"
#include "vtkImageData.h"
#include "vtkDoubleArray.h"

#include "DiffusionTensorEstimationCLP.h"


int main( int argc, const char * argv[] )
{

  PARSE_ARGS;
  {
  //vtkNRRDReader *reader = vtkNRRDReader::New();
  vtkSmartPointer<vtkNRRDReader> reader = vtkNRRDReader::New();
  reader->SetFileName(inputVolume.c_str());
  reader->Update();

  vtkSmartPointer<vtkDoubleArray> bValues = vtkDoubleArray::New();
  vtkSmartPointer<vtkDoubleArray> grads = vtkDoubleArray::New();
  vtkSmartPointer<vtkMRMLNRRDStorageNode> helper = vtkMRMLNRRDStorageNode::New();
  
  if ( !helper->ParseDiffusionInformation(reader,grads,bValues) )
    {
    std::cerr << argv[0] << ": Error parsing Diffusion information" << std::endl;
    return EXIT_FAILURE;
    }
  vtkSmartPointer<vtkTeemEstimateDiffusionTensor> estim = vtkTeemEstimateDiffusionTensor::New();

  estim->SetInput(reader->GetOutput());
  estim->SetNumberOfGradients(grads->GetNumberOfTuples());
  estim->SetDiffusionGradients(grads);
  estim->SetBValues(bValues);

  // Compute Transformation that brings the gradients to ijk
  double *sp = reader->GetOutput()->GetSpacing();
  vtkSmartPointer<vtkMatrix4x4> mf = vtkMatrix4x4::New();
  mf->DeepCopy(reader->GetMeasurementFrameMatrix());
  vtkSmartPointer<vtkMatrix4x4> rasToIjkRotation = vtkMatrix4x4::New();
  rasToIjkRotation->DeepCopy(reader->GetRasToIjkMatrix());

  //Set Translation to zero
  for (int i=0;i<3;i++)
    {
    rasToIjkRotation->SetElement(i,3,0);
    }
  //Remove scaling in rasToIjk to make a real roation matrix
  double col[3];
  for (int jjj = 0; jjj < 3; jjj++) 
    {
    for (int iii = 0; iii < 3; iii++)
      {
      col[iii]=rasToIjkRotation->GetElement(iii,jjj);
      }
    vtkMath::Normalize(col);
    for (int iii = 0; iii < 3; iii++)
      {
      rasToIjkRotation->SetElement(iii,jjj,col[iii]);
     }  
  }

  vtkSmartPointer<vtkTransform> trans = vtkTransform::New();
  trans->PostMultiply();
  trans->SetMatrix(mf);
  trans->Concatenate(rasToIjkRotation);
  trans->Update();

  estim->SetTransform(trans);
  if (estimationMethod == std::string("Least Squares"))
    {
    estim->SetEstimationMethodToLLS();
    }
  else if(estimationMethod == std::string("Weighted Least Squares"))
    {
    estim->SetEstimationMethodToWLS();
    }
  else if (estimationMethod == std::string("Non Linear"))
    {
    estim->SetEstimationMethodToNLS();
    }

  estim->Update();
  vtkImageData *tensorImage = estim->GetOutput();
  tensorImage->GetPointData()->SetScalars(NULL);

  //Save tensor
  vtkSmartPointer<vtkNRRDWriter> writer = vtkNRRDWriter::New();
  writer->SetInput(tensorImage);
  writer->SetFileName( outputTensor.c_str() );
  writer->UseCompressionOn();
  //Compute IjkToRas (used by Writer)
  reader->GetRasToIjkMatrix()->Invert();
  writer->SetIJKToRASMatrix( reader->GetRasToIjkMatrix() );
  //Compute measurement frame: Take into account that we have transformed
  //the gradients so tensor components are defined in ijk.
  rasToIjkRotation->Invert();
  writer->SetMeasurementFrameMatrix( rasToIjkRotation );
  writer->Write();

  //Save baseline
  vtkSmartPointer<vtkNRRDWriter> writer2 = vtkNRRDWriter::New();
  writer2->SetInput(estim->GetBaseline());
  writer2->SetFileName( outputBaseline.c_str() );
  writer2->UseCompressionOn();
  writer2->SetIJKToRASMatrix( reader->GetRasToIjkMatrix() );
  writer2->Write();
  }
  return EXIT_SUCCESS;
}
