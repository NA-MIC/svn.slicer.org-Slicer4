/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH)
  All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer

=========================================================================auto=*/

// MRML includes
#include "vtkMRMLCoreTestingMacros.h"
#include "vtkMRMLModelNode.h"

// VTK includes
#include <vtkDataSetAttributes.h>
#include <vtkPolyData.h>

//---------------------------------------------------------------------------
int ExerciseBasicMethods();
bool TestActiveScalars();

//---------------------------------------------------------------------------
int vtkMRMLModelNodeTest1(int , char * [] )
{
  /*
  if (ExerciseBasicMethods() != EXIT_SUCCESS)
    {
    std::cerr << __LINE__ << ": ExerciseBasicMethods() failed" << std::endl;
    return EXIT_FAILURE;
    }
  */
  if (!TestActiveScalars())
    {
    std::cerr << __LINE__ << ": TestActiveScalars() failed" << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
int ExerciseBasicMethods()
{
  vtkSmartPointer< vtkMRMLModelNode > node1 = vtkSmartPointer< vtkMRMLModelNode >::New();
  EXERCISE_BASIC_OBJECT_METHODS( node1 );
  EXERCISE_BASIC_DISPLAYABLE_MRML_METHODS(vtkMRMLModelNode, node1);
  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
bool TestActiveScalars()
{
  vtkSmartPointer< vtkMRMLModelNode > node1 = vtkSmartPointer< vtkMRMLModelNode >::New();

  vtkSmartPointer<vtkIntArray> testingArray = vtkSmartPointer <vtkIntArray>::New();
  testingArray->SetName("testingArray");
  node1->AddPointScalars(testingArray);

  vtkSmartPointer<vtkIntArray> testingArray2 = vtkSmartPointer <vtkIntArray>::New();
  testingArray2->SetName("testingArray2");
  node1->AddCellScalars(testingArray2);

  int attribute = vtkDataSetAttributes::SCALARS;
  node1->SetActivePointScalars("testingArray", attribute);
  node1->SetActiveCellScalars("testingArray2", attribute);

  const char *name = node1->GetActivePointScalarName(vtkDataSetAttributes::SCALARS);
  std::cout << "Active point scalars name = " << (name  == NULL ? "null" : name) << std::endl;
  name = node1->GetActiveCellScalarName(vtkDataSetAttributes::SCALARS);
  std::cout << "Active cell scalars name = " << (name == NULL ? "null" : name) << std::endl;
  node1->RemoveScalars("testingArray");

  return true;
}
