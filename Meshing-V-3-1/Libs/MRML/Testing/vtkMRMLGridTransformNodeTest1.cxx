/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
  All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer

=========================================================================auto=*/

#include "vtkMRMLGridTransformNode.h"

#include <stdlib.h>
#include <iostream>

#include "TestingMacros.h"

int vtkMRMLGridTransformNodeTest1(int , char * [] )
{
  vtkSmartPointer< vtkMRMLGridTransformNode > node1 = vtkSmartPointer< vtkMRMLGridTransformNode >::New();

  EXERCISE_BASIC_OBJECT_METHODS( node1 );

  EXERCISE_BASIC_TRANSFORM_MRML_METHODS(vtkMRMLGridTransformNode, node1);

  return EXIT_SUCCESS;
}
