/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH)
  All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer

=========================================================================auto=*/

// MRML includes
#include "vtkMRMLCoreTestingMacros.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLStorageNode.h"

// VTK includes
#include <vtkNew.h>
#include <vtkObjectFactory.h>

//---------------------------------------------------------------------------
class vtkMRMLStorageNodeTestHelper1 : public vtkMRMLStorageNode
{
public:
  // Provide a concrete New.
  static vtkMRMLStorageNodeTestHelper1 *New();

  vtkTypeMacro(vtkMRMLStorageNodeTestHelper1,vtkMRMLStorageNode);

  virtual vtkMRMLNode* CreateNodeInstance()
    {
    return vtkMRMLStorageNodeTestHelper1::New();
    }
  virtual const char* GetNodeTagName()
    {
    return "vtkMRMLStorageNodeTestHelper1";
    }

  virtual bool CanApplyNonLinearTransforms() { return false; }
  virtual void ApplyTransform(vtkAbstractTransform* vtkNotUsed(transform)) { return; }

  bool CanReadInReferenceNode(vtkMRMLNode * refNode)
    {
    return refNode->IsA(this->SupportedClass);
    }
  int ReadDataInternal(vtkMRMLNode * vtkNotUsed(refNode))
    {
    return this->ReadDataReturnValue;
    }

  const char* SupportedClass;
  int ReadDataReturnValue;
protected:
  vtkMRMLStorageNodeTestHelper1()
    :SupportedClass(0)
    ,ReadDataReturnValue(0)
  {}
};
vtkStandardNewMacro(vtkMRMLStorageNodeTestHelper1);

//---------------------------------------------------------------------------
int TestBasics();
int TestReadData();
int TestWriteData();

//---------------------------------------------------------------------------
int vtkMRMLStorageNodeTest1(int , char * [] )
{
  CHECK_EXIT_SUCCESS(TestBasics());
  CHECK_EXIT_SUCCESS(TestReadData());
  CHECK_EXIT_SUCCESS(TestWriteData());
  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
int TestBasics()
{
  vtkNew< vtkMRMLStorageNodeTestHelper1 > node1;
  EXERCISE_ALL_BASIC_MRML_METHODS(node1.GetPointer());
  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
int TestReadData(int referenceNodeType,
                  const char* supportedClass,
                  int readDataReturn,
                  int expectedRes)
{
  vtkNew<vtkMRMLStorageNodeTestHelper1> storageNode;
  storageNode->SupportedClass = supportedClass;
  storageNode->ReadDataReturnValue = readDataReturn;
  storageNode->SetFileName("file.ext");
  vtkNew<vtkMRMLLinearTransformNode> transformNode;
  vtkNew<vtkMRMLModelNode> modelNode;
  vtkMRMLNode* referenceNode = (referenceNodeType == 0 ? vtkMRMLNode::SafeDownCast(0):
                               (referenceNodeType == 1 ? vtkMRMLNode::SafeDownCast(transformNode.GetPointer()) :
                                  vtkMRMLNode::SafeDownCast(modelNode.GetPointer())));
  int res = storageNode->ReadData(referenceNode);
  std::cout << "StoredTime: " << storageNode->GetStoredTime() << std::endl;
  CHECK_INT(res, expectedRes);

  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
int TestReadData()
{
  TESTING_OUTPUT_ASSERT_ERRORS_BEGIN();
  CHECK_EXIT_SUCCESS(TestReadData(0, "invalid", 0, 0));
  TESTING_OUTPUT_ASSERT_ERRORS_END();

  TESTING_OUTPUT_ASSERT_ERRORS_BEGIN();
  CHECK_EXIT_SUCCESS(TestReadData(0, "invalid", 1, 0));
  TESTING_OUTPUT_ASSERT_ERRORS_END();

  TESTING_OUTPUT_ASSERT_ERRORS_BEGIN();
  CHECK_EXIT_SUCCESS(TestReadData(0, "vtkMRMLModelNode", 0, 0));
  TESTING_OUTPUT_ASSERT_ERRORS_END();

  TESTING_OUTPUT_ASSERT_ERRORS_BEGIN();
  CHECK_EXIT_SUCCESS(TestReadData(0, "vtkMRMLModelNode", 1, 0));
  TESTING_OUTPUT_ASSERT_ERRORS_END();

  CHECK_EXIT_SUCCESS(TestReadData(1, "invalid", 0, 0));
  CHECK_EXIT_SUCCESS(TestReadData(1, "invalid", 1, 0));
  CHECK_EXIT_SUCCESS(TestReadData(1, "vtkMRMLModelNode", 0, 0));
  CHECK_EXIT_SUCCESS(TestReadData(1, "vtkMRMLModelNode", 1, 0));
  CHECK_EXIT_SUCCESS(TestReadData(2, "invalid", 0, 0));
  CHECK_EXIT_SUCCESS(TestReadData(2, "invalid", 1, 0));
  CHECK_EXIT_SUCCESS(TestReadData(2, "vtkMRMLModelNode", 0, 0));
  CHECK_EXIT_SUCCESS(TestReadData(2, "vtkMRMLModelNode", 1, 1));

  return EXIT_SUCCESS;
}

//---------------------------------------------------------------------------
int TestWriteData()
{
  // TODO
  return EXIT_SUCCESS;
}
