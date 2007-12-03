/*=auto=========================================================================

Portions (c) Copyright 2007 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: $
Date:      $Date: $
Version:   $Revision: $

=========================================================================auto=*/

#include "vtkObjectFactory.h"

#include "vtkIGTMessageAttributeSet.h"
#include "vtkIGTMessageAttributeBase.h"
#include "vtkIGTMessageGenericAttribute.h"
#include "vtkIGTMessageImageDataAttribute.h"

vtkStandardNewMacro(vtkIGTMessageAttributeSet);
vtkCxxRevisionMacro(vtkIGTMessageAttributeSet, "$Revision: 1.0 $");


vtkIGTMessageAttributeSet::vtkIGTMessageAttributeSet()
{
  this->AttributeMap.clear();
  this->OpenTrackerStream = NULL;
}


vtkIGTMessageAttributeSet::~vtkIGTMessageAttributeSet()
{
}


void vtkIGTMessageAttributeSet::PrintSelf(ostream& os, vtkIndent indent)
{
}


template <typename T>
void vtkIGTMessageAttributeSet::AddAttribute(const char* key, const char* type_name)
{
  vtkIGTMessageAttributeBase* attr;

  if (strcmp(type_name, "bool") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<bool>::New();
    attr->SetTypeID(TYPE_BOOL);
    }
  else if (strcmp(type_name, "char") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<char>::New();
    attr->SetTypeID(TYPE_CHAR);
    }
  else if (strcmp(type_name, "signed_char") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<signed char>::New();
    attr->SetTypeID(TYPE_SIGNED_CHAR);
    }
  else if (strcmp(type_name, "unsigned_char") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned char>::New();
    attr->SetTypeID(TYPE_UNSIGNED_CHAR);
    }
  else if (strcmp(type_name, "int") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<int>::New();
    attr->SetTypeID(TYPE_INT);
    }
  else if (strcmp(type_name, "long") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<long>::New();
    attr->SetTypeID(TYPE_LONG);
    }
  else if (strcmp(type_name, "short") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<short>::New();
    attr->SetTypeID(TYPE_SHORT);
    }
  else if (strcmp(type_name, "unsigned_int") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned int>::New();
    attr->SetTypeID(TYPE_UNSIGNED_INT);
    }
  else if (strcmp(type_name, "unsigned_long") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned long>::New();
    attr->SetTypeID(TYPE_UNSIGNED_LONG);
    }
  else if (strcmp(type_name, "unsigned_short") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned short>::New();
    attr->SetTypeID(TYPE_UNSIGNED_SHORT);
    }
  else if (strcmp(type_name, "double") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<double>::New();
    attr->SetTypeID(TYPE_DOUBLE);
    }
  else if (strcmp(type_name, "long_double") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<long double>::New();
    attr->SetTypeID(TYPE_LONG_DOUBLE);
    }
  else if (strcmp(type_name, "float") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<float>::New();
    attr->SetTypeID(TYPE_FLOAT);
    }
  else if (strcmp(type_name, "string") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<std::string>::New();
    attr->SetTypeID(TYPE_STRING);
    }
  else if (strcmp(type_name, "vector<float>") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<std::vector<float> >::New();
    attr->SetTypeID(TYPE_VECTOR_FLOAT);
    }
  else if (strcmp(type_name, "MedScanImage") == 0)
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageImageDataAttribute::New();
    attr->SetTypeID(TYPE_VTK_IMAGE_DATA);
    }

  this->AttributeMap[key] = attr;
}


template <typename T>
int vtkIGTMessageAttributeSet::GetAttribute(const char* key, T* ptr)
{
  AttributeMapType::iterator iter = this->AttributeMap.find(key);
  if (iter != this->AttributeMap.end())
    {
    vtkIGTMessageAttributeBase* attr = iter->second;
    return attr->GetAttribute(ptr);
    }

  return 0;

}


void vtkIGTMessageAttributeSet::SetHandlerFunction(MessageHandlingFunction* func)
{
  this->HandlerFunction = func;
}

vtkIGTMessageAttributeSet::MessageHandlingFunction* vtkIGTMessageAttributeSet::GetHandlerFunction()
{
  return this->HandlerFunction;
}

