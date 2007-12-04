#include "vtkIGTMessageImageDataAttribute.h"
#include "vtkIGTMessageGenericAttribute.h"


template <typename T>
void vtkIGTMessageAttributeSet::AddAttribute(const char* key, T* dummy)
{
  vtkIGTMessageAttributeBase* attr;
  
  if (typeid(T) == typeid(bool))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<bool>::New();
    attr->SetTypeID(TYPE_BOOL);
    }
  else if (typeid(T) == typeid(char))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<char>::New();
    attr->SetTypeID(TYPE_CHAR);
    }
  else if (typeid(T) == typeid(signed char))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<signed char>::New();
    attr->SetTypeID(TYPE_SIGNED_CHAR);
    }
  else if (typeid(T) == typeid(unsigned char))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned char>::New();
    attr->SetTypeID(TYPE_UNSIGNED_CHAR);
    }
  else if (typeid(T) == typeid(int))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<int>::New();
    attr->SetTypeID(TYPE_INT);
    }
  else if (typeid(T) == typeid(long))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<long>::New();
    attr->SetTypeID(TYPE_LONG);
    }
  else if (typeid(T) == typeid(short))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<short>::New();
    attr->SetTypeID(TYPE_SHORT);
    }
  else if (typeid(T) == typeid(unsigned int))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned int>::New();
    attr->SetTypeID(TYPE_UNSIGNED_INT);
    }
  else if (typeid(T) == typeid(unsigned long))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned long>::New();
    attr->SetTypeID(TYPE_UNSIGNED_LONG);
    }
  else if (typeid(T) == typeid(unsigned short))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<unsigned short>::New();
    attr->SetTypeID(TYPE_UNSIGNED_SHORT);
    }
  else if (typeid(T) == typeid(double))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<double>::New();
    attr->SetTypeID(TYPE_DOUBLE);
    }
  else if (typeid(T) == typeid(long double))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<long double>::New();
    attr->SetTypeID(TYPE_LONG_DOUBLE);
    }
  else if (typeid(T) == typeid(float))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<float>::New();
    attr->SetTypeID(TYPE_FLOAT);
    }
  else if (typeid(T) == typeid(std::string))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<std::string>::New();
    attr->SetTypeID(TYPE_STRING);
    }
  else if (typeid(T) == typeid(std::vector<float>))
    {
    attr = (vtkIGTMessageAttributeBase*)vtkIGTMessageGenericAttribute<std::vector<float> >::New();
    attr->SetTypeID(TYPE_VECTOR_FLOAT);
    }
  else if (typeid(T) == typeid(vtkImageData))
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


