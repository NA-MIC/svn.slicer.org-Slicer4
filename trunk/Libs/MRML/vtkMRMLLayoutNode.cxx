#include <string>
#include <iostream>
#include <sstream>

#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkMRMLLayoutNode.h"
#include "vtkMRMLScene.h"


//------------------------------------------------------------------------------
vtkMRMLLayoutNode* vtkMRMLLayoutNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLLayoutNode");
  if(ret)
    {
    return (vtkMRMLLayoutNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLLayoutNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLLayoutNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLLayoutNode");
  if(ret)
    {
    return (vtkMRMLLayoutNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLLayoutNode;
}

//----------------------------------------------------------------------------
vtkMRMLLayoutNode::vtkMRMLLayoutNode()
{

  this->SingletonTag = "vtkMRMLLayoutNode";
  this->HideFromEditors = 1;
  this->GUIPanelVisibility = 1;
  this->BottomPanelVisibility = 1;
  this->GUIPanelLR = 0;
  this->ViewArrangement = -1;
  this->NumberOfCompareViewRows = 0;
  this->NumberOfCompareViewColumns = 0;
  this->NumberOfCompareViewLightboxRows = 1;
  this->NumberOfCompareViewLightboxColumns = 1;
  this->MainPanelSize = 400;
  this->SecondaryPanelSize = 400;
  return;

}

//----------------------------------------------------------------------------
vtkMRMLLayoutNode::~vtkMRMLLayoutNode()
{
}


//----------------------------------------------------------------------------
void vtkMRMLLayoutNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes, since the parsing of the string is dependent on the
  // order here. 
  
  Superclass::WriteXML(of, nIndent);
  vtkIndent indent(nIndent);

  of << indent << " currentViewArrangement=\"" << this->ViewArrangement << "\"";
  of << indent << " guiPanelVisibility=\"" << this->GUIPanelVisibility << "\"";
  of << indent << " bottomPanelVisiblity =\"" << this->BottomPanelVisibility << "\"";
  of << indent << " guiPanelLR=\"" << this->GUIPanelLR << "\"";
  of << indent << " numberOfCompareViewRows=\"" << this->NumberOfCompareViewRows << "\"";
  of << indent << " numberOfCompareViewColumns=\"" << this->NumberOfCompareViewColumns << "\"";
  of << indent << " numberOfLightboxRows=\"" << this->NumberOfCompareViewLightboxRows << "\"";
  of << indent << " numberOfLightboxColumns=\"" << this->NumberOfCompareViewLightboxColumns << "\"";
  of << indent << " mainPanelSize=\"" << this->MainPanelSize << "\"";
  of << indent << " secondaryPanelSize=\"" << this->SecondaryPanelSize << "\"";
}


//----------------------------------------------------------------------------
void vtkMRMLLayoutNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);
  
  const char* attName;
  const char* attValue;
  
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "currentViewArrangement")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->ViewArrangement;
      }
    else if (!strcmp (attName, "guiPanelVisibility"))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->GUIPanelVisibility;
      }
    else if ( !strcmp ( attName, "bottomPanelVisibility" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->BottomPanelVisibility;
      }
    else if ( !strcmp (attName, "guiPanelLR" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->GUIPanelLR;
      }
    else if ( !strcmp (attName, "numberOfCompareViewRows" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->NumberOfCompareViewRows;
      }
    else if ( !strcmp (attName, "numberOfCompareViewColumns" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->NumberOfCompareViewColumns;
      }
    else if ( !strcmp (attName, "numberOfLightboxRows" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->NumberOfCompareViewLightboxRows;
      }
    else if ( !strcmp (attName, "numberOfLightboxColumns" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->NumberOfCompareViewLightboxColumns;
      }
    else if ( !strcmp (attName, "mainPanelSize" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->MainPanelSize;
      }
    else if ( !strcmp (attName, "secondaryPanelSize" ))
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->SecondaryPanelSize;
      }
    }

}
    


//----------------------------------------------------------------------------
void vtkMRMLLayoutNode::SetViewArrangement ( int arrNew )
{
  this->ViewArrangement = arrNew;
  this->Modified();
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, LabelText, ID
void vtkMRMLLayoutNode::Copy(vtkMRMLNode *anode)
{
//  vtkObject::Copy(anode);
  vtkMRMLLayoutNode *node = (vtkMRMLLayoutNode *) anode;
  this->SetViewArrangement (node->GetViewArrangement() );
  this->SetGUIPanelVisibility(node->GetGUIPanelVisibility()) ;
  this->SetBottomPanelVisibility (node->GetBottomPanelVisibility());
  this->SetGUIPanelLR ( node->GetGUIPanelLR());
  this->SetNumberOfCompareViewRows ( node->GetNumberOfCompareViewRows() );
  this->SetNumberOfCompareViewColumns ( node->GetNumberOfCompareViewColumns() );
  this->SetNumberOfCompareViewLightboxRows ( node->GetNumberOfCompareViewLightboxRows() );
  this->SetNumberOfCompareViewLightboxColumns ( node->GetNumberOfCompareViewLightboxColumns() );

  this->SetMainPanelSize( node->GetMainPanelSize() );
  this->SetSecondaryPanelSize( node->GetSecondaryPanelSize() );
  
}

//----------------------------------------------------------------------------
void vtkMRMLLayoutNode::PrintSelf(ostream& os, vtkIndent indent)
{  

  Superclass::PrintSelf(os,indent);
  
  // Layout:
  os << indent << "ViewArrangement: " << this->ViewArrangement  << "\n";
  os << indent << "GUIPanelVisibility: " << this->GUIPanelVisibility  << "\n";
  os << indent << "GUIPanelLR: " << this->GUIPanelLR  << "\n";
  os << indent << "BottomPanelVisibility: " << this->BottomPanelVisibility  << "\n";
  os << indent << "NumberOfCompareViewRows: " << this->NumberOfCompareViewRows << "\n";
  os << indent << "NumberOfCompareViewColumns: " << this->NumberOfCompareViewColumns << "\n";
  os << indent << "NumberOfCompareViewLightboxRows: " << this->NumberOfCompareViewLightboxRows << "\n";
  os << indent << "NumberOfCompareViewLightboxColumns: " << this->NumberOfCompareViewLightboxColumns << "\n";
  os << indent << "Main panel size: " << this->MainPanelSize << "\n";
  os << indent << "Secondary panel size: " << this->SecondaryPanelSize << "\n";
}


