/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLColorNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.0 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLColorNode.h"
#include "vtkMRMLScene.h"

//------------------------------------------------------------------------------
vtkMRMLColorNode* vtkMRMLColorNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLColorNode");
  if(ret)
    {
    return (vtkMRMLColorNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLColorNode;
}

//-----------------------------------------------------------------------------
vtkMRMLNode* vtkMRMLColorNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLColorNode");
  if(ret)
    {
    return (vtkMRMLColorNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLColorNode;
}


//----------------------------------------------------------------------------
vtkMRMLColorNode::vtkMRMLColorNode()
{
  this->Name = NULL;
  this->SetName("");
  this->FileName = NULL;
  this->Type = -1;
}

//----------------------------------------------------------------------------
vtkMRMLColorNode::~vtkMRMLColorNode()
{
  if (this->FileName)
    {  
    delete [] this->FileName;
    this->FileName = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkMRMLColorNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);
  
  vtkIndent indent(nIndent);
  
  of << " type=\"" << this->GetType() << "\"";

  if (this->FileName != NULL)
    {
    of << " filename=\"" << this->FileName << "\"";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLColorNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "name"))
      {
      this->SetName(attValue);
      }
    else if (!strcmp(attName, "type")) 
      {
      int type;
      std::stringstream ss;
      ss << attValue;
      ss >> type;
      this->SetType(type);
      }
    else if (!strcmp(attName, "filename"))
      {
      this->SetFileName(attValue);
      // read in the file with the colours
      std::cout << "Reading file " << this->FileName << endl;
      this->ReadFile();
      }
    }
  vtkDebugMacro("Finished reading in xml attributes, list id = " << this->GetID() << " and name = " << this->GetName() << endl);
}

//----------------------------------------------------------------------------
vtkLookupTable * vtkMRMLColorNode::GetLookupTable()
{
  vtkWarningMacro("Subclass has not implemented GetLookupTable, returning NULL");
  return NULL;
}

//----------------------------------------------------------------------------
void vtkMRMLColorNode::ReadFile ()
{
  vtkErrorMacro("Subclass has not implemented ReadFile.");
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLColorNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  vtkMRMLColorNode *node = (vtkMRMLColorNode *) anode;

  if (node->Type != -1)
    {
    this->SetType(node->Type);
    }
  this->SetFileName(node->FileName);
}

//----------------------------------------------------------------------------
void vtkMRMLColorNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);

  os << indent << "Name: " <<
      (this->Name ? this->Name : "(none)") << "\n";
  

  os << indent << "Type: (" << this->GetTypeAsString() << ")\n";

  if (this->Names.size() > 0)
    {
    os << indent << "Color Names:\n";
    for (unsigned int i = 0; (int)i < this->Names.size(); i++)
      {
      os << indent << indent << i << " " << this->GetColorName(i) << endl;
      }
    }
}

//-----------------------------------------------------------

void vtkMRMLColorNode::UpdateScene(vtkMRMLScene *scene)
{
    Superclass::UpdateScene(scene);
    /*
    if (this->GetStorageNodeID() == NULL) 
    {
        //vtkErrorMacro("No reference StorageNodeID found");
        return;
    }

    vtkMRMLNode* mnode = scene->GetNodeByID(this->StorageNodeID);
    if (mnode) 
    {
        vtkMRMLStorageNode *node  = dynamic_cast < vtkMRMLStorageNode *>(mnode);
        node->ReadData(this);
        //this->SetAndObservePolyData(this->GetPolyData());
    }
    */
}


//---------------------------------------------------------------------------
void vtkMRMLColorNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
/*
  vtkMRMLColorDisplayNode *dnode = this->GetDisplayNode();
  if (dnode != NULL && dnode == vtkMRMLColorDisplayNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
        this->InvokeEvent(vtkMRMLColorNode::DisplayModifiedEvent, NULL);
    }
*/
  return;
}

//---------------------------------------------------------------------------
int vtkMRMLColorNode::GetFirstType()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return -1;
}

//---------------------------------------------------------------------------
int vtkMRMLColorNode::GetLastType()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return -1;
}

//---------------------------------------------------------------------------
const char * vtkMRMLColorNode::GetTypeAsString()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return "(unknown)";
}

//---------------------------------------------------------------------------
void vtkMRMLColorNode::SetType(int type)
{
  if (this->Type == type)
    {
    vtkDebugMacro("SetType: type is already set to " << type);
    return;
    }
    
  this->Type = type;

  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Type to " << type);

  // subclass should over ride this and define colours according to the node
  // type
  
  // invoke a modified event
  this->Modified();
    
  // invoke a type  modified event
  this->InvokeEvent(vtkMRMLColorNode::TypeModifiedEvent);
}

//---------------------------------------------------------------------------
void vtkMRMLColorNode::SetNamesFromColors()
{
  vtkErrorMacro("Subclass has not defined this method.");
}

//---------------------------------------------------------------------------
const char *vtkMRMLColorNode::GetColorName(int ind)
{
    if (ind < (int)this->Names.size() && ind >= 0)
    {
    if (strcmp(this->Names[ind].c_str(), "") == 0)
      {
      return "(none)";
      }
    else
      {
      return this->Names[ind].c_str();
      }
    }
  else
    {
    vtkDebugMacro("vtkMRMLColorNode::GetColorName: index " << ind << " is out of range 0 - " << this->Names.size());
    return "invalid";
    }
}

//---------------------------------------------------------------------------
const char *vtkMRMLColorNode::GetColorNameWithoutSpaces(int ind, const char *subst)
{
  std::string name = std::string(this->GetColorName(ind));
  if (strstr(name.c_str(), " ") != NULL)
    {
    std::string::size_type spaceIndex = name.find( " ", 0 );
    while (spaceIndex != std::string::npos)
      {
      name.replace(spaceIndex, 1, subst, 0, strlen(subst));
      spaceIndex = name.find( " ", spaceIndex );
      }
    return name.c_str();
    }
  else
    {
    // no spaces, return it as is
    return name.c_str();
    }
}

//---------------------------------------------------------------------------
void vtkMRMLColorNode::SetColorName(int ind, const char *name)
{
    if (ind < (int)this->Names.size() && ind >= 0)
    {
    this->Names[ind] = std::string(name);
    }
  else
    {
    vtkErrorMacro("ERROR: SetColorName, index was out of bounds: " << ind << ", current size is " << this->Names.size() << endl);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLColorNode::SetColorNameWithSpaces(int ind, const char *name, const char *subst)
{
 
  std::string nameString = std::string(name);
  std::string substString = std::string(subst);
   // does the input name have the subst character in it?
  if (strstr(name, substString.c_str()) != NULL)
    {
    this->SetColorName(ind, nameString.c_str());
    }
  else
    {
    // no substitutions necessary
    this->SetColorName(ind, name);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLColorNode::AddColorName(const char *name)
{
  this->Names.push_back(std::string(name));
}

