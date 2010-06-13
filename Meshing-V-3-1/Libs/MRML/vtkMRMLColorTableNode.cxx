/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLColorTableNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.0 $

=========================================================================auto=*/
#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkMRMLColorTableNode.h"
#include "vtkMRMLScene.h"

#include "vtkLookupTable.h"
//#include "vtkFSLookupTable.h"
#include "vtkMRMLStorageNode.h"
#include "vtkMRMLColorTableStorageNode.h"
//------------------------------------------------------------------------------
vtkMRMLColorTableNode* vtkMRMLColorTableNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLColorTableNode");
  if(ret)
    {
    return (vtkMRMLColorTableNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLColorTableNode;
}

//-----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLColorTableNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLColorTableNode");
  if(ret)
    {
    return (vtkMRMLColorTableNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLColorTableNode;
}


//----------------------------------------------------------------------------
vtkMRMLColorTableNode::vtkMRMLColorTableNode()
{

  this->SetName("");
  this->SetDescription("Color Table");
  this->LookupTable = NULL;
  this->LastAddedColor = -1;
}

//----------------------------------------------------------------------------
vtkMRMLColorTableNode::~vtkMRMLColorTableNode()
{
  if (this->LookupTable)
    {
    this->LookupTable->Delete();
    }
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their FullRainbows
  
  Superclass::WriteXML(of, nIndent);
  
  vtkIndent indent(nIndent);
  
  // only print out the look up table if ?
  if (this->LookupTable != NULL) // && this->Type != this->File
    {
    of << " numcolors=\"" << this->LookupTable->GetNumberOfTableValues() << "\"";
    of << " colors=\"";
    for (int i = 0; i < this->LookupTable->GetNumberOfTableValues(); i++)
      {
      double *rgba;
      rgba = this->LookupTable->GetTableValue(i);
      of <<  i << " '" << this->GetColorNameWithoutSpaces(i, "_") << "' " << rgba[0] << " " << rgba[1] << " " << rgba[2] << " " << rgba[3] << " ";
      }
    of << "\"";
    } 
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  int numColours;
  while (*atts != NULL) 
  {
      attName = *(atts++);
      attValue = *(atts++);
      if (!strcmp(attName, "name"))
      {
          this->SetName(attValue);
      }
      else if (!strcmp(attName, "id"))
      {
          // handled at the vtkMRMLNode level
      }
      else if (!strcmp(attName, "numcolors"))
        {
        std::stringstream ss;
        ss << attValue;
        ss >> numColours;
        vtkDebugMacro("Setting the look up table size to " << numColours << "\n");
        this->LookupTable->SetNumberOfTableValues(numColours);
        this->Names.clear();
        this->Names.resize(numColours);
        }
      else  if (!strcmp(attName, "colors")) 
      {
      std::stringstream ss;
      for (int i = 0; i < this->LookupTable->GetNumberOfTableValues(); i++)
        {
        vtkDebugMacro("Reading colour " << i << " of " << this->LookupTable->GetNumberOfTableValues() << endl);
        ss << attValue;
        // index name r g b a
        int index;
        std::string name;
        double r, g, b, a;
        ss >> index;
        ss >> name;          
        ss >> r;
        ss >> g;
        ss >> b;
        ss >> a;
        vtkDebugMacro("Adding colour at index " << index << ", r = " << r << ", g = " << g << ", b = " << b << ", a = " << a << " and then setting name to " << name.c_str() << endl);
        
        if (this->SetColorNameWithSpaces(index, name.c_str(), "_") != 0)
          {
          this->LookupTable->SetTableValue(index, r, g, b, a);
          }
        }
      // set the table range
      if ( this->LookupTable->GetNumberOfTableValues() > 0 )
        {
        this->LookupTable->SetRange(0,  this->LookupTable->GetNumberOfTableValues() - 1);
        }
      this->NamesInitialisedOn();
      }
      else if (!strcmp(attName, "type")) 
      {
      int type;
      std::stringstream ss;
      ss << attValue;
      ss >> type;
      this->SetType(type);
      }      
      else
      {
          vtkDebugMacro ("Unknown attribute name " << attName << endl);
      }
  }
  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLColorTableNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLColorTableNode *node = (vtkMRMLColorTableNode *) anode;
  if (node->LookupTable)
    {
    this->SetLookupTable(node->LookupTable);
    }
  this->EndModify(disabledModify);

}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);

  os << indent << "Name: " <<
      (this->Name ? this->Name : "(none)") << "\n";
  

  os << indent << "Type: (" << this->GetTypeAsString() << ")\n";

  if (this->LookupTable != NULL)
    {
    os << indent << "Look up table:\n";
    this->LookupTable->PrintSelf(os, indent.GetNextIndent());
    }
/*  if (this->FSLookupTable != NULL)
    {
    os << indent << "FreeSurfer look up table:\n";
    this->FSLookupTable->PrintSelf(os, indent.GetNextIndent());
    }
*/
  if (this->Names.size() > 0)
    {
    os << indent << "Color Names:\n";
    for (unsigned int i = 0; i < this->Names.size(); i++)
      {
      os << indent << indent << i << " " << this->GetColorName(i) << endl;
      if ( i > 10 )
        {
        os << indent << indent << "..." << endl;
        break;
        }
      }
    }
}

//-----------------------------------------------------------

void vtkMRMLColorTableNode::UpdateScene(vtkMRMLScene *scene)
{
  // UpdateScene on the displayable node superclass will set up the storage
  // node and call ReadData
  Superclass::UpdateScene(scene); 
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToFullRainbow()
{
    this->SetType(this->FullRainbow);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToGrey()
{
    this->SetType(this->Grey);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToIron()
{
    this->SetType(this->Iron);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToRainbow()
{
    this->SetType(this->Rainbow);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToOcean()
{
    this->SetType(this->Ocean);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToDesert()
{
    this->SetType(this->Desert);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToInvGrey()
{
    this->SetType(this->InvGrey);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToReverseRainbow()
{
    this->SetType(this->ReverseRainbow);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToFMRI()
{
    this->SetType(this->FMRI);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToFMRIPA()
{
    this->SetType(this->FMRIPA);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToLabels()
{
    this->SetType(this->Labels);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToRandom()
{
  
  this->SetType(this->Random);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToUser()
{
  this->SetType(this->User);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToFile()
{
    this->SetType(this->File);
}


//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToRed()
{
    this->SetType(this->Red);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToGreen()
{
    this->SetType(this->Green);
}
//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToBlue()
{
    this->SetType(this->Blue);
}
//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCyan()
{
    this->SetType(this->Cyan);
}
//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToMagenta()
{
    this->SetType(this->Magenta);
}
//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToYellow()
{
    this->SetType(this->Yellow);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarm1()
{
    this->SetType(this->Warm1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarm2()
{
    this->SetType(this->Warm2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarm3()
{
    this->SetType(this->Warm3);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCool1()
{
    this->SetType(this->Cool1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCool2()
{
    this->SetType(this->Cool2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCool3()
{
    this->SetType(this->Cool3);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmShade1()
{
    this->SetType(this->WarmShade1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmShade2()
{
    this->SetType(this->WarmShade2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmShade3()
{
    this->SetType(this->WarmShade3);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolShade1()
{
    this->SetType(this->CoolShade1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolShade2()
{
    this->SetType(this->CoolShade2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolShade3()
{
    this->SetType(this->CoolShade3);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmTint1()
{
    this->SetType(this->WarmTint1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmTint2()
{
    this->SetType(this->WarmTint2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToWarmTint3()
{
    this->SetType(this->WarmTint3);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolTint1()
{
    this->SetType(this->CoolTint1);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolTint2()
{
    this->SetType(this->CoolTint2);
}

//----------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetTypeToCoolTint3()
{
    this->SetType(this->CoolTint3);
}


//----------------------------------------------------------------------------
const char* vtkMRMLColorTableNode::GetTypeAsIDString()
{
  if (this->Type == this->FullRainbow)
    {
    return "vtkMRMLColorTableNodeFullRainbow";
    }
  if (this->Type == this->Grey)
    {
    return "vtkMRMLColorTableNodeGrey";
    }
  if (this->Type == this->Iron)
    {
    return "vtkMRMLColorTableNodeIron";
    }
  if (this->Type == this->Rainbow)
    {
    return "vtkMRMLColorTableNodeRainbow";
    }
  if (this->Type == this->Ocean)
    {
    return "vtkMRMLColorTableNodeOcean";
    }
  if (this->Type == this->Desert)
    {
    return "vtkMRMLColorTableNodeDesert";
    }
  if (this->Type == this->InvGrey)
    {
    return "vtkMRMLColorTableNodeInvertedGrey";
    }
  if (this->Type == this->ReverseRainbow)
    {
    return "vtkMRMLColorTableNodeReverseRainbow";
    }
  if (this->Type == this->FMRI)
    {
    return "vtkMRMLColorTableNodefMRI";
    }
  if (this->Type == this->FMRIPA)
    {
    return "vtkMRMLColorTableNodefMRIPA";
    }
  if (this->Type == this->Labels)
    {
    return "vtkMRMLColorTableNodeLabels";
    }
  if (this->Type == this->Random)
    {
    return "vtkMRMLColorTableNodeRandom";
    }
  if (this->Type == this->User)
    {
      return "vtkMRMLColorTableNodeUser";
    }
  if (this->Type == this->File)
    {
    return "vtkMRMLColorTableNodeFile";
    }
  if (this->Type == this->Red)
    {
    return "vtkMRMLColorTableNodeRed";
    }
  if (this->Type == this->Green)
    {
    return "vtkMRMLColorTableNodeGreen";
    }
  if (this->Type == this->Blue)
    {
    return "vtkMRMLColorTableNodeBlue";
    }
  if (this->Type == this->Yellow)
    {
    return "vtkMRMLColorTableNodeYellow";
    }
  if (this->Type == this->Magenta)
    {
    return "vtkMRMLColorTableNodeMagenta";
    }
  if (this->Type == this->Cyan)
    {
    return "vtkMRMLColorTableNodeCyan";
    }
  if (this->Type == this->Warm1)
    {
    return "vtkMRMLColorTableNodeWarm1";
    }
  if (this->Type == this->Warm2)
    {
    return "vtkMRMLColorTableNodeWarm2";
    }
  if (this->Type == this->Warm3)
    {
    return "vtkMRMLColorTableNodeWarm3";
    }
  if (this->Type == this->Cool1)
    {
    return "vtkMRMLColorTableNodeCool1";
    }
  if (this->Type == this->Cool2)
    {
    return "vtkMRMLColorTableNodeCool2";
    }
  if (this->Type == this->Cool3)
    {
    return "vtkMRMLColorTableNodeCool3";
    }
  if (this->Type == this->WarmShade1)
    {
    return "vtkMRMLColorTableNodeWarmShade1";
    }
  if (this->Type == this->WarmShade2)
    {
    return "vtkMRMLColorTableNodeWarmShade2";
    }
  if (this->Type == this->WarmShade3)
    {
    return "vtkMRMLColorTableNodeWarmShade3";
    }
  if (this->Type == this->CoolShade1)
    {
    return "vtkMRMLColorTableNodeCoolShade1";
    }
  if (this->Type == this->CoolShade2)
    {
    return "vtkMRMLColorTableNodeCoolShade2";
    }
  if (this->Type == this->CoolShade3)
    {
    return "vtkMRMLColorTableNodeCoolShade3";
    }
  if (this->Type == this->WarmTint1)
    {
    return "vtkMRMLColorTableNodeWarmTint1";
    }
  if (this->Type == this->WarmTint2)
    {
    return "vtkMRMLColorTableNodeWarmTint2";
    }
  if (this->Type == this->WarmTint3)
    {
    return "vtkMRMLColorTableNodeWarmTint3";
    }
  if (this->Type == this->CoolTint1)
    {
    return "vtkMRMLColorTableNodeCoolTint1";
    }
  if (this->Type == this->CoolTint2)
    {
    return "vtkMRMLColorTableNodeCoolTint2";
    }
  if (this->Type == this->CoolTint3)
    {
    return "vtkMRMLColorTableNodeCoolTint3";
    }
  return "(unknown)";
}

//----------------------------------------------------------------------------
const char* vtkMRMLColorTableNode::GetTypeAsString()
{
  if (this->Type == this->FullRainbow)
    {
    return "FullRainbow";
    }
  if (this->Type == this->Grey)
    {
    return "Grey";
    }
  if (this->Type == this->Iron)
    {
    return "Iron";
    }
  if (this->Type == this->Rainbow)
    {
    return "Rainbow";
    }
  if (this->Type == this->Ocean)
    {
    return "Ocean";
    }
  if (this->Type == this->Desert)
    {
    return "Desert";
    }
  if (this->Type == this->InvGrey)
    {
    return "InvertedGrey";
    }
  if (this->Type == this->ReverseRainbow)
    {
    return "ReverseRainbow";
    }
  if (this->Type == this->FMRI)
    {
    return "fMRI";
    }
  if (this->Type == this->FMRIPA)
    {
    return "fMRIPA";
    }
  if (this->Type == this->Labels)
    {
    return "Labels";
    }
  if (this->Type == this->Random)
    {
    return "Random";
    }
  if (this->Type == this->User)
    {
      return "UserDefined";
    }
  if (this->Type == this->File)
    {
    return "File";
    }
  if (this->Type == this->Red)
    {
    return "Red";
    }
  if (this->Type == this->Green)
    {
    return "Green";
    }
  if (this->Type == this->Blue)
    {
    return "Blue";
    }
  if (this->Type == this->Cyan)
    {
    return "Cyan";
    }
  if (this->Type == this->Magenta)
    {
    return "Magenta";
    }
  if (this->Type == this->Yellow)
    {
    return "Yellow";
    }
  if (this->Type == this->Warm1)
    {
    return "Warm1";
    }
  if (this->Type == this->Warm2)
    {
    return "Warm2";
    }
  if (this->Type == this->Warm3)
    {
    return "Warm3";
    }
  if (this->Type == this->Cool1)
    {
    return "Cool1";
    }
  if (this->Type == this->Cool2)
    {
    return "Cool2";
    }
  if (this->Type == this->Cool3)
    {
    return "Cool3";
    }
  if (this->Type == this->WarmShade1)
    {
    return "WarmShade1";
    }
  if (this->Type == this->WarmShade2)
    {
    return "WarmShade2";
    }
  if (this->Type == this->WarmShade3)
    {
    return "WarmShade3";
    }
  if (this->Type == this->CoolShade1)
    {
    return "CoolShade1";
    }
  if (this->Type == this->CoolShade2)
    {
    return "CoolShade2";
    }
  if (this->Type == this->CoolShade3)
    {
    return "CoolShade3";
    }
  if (this->Type == this->WarmTint1)
    {
    return "WarmTint1";
    }
  if (this->Type == this->WarmTint2)
    {
    return "WarmTint2";
    }
  if (this->Type == this->WarmTint3)
    {
    return "WarmTint3";
    }
  if (this->Type == this->CoolTint1)
    {
    return "CoolTint1";
    }
  if (this->Type == this->CoolTint2)
    {
    return "CoolTint2";
    }
  if (this->Type == this->CoolTint3)
    {
    return "CoolTint3";
    }
  return "(unknown)";
}

//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
/*
  vtkMRMLColorTableDisplayNode *dnode = this->GetDisplayNode();
  if (dnode != NULL && dnode == vtkMRMLColorTableDisplayNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
        this->InvokeEvent(vtkMRMLColorTableNode::DisplayModifiedEvent, NULL);
    }
*/
  return;
}

//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetType(int type)
{
  if (this->GetLookupTable() != NULL &&
      this->Type == type)
    {
    vtkDebugMacro("SetType: type is already set to " << type <<  " = " << this->GetTypeAsString());
    return;
    }
    
    this->Type = type;

    vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting Type to " << type << " = " << this->GetTypeAsString());

    //this->LookupTable->Delete();
    if (this->GetLookupTable() == NULL)
      {
      vtkDebugMacro("vtkMRMLColorTableNode::SetType Creating a new lookup table (was null) of type " << this->GetTypeAsString() << "\n");
      vtkLookupTable *table = vtkLookupTable::New();
      this->SetLookupTable(table);
      table->Delete();
      // as a FullRainbow, set the table range to 255
      this->GetLookupTable()->SetTableRange(0, 255);
      }

    // delay setting names from colours until asked for one
    if (this->Type == this->FullRainbow)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0, 1);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A full rainbow of 256 colors, goes from red to red with all rainbow colors in between. Useful for colourful display of a label map");
      }
    else if (this->Type == this->Grey)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0, 0);
      this->GetLookupTable()->SetSaturationRange(0, 0);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A grey scale ranging from black at 0 to white at 255. Useful for displaying MRI volumes.");
      }
    else if (this->Type == this->Red)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0, 0);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A red scale of 256 values. Useful for layering with Cyan");
      }
    else if (this->Type == this->Green)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.333, 0.333);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A green scale of 256 values, useful for layering with Magenta");
      }
    else if (this->Type == this->Blue)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.667, 0.667);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A blue scale of 256 values from black to pure blue, useful for layering with Yellow");
      }
    else if (this->Type == this->Yellow)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.167, 0.167);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A yellow scale of 256 values, from black to pure yellow, useful for layering with blue (it's complementary, layering yields gray)");
      }
    else if (this->Type == this->Cyan)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.5, 0.5);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A cyan ramp of 256 values, from black to cyan, complementary ramp to red, layering yeilds gray");
      }
    else if (this->Type == this->Magenta)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.833, 0.833);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A magenta scale of 256 colors from black to magenta, complementary ramp to green, layering yeilds gray ");
      }
    else if (this->Type == this->WarmShade1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.167, 0.0);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to red, of 256 colors, ramp of warm colors with variation in value that's complementary to CoolShade1 ");
      }
    else if (this->Type == this->WarmShade2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(.333, 0.167);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to yellow, through green, of 256 colors, ramp of warm colors with variation in value that's complementary to CoolShade2 ");
      }
    else if (this->Type == this->WarmShade3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.5, 0.333);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to green, of 256 colours, ramp of warm colors with variation in value that's complementary to CoolShade3 ");
      }
    else if (this->Type == this->CoolShade1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.667, 0.5);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to cyan, 256 colours, ramp of cool colours with variation in value that is complementary to WarmShade1 ");
      }
    else if (this->Type == this->CoolShade2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.833, 0.667);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to blue through purple, 256 colors, ramp of cool colours with variation in value that is complementary to WarmShade2 ");
      }
    else if (this->Type == this->CoolShade3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(1.0, 0.833);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(0, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from black to magenta, 256 colours, ramp of cool colours with variation in value that is complementary to WarmShade3");
      }
    else if (this->Type == this->WarmTint1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.167, 0.0);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to red, 256 colours, ramp of warm colours with variation in saturation that's complementary to CoolTint1");
      }
    else if (this->Type == this->WarmTint2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(.333, 0.167);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to yellow, 256 colours, ramp of warm colours with variation in saturation that's complementary to CoolTint2");
      }
    else if (this->Type == this->WarmTint3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.5, 0.333);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to green, 256 colours, ramp of warm colours with variation in saturation that's complementary to CoolTint3");
      }
    else if (this->Type == this->CoolTint1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.667, 0.5);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to cyan, 256 colours, ramp of cool colours with variations in saturation that's complementary to WarmTint1");
      }
    else if (this->Type == this->CoolTint2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.833, 0.667);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to blue, 256 colours, ramp of cool colours with variations in saturation that's complementary to WarmTint2");
      }
    else if (this->Type == this->CoolTint3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(1.0, 0.833);
      this->GetLookupTable()->SetSaturationRange(0, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from white to magenta, 256 colours, ramp of cool colours with variations in saturation that's complementary to WarmTint3");
      }
    else if (this->Type == this->Warm1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.167, 0.0);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from yellow to red, of 256 colors, ramp of warm colours that's complementary to Cool1");
      }
    else if (this->Type == this->Warm2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(.333, 0.167);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from green to yellow, 256 colours, ramp of warm colours that's complementary to Cool2");
      }
    else if (this->Type == this->Warm3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.5, 0.333);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from cyan to green, 256 colours, ramp of warm colours that's complementary to Cool3");
      }
    else if (this->Type == this->Cool1)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.667, 0.5);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from blue to cyan, 256 colours, ramp of cool colours that's complementary to Warm1");
      }
    else if (this->Type == this->Cool2)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(0.833, 0.667);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from magenta to blue, 256 colours, ramp of cool colours that's complementary to Warm2");
      }
    else if (this->Type == this->Cool3)
      {
      // from vtkSlicerSliceLayerLogic.cxx
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->SetTableRange(0, 255);
      this->GetLookupTable()->SetHueRange(1.0, 0.833);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetAlphaRange(1, 1); // not used
      this->GetLookupTable()->Build();
      this->SetNamesFromColors();
      this->SetDescription("A scale from red to magenta, ramp of cool colours that's complementary to Warm3");
      }
    else if (this->Type == this->Iron)
      {
      this->GetLookupTable()->SetNumberOfTableValues(156);
      this->GetLookupTable()->SetTableRange(0, 156);
      this->GetLookupTable()->SetHueRange(0, 0.15);
      this->GetLookupTable()->SetSaturationRange(1,1);
      this->GetLookupTable()->SetValueRange(1,1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("A scale from red to  yellow, 157 colours, useful for ");
      }

    else if (this->Type == this->Rainbow)
      {
      this->GetLookupTable()->SetNumberOfTableValues(256);
      this->GetLookupTable()->SetHueRange(0, 0.8);
      this->GetLookupTable()->SetSaturationRange(1,1);
      this->GetLookupTable()->SetValueRange(1,1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("Goes from red to purple, passing through the colors of the rainbow in between. Useful for a colorful display of a label map");
      }

    else if (this->Type == this->Ocean)
      {
      this->GetLookupTable()->SetNumberOfTableValues(256);
      this->GetLookupTable()->SetHueRange(0.666667, 0.5);
      this->GetLookupTable()->SetSaturationRange(1,1);
      this->GetLookupTable()->SetValueRange(1,1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("A lighter blue scale of 256 values, useful for showing registration results.");
      }
    else if (this->Type == this->Desert)
      {
      this->GetLookupTable()->SetNumberOfTableValues(256);
      this->GetLookupTable()->SetHueRange(0, 0.1);
      this->GetLookupTable()->SetSaturationRange(1,1);
      this->GetLookupTable()->SetValueRange(1,1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("Red to yellow/orange scale, 256 colours, useful for ");
      }
    
    else if (this->Type == this->InvGrey)
      {
      this->GetLookupTable()->SetNumberOfTableValues(256);
      this->GetLookupTable()->SetHueRange(0,0);
      this->GetLookupTable()->SetSaturationRange(0,0);
      this->GetLookupTable()->SetValueRange(1,0);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("A white to black scale, 256 colours, useful to highlight negative versions, or to flip intensities of signal values.");
      }

    else if (this->Type == this->ReverseRainbow)
      {
      this->GetLookupTable()->SetNumberOfTableValues(256);
      this->GetLookupTable()->SetHueRange(0.8, 1);
      this->GetLookupTable()->SetSaturationRange(1,1);
      this->GetLookupTable()->SetValueRange(1,1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("A colourful display option, 256 colours going from purple to red");
      }
    
    else if (this->Type == this->FMRI)
      {
      // Use different numbers of table values for neg and pos
      // to make sure -1 is represented by blue

      // From green to blue
      vtkLookupTable *neg = vtkLookupTable::New();
      neg->SetNumberOfTableValues(23);
      neg->SetHueRange(0.5, 0.66667);
      neg->SetSaturationRange( 1, 1);
      neg->SetValueRange(1, 1);
      neg->SetRampToLinear();
      neg->Build();

      // From red to yellow
      vtkLookupTable *pos = vtkLookupTable::New();
      pos->SetNumberOfTableValues(20);
      pos->SetHueRange(0,0.16667);
      pos->SetSaturationRange(1,1);
      pos->SetValueRange(1,1);
      pos->SetRampToLinear();
      pos->Build();

      this->GetLookupTable()->SetNumberOfTableValues(43);
      this->GetLookupTable()->SetTableRange(0,43);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();

      for (int i = 0; i < 23; i++)
        {
        this->GetLookupTable()->SetTableValue(i, neg->GetTableValue(i));
        }
      for (int i = 0; i < 20; i++)
        {
        this->GetLookupTable()->SetTableValue(i+23, pos->GetTableValue(i));
        }
      
      pos->Delete();
      neg->Delete();
      this->SetNamesFromColors();
      this->SetDescription("A combination of Ocean (0-22) and Desert (23-42), useful for displaying functional MRI volumes (highlights activation)");
      }
    
    else if (this->Type == this->FMRIPA)
      {
      int size = 20;
      this->GetLookupTable()->SetNumberOfTableValues(size);
      this->GetLookupTable()->SetTableRange(0,size);
      this->GetLookupTable()->SetHueRange(0, 0.16667);
      this->GetLookupTable()->SetSaturationRange(1, 1);
      this->GetLookupTable()->SetValueRange(1, 1);
      this->GetLookupTable()->SetRampToLinear();
      this->GetLookupTable()->ForceBuild();
      this->SetNamesFromColors();
      this->SetDescription("A small fMRI positive activation scale going from red to yellow from 0-19, useful for displaying functional MRI volumes when don't need the blue of the fMRI scale.");
      }

    else if (this->Type == this->Labels)
      {
      // from Slicer2's Colors.xml
      this->GetLookupTable()->SetNumberOfTableValues(257);
      this->GetLookupTable()->SetTableRange(0,257);
      this->Names.clear();
      this->Names.resize(this->GetLookupTable()->GetNumberOfTableValues());
      
      if (this->SetColorName(0, "Black") != 0)
        {
        this->GetLookupTable()->SetTableValue(0, 0, 0, 0, 0.0);
        }

      if (this->SetColorName(1, "jake") != 0)
        {
        this->GetLookupTable()->SetTableValue(1, 0.2, 0.5, 0.8, 1.0);
        }
      if (this->SetColorName(2, "Peach") != 0)
        {
        this->GetLookupTable()->SetTableValue(2, 1.0, 0.8, 0.7, 1.0);
        }
      if (this->SetColorName(3, "Brain") != 0)
        {
        this->GetLookupTable()->SetTableValue(3, 1.0, 1.0, 1.0, 1.0);
        }
      if (this->SetColorName(4, "Ventricles") != 0)
        {
        this->GetLookupTable()->SetTableValue(4, 0.4, 0.7, 1.0, 1.0);
        }

      if (this->SetColorName(5, "Vessels") != 0)
        {
        this->GetLookupTable()->SetTableValue(5, 0.9, 0.5, 0.5, 1.0);
        }
      if (this->SetColorName(6, "Tumor") != 0)
        {
        this->GetLookupTable()->SetTableValue(6, 0.5, 0.9, 0.5, 1.0);
        }
      if (this->SetColorName(7, "fMRI-high") != 0)
        {
        this->GetLookupTable()->SetTableValue(7, 0.5, 0.9, 0.9, 1.0);
        }
      if (this->SetColorName(8, "fMRI-low") != 0)
        {
        this->GetLookupTable()->SetTableValue(8, 0.9, 0.9, 0.5, 1.0);
        }
      if (this->SetColorName(9, "Pre-Gyrus") != 0)
        {
        this->GetLookupTable()->SetTableValue(9, 0.9, 0.7, 0.9, 1.0);
        }
        if (this->SetColorName(10, "Post-Gyrus") != 0)
        {
        this->GetLookupTable()->SetTableValue(10, 0.9, 0.9, 0.5, 1.0);
        }
      for (int offset = 0; offset <= 240; offset += 10)
        {
        if (this->SetColorName(offset + 11, "jake") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 11, 0.2, 0.5, 0.8, 1.0);
          }        
        if (this->SetColorName(offset + 12, "elwood") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 12, 0.2, 0.8, 0.5, 1.0);
          }
        if (this->SetColorName(offset + 13, "gato") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 13, 0.8, 0.5, 0.2, 1.0);
          }
        if (this->SetColorName(offset + 14, "avery") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 14, 0.8, 0.2, 0.5, 1.0);
          }
        if (this->SetColorName(offset + 15, "mambazo") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 15, 0.5, 0.2, 0.8, 1.0);
          }
        if (this->SetColorName(offset + 16, "domino") != 0)
          {
          this->GetLookupTable()->SetTableValue(offset + 16, 0.5, 0.8, 0.2, 1.0);
          }
        if (offset <= 230)
          {
          // these ones don't go past 256
          if (this->SetColorName(offset + 17, "monk") != 0)
            {
            this->GetLookupTable()->SetTableValue(offset + 17, 0.2, 0.2, 0.8, 1.0);
            }
          if (this->SetColorName(offset + 18, "forest") != 0)
            {
            this->GetLookupTable()->SetTableValue(offset + 18, 0.8, 0.8, 0.2, 1.0);
            }
          if (this->SetColorName(offset + 19, "dylan") != 0)
            {
            this->GetLookupTable()->SetTableValue(offset + 19, 0.2, 0.8, 0.8, 1.0);
            }
          if (this->SetColorName(offset + 20, "kales") != 0)
            {
            this->GetLookupTable()->SetTableValue(offset + 20, 0.5, 0.5, 0.5, 1.0);
            }
          }
        }
      /*
      this->SetColorName(300, "fMRI-neg");
      this->GetLookupTable()->SetTableValue(300, 0.0, 0.8, 1.0, 1.0);

      this->SetColorName(301, "fMRI-pos");
      this->GetLookupTable()->SetTableValue(301, 1.0, 1.0, 0.0, 1.0);
      */
      this->NamesInitialisedOn();
      this->SetDescription("A legacy colour table that contains some anatomical mapping");
      }
    else if (this->Type == this->Random)
      {
      int size = 255;
      
      this->GetLookupTable()->SetTableValue(0, 0, 0, 0, 0);
      this->GetLookupTable()->SetRange(0, size);
      this->GetLookupTable()->SetNumberOfTableValues(size + 1);
      for (int i = 1; i <= size; i++)
        {
        // table values have to be 0-1
        double r = (rand()%255)/255.0;
        double g = (rand()%255)/255.0;
        double b = (rand()%255)/255.0;
       
        this->GetLookupTable()->SetTableValue(i, r, g, b, 1.0);
        }
      this->SetNamesFromColors();      
      this->SetDescription("A random selection of 256 rgb colours, useful to distinguish between a small number of labeled regions (especially outside of the brain)");
      }

    else if (this->Type == this->User)
      {
      this->LookupTable->SetNumberOfTableValues(0);
      this->LastAddedColor = -1;
      vtkDebugMacro("Set type to user, call SetNumberOfColors, then AddColor..");
      this->SetDescription("A user defined colour table, use the editor to specify it");
      }

    else if (this->Type == this->File)
      {
      vtkDebugMacro("Set type to file, set up a storage node, set it's FileName and call ReadData on it...");
      this->SetDescription("A color table read in from a text file, each line of the format: IntegerLabel  Name  R  G  B  Alpha");
      }
    
    else
      {
      vtkErrorMacro("vtkMRMLColorTableNode: SetType ERROR, unknown type " << type << endl);
      return;
      }
    // invoke a modified event
    this->Modified();
    
    // invoke a type  modified event
    this->InvokeEvent(vtkMRMLColorTableNode::TypeModifiedEvent);
}

//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetNamesFromColors()
{
  if (this->GetNamesInitialised())
    {
    vtkWarningMacro("SetNamesFromColors is over riding already set names for node " << this->GetName());
    }
  
  int size = this->GetLookupTable()->GetNumberOfColors();
  double *rgba;
  bool errorCondition = false;
  // reset the names
  this->Names.clear();
  this->Names.resize(size);
  for (int i = 0; i < size; i++)
    {
    rgba = this->GetLookupTable()->GetTableValue(i);
    std::stringstream ss;
    ss << "R=";
    ss << rgba[0];
    ss << " G=";
    ss << rgba[1];
    ss << " B=";
    ss << rgba[2];
    ss << " A=";
    ss << rgba[3];
    vtkDebugMacro("SetNamesFromColors: " << i << " Name = " << ss.str().c_str());
    if (this->SetColorName(i, ss.str().c_str()) == 0)
      {
      vtkErrorMacro("SetNamesFromColors: Error setting color name " << i << " Name = " << ss.str().c_str());
      errorCondition = true;
      break;
      }
    }
  if (!errorCondition)
    {
    this->NamesInitialisedOn();
    }
}

//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::SetNumberOfColors(int n)
{
  if (this->GetLookupTable() == NULL)
    {
    vtkErrorMacro("SetNumberofColors: lookup table is null, set the type first.");
    return;
    }
  if (this->GetType() != this->User &&
      this->GetType() != this->File)
    {
      vtkErrorMacro("vtkMRMLColorTableNode::SetNumberOfColors: ERROR: can't set number of colours if not a user defined colour table, reset the type first to User or File\n");
      return;
    }

  if (this->GetLookupTable()->GetNumberOfTableValues() != n)
    {
    this->GetLookupTable()->SetNumberOfTableValues(n);
    }

  if (this->Names.size() != (unsigned int)n)
    {
    this->Names.resize(n);
    }
  
}

//---------------------------------------------------------------------------
int vtkMRMLColorTableNode::GetNumberOfColors()
{
  if (this->GetLookupTable() != NULL)
    {
      return this->GetLookupTable()->GetNumberOfTableValues();
    }
  else
    {
      return 0;
    }
}
//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::AddColor(const char *name, double r, double g, double b)
{
 if (this->GetType() != this->User &&
     this->GetType() != this->File)
    {
      vtkErrorMacro("vtkMRMLColorTableNode::AddColor: ERROR: can't add a colour if not a user defined colour table, reset the type first to User or File\n");
      return;
    }
 this->LastAddedColor++;
 this->SetColor(this->LastAddedColor, name, r, g, b);
}

//---------------------------------------------------------------------------
int vtkMRMLColorTableNode::SetColor(int entry, const char *name, double r, double g, double b)
{
  if (this->GetType() != this->User)
    {
      vtkErrorMacro( "vtkMRMLColorTableNode::SetColor: ERROR: can't set a colour if not a user defined colour table, reset the type first to User\n");
      return 0;
    }
  if (entry < 0 ||
      entry >= this->GetLookupTable()->GetNumberOfTableValues())
    {
    vtkErrorMacro( "vtkMRMLColorTableNode::SetColor: requested entry " << entry << " is out of table range: 0 - " << this->GetLookupTable()->GetNumberOfTableValues() << ", call SetNumberOfColors" << endl);
      return 0;
    }

  this->GetLookupTable()->SetTableValue(entry, r, g, b, 1.0);
  if (strcmp(this->GetColorName(entry), name) != 0)
    {
    if (this->SetColorName(entry, name) == 0)
      {
      vtkWarningMacro("SetColor: error setting color name " << name << " for entry " << entry);
      return 0;
      }
    }

  // trigger a modified event
  this->InvokeEvent (vtkCommand::ModifiedEvent);
  return 1;
}


//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::ClearNames()
{
  this->Names.clear();
  this->NamesInitialisedOff();
}

//---------------------------------------------------------------------------
void vtkMRMLColorTableNode::Reset()
{
  int disabledModify = this->StartModify();

  // only call reset if this is a user node
  if (this->GetType() == vtkMRMLColorTableNode::User)
    {
    int type = this->GetType();
    Superclass::Reset();
    this->SetType(type);
    }

  this->EndModify(disabledModify);
}

//---------------------------------------------------------------------------
int vtkMRMLColorTableNode::GetColorIndexByName(const char *name)
{
  if (this->GetNamesInitialised() && name != NULL)
    {
    std::string strName = name;
    for (unsigned int i = 0; i < this->Names.size(); i++)
      {
      if (strName.compare(this->GetColorName(i)) == 0)
        {
        return i;
        }
      }
    }
    return -1;
}
