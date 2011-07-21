
// MRML includes
#include "vtkMRMLInteractionNode.h"
#include "vtkMRMLScene.h"

// VTK includes
#include <vtkObjectFactory.h>
#include <vtkCallbackCommand.h>

// STD includes
#include <string>
#include <iostream>
#include <sstream>

//------------------------------------------------------------------------------
vtkMRMLInteractionNode* vtkMRMLInteractionNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLInteractionNode");
  if(ret)
    {
    return (vtkMRMLInteractionNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLInteractionNode;
}

//-----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLInteractionNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLInteractionNode");
  if(ret)
    {
    return (vtkMRMLInteractionNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLInteractionNode;
}


//----------------------------------------------------------------------------
vtkMRMLInteractionNode::vtkMRMLInteractionNode()
{
  this->SetSingletonTag("vtkMRMLInteractionNode");

  this->CurrentInteractionMode = vtkMRMLInteractionNode::ViewTransform;
  this->LastInteractionMode = vtkMRMLInteractionNode::ViewTransform;
  this->PickModePersistence = 0;
  this->PlaceModePersistence = 0;
  this->TransformModePersistence = 1;

}

//----------------------------------------------------------------------------
vtkMRMLInteractionNode::~vtkMRMLInteractionNode()
{
}

//----------------------------------------------------------------------------
int vtkMRMLInteractionNode::GetInteractionModeByString ( const char * modeString )
{
  if (modeString == NULL)
    {
    return (-1);
    }
  if ( !strcmp (modeString, "Pick" ))
    {
    return ( vtkMRMLInteractionNode::PickManipulate);
    }
  else if ( !strcmp (modeString, "PickManipulate" ))
    {
    return ( vtkMRMLInteractionNode::PickManipulate);
    }
  else if ( !strcmp (modeString, "Place" ))
    {
    return ( vtkMRMLInteractionNode::Place);
    }
  else if ( !strcmp (modeString, "Put" ))
    {
    return ( vtkMRMLInteractionNode::Place );
    }
  else if ( !strcmp (modeString, "ViewTransform" ))
    {
    return ( vtkMRMLInteractionNode::ViewTransform);
    }
  else if ( !strcmp (modeString, "Transform" ))
    {
    return ( vtkMRMLInteractionNode::ViewTransform);
    }
  else if ( !strcmp (modeString, "SelectRegion" ))
    {
    return ( vtkMRMLInteractionNode::SelectRegion);
    }
  else if ( !strcmp (modeString, "LassoRegion" ))
    {
    return ( vtkMRMLInteractionNode::LassoRegion);
    }
  else
    {
    return (-1);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::NormalizeAllMouseModes()
{
  this->PickModePersistence = 0;
  this->PlaceModePersistence = 0;
  this->TransformModePersistence = 1;
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SetPickModePersistence ( int val )
{
  if (this->PickModePersistence != val)
    {
    this->PickModePersistence = val;
    this->InvokeEvent(this->InteractionModePersistenceChangedEvent, NULL);
    }
}


//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SetPlaceModePersistence ( int val )
{
  if (this->PlaceModePersistence != val)
    {
    this->PlaceModePersistence = val;
    this->InvokeEvent(this->InteractionModePersistenceChangedEvent, NULL);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SetTransformModePersistence ( int val )
{
  if (this->TransformModePersistence != val)
    {
    this->TransformModePersistence = val;
    this->InvokeEvent(this->InteractionModePersistenceChangedEvent, NULL);
    }
}


//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SetLastInteractionMode ( int mode )
{
  if (this->LastInteractionMode != mode)
    {
    this->LastInteractionMode = mode;
    }
}


//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SetCurrentInteractionMode ( int mode )
{
  if ( this->CurrentInteractionMode == mode )
    {
    return;
    }
  switch (mode)
    {
    case vtkMRMLInteractionNode::PickManipulate:
      this->CurrentInteractionMode = mode;
      this->InvokeEvent(this->InteractionModeChangedEvent, NULL);
      break;
    case vtkMRMLInteractionNode::SelectRegion:
      this->CurrentInteractionMode = mode;
      this->InvokeEvent(this->InteractionModeChangedEvent, NULL);
      break;
    case vtkMRMLInteractionNode::LassoRegion:
      this->CurrentInteractionMode = mode;
      this->InvokeEvent(this->InteractionModeChangedEvent, NULL);
      break;
    case vtkMRMLInteractionNode::Place:
      this->CurrentInteractionMode = mode;
      this->InvokeEvent(this->InteractionModeChangedEvent, NULL);
      break;
    case vtkMRMLInteractionNode::ViewTransform:      
      this->CurrentInteractionMode = mode;
      this->InvokeEvent(this->InteractionModeChangedEvent, NULL);
      break;
    default:
      break;
    }
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::WriteXML(ostream& of, int nIndent)
{
  // Write all attributes not equal to their defaults
  
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  // configure stereo
  if ( this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::PickManipulate )
    {
    of << indent << " currentInteractionMode=\"" << "PickManipulate" << "\"";    
    }
  else if ( this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::SelectRegion )
    {
    of << indent << " currentInteractionMode=\"" << "SelectRegion" << "\"";    
    }
  else if ( this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::LassoRegion )
    {
    of << indent << " currentInteractionMode=\"" << "LassoRegion" << "\"";    
    }
  else if ( this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::Place )
    {
    of << indent << " currentInteractionMode=\"" << "Place" << "\"";    
    }
  else if ( this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::ViewTransform )
    {
    of << indent << " currentInteractionMode=\"" << "ViewTransform" << "\"";    
    }

  if ( this->GetLastInteractionMode() == vtkMRMLInteractionNode::PickManipulate )
    {
    of << indent << " lastInteractionMode=\"" << "PickManipulate" << "\"";    
    }
  else if ( this->GetLastInteractionMode() == vtkMRMLInteractionNode::SelectRegion )
    {
    of << indent << " lastInteractionMode=\"" << "SelectRegion" << "\"";    
    }
  else if ( this->GetLastInteractionMode() == vtkMRMLInteractionNode::LassoRegion )
    {
    of << indent << " lastInteractionMode=\"" << "LassoRegion" << "\"";    
    }
  else if ( this->GetLastInteractionMode() == vtkMRMLInteractionNode::Place )
    {
    of << indent << " lastInteractionMode=\"" << "Place" << "\"";    
    }
  else if ( this->GetLastInteractionMode() == vtkMRMLInteractionNode::ViewTransform )
    {
    of << indent << " lastInteractionMode=\"" << "ViewTransform" << "\"";    
    }

}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);


    if (!strcmp(attName, "currentInteractionMode")) 
      {
      if (!strcmp(attValue,"PickManipulate")) 
        {
        this->CurrentInteractionMode = vtkMRMLInteractionNode::PickManipulate;
        }
      else if ( !strcmp (attValue, "SelectRegion" ))
        {
        this->CurrentInteractionMode = vtkMRMLInteractionNode::SelectRegion;
        }
      else if ( !strcmp (attValue, "LassoRegion" ))
        {
        this->CurrentInteractionMode = vtkMRMLInteractionNode::LassoRegion;
        }
      else if ( !strcmp (attValue, "Place" ))
        {
        this->CurrentInteractionMode = vtkMRMLInteractionNode::Place;
        }
      else if ( !strcmp (attValue, "ViewTransform" ))
        {
        this->CurrentInteractionMode = vtkMRMLInteractionNode::ViewTransform;
        }
      }
    else if (!strcmp(attName, "lastInteractionMode"))
      {
      if (!strcmp(attValue,"PickManipulate")) 
        {
        this->LastInteractionMode = vtkMRMLInteractionNode::PickManipulate;
        }
      else if ( !strcmp (attValue, "SelectRegion" ))
        {
        this->LastInteractionMode = vtkMRMLInteractionNode::SelectRegion;
        }
      else if ( !strcmp (attValue, "LassoRegion" ))
        {
        this->LastInteractionMode = vtkMRMLInteractionNode::LassoRegion;
        }
      else if ( !strcmp (attValue, "Place" ))
        {
        this->LastInteractionMode = vtkMRMLInteractionNode::Place;
        }
      else if ( !strcmp (attValue, "ViewTransform" ))
        {
        this->LastInteractionMode = vtkMRMLInteractionNode::ViewTransform;
        }
      }
    }
    
  this->EndModify(disabledModify);

}




//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLInteractionNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLInteractionNode *node = (vtkMRMLInteractionNode *) anode;

  this->SetCurrentInteractionMode (node->GetCurrentInteractionMode());

  this->SetLastInteractionMode ( node->GetLastInteractionMode());

  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
  Superclass::PrintSelf(os,indent);
  os << indent << "CurrentInteractionMode:        " << this->GetInteractionModeAsString(this->CurrentInteractionMode) << "\n";
  os << indent << "LastInteractionMode:        " <<  this->GetInteractionModeAsString(this->LastInteractionMode) << "\n";
  
  os << indent << "PickModePersistence: " << this->GetPickModePersistence() << "\n";
  os << indent << "PlaceModePersistence: " << this->GetPlaceModePersistence() << "\n";
  os << indent << "TransformModePersistence: " << this->GetTransformModePersistence() << "\n";
}



//---------------------------------------------------------------------------
const char * vtkMRMLInteractionNode::GetInteractionModeAsString(int mode)
{
  if (mode == this->PickManipulate)
    {
    return "PickManipulate";
    }
  if (mode == this->SelectRegion)
    {
    return "SelectRegion";
    }
  if (mode == this->LassoRegion)
    {
    return "LassoRegion";
    }
  if (mode == this->Place)
    {
    return "Place";
    }
  if (mode == this->ViewTransform)
    {
    return "ViewTransform";
    }
  return "(unknown)";
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SwitchToPersistentPickMode()
{
  if (this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::PickManipulate &&
      this->GetPickModePersistence() == 1)
    {
    return;
    }
  this->NormalizeAllMouseModes();
  this->SetLastInteractionMode(this->GetCurrentInteractionMode());
  this->SetPickModePersistence(1);
  this->SetCurrentInteractionMode(vtkMRMLInteractionNode::PickManipulate);
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SwitchToSinglePickMode()
{
  if (this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::PickManipulate &&
      this->GetPickModePersistence() == 0)
    {
    return;
    }
  this->NormalizeAllMouseModes();
  this->SetLastInteractionMode(this->GetCurrentInteractionMode());
  this->SetPickModePersistence(0);
  this->SetCurrentInteractionMode(vtkMRMLInteractionNode::PickManipulate);
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SwitchToPersistentPlaceMode()
{
  if (this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::Place &&
      this->GetPlaceModePersistence() == 1)
    {
    return;
    }
  this->NormalizeAllMouseModes();
  this->SetLastInteractionMode(this->GetCurrentInteractionMode());
  this->SetPlaceModePersistence(1);
  this->SetCurrentInteractionMode(vtkMRMLInteractionNode::Place);
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SwitchToSinglePlaceMode()
{
  if (this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::Place &&
      this->GetPlaceModePersistence() == 0)
    {
    return;
    }
  this->NormalizeAllMouseModes();
  this->SetLastInteractionMode(this->GetCurrentInteractionMode());
  this->SetPlaceModePersistence(0);
  this->SetCurrentInteractionMode(vtkMRMLInteractionNode::Place);
}

//----------------------------------------------------------------------------
void vtkMRMLInteractionNode::SwitchToViewTransformMode()
{
  if (this->GetCurrentInteractionMode() == vtkMRMLInteractionNode::ViewTransform &&
      this->GetTransformModePersistence() == 1)
    {
    return;
    }
  this->SetLastInteractionMode(this->GetCurrentInteractionMode());
  // only set transform mode persistence, keep the state of the pick and place
  // mode persistence
  this->SetTransformModePersistence(1);
  this->SetCurrentInteractionMode(vtkMRMLInteractionNode::ViewTransform);
}
