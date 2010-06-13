/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women\"s Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLSliceCompositeNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkMRMLSliceCompositeNode.h"
#include "vtkMRMLScene.h"

#include "vtkMatrix4x4.h"

//------------------------------------------------------------------------------
vtkMRMLSliceCompositeNode* vtkMRMLSliceCompositeNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLSliceCompositeNode");
  if(ret)
    {
    return (vtkMRMLSliceCompositeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLSliceCompositeNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLSliceCompositeNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLSliceCompositeNode");
  if(ret)
    {
    return (vtkMRMLSliceCompositeNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLSliceCompositeNode;
}

//----------------------------------------------------------------------------
vtkMRMLSliceCompositeNode::vtkMRMLSliceCompositeNode()
{
  this->BackgroundVolumeID = NULL;
  this->ForegroundVolumeID = NULL;
  this->LabelVolumeID = NULL;
  this->Compositing = 0;
  this->ForegroundOpacity = 0.0; // start by showing only the background volume
  this->LabelOpacity = 1.0; // Show the label if there is one
  this->LinkedControl = 0;
  this->ForegroundGrid = 0;
  this->BackgroundGrid = 0;
  this->LabelGrid = 1;
  this->FiducialVisibility = 1;
  this->FiducialLabelVisibility = 1;
  this->AnnotationSpace = vtkMRMLSliceCompositeNode::IJKAndRAS;
  this->AnnotationMode = vtkMRMLSliceCompositeNode::All;
  this->SliceIntersectionVisibility = 0;
  this->DoPropagateVolumeSelection = true;
}

//----------------------------------------------------------------------------
vtkMRMLSliceCompositeNode::~vtkMRMLSliceCompositeNode()
{
  if (this->BackgroundVolumeID)
    {
    this->SetBackgroundVolumeID(NULL);
    }
  if (this->ForegroundVolumeID)
    {
    this->SetForegroundVolumeID(NULL);
    }
  if (this->LabelVolumeID)
    {
    this->SetLabelVolumeID(NULL);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLSliceCompositeNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);

  vtkIndent indent(nIndent);

  of << indent << " backgroundVolumeID=\"" << 
   (this->BackgroundVolumeID ? this->BackgroundVolumeID : "") << "\"";
  of << indent << " foregroundVolumeID=\"" << 
   (this->ForegroundVolumeID ? this->ForegroundVolumeID : "") << "\"";
  of << indent << " labelVolumeID=\"" << 
   (this->LabelVolumeID ? this->LabelVolumeID : "") << "\"";

  of << indent << " compositing=\"" << this->Compositing << "\"";
  of << indent << " labelOpacity=\"" << this->LabelOpacity << "\"";
  of << indent << " linkedControl=\"" << this->LinkedControl << "\"";
  of << indent << " foregroundGrid=\"" << this->ForegroundGrid << "\"";
  of << indent << " backgroundGrid=\"" << this->BackgroundGrid << "\"";
  of << indent << " labelGrid=\"" << this->LabelGrid << "\"";
  of << indent << " fiducialVisibility=\"" << this->FiducialVisibility << "\"";
  of << indent << " fiducialLabelVisibility=\"" << this->FiducialLabelVisibility << "\"";
  of << indent << " sliceIntersectionVisibility=\"" << this->SliceIntersectionVisibility << "\"";
  if (this->GetLayoutName() != NULL)
    {
    of << indent << " layoutName=\"" << this->GetLayoutName() << "\"";
    }

  if ( this->AnnotationSpace == vtkMRMLSliceCompositeNode::XYZ)
    {
    of << indent << " annotationSpace=\"" << "xyz" << "\"";
    }
  else if ( this->AnnotationSpace == vtkMRMLSliceCompositeNode::IJK)
    {
    of << indent << " annotationSpace=\"" << "ijk" << "\"";
    }
  else if ( this->AnnotationSpace == vtkMRMLSliceCompositeNode::RAS)
    {
    of << indent << " annotationSpace=\"" << "RAS" << "\"";
    }

  if ( this->AnnotationMode == vtkMRMLSliceCompositeNode::NoAnnotation )
    {
    of << indent << " annotationMode=\"" << "NoAnnotation" << "\"";
    }
  else if ( this->AnnotationMode == vtkMRMLSliceCompositeNode::All )
    {
    of << indent << " annotationMode=\"" << "All" << "\"";
    }
  if ( this->AnnotationMode == vtkMRMLSliceCompositeNode::LabelValuesOnly )
    {
    of << indent << " annotationMode=\"" << "LabelValuesOnly" << "\"";
    }
  if ( this->AnnotationMode == vtkMRMLSliceCompositeNode::LabelAndVoxelValuesOnly )
    {
    of << indent << " annotationMode=\"" << "LabelAndVoxelValuesOnly" << "\"";
    }
  of << indent << " doPropagateVolumeSelection=\"" << (int)this->DoPropagateVolumeSelection << "\"";
}

//-----------------------------------------------------------
void vtkMRMLSliceCompositeNode::UpdateReferences()
{
   Superclass::UpdateReferences();

  if (this->BackgroundVolumeID != NULL && this->Scene->GetNodeByID(this->BackgroundVolumeID) == NULL)
    {
    this->SetBackgroundVolumeID(NULL);
    }
  if (this->ForegroundVolumeID != NULL && this->Scene->GetNodeByID(this->ForegroundVolumeID) == NULL)
    {
    this->SetForegroundVolumeID(NULL);
    }
  if (this->LabelVolumeID != NULL && this->Scene->GetNodeByID(this->LabelVolumeID) == NULL)
    {
    this->SetLabelVolumeID(NULL);
    }


}
//----------------------------------------------------------------------------
void vtkMRMLSliceCompositeNode::UpdateReferenceID(const char *oldID, const char *newID)
{
  Superclass::UpdateReferenceID(oldID, newID);
  if (this->BackgroundVolumeID && !strcmp(oldID, this->BackgroundVolumeID))
    {
    this->SetBackgroundVolumeID(newID);
    }
  if (this->ForegroundVolumeID && !strcmp(oldID, this->ForegroundVolumeID))
    {
    this->SetForegroundVolumeID(newID);
    }
  if (this->LabelVolumeID && !strcmp(oldID, this->LabelVolumeID))
    {
    this->SetLabelVolumeID(newID);
    }
}

//----------------------------------------------------------------------------
void vtkMRMLSliceCompositeNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "backgroundVolumeID")) 
      {
      if (attValue && *attValue == '\0')
        {
        this->SetBackgroundVolumeID(NULL);
        }
      else
        {
        this->SetBackgroundVolumeID(attValue);
        //this->Scene->AddReferencedNodeID(this->BackgroundVolumeID, this);
        }
      }
    else if (!strcmp(attName, "foregroundVolumeID")) 
      {
      if (attValue && *attValue == '\0')
        {
        this->SetForegroundVolumeID(NULL);
        }
      else
        {
        this->SetForegroundVolumeID(attValue);
        //this->Scene->AddReferencedNodeID(this->ForegroundVolumeID, this);
        }
      }
    else if (!strcmp(attName, "labelVolumeID")) 
      {
      if (attValue && *attValue == '\0')
        {
        this->SetLabelVolumeID(NULL);
        }
      else
        {
        this->SetLabelVolumeID(attValue);
        //this->Scene->AddReferencedNodeID(this->LabelVolumeID, this);
        }
      }
    else if (!strcmp(attName, "compositing"))
      {
      this->SetCompositing( atoi(attValue) );
      }
    else if (!strcmp(attName, "foregroundOpacity")) 
      {
      this->SetForegroundOpacity( atof(attValue) );
      }
    else if (!strcmp(attName, "labelOpacity")) 
      {
      this->SetLabelOpacity( atof(attValue) );
      }
    else if (!strcmp(attName, "linkedControl")) 
      {
      this->SetLinkedControl( atoi(attValue) );
      }    
    else if (!strcmp(attName, "foregroundGrid")) 
      {
      this->SetForegroundGrid( atoi(attValue) );
      }
    else if (!strcmp(attName, "backGrid")) 
      {
      this->SetBackgroundGrid( atoi(attValue) );
      }
    else if (!strcmp(attName, "labelGrid")) 
      {
      this->SetLabelGrid( atoi(attValue) );
      }    
    else if (!strcmp(attName, "fiducialVisibility")) 
      {
      this->SetFiducialVisibility( atoi(attValue) );
      }    
    else if (!strcmp(attName, "fiducialLabelVisibility")) 
      {
      this->SetFiducialLabelVisibility( atoi(attValue) );
      }    
    else if (!strcmp(attName, "sliceIntersectionVisibility")) 
      {
      this->SetSliceIntersectionVisibility( atoi(attValue) );
      }    
   else if (!strcmp(attName, "layoutName")) 
      {
      this->SetLayoutName( attValue );
      }

    else if(!strcmp (attName, "annotationSpace" ))
      {
      if (!strcmp (attValue, "xyz"))
        {
        this->SetAnnotationSpace (vtkMRMLSliceCompositeNode::XYZ);
        }
      else if (!strcmp (attValue, "ijk"))
        {
        this->SetAnnotationSpace (vtkMRMLSliceCompositeNode::IJK);
        }
      else if (!strcmp (attValue, "RAS"))
        {
        this->SetAnnotationSpace  (vtkMRMLSliceCompositeNode::RAS);
        }
      }
    else if(!strcmp (attName, "annotationMode" ))
      {
      if (!strcmp (attValue, "NoAnnotation"))
        {
        this->SetAnnotationMode (vtkMRMLSliceCompositeNode::NoAnnotation);
        }
      else if (!strcmp (attValue, "All"))
        {
        this->SetAnnotationMode (vtkMRMLSliceCompositeNode::All);
        }
      else if (!strcmp (attValue, "LabelValuesOnly"))
        {
        this->SetAnnotationMode (vtkMRMLSliceCompositeNode::LabelValuesOnly);
        }
      else if (!strcmp (attValue, "LabelAndVoxelValuesOnly"))
        {
        this->SetAnnotationMode (vtkMRMLSliceCompositeNode::LabelAndVoxelValuesOnly);
        }
      }
    else if(!strcmp (attName, "doPropagateVolumeSelection" ))
      {
      this->SetDoPropagateVolumeSelection(atoi(attValue)?true:false);
      }
    }

  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
// Copy the node\"s attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, SliceID
void vtkMRMLSliceCompositeNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLSliceCompositeNode *node = vtkMRMLSliceCompositeNode::SafeDownCast(anode);

  this->SetBackgroundVolumeID(node->GetBackgroundVolumeID());
  this->SetForegroundVolumeID(node->GetForegroundVolumeID());
  this->SetLabelVolumeID(node->GetLabelVolumeID());
  this->SetCompositing(node->GetCompositing());
  this->SetForegroundOpacity(node->GetForegroundOpacity());
  this->SetLabelOpacity(node->GetLabelOpacity());
  this->SetLinkedControl (node->GetLinkedControl());
  this->SetForegroundGrid ( node->GetForegroundGrid());
  this->SetBackgroundGrid ( node->GetBackgroundGrid());
  this->SetLabelGrid ( node->GetLabelGrid());
  this->SetFiducialVisibility ( node->GetFiducialVisibility ( ) );
  this->SetFiducialLabelVisibility ( node->GetFiducialLabelVisibility ( ) );
  this->SetAnnotationSpace ( node->GetAnnotationSpace() );
  this->SetAnnotationMode ( node->GetAnnotationMode() );
  this->SetDoPropagateVolumeSelection (node->GetDoPropagateVolumeSelection());
   
  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
void vtkMRMLSliceCompositeNode::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);

  os << indent << "BackgroundVolumeID: " << 
   (this->BackgroundVolumeID ? this->BackgroundVolumeID : "(none)") << "\n";
  os << indent << "ForegroundVolumeID: " << 
   (this->ForegroundVolumeID ? this->ForegroundVolumeID : "(none)") << "\n";
  os << indent << "LabelVolumeID: " << 
    (this->LabelVolumeID ? this->LabelVolumeID : "(none)") << "\n";
  os << indent << "Compositing: " << this->Compositing << "\n";
  os << indent << "ForegroundOpacity: " << this->ForegroundOpacity << "\n";
  os << indent << "LabelOpacity: " << this->LabelOpacity << "\n";
  os << indent << "LinkedControl: " << this->LinkedControl << "\n";
  os << indent << "ForegroundGrid: " << this->ForegroundGrid << "\n";
  os << indent << "BackgroundGrid: " << this->BackgroundGrid << "\n";
  os << indent << "LabelGrid: " << this->LabelGrid << "\n";
  os << indent << "FiducialVisibility: " << this->FiducialVisibility << "\n";
  os << indent << "FiducialLabelVisibility: " << this->FiducialLabelVisibility << "\n";
  os << indent << "SliceIntersectionVisibility: " << this->SliceIntersectionVisibility << "\n";
  os << indent << "AnnotationSpace: " << this->AnnotationSpace << "\n";
  os << indent << "AnnotationMode: " << this->AnnotationMode << "\n";
  os << indent << "DoPropagateVolumeSelection: " << this->DoPropagateVolumeSelection << "\n";
}

//----------------------------------------------------------------------------
void vtkMRMLSliceCompositeNode::UpdateScene(vtkMRMLScene* scene)
{
  if (scene == NULL)
    {
    vtkErrorMacro("UpdateScene: input scene is null!");
    return;
    }
  vtkMRMLSliceCompositeNode *node= NULL;
  int nnodes = scene->GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode");
  for (int n=0; n<nnodes; n++)
    {
    node = vtkMRMLSliceCompositeNode::SafeDownCast (
          scene->GetNthNodeByClass(n, "vtkMRMLSliceCompositeNode"));
    if (node != this && !strcmp(node->GetLayoutName(), this->GetLayoutName()))
      {
      break;
      }
    node = NULL;
    }
  if (node != NULL)
    {
    //scene->RemoveNodeNoNotify(node);
    scene->RemoveNode(node);
    }
}
// End
