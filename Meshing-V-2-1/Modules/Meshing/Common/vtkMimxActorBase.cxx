/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkMimxActorBase.cxx,v $
Language:  C++
Date:      $Date: 2008/07/28 15:06:02 $
Version:   $Revision: 1.13 $

 Musculoskeletal Imaging, Modelling and Experimentation (MIMX)
 Center for Computer Aided Design
 The University of Iowa
 Iowa City, IA 52242
 http://www.ccad.uiowa.edu/mimx/

Copyright (c) The University of Iowa. All rights reserved.
See MIMXCopyright.txt or http://www.ccad.uiowa.edu/mimx/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "vtkMimxActorBase.h"
#include "vtkActor.h"
#include "vtkDataSet.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"


vtkCxxRevisionMacro(vtkMimxActorBase, "$Revision: 1.13 $");


//vtkStandardNewMacro(vtkActorBase);

vtkMimxActorBase::vtkMimxActorBase()
{
  this->FileName = new char[256];
  this->FilePath = new char[256];
  this->UniqueId = new char[64];
  this->FoundationName = new char[256];
  this->DataType = 0;
  // added for Slicer
  this->SavedDisplayNode = NULL;
}

vtkMimxActorBase::~vtkMimxActorBase()
{
  delete this->FileName;
  delete this->FilePath;
  delete this->UniqueId;
  delete this->FoundationName;
}

//void vtkMimxActorBase::SetFileName( const char *InputFileName)
//{
//  strcpy(this->FileName, InputFileName);
//}

void vtkMimxActorBase::SetFilePath(const char *InputFilePath)
{
  strcpy(this->FilePath, InputFilePath);
}

void vtkMimxActorBase::SetUniqueId( const char *Id)
{
  strcpy(this->UniqueId, Id);
}

void vtkMimxActorBase::SetFoundationName(const char *created)
{
  strcpy(this->FoundationName, created);
  strcpy(this->FileName, created);
}

void vtkMimxActorBase::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

void vtkMimxActorBase::SetObjectName(const char* aFileName, vtkIdType &Count)
{
  char tempbuffer[1024];
  strcpy(tempbuffer, this->FoundationName);
  strcat(tempbuffer, "_");
  strcat(tempbuffer, aFileName);
  char buffer[10];
  sprintf(buffer, "%ld", (long) Count);
  strcat(tempbuffer, buffer);
  strcpy(this->FileName, tempbuffer);
}

// added for Slicer integration

void vtkMimxActorBase::SaveVisibility(void) {this->SavedVisibility = (this->Actor->GetVisibility())?true:false;}
void vtkMimxActorBase::RestoreVisibility(void) {this->SetVisibility(this->SavedVisibility);}


void vtkMimxActorBase::Hide()
{
    if (this->SavedDisplayNode != NULL)
        this->SavedDisplayNode->SetVisibility(0);
    if (this->Actor != NULL)
        this->Actor->SetVisibility(0);
}


void vtkMimxActorBase::Show()
{
    if (this->SavedDisplayNode != NULL)
        this->SavedDisplayNode->SetVisibility(1);
    if (this->Actor != NULL)
        this->Actor->SetVisibility(1);
}


void  vtkMimxActorBase::SetVisibility(int i)
{
    switch (i)
   {
      case 0: {this->Hide(); break;}
      case 1: {this->Show(); break;}
    }
}
