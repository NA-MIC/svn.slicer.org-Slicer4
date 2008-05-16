/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMimxErrorCallback.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMimxErrorCallback.h"

#include "vtkSetGet.h"
#include "vtkObject.h"
#include "vtkKWMessageDialog.h"

#include <string.h>
#include <ctype.h>

//----------------------------------------------------------------
vtkMimxErrorCallback::vtkMimxErrorCallback() 
{ 
  this->ClientData = NULL;
  this->Callback = NULL; 
  this->ClientDataDeleteCallback = NULL;
  this->KWMessageDialog = vtkKWMessageDialog::New();
  this->State = 0;
}
  
vtkMimxErrorCallback::~vtkMimxErrorCallback() 
{ 
  if (this->ClientDataDeleteCallback)
    {
    this->ClientDataDeleteCallback(this->ClientData);
    }
  this->KWMessageDialog->Delete();
}
 
void vtkMimxErrorCallback::Execute(vtkObject *,unsigned long, void *calldata)
{
  if (this->Callback)
    {
    this->Callback(this->ClientData);
    }
  // displaying the error messages
  const char* message = reinterpret_cast<const char*>( calldata );
  this->ErrorMessage(message);
}
//------------------------------------------------------------------------------
void vtkMimxErrorCallback::ErrorMessage(const char *Message)
{
        this->State = 1;
        this->KWMessageDialog->SetStyleToCancel();
        this->KWMessageDialog->SetCancelButtonText("OK");
        this->KWMessageDialog->SetApplication(this->KWApplication);
        this->KWMessageDialog->Create();
        this->KWMessageDialog->SetTitle("Your attention please!");
        this->KWMessageDialog->SetText(Message);
        this->KWMessageDialog->Invoke();
}
