/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Laboratory for Percutaneous Surgery (PerkLab)
  Queen's University, Kingston, ON, Canada. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Csaba Pinter, PerkLab, Queen's University
  and was supported through the Applied Cancer Research Unit program of Cancer Care
  Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care

==============================================================================*/

// DICOMLib includes
#include "vtkSlicerDICOMLoadable.h"

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkObjectFactory.h>

//------------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerDICOMLoadable);

//----------------------------------------------------------------------------
vtkSlicerDICOMLoadable::vtkSlicerDICOMLoadable()
{
  this->Name = NULL;
  this->SetName("Unknown");
  this->Tooltip = NULL;
  this->SetTooltip("No further information available");
  this->Warning = NULL;
  this->Selected = false;
  this->Confidence = 0.5;

  this->Files = NULL;
  vtkSmartPointer<vtkStringArray> files = vtkSmartPointer<vtkStringArray>::New();
  this->SetFiles(files);
}

//----------------------------------------------------------------------------
vtkSlicerDICOMLoadable::~vtkSlicerDICOMLoadable()
{
  this->SetFiles(NULL);
}

//----------------------------------------------------------------------------
void vtkSlicerDICOMLoadable::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Name:   " << (this->Name?this->Name:"NULL") << "\n";
  os << indent << "Tooltip:   " << (this->Tooltip?this->Tooltip:"NULL") << "\n";
  os << indent << "Warning:   " << (this->Warning?this->Warning:"NULL") << "\n";
  os << indent << "Files:   " << (this->Files?"":"NULL") << "\n";
  if (this->Files)
    {
    for (int fileIndex=0; fileIndex<this->Files->GetNumberOfValues(); ++fileIndex)
      {
      os << indent << "  " << this->Files->GetValue(fileIndex) << "\n";
      }
    }
  os << indent << "Selected:   " << (this->Selected?"true":"false") << "\n";
  os << indent << "Confidence:   " << this->Confidence << "\n";
}

//----------------------------------------------------------------------------
void vtkSlicerDICOMLoadable::AddFile(const char* file)
{
  this->Files->InsertNextValue(file);
}
