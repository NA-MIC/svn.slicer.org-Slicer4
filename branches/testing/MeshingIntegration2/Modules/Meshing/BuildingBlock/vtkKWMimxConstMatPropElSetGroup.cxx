/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxConstMatPropElSetGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.8 $

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

#include "vtkKWMimxConstMatPropElSetGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWMimxMainMenuGroup.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkMimxMeshActor.h"
#include "vtkUnstructuredGrid.h"

#include "vtkKWApplication.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWEntry.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWOptions.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWPushButton.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkFieldData.h"
#include "vtkFloatArray.h"
#include "vtkCellData.h"
#include "vtkIntArray.h"
#include "vtkStringArray.h"
#include "vtkDoubleArray.h"
#include "vtkKWMessageDialog.h"
#include "vtkKWCheckButton.h"
#include "vtkKWCheckButtonWithLabel.h"


#include "vtkRenderer.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxConstMatPropElSetGroup);
vtkCxxRevisionMacro(vtkKWMimxConstMatPropElSetGroup, "$Revision: 1.8 $");

//----------------------------------------------------------------------------
vtkKWMimxConstMatPropElSetGroup::vtkKWMimxConstMatPropElSetGroup()
{
  this->ObjectListComboBox = NULL;
  this->ElementSetComboBox = NULL;
  this->YoungsModulusEntry = NULL;
  this->PoissonsRatioEntry = NULL;
  this->ViewFrame = NULL;
  this->ViewPropertyButton = NULL;
  this->ViewLegendButton = NULL;
  this->ClippingPlaneMenuButton = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxConstMatPropElSetGroup::~vtkKWMimxConstMatPropElSetGroup()
{
  if(this->ObjectListComboBox)
     this->ObjectListComboBox->Delete();
  if(this->ElementSetComboBox)
          this->ElementSetComboBox->Delete();
  if(this->YoungsModulusEntry)
          this->YoungsModulusEntry->Delete();
  if(this->PoissonsRatioEntry)
          this->PoissonsRatioEntry->Delete();
        if(this->ViewFrame)
          this->ViewFrame->Delete();  
        if(this->ViewPropertyButton)
          this->ViewPropertyButton->Delete();
  if(this->ViewLegendButton)
          this->ViewLegendButton->Delete();
  if(this->ClippingPlaneMenuButton)
          this->ClippingPlaneMenuButton->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  this->Superclass::CreateWidget();
  if(!this->ObjectListComboBox) 
  {
     this->ObjectListComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Assign Material Properties (Constant)");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  ObjectListComboBox->SetParent(this->MainFrame->GetFrame());
  ObjectListComboBox->Create();
  ObjectListComboBox->SetLabelText("Mesh : ");
  ObjectListComboBox->SetLabelWidth(15);
  ObjectListComboBox->GetWidget()->ReadOnlyOn();
  ObjectListComboBox->GetWidget()->SetCommand(this, "SelectionChangedCallback");
  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ObjectListComboBox->GetWidgetName());

  // for element set listing
  if(!this->ElementSetComboBox) 
  {
          this->ElementSetComboBox = vtkKWComboBoxWithLabel::New();
  }
  ElementSetComboBox->SetParent(this->MainFrame->GetFrame());
  ElementSetComboBox->Create();
  ElementSetComboBox->SetLabelText("Element Set : ");
  ElementSetComboBox->SetLabelWidth(15);
  ElementSetComboBox->GetWidget()->ReadOnlyOn();
  this->ElementSetComboBox->GetWidget()->SetCommand(this, "ElementSetChangedCallback");
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          ElementSetComboBox->GetWidgetName());

  //Young's modulus
  if (!this->YoungsModulusEntry)
          this->YoungsModulusEntry = vtkKWEntryWithLabel::New();

  this->YoungsModulusEntry->SetParent(this->MainFrame->GetFrame());
  this->YoungsModulusEntry->Create();
  //this->YoungsModulusEntry->SetWidth(10);
  this->YoungsModulusEntry->SetLabelWidth(15);
  this->YoungsModulusEntry->SetLabelText("Young's Modulus : ");
//  this->YoungsModulusEntry->GetWidget()->SetCommand(this, "RadiusChangeCallback");
  this->YoungsModulusEntry->GetWidget()->SetRestrictValueToDouble();

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->YoungsModulusEntry->GetWidgetName());

  // Poisson's ratio
  if (!this->PoissonsRatioEntry)
          this->PoissonsRatioEntry = vtkKWEntryWithLabel::New();

  this->PoissonsRatioEntry->SetParent(this->MainFrame->GetFrame());
  this->PoissonsRatioEntry->Create();
  //this->PoissonsRatioEntry->SetWidth(4);
  this->PoissonsRatioEntry->SetLabelWidth(15);
  this->PoissonsRatioEntry->SetLabelText("Poisson's Ratio : ");
  //  this->YoungsModulusEntry->GetWidget()->SetCommand(this, "RadiusChangeCallback");
  this->PoissonsRatioEntry->GetWidget()->SetRestrictValueToDouble();

  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          this->PoissonsRatioEntry->GetWidgetName());

  if (!this->ViewFrame)
    this->ViewFrame = vtkKWFrameWithLabel::New();
  this->ViewFrame->SetParent( this->MainFrame->GetFrame() );
  this->ViewFrame->Create();
  this->ViewFrame->SetLabelText("View");
  this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2",
              this->ViewFrame->GetWidgetName() );    
  this->ViewFrame->CollapseFrame();
  
  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "ConstMatPropElSetApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());

  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "ConstMatPropElSetCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());
    
  
    
  if (!this->ViewPropertyButton)
    this->ViewPropertyButton = vtkKWCheckButtonWithLabel::New();
  this->ViewPropertyButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewPropertyButton->Create();
  this->ViewPropertyButton->GetWidget()->SetCommand(this, "ViewMaterialPropertyCallback");
  this->ViewPropertyButton->GetWidget()->SetText("View Material Properties");
  this->ViewPropertyButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
        this->ViewPropertyButton->GetWidgetName());

  if (!this->ViewLegendButton)
    this->ViewLegendButton = vtkKWCheckButtonWithLabel::New();
  this->ViewLegendButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewLegendButton->Create();
  this->ViewLegendButton->GetWidget()->SetCommand(this, "ViewPropertyLegendCallback");
  this->ViewLegendButton->GetWidget()->SetText("View Legend");
  this->ViewLegendButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
        this->ViewLegendButton->GetWidgetName());
  
  if(!this->ClippingPlaneMenuButton)    
                this->ClippingPlaneMenuButton = vtkKWMenuButtonWithLabel::New();
        this->ClippingPlaneMenuButton->SetParent(this->ViewFrame->GetFrame());
        this->ClippingPlaneMenuButton->Create();
        this->ClippingPlaneMenuButton->SetBorderWidth(0);
        this->ClippingPlaneMenuButton->SetReliefToGroove();
        this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 0 );
        this->ClippingPlaneMenuButton->SetLabelText("Clipping Plane :");
        this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
                this->ClippingPlaneMenuButton->GetWidgetName());
        this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Off",this, "ClippingPlaneCallback 1");
        this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "On",this, "ClippingPlaneCallback 2");
  this->ClippingPlaneMenuButton->GetWidget()->GetMenu()->AddRadioButton(
                "Invert",this, "ClippingPlaneCallback 3");
        this->ClippingPlaneMenuButton->GetWidget()->SetValue("Off");
          

}
//----------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxConstMatPropElSetGroup::ConstMatPropElSetApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();
        if(!strcmp(this->ObjectListComboBox->GetWidget()->GetValue(),""))
        {
                callback->ErrorMessage("FE Mesh selection required");
                return 0;
        }

  vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
  const char *name = combobox->GetValue();
  strcpy(this->meshName, name);
  
        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid FE Mesh");
                combobox->SetValue("");
                return 0;
        }
        
        float youngsmodulus = this->YoungsModulusEntry->GetWidget()->GetValueAsDouble();
        float poissonsratio = this->PoissonsRatioEntry->GetWidget()->GetValueAsDouble();

        if(youngsmodulus < 0)
        {
                callback->ErrorMessage("Young's Modulus cannot be < 0");
                return 0;
        }

        if(poissonsratio < -1.0)
        {
                callback->ErrorMessage("poissons ratio value should be >= -1.0");
                return 0;
        }
        
        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(name)));
  vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
          this->FEMeshList->GetItem(combobox->GetValueIndex(name)))->GetDataSet();
        
  const char *elementsetname = this->ElementSetComboBox->GetWidget()->GetValue();
  strcpy(this->elementSetName, elementsetname);
  
  if(!strcmp(elementsetname,""))
  {
          callback->ErrorMessage("Choose valid element set name");
          return 0;
  }

  char imagebased[256];
  strcpy(imagebased, elementsetname);
  strcat(imagebased, "_Image_Based_Material_Property");

  vtkDoubleArray *matarray = vtkDoubleArray::SafeDownCast(
          ugrid->GetCellData()->GetArray(imagebased));
  if(matarray)
  {
          vtkKWMessageDialog *Dialog = vtkKWMessageDialog::New();
          Dialog->SetStyleToYesNo();
          Dialog->SetApplication(this->GetApplication());
          Dialog->Create();
          Dialog->SetTitle("Your attention please!");
          Dialog->SetText("Image based material property already exists. Would you like to overwrite ?");
          Dialog->Invoke();
          if(Dialog->GetStatus() == vtkKWMessageDialog::StatusCanceled)
          {
                  meshActor->StoreConstantPoissonsRatio(elementsetname, poissonsratio);
                  Dialog->Delete();
                  this->ViewPropertyButton->GetWidget()->SetEnabled( 1 );
                  this->ViewLegendButton->GetWidget()->SetEnabled( 1 );
                  this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 1 );
                  this->GetMimxMainWindow()->SetStatusText("Assigned Material Properties");       
                  return 1;
          }
          else
          {
                  Dialog->Delete();
          }
  }
  
  meshActor->StoreConstantMaterialProperty(elementsetname, youngsmodulus);
  meshActor->StoreConstantPoissonsRatio(elementsetname, poissonsratio);
  
  this->ViewPropertyButton->GetWidget()->SetEnabled( 1 );
  this->ViewLegendButton->GetWidget()->SetEnabled( 1 );
  this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 1 );
  this->GetMimxMainWindow()->SetStatusText("Assigned Material Properties");       

  return 1;
}
//----------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::ConstMatPropElSetCancelCallback()
{
//  this->MainFrame->UnpackChildren();
  this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
  this->MenuGroup->SetMenuButtonsEnabled(1);
    this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//-----------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::UpdateObjectLists()
{
        this->ObjectListComboBox->GetWidget()->DeleteAllValues();
        
        int defaultItem = -1;
        for (int i = 0; i < this->FEMeshList->GetNumberOfItems(); i++)
        {
                ObjectListComboBox->GetWidget()->AddValue(
                        this->FEMeshList->GetItem(i)->GetFileName());
          int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                                         this->FEMeshList->GetItem(i)->GetActor());
    if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }                               
        }
        if (defaultItem != -1)
  {
    ObjectListComboBox->GetWidget()->SetValue(
          this->FEMeshList->GetItem(defaultItem)->GetFileName());
  }
        this->SelectionChangedCallback(ObjectListComboBox->GetWidget()->GetValue());
}
//--------------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::SelectionChangedCallback(const char *Selection)
{
        if(!strcmp(Selection,""))
        {
                return;
        }
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(Selection)))->GetDataSet();

        // populate the element set list
        this->ElementSetComboBox->GetWidget()->DeleteAllValues();
        int i;
        vtkStringArray *strarray = vtkStringArray::SafeDownCast(
                ugrid->GetFieldData()->GetAbstractArray("Element_Set_Names"));

        int numarrrays = strarray->GetNumberOfValues();

        for (i=0; i<numarrrays; i++)
        {
                this->ElementSetComboBox->GetWidget()->AddValue(
                        strarray->GetValue(i));
        }
        this->ElementSetComboBox->GetWidget()->SetValue( strarray->GetValue(0) );
}
//-------------------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::ElementSetChangedCallback(const char *Selection)
{
        if(!strcmp(Selection,""))
        {
                return;
        }
        // get the femesh
        vtkKWComboBox *combobox = this->ObjectListComboBox->GetWidget();
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(
                ObjectListComboBox->GetWidget()->GetValue())))->GetDataSet();

        char young[256];
        strcpy(young, Selection);
        strcat(young, "_Constant_Youngs_Modulus");

        char poisson[256];
        strcpy(poisson, Selection);
        strcat(poisson, "_Constant_Poissons_Ratio");

        // check if the field data exists.
        vtkFloatArray *Earray = vtkFloatArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(young));        

        vtkFloatArray *Nuarray = vtkFloatArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(poisson));      

        float youngsmodulus = -1.0;
        float poissonsratio = -1.0;
        if(Earray)      youngsmodulus = Earray->GetValue(0);
        if(Nuarray)     poissonsratio = Nuarray->GetValue(0);
        // youngs modulus and poissons ratio data exists
        this->YoungsModulusEntry->GetWidget()->SetValueAsDouble(youngsmodulus);
        this->PoissonsRatioEntry->GetWidget()->SetValueAsDouble(poissonsratio);
}

//-------------------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::ViewMaterialPropertyCallback( int mode )
{
/*
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
           
  if ( mode )
  {
    meshActor->SetMeshScalarName( metricFieldName );
    meshActor->SetMeshScalarVisibility(true);
     this->GetMimxMainWindow()->GetRenderWidget()->Render();
  }
  else
  {
    meshActor->SetMeshScalarVisibility(false);
    this->ViewQualityLegendCallback(0);
    this->ViewLegendButton->GetWidget()->SetSelectedState( 0 );
    this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
  }
*/  
}

//-------------------------------------------------------------------------------------
void vtkKWMimxConstMatPropElSetGroup::ViewPropertyLegendCallback( int mode )
{
/*
  std::cout << "In ViewQualityLegendCallback - Mode: " << mode  << std::endl;
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
           
  if ( mode )
  {
    meshActor->SetMeshLegendVisibility(true);
    this->GetMimxMainWindow()->GetRenderWidget()->Render();
  }
  else
  {
    meshActor->SetMeshLegendVisibility(false);
    this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
  }
*/
}

//-------------------------------------------------------------------------------------
int vtkKWMimxConstMatPropElSetGroup::ClippingPlaneCallback( int mode )
{
/*
  vtkKWComboBox *combobox = this->MeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( this->meshName )));
           
  if (mode == 1)
  {
    meshActor->DisableMeshCuttingPlane();
  }
  else if (mode == 2)
  {
    meshActor->EnableMeshCuttingPlane();
    meshActor->SetInvertCuttingPlane( false );
  }
  else
  {
    meshActor->EnableMeshCuttingPlane();
    meshActor->SetInvertCuttingPlane( true );  
  }
  this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
*/
        return 1;
}

//----------------------------------------------------------------------------------------
