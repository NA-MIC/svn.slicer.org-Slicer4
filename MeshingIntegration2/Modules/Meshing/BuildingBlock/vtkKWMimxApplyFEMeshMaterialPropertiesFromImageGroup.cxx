/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup.cxx,v $
Language:  C++
Date:      $Date: 2008/05/05 19:30:08 $
Version:   $Revision: 1.25 $

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

#include "vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxMainNotebook.h"

#include "vtkLinkedListWrapper.h"

#include "vtkActor.h"
#include "vtkCellData.h"
#include "vtkIntArray.h"
#include "vtkMimxSurfacePolyDataActor.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkMimxUnstructuredGridActor.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMimxImageActor.h"
#include "vtkMimxApplyImageBasedMaterialProperties.h"
#include "vtkRenderer.h"

#include "vtkKWApplication.h"
#include "vtkKWFileBrowserDialog.h"
#include "vtkKWEvent.h"
#include "vtkKWFrame.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButton.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWNotebook.h"
#include "vtkKWOptions.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWTkUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkKWComboBoxWithLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWPushButton.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include "vtkStringArray.h"
#include "vtkFloatArray.h"
#include "vtkKWMessageDialog.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkMimxMeshActor.h"
#include "vtkKWCheckButtonWithLabel.h"
#include "vtkKWCheckButton.h"

#include "itkImage.h"

#include <vtksys/stl/list>
#include <vtksys/stl/algorithm>

// define the option types
#define VTK_KW_OPTION_NONE         0
#define VTK_KW_OPTION_LOAD                 1

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup);
vtkCxxRevisionMacro(vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup, "$Revision: 1.25 $");

//----------------------------------------------------------------------------
vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup()
{
  this->FEMeshListComboBox = NULL;
  this->ImageListComboBox = NULL;
  this->ElementSetComboBox = NULL;
  this->PoissonsRatioEntry = NULL;
  
  this->ViewFrame = NULL;
  this->ViewPropertyButton = NULL;
  this->ViewLegendButton = NULL;
  this->ClippingPlaneMenuButton = NULL;
}

//----------------------------------------------------------------------------
vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::~vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup()
{
  if(this->FEMeshListComboBox)
     this->FEMeshListComboBox->Delete();
  if(this->ImageListComboBox)
    this->ImageListComboBox->Delete();
  if(this->ElementSetComboBox)
          this->ElementSetComboBox->Delete();
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
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::CreateWidget()
{
  if(this->IsCreated())
  {
    vtkErrorMacro("class already created");
    return;
  }

  this->Superclass::CreateWidget();
  if(!this->FEMeshListComboBox) 
  {
     this->FEMeshListComboBox = vtkKWComboBoxWithLabel::New();
  }
  if(!this->ImageListComboBox)  
  {
    this->ImageListComboBox = vtkKWComboBoxWithLabel::New();
  }
  this->MainFrame->SetParent(this->GetParent());
  this->MainFrame->Create();
  this->MainFrame->SetLabelText("Assign Material Properties (Image)");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand n -padx 2 -pady 0 -fill x", 
    this->MainFrame->GetWidgetName());

  FEMeshListComboBox->SetParent(this->MainFrame->GetFrame());
  FEMeshListComboBox->Create();
  this->FEMeshListComboBox->GetWidget()->SetCommand(
          this, "FEMeshSelectionChangedCallback");
  FEMeshListComboBox->SetLabelText("Mesh : ");
  FEMeshListComboBox->SetLabelWidth( 15 );
  FEMeshListComboBox->GetWidget()->ReadOnlyOn();
  //FEMeshListComboBox->GetWidget()->SetBalloonHelpString("Surface onto which the resulting FE Mesh projected");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    FEMeshListComboBox->GetWidgetName());

  ImageListComboBox->SetParent(this->MainFrame->GetFrame());
  ImageListComboBox->Create();
  ImageListComboBox->SetLabelText("Image : ");
  ImageListComboBox->SetLabelWidth(15);
  ImageListComboBox->GetWidget()->ReadOnlyOn();
//  ImageListComboBox->GetWidget()->SetBalloonHelpString("Building Block from which F E Mesh is generated");

  this->GetApplication()->Script(
    "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
    ImageListComboBox->GetWidgetName());

  if(!this->ElementSetComboBox) 
  {
          this->ElementSetComboBox = vtkKWComboBoxWithLabel::New();
  }
  ElementSetComboBox->SetParent(this->MainFrame->GetFrame());
  ElementSetComboBox->Create();
  ElementSetComboBox->GetWidget()->SetCommand(this, "ElementSetChangedCallback");
  ElementSetComboBox->SetLabelText("Element Set : ");
  ElementSetComboBox->SetLabelWidth(15);
  ElementSetComboBox->GetWidget()->ReadOnlyOn();
  this->GetApplication()->Script(
          "pack %s -side top -anchor nw -expand y -padx 2 -pady 6 -fill x", 
          ElementSetComboBox->GetWidgetName());

  // Poisson's ratio
  if (!this->PoissonsRatioEntry)
          this->PoissonsRatioEntry = vtkKWEntryWithLabel::New();

  this->PoissonsRatioEntry->SetParent(this->MainFrame->GetFrame());
  this->PoissonsRatioEntry->Create();
  //this->PoissonsRatioEntry->SetWidth(4);
  this->PoissonsRatioEntry->SetLabelText("Poisson's Ratio : ");
  this->PoissonsRatioEntry->SetLabelWidth(15);
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
  
  if (!this->ViewPropertyButton)
    this->ViewPropertyButton = vtkKWCheckButtonWithLabel::New();
  this->ViewPropertyButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewPropertyButton->Create();
  this->ViewPropertyButton->GetWidget()->SetCommand(this, "ViewMaterialPropertyCallback");
  this->ViewPropertyButton->GetWidget()->SetText("View Material Properties");
//  this->ViewPropertyButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
        this->ViewPropertyButton->GetWidgetName());

  if (!this->ViewLegendButton)
    this->ViewLegendButton = vtkKWCheckButtonWithLabel::New();
  this->ViewLegendButton->SetParent(this->ViewFrame->GetFrame());
  this->ViewLegendButton->Create();
  this->ViewLegendButton->GetWidget()->SetCommand(this, "ViewPropertyLegendCallback");
  this->ViewLegendButton->GetWidget()->SetText("View Legend");
 // this->ViewLegendButton->GetWidget()->SetEnabled( 0 );
  this->GetApplication()->Script(
        "pack %s -side top -anchor nw -expand n -fill x -padx 2 -pady 2", 
        this->ViewLegendButton->GetWidgetName());
  
  if(!this->ClippingPlaneMenuButton)    
                this->ClippingPlaneMenuButton = vtkKWMenuButtonWithLabel::New();
        this->ClippingPlaneMenuButton->SetParent(this->ViewFrame->GetFrame());
        this->ClippingPlaneMenuButton->Create();
        this->ClippingPlaneMenuButton->SetBorderWidth(0);
        this->ClippingPlaneMenuButton->SetReliefToGroove();
//      this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 0 );
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
        
        
  this->ApplyButton->SetParent(this->MainFrame->GetFrame());
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Apply");
  this->ApplyButton->SetCommand(this, "ApplyFEMeshMaterialPropertiesFromImageApplyCallback");
  this->GetApplication()->Script(
          "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
          this->ApplyButton->GetWidgetName());
/*
  this->DoneButton->SetParent(this->MainFrame->GetFrame());
  this->DoneButton->Create();
  this->DoneButton->SetText("Done");
  this->DoneButton->SetCommand(this, "ApplyFEMeshMaterialPropertiesFromImageDoneCallback");
  this->GetApplication()->Script(
    "pack %s -side left -anchor nw -expand y -padx 20 -pady 6", 
    this->DoneButton->GetWidgetName());
*/
  this->CancelButton->SetParent(this->MainFrame->GetFrame());
  this->CancelButton->Create();
  this->CancelButton->SetText("Cancel");
  this->CancelButton->SetCommand(this, "ApplyFEMeshMaterialPropertiesFromImageCancelCallback");
  this->GetApplication()->Script(
    "pack %s -side right -anchor ne -expand y -padx 20 -pady 6", 
    this->CancelButton->GetWidgetName());

}
//----------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::Update()
{
        this->UpdateEnableState();
}
//---------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::UpdateEnableState()
{
        this->UpdateObjectLists();
        this->Superclass::UpdateEnableState();
}
//----------------------------------------------------------------------------
int vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::
        ApplyFEMeshMaterialPropertiesFromImageApplyCallback()
{
        vtkMimxErrorCallback *callback = this->GetMimxMainWindow()->GetErrorCallback();

  if(!strcmp(this->FEMeshListComboBox->GetWidget()->GetValue(),""))
  {
        callback->ErrorMessage("FE Mesh not chosen");
        return 0;
  }
    if(!strcmp(this->ImageListComboBox->GetWidget()->GetValue(),""))
  {
          callback->ErrorMessage("Image not chosen");
          return 0;
        }
    vtkKWComboBox *combobox = this->FEMeshListComboBox->GetWidget();
    const char *name = combobox->GetValue();
  strcpy(this->meshName, name);
        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid FE mesh");
                combobox->SetValue("");
                return 0;
        }

        const char *elementsetname = this->ElementSetComboBox->GetWidget()->GetValue();
        if(!strcmp(elementsetname,""))
        {
                callback->ErrorMessage("Choose valid element set");
                return 0;
        }
        strcpy(this->elementSetName, elementsetname);

        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex(name)));

        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(this->FEMeshList
                        ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

        typedef itk::Image<signed short, 3>  ImageType;
        ImageType::Pointer itkimage;

        combobox = this->ImageListComboBox->GetWidget();
        name = combobox->GetValue();

        num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                callback->ErrorMessage("Choose valid image");
                combobox->SetValue("");
                return 0;
        }

        float poissonsratio = this->PoissonsRatioEntry->GetWidget()->GetValueAsDouble();
        if(poissonsratio < -1.0 || poissonsratio > 0.5)
        {
                callback->ErrorMessage("Poisson's ratio should be beetween -1.0 and 0.5");
                return 0;
        }
        itkimage = vtkMimxImageActor::SafeDownCast(
                this->ImageList->GetItem(combobox->GetValueIndex(name)))->GetITKImage();

        char young[256];
        strcpy(young, elementsetname);
        strcat(young, "_Constant_Youngs_Modulus");

        vtkFloatArray *youngarray = vtkFloatArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(young));

        if(youngarray)
        {
                vtkKWMessageDialog *Dialog = vtkKWMessageDialog::New();
                Dialog->SetStyleToYesNo();
                Dialog->SetApplication(this->GetApplication());
                Dialog->Create();
                Dialog->SetTitle("Your attention please!");
                Dialog->SetText("Constant material property already exists for this data set. Would you like to overwrite ?");
                Dialog->Invoke();
                if(Dialog->GetStatus() == vtkKWMessageDialog::StatusCanceled)
                {
                        Dialog->Delete();
                        return 1;
                }
                Dialog->Delete();
        }

        vtkMimxApplyImageBasedMaterialProperties *applymatprops = 
                vtkMimxApplyImageBasedMaterialProperties::New();
        applymatprops->SetInput(ugrid);
        applymatprops->SetITKImage(itkimage);
        callback->SetState(0);
        applymatprops->AddObserver(vtkCommand::ErrorEvent, callback, 1.0);
        applymatprops->SetElementSetName(
                this->ElementSetComboBox->GetWidget()->GetValue());
        applymatprops->Update();
        if (!callback->GetState())
        {
                ugrid->Initialize();
                ugrid->DeepCopy(applymatprops->GetOutput());
                ugrid->GetFieldData()->RemoveArray(young);
                applymatprops->RemoveObserver(callback);
                applymatprops->Delete();
                meshActor->StoreImageBasedMaterialProperty(
                        this->ElementSetComboBox->GetWidget()->GetValue());
                meshActor->StoreConstantPoissonsRatio(
                        this->ElementSetComboBox->GetWidget()->GetValue(), poissonsratio);
                this->ViewPropertyButton->GetWidget()->SetEnabled( 1 );
                this->ViewLegendButton->GetWidget()->SetEnabled( 1 );
                this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 1 );
                return 1;
        }
        applymatprops->RemoveObserver(callback);
        applymatprops->Delete();
        this->ViewPropertyButton->GetWidget()->SetEnabled( 0 );
        this->ViewLegendButton->GetWidget()->SetEnabled( 0 );
        this->ClippingPlaneMenuButton->GetWidget()->SetEnabled( 0 );

  this->GetMimxMainWindow()->SetStatusText("Assigned Material Properties");
  
        return 0;
}
//----------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
//----------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::ApplyFEMeshMaterialPropertiesFromImageCancelCallback()
{
        this->GetApplication()->Script("pack forget %s", this->MainFrame->GetWidgetName());
        this->MenuGroup->SetMenuButtonsEnabled(1);
          this->GetMimxMainWindow()->GetMainUserInterfacePanel()->GetMimxMainNotebook()->SetEnabled(1);
}
//------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::UpdateObjectLists()
{
  this->FEMeshListComboBox->GetWidget()->DeleteAllValues();
  
  int defaultItem = -1;
  for (int i = 0; i < this->FEMeshList->GetNumberOfItems(); i++)
  {
          FEMeshListComboBox->GetWidget()->AddValue(
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
    FEMeshListComboBox->GetWidget()->SetValue(
          this->FEMeshList->GetItem(defaultItem)->GetFileName());
    /* Insert the Element Set Name */
    
  }
  
  this->ImageListComboBox->GetWidget()->DeleteAllValues();
  
  defaultItem = -1;
  for (int i = 0; i < this->ImageList->GetNumberOfItems(); i++)
  {
          ImageListComboBox->GetWidget()->AddValue(
                  this->ImageList->GetItem(i)->GetFileName());
                int viewedItem = this->GetMimxMainWindow()->GetRenderWidget()->GetRenderer()->HasViewProp(
                        this->ImageList->GetItem(i)->GetActor());
    if ( (defaultItem == -1) && ( viewedItem ) )
                {
                  defaultItem = i;
                }
  }
  
  if (defaultItem != -1)
  {
    ImageListComboBox->GetWidget()->SetValue(
          this->ImageList->GetItem(defaultItem)->GetFileName());
  }
  this->FEMeshSelectionChangedCallback(NULL);
}
//------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::
        ApplyFEMeshMaterialPropertiesFromImageDoneCallback()
{
        if(this->ApplyFEMeshMaterialPropertiesFromImageApplyCallback())
                this->ApplyFEMeshMaterialPropertiesFromImageCancelCallback();
}
//---------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::
        FEMeshSelectionChangedCallback(const char *Selection)
{
        if(!this->FEMeshList->GetNumberOfItems())
        {
                return;
        }

        const char *selection = this->FEMeshListComboBox->GetWidget()->GetValue();
        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(this->FEMeshList->
                GetItem(FEMeshListComboBox->GetWidget()->GetValueIndex(selection)))->GetDataSet();

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
//-------------------------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::
        ElementSetChangedCallback(const char *Selection)
{
        if(!strcmp(Selection,""))
        {
                return;
        }

        if(!strcmp(this->FEMeshListComboBox->GetWidget()->GetValue(),""))
        {
                return;
        }

        char poisson[256];
        strcpy(poisson, Selection);
        strcat(poisson, "_Constant_Poissons_Ratio");

        vtkKWComboBox *combobox = this->FEMeshListComboBox->GetWidget();
        const char *name = combobox->GetValue();
        int num = combobox->GetValueIndex(name);
        if(num < 0 || num > combobox->GetNumberOfValues()-1)
        {
                return;
        }

        vtkUnstructuredGrid *ugrid = vtkMimxMeshActor::SafeDownCast(this->FEMeshList
                ->GetItem(combobox->GetValueIndex(name)))->GetDataSet();

        vtkFloatArray *Nuarray = vtkFloatArray::SafeDownCast(
                ugrid->GetFieldData()->GetArray(poisson));      

        float poissonsratio = -1.0;
        if (Nuarray)
        {
                poissonsratio = Nuarray->GetValue(0);
        }
        this->PoissonsRatioEntry->GetWidget()->SetValueAsDouble(poissonsratio);
}
//-------------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::ViewMaterialPropertyCallback( int mode )
{
  vtkKWComboBox *combobox = this->FEMeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( 
                   this->FEMeshListComboBox->GetWidget()->GetValue())));         
  if ( mode )
  {
    char scalarName[256];
    strcpy(scalarName, this->ElementSetComboBox->GetWidget()->GetValue());
          strcat(scalarName, "_Image_Based_Material_Property");
    meshActor->SetElementSetScalarName(this->ElementSetComboBox->GetWidget()->GetValue(), scalarName);
    meshActor->SetElementSetScalarVisibility(this->ElementSetComboBox->GetWidget()->GetValue(), true);
    this->GetMimxMainWindow()->GetRenderWidget()->Render();
  }
  else
  {
    meshActor->SetElementSetScalarVisibility(this->ElementSetComboBox->GetWidget()->GetValue(), false);
    this->ViewPropertyLegendCallback(0);
    this->ViewLegendButton->GetWidget()->SetSelectedState( 0 );
    this->GetMimxMainWindow()->GetRenderWidget()->Render(); 
  }
}

//-------------------------------------------------------------------------------------
void vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::ViewPropertyLegendCallback( int mode )
{
  vtkKWComboBox *combobox = this->FEMeshListComboBox->GetWidget();
  vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
           this->FEMeshList->GetItem(combobox->GetValueIndex( combobox->GetValue() )));
           
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
}

//-------------------------------------------------------------------------------------
int vtkKWMimxApplyFEMeshMaterialPropertiesFromImageGroup::ClippingPlaneCallback( int mode )
{
        vtkKWComboBox *combobox = this->FEMeshListComboBox->GetWidget();
        vtkMimxMeshActor *meshActor = vtkMimxMeshActor::SafeDownCast(
                this->FEMeshList->GetItem(combobox->GetValueIndex( combobox->GetValue())));

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
        return 1;
}
//-------------------------------------------------------------------------------------
