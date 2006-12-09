
#include "vtkObjectFactory.h"
#include "vtkSlicerSlicesControlIcons.h"
#include "vtkKWIcon.h"
#include "vtkKWResourceUtilities.h"


//---------------------------------------------------------------------------
vtkStandardNewMacro ( vtkSlicerSlicesControlIcons );
vtkCxxRevisionMacro ( vtkSlicerSlicesControlIcons, "$Revision: 1.0 $");

//---------------------------------------------------------------------------
vtkSlicerSlicesControlIcons::vtkSlicerSlicesControlIcons ( )
{

  this->FgIcon = vtkKWIcon::New ( );
  this->BgIcon = vtkKWIcon::New ( );
  this->ToggleFgBgIcon = vtkKWIcon::New ( );
  this->LabelOpacityIcon = vtkKWIcon::New ( );
  this->LinkControlsIcon = vtkKWIcon::New ( );
  this->UnlinkControlsIcon = vtkKWIcon::New ( );
  this->InterpolationOnIcon = vtkKWIcon::New ( );
  this->InterpolationOffIcon = vtkKWIcon::New ( );
  this->AnnotationIcon = vtkKWIcon::New ( );
  this->SpatialUnitsIcon = vtkKWIcon::New ( );
  this->CrossHairIcon = vtkKWIcon::New ( );
  this->GridIcon = vtkKWIcon::New ( );
  this->SetFgIcon = vtkKWIcon::New ( );
  this->SetBgIcon = vtkKWIcon::New ( );
  this->SetLbIcon = vtkKWIcon::New ( );
  this->SetOrIcon = vtkKWIcon::New ( );

    //--- read in image data and assign to Icons.
    this->AssignImageDataToIcons ( );

}



//---------------------------------------------------------------------------
vtkSlicerSlicesControlIcons::~vtkSlicerSlicesControlIcons ( )
{

  if ( this->FgIcon )
    {
    this->FgIcon->Delete ( );
    this->FgIcon = NULL;
    }
  if ( this->BgIcon )
    {
    this->BgIcon->Delete ( );
    this->BgIcon = NULL;
    }
  if ( this->ToggleFgBgIcon )
    {
    this->ToggleFgBgIcon->Delete ( );
    this->ToggleFgBgIcon = NULL;
    }
  if ( this->LabelOpacityIcon )
    {
    this->LabelOpacityIcon->Delete ( );
    this->LabelOpacityIcon = NULL;
    }
  if ( this->LinkControlsIcon )
    {
    this->LinkControlsIcon->Delete ( );
    this->LinkControlsIcon = NULL;
    }
  if ( this->UnlinkControlsIcon )
    {
    this->UnlinkControlsIcon->Delete ( );
    this->UnlinkControlsIcon = NULL;
    }
  if ( this->InterpolationOnIcon )
    {
    this->InterpolationOnIcon->Delete( );
    this->InterpolationOnIcon = NULL;
    }
  if ( this->InterpolationOffIcon )
    {
    this->InterpolationOffIcon->Delete ( );
    this->InterpolationOffIcon = NULL;
    }
  if ( this->AnnotationIcon )
    {
    this->AnnotationIcon->Delete ( );
    this->AnnotationIcon = NULL;
    }
  if ( this->SpatialUnitsIcon )
    {
    this->SpatialUnitsIcon->Delete ( );
    this->SpatialUnitsIcon = NULL;
    }
  if ( this->CrossHairIcon )
    {
    this->CrossHairIcon->Delete ( );
    this->CrossHairIcon = NULL;
    }
  if ( this->GridIcon )
    {
    this->GridIcon->Delete ( );
    this->GridIcon = NULL;
    }
  if ( this->SetFgIcon )
    {
    this->SetFgIcon->Delete ( );
    this->SetFgIcon = NULL;
    }
  if ( this->SetBgIcon )
    {
    this->SetBgIcon->Delete ( );
    this->SetBgIcon = NULL;
    }
  if ( this->SetLbIcon )
    {
    this->SetLbIcon->Delete ( );
    this->SetLbIcon = NULL;
    }
  if ( this->SetOrIcon )
    {
    this->SetOrIcon->Delete ( );
    this->SetOrIcon = NULL;
    }  

}


                                                              
//---------------------------------------------------------------------------
void vtkSlicerSlicesControlIcons::AssignImageDataToIcons ( ) {


    this->FgIcon->SetImage ( image_SlicesFadeToFG,
                             image_SlicesFadeToFG_width,
                             image_SlicesFadeToFG_height,
                             image_SlicesFadeToFG_pixel_size,
                             image_SlicesFadeToFG_length, 0 );
    this->BgIcon->SetImage ( image_SlicesFadeToBG,
                             image_SlicesFadeToBG_width,
                             image_SlicesFadeToBG_height,
                             image_SlicesFadeToBG_pixel_size,
                             image_SlicesFadeToBG_length, 0 );
    this->ToggleFgBgIcon->SetImage ( image_SlicesToggleFgBg,
                             image_SlicesToggleFgBg_width,
                             image_SlicesToggleFgBg_height,
                             image_SlicesToggleFgBg_pixel_size,
                             image_SlicesToggleFgBg_length, 0 );
    this->LabelOpacityIcon->SetImage ( image_SlicesLabelOpacity,
                             image_SlicesLabelOpacity_width,
                             image_SlicesLabelOpacity_height,
                             image_SlicesLabelOpacity_pixel_size,
                             image_SlicesLabelOpacity_length, 0 );
    this->LinkControlsIcon->SetImage ( image_SlicesLinkSliceControls,
                             image_SlicesLinkSliceControls_width,
                             image_SlicesLinkSliceControls_height,
                             image_SlicesLinkSliceControls_pixel_size,
                             image_SlicesLinkSliceControls_length, 0 );
    this->UnlinkControlsIcon->SetImage ( image_SlicesUnlinkSliceControls,
                             image_SlicesUnlinkSliceControls_width,
                             image_SlicesUnlinkSliceControls_height,
                             image_SlicesUnlinkSliceControls_pixel_size,
                             image_SlicesUnlinkSliceControls_length, 0 );
    this->InterpolationOnIcon->SetImage ( image_SlicesInterpolationOn,
                             image_SlicesInterpolationOn_width,
                             image_SlicesInterpolationOn_height,
                             image_SlicesInterpolationOn_pixel_size,
                             image_SlicesInterpolationOn_length, 0 );
    this->InterpolationOffIcon->SetImage ( image_SlicesInterpolationOff,
                             image_SlicesInterpolationOff_width,
                             image_SlicesInterpolationOff_height,
                             image_SlicesInterpolationOff_pixel_size,
                             image_SlicesInterpolationOff_length, 0 );
    this->AnnotationIcon->SetImage ( image_SlicesAnnotation,
                             image_SlicesAnnotation_width,
                             image_SlicesAnnotation_height,
                             image_SlicesAnnotation_pixel_size,
                             image_SlicesAnnotation_length, 0 );
    this->SpatialUnitsIcon->SetImage ( image_SlicesSpatialUnit,
                             image_SlicesSpatialUnit_width,
                             image_SlicesSpatialUnit_height,
                             image_SlicesSpatialUnit_pixel_size,
                             image_SlicesSpatialUnit_length, 0 );
    this->CrossHairIcon->SetImage ( image_SlicesCrosshair,
                             image_SlicesCrosshair_width,
                             image_SlicesCrosshair_height,
                             image_SlicesCrosshair_pixel_size,
                             image_SlicesCrosshair_length, 0 );
    this->GridIcon->SetImage ( image_SlicesGrid,
                             image_SlicesGrid_width,
                             image_SlicesGrid_height,
                             image_SlicesGrid_pixel_size,
                             image_SlicesGrid_length, 0 );
   this->SetFgIcon->SetImage ( image_SliceFG,
                             image_SliceFG_width,
                             image_SliceFG_height,
                             image_SliceFG_pixel_size,
                             image_SliceFG_length, 0 );
    this->SetBgIcon->SetImage ( image_SliceBG,
                             image_SliceBG_width,
                             image_SliceBG_height,
                             image_SliceBG_pixel_size,
                             image_SliceBG_length, 0 );
    this->SetLbIcon->SetImage ( image_SliceLB,
                             image_SliceLB_width,
                             image_SliceLB_height,
                             image_SliceLB_pixel_size,
                             image_SliceLB_length, 0 );
    this->SetOrIcon->SetImage ( image_SliceOR,
                             image_SliceOR_width,
                             image_SliceOR_height,
                             image_SliceOR_pixel_size,
                             image_SliceOR_length, 0 );    
}



//---------------------------------------------------------------------------
void vtkSlicerSlicesControlIcons::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerSlicesControlIcons: " << this->GetClassName ( ) << "\n";
    os << indent << "FgIcon: " << this->GetFgIcon() << "\n";
    os << indent << "BgIcon: " << this->GetBgIcon() << "\n";
    os << indent << "ToggleFgBgIcon: " << this->GetToggleFgBgIcon() << "\n";
    os << indent << "LabelOpacityIcon: " << this->GetLabelOpacityIcon() << "\n";
    os << indent << "LinkControlsIcon: " << this->GetLinkControlsIcon() << "\n";
    os << indent << "UnlinkControlsIcon: " << this->GetUnlinkControlsIcon() << "\n";
    os << indent << "InterpolationOnIcon: " << this->GetInterpolationOnIcon() << "\n";
    os << indent << "InterpolationOffIcon: " << this->GetInterpolationOffIcon() << "\n";
    os << indent << "AnnotationIcon: " << this->GetAnnotationIcon() << "\n";
    os << indent << "SpatialUnitsIcon: " << this->GetSpatialUnitsIcon() << "\n";
    os << indent << "CrossHairIcon: " << this->GetCrossHairIcon() << "\n";
    os << indent << "GridHairIcon: " << this->GetGridIcon() << "\n";
     os << indent << "SetFgIcon: " << this->GetSetFgIcon() << "\n";
    os << indent << "SetBgIcon: " << this->GetSetBgIcon() << "\n";
    os << indent << "SetLbIcon: " << this->GetSetLbIcon() << "\n";
    os << indent << "SetOrIcon: " << this->GetSetOrIcon() << "\n";
}
