
#include "vtkObjectFactory.h"
#include "vtkSlicerViewControlIcons.h"
#include "vtkKWIcon.h"
#include "vtkKWResourceUtilities.h"


//---------------------------------------------------------------------------
vtkStandardNewMacro ( vtkSlicerViewControlIcons );
vtkCxxRevisionMacro ( vtkSlicerViewControlIcons, "$Revision: 1.0 $");

//---------------------------------------------------------------------------
vtkSlicerViewControlIcons::vtkSlicerViewControlIcons ( )
{

    //--- "View Axis" images in the ViewControlFrame
    this->ViewAxisAIconLO = vtkKWIcon::New ( );
    this->ViewAxisAIconHI = vtkKWIcon::New ( );
    this->ViewAxisPIconLO = vtkKWIcon::New ( );
    this->ViewAxisPIconHI = vtkKWIcon::New ( );    
    this->ViewAxisRIconLO = vtkKWIcon::New ( );
    this->ViewAxisRIconHI = vtkKWIcon::New ( );    
    this->ViewAxisLIconLO = vtkKWIcon::New ( );
    this->ViewAxisLIconHI = vtkKWIcon::New ( );    
    this->ViewAxisSIconLO = vtkKWIcon::New ( );
    this->ViewAxisSIconHI = vtkKWIcon::New ( );    
    this->ViewAxisIIconLO = vtkKWIcon::New ( );
    this->ViewAxisIIconHI = vtkKWIcon::New ( );    
    this->ViewAxisTopCornerIcon = vtkKWIcon::New ( );    
    this->ViewAxisBottomCornerIcon = vtkKWIcon::New ( );    
    this->ViewAxisCenterIcon = vtkKWIcon::New ( );    

    this->SpinButtonIcon = vtkKWIcon::New ( );
    this->RockButtonIcon = vtkKWIcon::New ( );
    this->OrthoButtonIcon = vtkKWIcon::New ( );
    this->PerspectiveButtonIcon = vtkKWIcon::New ( );
    this->CenterButtonIcon = vtkKWIcon::New ( );
    this->SelectButtonIcon = vtkKWIcon::New ( );
    this->StereoButtonIcon = vtkKWIcon::New ( );
    this->LookFromButtonIcon = vtkKWIcon::New ( );
    this->RotateAroundButtonIcon = vtkKWIcon::New ( );
    
    //--- "RotateAround Axis" images in the ViewControlFrame
    this->RotateAroundAIconLO = vtkKWIcon::New ( );
    this->RotateAroundAIconHI = vtkKWIcon::New ( );
    this->RotateAroundPIconLO = vtkKWIcon::New ( );
    this->RotateAroundPIconHI = vtkKWIcon::New ( );    
    this->RotateAroundRIconLO = vtkKWIcon::New ( );
    this->RotateAroundRIconHI = vtkKWIcon::New ( );    
    this->RotateAroundLIconLO = vtkKWIcon::New ( );
    this->RotateAroundLIconHI = vtkKWIcon::New ( );    
    this->RotateAroundSIconLO = vtkKWIcon::New ( );
    this->RotateAroundSIconHI = vtkKWIcon::New ( );    
    this->RotateAroundIIconLO = vtkKWIcon::New ( );
    this->RotateAroundIIconHI = vtkKWIcon::New ( );    
    this->RotateAroundTopCornerIcon = vtkKWIcon::New ( );    
    this->RotateAroundBottomCornerIcon = vtkKWIcon::New ( );    
    this->RotateAroundMiddleIcon = vtkKWIcon::New ( );    

    //--- "Look From" images in the ViewControlFrame
    this->LookFromAIconLO = vtkKWIcon::New ( );
    this->LookFromAIconHI = vtkKWIcon::New ( );
    this->LookFromPIconLO = vtkKWIcon::New ( );
    this->LookFromPIconHI = vtkKWIcon::New ( );    
    this->LookFromRIconLO = vtkKWIcon::New ( );
    this->LookFromRIconHI = vtkKWIcon::New ( );    
    this->LookFromLIconLO = vtkKWIcon::New ( );
    this->LookFromLIconHI = vtkKWIcon::New ( );    
    this->LookFromSIconLO = vtkKWIcon::New ( );
    this->LookFromSIconHI = vtkKWIcon::New ( );    
    this->LookFromIIconLO = vtkKWIcon::New ( );
    this->LookFromIIconHI = vtkKWIcon::New ( );    
    this->LookFromTopCornerIcon = vtkKWIcon::New ( );    
    this->LookFromBottomCornerIcon = vtkKWIcon::New ( );    
    this->LookFromMiddleIcon = vtkKWIcon::New ( );    
    
    //--- "Zoom in and out icons
    this->NavZoomInIcon = vtkKWIcon::New();
    this->NavZoomOutIcon = vtkKWIcon::New();

    //--- read in image data and assign to Icons.
    this->AssignImageDataToIcons ( );

}



//---------------------------------------------------------------------------
vtkSlicerViewControlIcons::~vtkSlicerViewControlIcons ( )
{

    // view from or rotate around icons
    if ( this->ViewAxisAIconLO ) {
        this->ViewAxisAIconLO->Delete ( );
        this->ViewAxisAIconLO = NULL;
    }
    if ( this->ViewAxisAIconHI ) {
        this->ViewAxisAIconHI->Delete ( );
        this->ViewAxisAIconHI = NULL;
    }
    if ( this->ViewAxisPIconLO ) {
        this->ViewAxisPIconLO->Delete ( );
        this->ViewAxisPIconLO = NULL;
    }
    if ( this->ViewAxisPIconHI ) {
        this->ViewAxisPIconHI->Delete ( );
        this->ViewAxisPIconHI = NULL;
    }
    if ( this->ViewAxisRIconLO ) {
        this->ViewAxisRIconLO->Delete ( );
        this->ViewAxisRIconLO = NULL;
    }
    if ( this->ViewAxisRIconHI ) {
        this->ViewAxisRIconHI->Delete ( );
        this->ViewAxisRIconHI = NULL;
    }
    if ( this->ViewAxisLIconLO ) {
        this->ViewAxisLIconLO->Delete ( );
        this->ViewAxisLIconLO = NULL;
    }
    if ( this->ViewAxisLIconHI ) {
        this->ViewAxisLIconHI->Delete ( );
        this->ViewAxisLIconHI = NULL;
    }
    if ( this->ViewAxisSIconLO ) {
        this->ViewAxisSIconLO->Delete ( );
        this->ViewAxisSIconLO = NULL;
    }
    if ( this->ViewAxisSIconHI ) {
        this->ViewAxisSIconHI->Delete ( );
        this->ViewAxisSIconHI = NULL;
    }
    if ( this->ViewAxisIIconLO ) {
        this->ViewAxisIIconLO->Delete ( );
        this->ViewAxisIIconLO = NULL;
    }
    if ( this->ViewAxisIIconHI ) {
        this->ViewAxisIIconHI->Delete ( );
        this->ViewAxisIIconHI = NULL;
    }
    if ( this->ViewAxisTopCornerIcon ) {
        this->ViewAxisTopCornerIcon->Delete ( );
        this->ViewAxisTopCornerIcon = NULL;
    }
    if ( this->ViewAxisBottomCornerIcon ) {
        this->ViewAxisBottomCornerIcon->Delete ( );
        this->ViewAxisBottomCornerIcon = NULL;
    }
    if ( this->ViewAxisCenterIcon ) {
        this->ViewAxisCenterIcon->Delete ( );
        this->ViewAxisCenterIcon = NULL;
    }
    if ( this->SpinButtonIcon )
      {
      this->SpinButtonIcon->Delete ( );
      this->SpinButtonIcon = NULL;
      }
    if ( this->RockButtonIcon )
      {
      this->RockButtonIcon->Delete ( );
      this->RockButtonIcon = NULL;
      }
    if ( this->OrthoButtonIcon )
      {
      this->OrthoButtonIcon->Delete ( );
      this->OrthoButtonIcon = NULL;
      }
    if ( this->PerspectiveButtonIcon )
      {
      this->PerspectiveButtonIcon->Delete ( );
      this->PerspectiveButtonIcon = NULL;
      }
    if ( this->CenterButtonIcon )
      {
      this->CenterButtonIcon->Delete ( );
      this->CenterButtonIcon = NULL;
      }
    if ( this->SelectButtonIcon )
      {
      this->SelectButtonIcon->Delete ( );
      this->SelectButtonIcon = NULL;
      }
    if ( this->StereoButtonIcon )
      {
      this->StereoButtonIcon->Delete ( );
      this->StereoButtonIcon = NULL;
      }
    if ( this->LookFromButtonIcon )
      {
      this->LookFromButtonIcon->Delete ( );
      this->LookFromButtonIcon = NULL;
      }
    if ( this->RotateAroundButtonIcon )
      {
      this->RotateAroundButtonIcon->Delete ( );
      this->RotateAroundButtonIcon = NULL;
      }

    // rotate around images
    if ( this->RotateAroundAIconLO ) {
        this->RotateAroundAIconLO->Delete ( );
        this->RotateAroundAIconLO = NULL;
    }
    if ( this->RotateAroundAIconHI ) {
        this->RotateAroundAIconHI->Delete ( );
        this->RotateAroundAIconHI = NULL;
    }
    if ( this->RotateAroundPIconLO ) {
        this->RotateAroundPIconLO->Delete ( );
        this->RotateAroundPIconLO = NULL;
    }
    if ( this->RotateAroundPIconHI ) {
        this->RotateAroundPIconHI->Delete ( );
        this->RotateAroundPIconHI = NULL;
    }
    if ( this->RotateAroundRIconLO ) {
        this->RotateAroundRIconLO->Delete ( );
        this->RotateAroundRIconLO = NULL;
    }
    if ( this->RotateAroundRIconHI ) {
        this->RotateAroundRIconHI->Delete ( );
        this->RotateAroundRIconHI = NULL;
    }
    if ( this->RotateAroundLIconLO ) {
        this->RotateAroundLIconLO->Delete ( );
        this->RotateAroundLIconLO = NULL;
    }
    if ( this->RotateAroundLIconHI ) {
        this->RotateAroundLIconHI->Delete ( );
        this->RotateAroundLIconHI = NULL;
    }
    if ( this->RotateAroundSIconLO ) {
        this->RotateAroundSIconLO->Delete ( );
        this->RotateAroundSIconLO = NULL;
    }
    if ( this->RotateAroundSIconHI ) {
        this->RotateAroundSIconHI->Delete ( );
        this->RotateAroundSIconHI = NULL;
    }
    if ( this->RotateAroundIIconLO ) {
        this->RotateAroundIIconLO->Delete ( );
        this->RotateAroundIIconLO = NULL;
    }
    if ( this->RotateAroundIIconHI ) {
        this->RotateAroundIIconHI->Delete ( );
        this->RotateAroundIIconHI = NULL;
    }
    if ( this->RotateAroundTopCornerIcon ) {
        this->RotateAroundTopCornerIcon->Delete ( );
        this->RotateAroundTopCornerIcon = NULL;
    }
    if ( this->RotateAroundBottomCornerIcon ) {
        this->RotateAroundBottomCornerIcon->Delete ( );
        this->RotateAroundBottomCornerIcon = NULL;
    }
    if ( this->RotateAroundMiddleIcon ) {
        this->RotateAroundMiddleIcon->Delete ( );
        this->RotateAroundMiddleIcon = NULL;
    }


    // look from images.
    if ( this->LookFromAIconLO ) {
        this->LookFromAIconLO->Delete ( );
        this->LookFromAIconLO = NULL;
    }
    if ( this->LookFromAIconHI ) {
        this->LookFromAIconHI->Delete ( );
        this->LookFromAIconHI = NULL;
    }
    if ( this->LookFromPIconLO ) {
        this->LookFromPIconLO->Delete ( );
        this->LookFromPIconLO = NULL;
    }
    if ( this->LookFromPIconHI ) {
        this->LookFromPIconHI->Delete ( );
        this->LookFromPIconHI = NULL;
    }
    if ( this->LookFromRIconLO ) {
        this->LookFromRIconLO->Delete ( );
        this->LookFromRIconLO = NULL;
    }
    if ( this->LookFromRIconHI ) {
        this->LookFromRIconHI->Delete ( );
        this->LookFromRIconHI = NULL;
    }
    if ( this->LookFromLIconLO ) {
        this->LookFromLIconLO->Delete ( );
        this->LookFromLIconLO = NULL;
    }
    if ( this->LookFromLIconHI ) {
        this->LookFromLIconHI->Delete ( );
        this->LookFromLIconHI = NULL;
    }
    if ( this->LookFromSIconLO ) {
        this->LookFromSIconLO->Delete ( );
        this->LookFromSIconLO = NULL;
    }
    if ( this->LookFromSIconHI ) {
        this->LookFromSIconHI->Delete ( );
        this->LookFromSIconHI = NULL;
    }
    if ( this->LookFromIIconLO ) {
        this->LookFromIIconLO->Delete ( );
        this->LookFromIIconLO = NULL;
    }
    if ( this->LookFromIIconHI ) {
        this->LookFromIIconHI->Delete ( );
        this->LookFromIIconHI = NULL;
    }
    if ( this->LookFromTopCornerIcon ) {
        this->LookFromTopCornerIcon->Delete ( );
        this->LookFromTopCornerIcon = NULL;
    }
    if ( this->LookFromBottomCornerIcon ) {
        this->LookFromBottomCornerIcon->Delete ( );
        this->LookFromBottomCornerIcon = NULL;
    }
    if ( this->LookFromMiddleIcon ) {
        this->LookFromMiddleIcon->Delete ( );
        this->LookFromMiddleIcon = NULL;
    }

    // zoom images
    if ( this->NavZoomInIcon ) {
        this->NavZoomInIcon->Delete ( );
        this->NavZoomInIcon = NULL;
    }
    if ( this->NavZoomOutIcon ) {
        this->NavZoomOutIcon->Delete ( );
        this->NavZoomOutIcon = NULL;
    }


}


                                                              
//---------------------------------------------------------------------------
void vtkSlicerViewControlIcons::AssignImageDataToIcons ( ) {
    // Rotate around Icons (9 of these tile one image in the view control frame;
    // six of them have a "LO" and "HI" state that changes
    // on mouseover.


  this->ViewAxisAIconHI->SetImage( image_ViewAxisAHi,
                                   image_ViewAxisAHi_width,
                                   image_ViewAxisAHi_height,
                                   image_ViewAxisAHi_pixel_size,
                                   image_ViewAxisAHi_length, 0);
  this->ViewAxisAIconLO->SetImage( image_ViewAxisALo,
                                   image_ViewAxisALo_width,
                                   image_ViewAxisALo_height,
                                   image_ViewAxisALo_pixel_size,
                                   image_ViewAxisALo_length, 0);
  this->ViewAxisPIconLO->SetImage( image_ViewAxisPLo,
                                   image_ViewAxisPLo_width,
                                   image_ViewAxisPLo_height,
                                   image_ViewAxisPLo_pixel_size,
                                   image_ViewAxisPLo_length, 0);
  this->ViewAxisPIconHI->SetImage( image_ViewAxisPHi,
                                   image_ViewAxisPHi_width,
                                   image_ViewAxisPHi_height,
                                   image_ViewAxisPHi_pixel_size,
                                   image_ViewAxisPHi_length, 0);
  this->ViewAxisRIconLO->SetImage( image_ViewAxisRLo,
                                   image_ViewAxisRLo_width,
                                   image_ViewAxisRLo_height,
                                   image_ViewAxisRLo_pixel_size,
                                   image_ViewAxisRLo_length, 0);
  this->ViewAxisRIconHI->SetImage( image_ViewAxisRHi,
                                   image_ViewAxisRHi_width,
                                   image_ViewAxisRHi_height,
                                   image_ViewAxisRHi_pixel_size,
                                   image_ViewAxisRHi_length, 0);
  this->ViewAxisLIconLO->SetImage( image_ViewAxisLLo,
                                   image_ViewAxisLLo_width,
                                   image_ViewAxisLLo_height,
                                   image_ViewAxisLLo_pixel_size,
                                   image_ViewAxisLLo_length, 0);
  this->ViewAxisLIconHI->SetImage( image_ViewAxisLHi,
                                   image_ViewAxisLHi_width,
                                   image_ViewAxisLHi_height,
                                   image_ViewAxisLHi_pixel_size,
                                   image_ViewAxisLHi_length, 0);
  this->ViewAxisSIconLO->SetImage( image_ViewAxisSLo,
                                   image_ViewAxisSLo_width,
                                   image_ViewAxisSLo_height,
                                   image_ViewAxisSLo_pixel_size,
                                   image_ViewAxisSLo_length, 0);
  this->ViewAxisSIconHI->SetImage( image_ViewAxisSHi,
                                   image_ViewAxisSHi_width,
                                   image_ViewAxisSHi_height,
                                   image_ViewAxisSHi_pixel_size,
                                   image_ViewAxisSHi_length, 0);
  this->ViewAxisIIconLO->SetImage( image_ViewAxisILo,
                                   image_ViewAxisILo_width,
                                   image_ViewAxisILo_height,
                                   image_ViewAxisILo_pixel_size,
                                   image_ViewAxisILo_length, 0 );
  this->ViewAxisIIconHI->SetImage( image_ViewAxisIHi,
                                   image_ViewAxisIHi_width,
                                   image_ViewAxisIHi_height,
                                   image_ViewAxisIHi_pixel_size,
                                   image_ViewAxisIHi_length, 0);
  this->ViewAxisBottomCornerIcon->SetImage( image_ViewAxisBottomCorner,
                                            image_ViewAxisBottomCorner_width,
                                            image_ViewAxisBottomCorner_height,
                                            image_ViewAxisBottomCorner_pixel_size,
                                            image_ViewAxisBottomCorner_length, 0);
  this->ViewAxisTopCornerIcon->SetImage( image_ViewAxisTopCorner,
                                         image_ViewAxisTopCorner_width,
                                         image_ViewAxisTopCorner_height,
                                         image_ViewAxisTopCorner_pixel_size,
                                         image_ViewAxisTopCorner_length, 0);
  this->ViewAxisCenterIcon->SetImage( image_ViewAxisMiddle,
                                      image_ViewAxisMiddle_width,
                                      image_ViewAxisMiddle_height,
                                      image_ViewAxisMiddle_pixel_size,
                                      image_ViewAxisMiddle_length, 0);
  this->SpinButtonIcon->SetImage ( image_ViewSpin,
                                   image_ViewSpin_width,
                                   image_ViewSpin_height,
                                   image_ViewSpin_pixel_size,
                                   image_ViewSpin_length, 0 );                                   
  this->RockButtonIcon->SetImage ( image_ViewRock,
                                   image_ViewRock_width,
                                   image_ViewRock_height,
                                   image_ViewRock_pixel_size,
                                   image_ViewRock_length, 0 );                                   
  this->OrthoButtonIcon->SetImage ( image_ViewOrtho,
                                   image_ViewOrtho_width,
                                   image_ViewOrtho_height,
                                   image_ViewOrtho_pixel_size,
                                   image_ViewOrtho_length, 0 );                                   
  this->PerspectiveButtonIcon->SetImage ( image_ViewPerspective,
                                   image_ViewPerspective_width,
                                   image_ViewPerspective_height,
                                   image_ViewPerspective_pixel_size,
                                   image_ViewPerspective_length, 0 );                                   
  this->CenterButtonIcon->SetImage ( image_ViewCenter,
                                   image_ViewCenter_width,
                                   image_ViewCenter_height,
                                   image_ViewCenter_pixel_size,
                                   image_ViewCenter_length, 0 );                                   
  this->SelectButtonIcon->SetImage ( image_ViewSelect,
                                   image_ViewSelect_width,
                                   image_ViewSelect_height,
                                   image_ViewSelect_pixel_size,
                                   image_ViewSelect_length, 0 );                                   
  this->StereoButtonIcon->SetImage ( image_ViewStereo,
                                   image_ViewStereo_width,
                                   image_ViewStereo_height,
                                   image_ViewStereo_pixel_size,
                                   image_ViewStereo_length, 0 );                                   
  this->LookFromButtonIcon->SetImage ( image_ViewAxisLookFrom,
                                   image_ViewAxisLookFrom_width,
                                   image_ViewAxisLookFrom_height,
                                   image_ViewAxisLookFrom_pixel_size,
                                   image_ViewAxisLookFrom_length, 0 );                                   
  this->RotateAroundButtonIcon->SetImage ( image_ViewAxisRotateAround,
                                   image_ViewAxisRotateAround_width,
                                   image_ViewAxisRotateAround_height,
                                   image_ViewAxisRotateAround_pixel_size,
                                   image_ViewAxisRotateAround_length, 0 );                                   


  this->RotateAroundAIconHI->SetImage( image_RotateAroundAHI,
                                         image_RotateAroundAHI_width,
                                         image_RotateAroundAHI_height,
                                         image_RotateAroundAHI_pixel_size, 0, 0);
    this->RotateAroundAIconLO->SetImage( image_RotateAroundALO,
                                         image_RotateAroundALO_width,
                                         image_RotateAroundALO_height,
                                         image_RotateAroundALO_pixel_size, 0, 0);
    this->RotateAroundPIconLO->SetImage( image_RotateAroundPLO,
                                         image_RotateAroundPLO_width,
                                         image_RotateAroundPLO_height,
                                         image_RotateAroundPLO_pixel_size, 0, 0);
    this->RotateAroundPIconHI->SetImage( image_RotateAroundPHI,
                                          image_RotateAroundPHI_width,
                                          image_RotateAroundPHI_height,
                                          image_RotateAroundPHI_pixel_size, 0, 0);
    this->RotateAroundRIconLO->SetImage( image_RotateAroundRLO,
                                         image_RotateAroundRLO_width,
                                         image_RotateAroundRLO_height,
                                         image_RotateAroundRLO_pixel_size, 0, 0);
    this->RotateAroundRIconHI->SetImage( image_RotateAroundRHI,
                                         image_RotateAroundRHI_width,
                                         image_RotateAroundRHI_height,
                                         image_RotateAroundRHI_pixel_size, 0, 0);
    this->RotateAroundLIconLO->SetImage( image_RotateAroundLLO,
                                         image_RotateAroundLLO_width,
                                         image_RotateAroundLLO_height,
                                         image_RotateAroundLLO_pixel_size, 0, 0);
    this->RotateAroundLIconHI->SetImage( image_RotateAroundLHI,
                                         image_RotateAroundLHI_width,
                                         image_RotateAroundLHI_height,
                                         image_RotateAroundLHI_pixel_size, 0, 0);
    this->RotateAroundSIconLO->SetImage( image_RotateAroundSLO,
                                         image_RotateAroundSLO_width,
                                         image_RotateAroundSLO_height,
                                         image_RotateAroundSLO_pixel_size, 0, 0);
    this->RotateAroundSIconHI->SetImage( image_RotateAroundSHI,
                                         image_RotateAroundSHI_width,
                                         image_RotateAroundSHI_height,
                                         image_RotateAroundSHI_pixel_size, 0, 0);
    this->RotateAroundIIconLO->SetImage( image_RotateAroundILO,
                                         image_RotateAroundILO_width,
                                         image_RotateAroundILO_height,
                                         image_RotateAroundILO_pixel_size, 0, 0 );
    this->RotateAroundIIconHI->SetImage( image_RotateAroundIHI,
                                         image_RotateAroundIHI_width,
                                         image_RotateAroundIHI_height,
                                         image_RotateAroundIHI_pixel_size, 0, 0);
    this->RotateAroundBottomCornerIcon->SetImage( image_RotateAroundBottomCorner,
                                                  image_RotateAroundBottomCorner_width,
                                                  image_RotateAroundBottomCorner_height,
                                                  image_RotateAroundBottomCorner_pixel_size, 0, 0);
    this->RotateAroundTopCornerIcon->SetImage( image_RotateAroundTopCorner,
                                               image_RotateAroundTopCorner_width,
                                               image_RotateAroundTopCorner_height,
                                               image_RotateAroundTopCorner_pixel_size, 0, 0);
    this->RotateAroundMiddleIcon->SetImage( image_RotateAroundMiddle,
                                            image_RotateAroundMiddle_width,
                                            image_RotateAroundMiddle_height,
                                            image_RotateAroundMiddle_pixel_size, 0, 0);

    // Look from Icons (9 of these tile an image in the view control frame;
    // six of them have "LO" and "HI" representations
    // that change state during mouseover.
    
    this->LookFromAIconHI->SetImage( image_LookFromAHI,
                                     image_LookFromAHI_width,
                                     image_LookFromAHI_height,
                                     image_LookFromAHI_pixel_size, 0, 0);
    this->LookFromAIconLO->SetImage( image_LookFromALO,
                                     image_LookFromALO_width,
                                     image_LookFromALO_height,
                                     image_LookFromALO_pixel_size, 0, 0 );
    this->LookFromPIconLO->SetImage( image_LookFromPLO,
                                     image_LookFromPLO_width,
                                     image_LookFromPLO_height,
                                     image_LookFromPLO_pixel_size, 0, 0);
    this->LookFromPIconHI->SetImage( image_LookFromPHI,
                                     image_LookFromPHI_width,
                                     image_LookFromPHI_height,
                                     image_LookFromPHI_pixel_size, 0, 0);
    this->LookFromRIconLO->SetImage(image_LookFromRLO,
                                    image_LookFromRLO_width,
                                    image_LookFromRLO_height,
                                    image_LookFromRLO_pixel_size, 0, 0);
    this->LookFromRIconHI->SetImage( image_LookFromRHI,
                                     image_LookFromRHI_width,
                                     image_LookFromRHI_height,
                                     image_LookFromRHI_pixel_size, 0, 0);
    this->LookFromLIconLO->SetImage( image_LookFromLLO,
                                     image_LookFromLLO_width,
                                     image_LookFromLLO_height,
                                     image_LookFromLLO_pixel_size, 0, 0);
    this->LookFromLIconHI->SetImage( image_LookFromLHI,
                                     image_LookFromLHI_width,
                                     image_LookFromLHI_height,
                                     image_LookFromLHI_pixel_size, 0, 0);
    this->LookFromSIconLO->SetImage( image_LookFromSLO,
                                     image_LookFromSLO_width,
                                     image_LookFromSLO_height,
                                     image_LookFromSLO_pixel_size, 0, 0);
    this->LookFromSIconHI->SetImage( image_LookFromSHI,
                                     image_LookFromSHI_width,
                                     image_LookFromSHI_height,
                                     image_LookFromSHI_pixel_size, 0, 0);
    this->LookFromIIconLO->SetImage( image_LookFromILO,
                                     image_LookFromILO_width,
                                     image_LookFromILO_height,
                                     image_LookFromILO_pixel_size, 0, 0);
    this->LookFromIIconHI->SetImage( image_LookFromIHI,
                                     image_LookFromIHI_width,
                                     image_LookFromIHI_height,
                                     image_LookFromIHI_pixel_size, 0, 0);
    this->LookFromBottomCornerIcon->SetImage( image_LookFromBottomCorner,
                                              image_LookFromBottomCorner_width,
                                              image_LookFromBottomCorner_height,
                                              image_LookFromBottomCorner_pixel_size, 0, 0);
    this->LookFromTopCornerIcon->SetImage( image_LookFromTopCorner,
                                           image_LookFromTopCorner_width,
                                           image_LookFromTopCorner_height,
                                           image_LookFromTopCorner_pixel_size, 0, 0);
    this->LookFromMiddleIcon->SetImage( image_LookFromMiddle,
                                        image_LookFromMiddle_width,
                                        image_LookFromMiddle_height,
                                        image_LookFromMiddle_pixel_size, 0, 0);

    this->NavZoomInIcon->SetImage( image_NavZoomIn,
                                   image_NavZoomIn_width,
                                   image_NavZoomIn_height,
                                   image_NavZoomIn_pixel_size, 0, 0);
    this->NavZoomOutIcon->SetImage( image_NavZoomOut,
                                    image_NavZoomOut_width,
                                    image_NavZoomOut_height,
                                    image_NavZoomOut_pixel_size, 0, 0);
    
}



//---------------------------------------------------------------------------
void vtkSlicerViewControlIcons::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerViewControlIcons: " << this->GetClassName ( ) << "\n";

    os << indent << "ViewAxisAIconLO" << this->GetViewAxisAIconLO() << "\n";
    os << indent << "ViewAxisAIconHI" << this->GetViewAxisAIconHI () << "\n";
    os << indent << "ViewAxisPIconLO" << this->GetViewAxisPIconLO () << "\n";
    os << indent << "ViewAxisPIconHI" << this->GetViewAxisPIconHI () << "\n";
    os << indent << "ViewAxisRIconLO" << this->GetViewAxisRIconLO () << "\n";
    os << indent << "ViewAxisRIconHI" << this->GetViewAxisRIconHI () << "\n";
    os << indent << "ViewAxisLIconLO" << this->GetViewAxisLIconLO () << "\n";
    os << indent << "ViewAxisLIconHI" << this->GetViewAxisLIconHI () << "\n";
    os << indent << "ViewAxisSIconLO" << this->GetViewAxisSIconLO () << "\n";
    os << indent << "ViewAxisSIconHI" << this->GetViewAxisSIconHI () << "\n";
    os << indent << "ViewAxisIIconLO" << this->GetViewAxisIIconLO () << "\n";
    os << indent << "ViewAxisIIconHI" << this->GetViewAxisIIconHI () << "\n";
    os << indent << "ViewAxisTopCornerIcon" << this->GetViewAxisTopCornerIcon () << "\n";
    os << indent << "ViewAxisBottomCornerIcon" << this->GetViewAxisBottomCornerIcon () << "\n";
    os << indent << "ViewAxisCenterIcon" << this->GetViewAxisCenterIcon () << "\n";

    os << indent << "RotateAroundAIconLO" << this->GetRotateAroundAIconLO () << "\n";
    os << indent << "RotateAroundAIconHI" << this->GetRotateAroundAIconHI () << "\n";
    os << indent << "RotateAroundPIconLO" << this->GetRotateAroundPIconLO () << "\n";
    os << indent << "RotateAroundPIconHI" << this->GetRotateAroundPIconHI () << "\n";
    os << indent << "RotateAroundRIconLO" << this->GetRotateAroundRIconLO () << "\n";
    os << indent << "RotateAroundRIconHI" << this->GetRotateAroundRIconHI () << "\n";
    os << indent << "RotateAroundLIconLO" << this->GetRotateAroundLIconLO () << "\n";
    os << indent << "RotateAroundLIconHI" << this->GetRotateAroundLIconHI () << "\n";
    os << indent << "RotateAroundSIconLO" << this->GetRotateAroundSIconLO () << "\n";
    os << indent << "RotateAroundSIconHI" << this->GetRotateAroundSIconHI () << "\n";
    os << indent << "RotateAroundIIconLO" << this->GetRotateAroundIIconLO () << "\n";
    os << indent << "RotateAroundIIconHI" << this->GetRotateAroundIIconHI () << "\n";
    os << indent << "RotateAroundTopCornerIcon" << this->GetRotateAroundTopCornerIcon () << "\n";
    os << indent << "RotateAroundBottomCornerIcon" << this->GetRotateAroundBottomCornerIcon () << "\n";
    os << indent << "RotateAroundMiddleIcon" << this->GetRotateAroundMiddleIcon () << "\n";

    os << indent << "LookFromAIconLO" << this->GetLookFromAIconLO () << "\n";
    os << indent << "LookFromAIconHI" << this->GetLookFromAIconHI () << "\n";
    os << indent << "LookFromPIconLO" << this->GetLookFromPIconLO () << "\n";
    os << indent << "LookFromPIconHI" << this->GetLookFromPIconHI () << "\n";
    os << indent << "LookFromRIconLO" << this->GetLookFromRIconLO () << "\n";
    os << indent << "LookFromRIconHI" << this->GetLookFromRIconHI () << "\n";
    os << indent << "LookFromLIconLO" << this->GetLookFromLIconLO () << "\n";
    os << indent << "LookFromLIconHI" << this->GetLookFromLIconHI () << "\n";
    os << indent << "LookFromSIconLO" << this->GetLookFromSIconLO () << "\n";
    os << indent << "LookFromSIconHI" << this->GetLookFromSIconHI () << "\n";
    os << indent << "LookFromIIconLO" << this->GetLookFromIIconLO () << "\n";
    os << indent << "LookFromIIconHI" << this->GetLookFromIIconHI () << "\n";
    os << indent << "LookFromTopCornerIcon" << this->GetLookFromTopCornerIcon () << "\n";
    os << indent << "LookFromBottomCornerIcon" << this->GetLookFromBottomCornerIcon () << "\n";
    os << indent << "LookFromMiddleIcon" << this->GetLookFromMiddleIcon () << "\n";

}
