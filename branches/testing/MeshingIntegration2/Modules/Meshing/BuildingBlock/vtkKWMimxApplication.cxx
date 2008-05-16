/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxApplication.cxx,v $
Language:  C++
Date:      $Date: 2008/05/01 02:34:12 $
Version:   $Revision: 1.14 $

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

#include "vtkKWMimxApplication.h"

#include "vtkObjectFactory.h"
#include "vtkKWWindowBase.h"
#include "vtkSlicerTheme.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWRegistryHelper.h"
#include "vtkKWMimxMainWindow.h"
#include "vtkKWTkUtilities.h"

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkKWMimxApplication );
vtkCxxRevisionMacro(vtkKWMimxApplication, "$Revision: 1.14 $");

//----------------------------------------------------------------------------
vtkKWMimxApplication::vtkKWMimxApplication()
{
  vtkKWFrameWithLabel::SetDefaultLabelFontWeightToNormal( );
  this->SlicerTheme = vtkSlicerTheme::New ( );
  
  strcpy (this->ApplicationFontSize, "small" );
  strcpy ( this->ApplicationFontFamily, "Arial" );
    
}

vtkKWMimxApplication::~vtkKWMimxApplication ( ) 
{
  if ( this->SlicerTheme )
    {
    this->SlicerTheme->Delete ( );
    this->SlicerTheme = NULL;
    }
}

void vtkKWMimxApplication::InstallDefaultTheme ( )
{
  InstallTheme( this->SlicerTheme );
}

//---------------------------------------------------------------------------
void vtkKWMimxApplication::InstallTheme ( vtkKWTheme *theme )
{
  if ( theme != NULL ) 
  {
    if ( vtkSlicerTheme::SafeDownCast (theme) == this->SlicerTheme ) {
        this->SetTheme (this->SlicerTheme );
    } else {
        this->SetTheme ( theme );
    }
  }
}

//----------------------------------------------------------------------------
void vtkKWMimxApplication::SetApplicationFontFamily ( const char *font)
{
  
  if ( this->SlicerTheme )
    {
    this->SlicerTheme->SetFontFamily ( font );
    this->Script ( "font configure %s -family %s", this->SlicerTheme->GetApplicationFont2(), font );
    this->Script ( "font configure %s -family %s", this->SlicerTheme->GetApplicationFont1(), font );
    this->Script ( "font configure %s -family %s", this->SlicerTheme->GetApplicationFont0(), font );
    strcpy ( this->ApplicationFontFamily, font );
    }

}
//----------------------------------------------------------------------------
const char *vtkKWMimxApplication::GetApplicationFontFamily () const
{
  return this->ApplicationFontFamily;
}


//----------------------------------------------------------------------------
void vtkKWMimxApplication::SetApplicationFontSize ( const char *size)
{
  
  if (this->SlicerTheme)
    {
    vtkSlicerFont *font = this->SlicerTheme->GetSlicerFonts();
    if ( font)
      {
      // check to see if m has a valid value:
      if ( font->IsValidFontSize ( size) )
        {
        int f2 = font->GetFontSize2( size );
        int f1 = font->GetFontSize1( size );
        int f0 = font->GetFontSize0( size );
        
        this->Script ( "font configure %s -size %d", this->SlicerTheme->GetApplicationFont2(), f2);
        this->Script ( "font configure %s -size %d", this->SlicerTheme->GetApplicationFont1(), f1);
        this->Script ( "font configure %s -size %d", this->SlicerTheme->GetApplicationFont0(), f0);
        
        strcpy (this->ApplicationFontSize, size );
        }
      }
    }    
}
//----------------------------------------------------------------------------
const char *vtkKWMimxApplication::GetApplicationFontSize () const
{
  return this->ApplicationFontSize;
}


//----------------------------------------------------------------------------
void vtkKWMimxApplication::AddAboutCopyrights(ostream &os)
{
  os << "IA-FEMesh is developed by the ";
  os << "Musculoskeletal Imaging, Modelling and Experimentation (MIMX) Program" << std::endl;
  os << "Center for Computer Aided Design" << std::endl;
  os << "The University of Iowa" << std::endl;
  os << "Iowa City, IA 52242" << std::endl;

  os << "See http://www.ccad.uiowa.edu/mimx/ for Copyright Information" << std::endl << std::endl;
  os << "IA-FEMesh is built upon:" << std::endl;
  os << "  Slicer3 http://www.na-mic.org/" << std::endl;
  os << "  VTK http://www.vtk.org/copyright.php" << std::endl;
  os << "  ITK http://www.itk.org/HTML/Copyright.htm" << std::endl;
  os << "  KWWidgets http://www.kitware.com/Copyright.htm" << std::endl;
  os << "  Tcl/Tk http://www.tcl.tk" << std::endl << std::endl;
  os << "Supported is provided by the National Institutes of Health Grants" << std::endl;
  os << "  5R21EB001501 and 5R01EB005973." << std::endl;

#if 0
  // example of the extra detail needed:
  //
     << tcl_major << "." << tcl_minor << "." << tcl_patch_level << endl
     << "  - Copyright (c) 1989-1994 The Regents of the University of "
     << "California." << endl
     << "  - Copyright (c) 1994 The Australian National University." << endl
     << "  - Copyright (c) 1994-1998 Sun Microsystems, Inc." << endl
     << "  - Copyright (c) 1998-2000 Ajuba Solutions." << endl;
#endif
}

