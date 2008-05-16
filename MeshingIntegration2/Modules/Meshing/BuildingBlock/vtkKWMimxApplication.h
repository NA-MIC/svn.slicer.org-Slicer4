/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxApplication.h,v $
Language:  C++
Date:      $Date: 2008/03/18 02:32:12 $
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

#ifndef __vtkKWMimxApplication_h
#define __vtkKWMimxApplication_h

#include "vtkKWApplication.h"
#include "vtkKWRegistryHelper.h"


class vtkSlicerTheme;

class vtkKWMimxApplication : public vtkKWApplication
{
public:
  static vtkKWMimxApplication* New();
  vtkTypeRevisionMacro(vtkKWMimxApplication,vtkKWApplication);
 
  virtual void AddAboutCopyrights(ostream &);
  virtual void InstallTheme ( vtkKWTheme *theme );
  virtual void InstallDefaultTheme ( );
  
  // Description:
  // Set/Get the application font family
  void SetApplicationFontFamily ( const char *family);
  const char *GetApplicationFontFamily ( ) const;

  // Description:
  // Set/Get the application font size
  void SetApplicationFontSize ( const char *size );
  const char *GetApplicationFontSize ( ) const;
  
protected:
  vtkKWMimxApplication();
  ~vtkKWMimxApplication();
  
  vtkSlicerTheme *SlicerTheme;

  char ApplicationFontSize [vtkKWRegistryHelper::RegistryKeyValueSizeMax];
  char ApplicationFontFamily [vtkKWRegistryHelper::RegistryKeyValueSizeMax];
  
private:
  vtkKWMimxApplication(const vtkKWMimxApplication&);   // Not implemented.
  void operator=(const vtkKWMimxApplication&);  // Not implemented.
  
  
};

#endif
