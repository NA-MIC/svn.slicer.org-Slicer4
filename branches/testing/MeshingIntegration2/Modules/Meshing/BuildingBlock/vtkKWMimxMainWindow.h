/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxMainWindow.h,v $
Language:  C++
Date:      $Date: 2008/05/01 02:34:12 $
Version:   $Revision: 1.15 $

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

#ifndef __vtkKWMimxMainWindow_h
#define __vtkKWMimxMainWindow_h

#include "vtkKWWindow.h"

// includes added during slicer integration
#include "vtkKWRenderWidget.h"

class vtkCallbackCommand;
class vtkRenderer;
class vtkPVAxesActor;
class vtkKWRenderWidget;
class vtkKWChangeColorButton;
class vtkMimxErrorCallback;
class vtkKWFrameWithScrollbar;
class vtkKWMenu;
//class vtkKWMimxMainNotebook;
class vtkKWMimxViewProperties;
class vtkLinkedListWrapperTree;
class vtkKWLoadSaveDialog;
class vtkKWMimxMainUserInterfacePanel;
class vtkKWMimxDisplayPropertiesGroup;


class vtkKWMimxMainWindow : public vtkKWWindow
{
public:
        static vtkKWMimxMainWindow* New();
        vtkTypeRevisionMacro(vtkKWMimxMainWindow,vtkKWWindow);
        vtkGetObjectMacro(RenderWidget, vtkKWRenderWidget);
        vtkSetObjectMacro(RenderWidget,vtkKWRenderWidget);
    vtkGetObjectMacro(TraceRenderWidget, vtkKWRenderWidget);
        vtkGetObjectMacro(ErrorCallback, vtkMimxErrorCallback);
        vtkGetObjectMacro(ViewProperties, vtkKWMimxViewProperties);
        vtkGetObjectMacro(PVAxesActor, vtkPVAxesActor);
        vtkGetObjectMacro(
                MainUserInterfacePanel, vtkKWMimxMainUserInterfacePanel);
        vtkKWRenderWidget *RenderWidget;
        vtkKWRenderWidget *TraceRenderWidget;
        vtkRenderer *AxesRenderer;
        vtkPVAxesActor *PVAxesActor;
        vtkCallbackCommand *CallbackCommand;
        
        void DisplayPropertyCallback();
        void SetTextColor(double color[3]);
        double *GetTextColor();
        void SetBackgroundColor(double color[3]);
        double *GetBackgroundColor();
        void SetApplicationFontFamily( const char *font );
        void SetApplicationFontSize( const char *size );
        
        //void ViewWindowProperties();
        //void CaptureWindowSnapshot();
        //void UpdateViewXaxis();
        //void UpdateViewYaxis();
        //void UpdateViewZaxis();
        //void UpdateViewXaxisNeg();
        //void UpdateViewYaxisNeg();
        //void UpdateViewZaxisNeg();
protected:
        vtkKWMimxMainWindow();
        ~vtkKWMimxMainWindow();
        virtual void CreateWidget();
        vtkKWChangeColorButton *ChangeColorButton;
        vtkMimxErrorCallback *ErrorCallback;
        vtkKWMenu *ViewMenu;
        vtkKWFrameWithScrollbar *MainNoteBookFrameScrollbar;
//      vtkKWMimxMainNotebook *MimxMainNotebook;
        vtkKWMimxViewProperties *ViewProperties;
        vtkLinkedListWrapperTree *DoUndoTree;
        vtkKWLoadSaveDialog *LoadSaveDialog;
        vtkKWMimxMainUserInterfacePanel *MainUserInterfacePanel;
        vtkKWMimxDisplayPropertiesGroup *DisplayPropertyDialog;
        
private:
        vtkKWMimxMainWindow(const vtkKWMimxMainWindow&);   // Not implemented.
        void operator=(const vtkKWMimxMainWindow&);  // Not implemented.
        
        double TextColor[3];
        double BackgroundColor[3];

};
void updateAxis(vtkObject* caller, unsigned long , void* arg, void* );

#endif
