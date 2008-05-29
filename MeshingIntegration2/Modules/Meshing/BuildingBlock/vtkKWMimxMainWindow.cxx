/*=========================================================================

Program:   MIMX Meshing Toolkit
Module:    $RCSfile: vtkKWMimxMainWindow.cxx,v $
Language:  C++
Date:      $Date: 2008/05/01 02:34:12 $
Version:   $Revision: 1.19 $

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

#include "vtkKWMimxMainWindow.h"
#include "vtkKWChangeColorButton.h"

#include "vtkMimxUnstructuredGridWidget.h"
#include "vtkMimxErrorCallback.h"
#include "vtkKWMimxViewProperties.h"
#include "vtkLinkedListWrapperTree.h"
#include "vtkWindowToImageFilter.h"
#include "vtkJPEGWriter.h"

#include "vtkKWApplication.h"
#include "vtkKWMenu.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWWindow.h"
#include "vtkCallbackCommand.h"
#include "vtkCamera.h"
#include "vtkObjectFactory.h"
#include "vtkPVAxesActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWMimxMainUserInterfacePanel.h"
#include <vtksys/SystemTools.hxx>
#include "vtkKWPushButton.h"
#include "vtkKWToolbarSet.h"
#include "vtkKWIcon.h"
#include "vtkKWToolbar.h"
#include "vtkKWTkUtilities.h"
#include "vtkKWMimxApplication.h"

//#include "vtkKWMimxMainNotebook.h"
#include "vtkKWFrameWithScrollbar.h"
#include "vtkKWNotebook.h"
#include "vtkKWWindow.h"

#include "vtkToolkits.h"

#include "Resources/mimxSnapshot.h"
#include "Resources/mimxPerspective.h"
#include "Resources/mimxParallel.h"
#include "Resources/mimxAutofit.h"
#include "Resources/mimxPosX.h"
#include "Resources/mimxNegX.h"
#include "Resources/mimxPosY.h"
#include "Resources/mimxNegY.h"
#include "Resources/mimxPosZ.h"
#include "Resources/mimxNegZ.h"


//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkKWMimxMainWindow );

vtkCxxRevisionMacro(vtkKWMimxMainWindow, "$Revision: 1.19 $");
//----------------------------------------------------------------------------
class vtkKeypress : public vtkCommand
{
public:
        static vtkKeypress *New() 
        { return new vtkKeypress; }
        virtual void Execute(vtkObject *caller, unsigned long, void*)
        {
                /*   vtkRenderer *renderer = reinterpret_cast<vtkRenderer*>(caller);
                cout << renderer->GetActiveCamera()->GetPosition()[0] << " "
                << renderer->GetActiveCamera()->GetPosition()[1] << " "
                << renderer->GetActiveCamera()->GetPosition()[2] << "\n";*/
        }
};

//----------------------------------------------------------------------------
vtkKWMimxMainWindow::vtkKWMimxMainWindow()
{
        this->RenderWidget = NULL;
        this->PVAxesActor = NULL;
        this->AxesRenderer = NULL;
        this->CallbackCommand = NULL;
        this->ChangeColorButton = NULL;
        this->ErrorCallback = vtkMimxErrorCallback::New();
        this->ViewMenu = NULL;
        this->MainNoteBookFrameScrollbar = NULL;
        //this->MimxMainNotebook = NULL;
        this->ViewProperties = NULL;
        this->DoUndoTree = NULL;
        this->LoadSaveDialog = NULL;
        this->MainUserInterfacePanel = NULL;
        this->DisplayPropertyDialog = NULL;
        
        TextColor[0] = TextColor[1] = TextColor[2] = 1.0;
        BackgroundColor[0] = BackgroundColor[1] = BackgroundColor[2] = 0.0; 
}

//----------------------------------------------------------------------------
vtkKWMimxMainWindow::~vtkKWMimxMainWindow()
{
//        if (this->AxesRenderer) {
//                this->AxesRenderer->Delete();
//                this->PVAxesActor->Delete();
//                this->CallbackCommand->Delete();
//        }
//        if(this->RenderWidget)
//        {
//                this->RenderWidget->Delete();
//        }
        if(this->ChangeColorButton)
        {
                this->ChangeColorButton->Delete();
                this->ChangeColorButton = NULL;
        }
        this->ErrorCallback->Delete();

        if(this->ViewMenu)
        {
                this->ViewMenu->Delete();
        }
        if(this->MainNoteBookFrameScrollbar)
        {
                this->MainNoteBookFrameScrollbar->Delete();
        }
        //if(this->MimxMainNotebook)
        //{
        //      this->MimxMainNotebook->Delete();
        //}
        if(this->ViewProperties)
        {
                this->ViewProperties->Delete();
        }
        if(this->DoUndoTree)
                this->DoUndoTree->Delete();
        if(this->LoadSaveDialog)
                this->LoadSaveDialog->Delete();
        if(this->MainUserInterfacePanel)
                this->MainUserInterfacePanel->Delete();
        if (this->DisplayPropertyDialog )
          this->DisplayPropertyDialog->Delete();
}
//----------------------------------------------------------------------------
void vtkKWMimxMainWindow::CreateWidget()
{
        // setting the position and size of the window
        if(this->IsCreated())
        {
                vtkErrorMacro("Class already created");
                return;
        }
      this->Superclass::CreateWidget();
//
//      // setting the render window
//      if(!this->RenderWidget)
//      {
//              this->RenderWidget = vtkKWRenderWidget::New();
//      }
//      this->RenderWidget->SetParent(this->GetViewFrame());
//      this->RenderWidget->Create();
//
//      this->GetApplication()->Script("pack %s -expand y -fill both -anchor c -expand y", 
//              this->RenderWidget->GetWidgetName());
//
//      this->GetMenu()->Unpack();
//
//      // creation of axes representation
//      // we need two renderers
//      if(!this->AxesRenderer)
//      {
//              this->AxesRenderer = vtkRenderer::New();
//              this->PVAxesActor = vtkPVAxesActor::New();
//              this->CallbackCommand = vtkCallbackCommand::New();
//      }
//      this->AxesRenderer->InteractiveOff();
//      this->RenderWidget->GetRenderWindow()->SetNumberOfLayers(2);
//      this->RenderWidget->GetRenderer()->SetLayer(0);
//      this->AxesRenderer->SetLayer(1);
//      this->AxesRenderer->SetViewport(0.0,0.0,0.25,0.25);
//      this->AxesRenderer->AddActor(this->PVAxesActor);
//      this->CallbackCommand->SetCallback(updateAxis);
//      this->CallbackCommand->SetClientData(this);
//      this->RenderWidget->GetRenderer()->AddObserver(vtkCommand::AnyEvent,this->CallbackCommand);
//      this->AxesRenderer->SetBackground(0.0,0.0,0.0);
//      this->RenderWidget->GetRenderer()->SetBackground(0.0,0.0,0.0);
//      this->RenderWidget->GetRenderWindow()->AddRenderer(this->AxesRenderer);
//
//      //vtkKeypress *Keypress = vtkKeypress::New();
//      this->RenderWidget->GetRenderWindowInteractor()->RemoveObserver(vtkCommand::KeyPressEvent);
//      this->RenderWidget->GetRenderWindowInteractor()->RemoveObserver(vtkCommand::KeyReleaseEvent);
//      //this->RenderWidget->GetRenderWindowInteractor()->AddObserver(vtkCommand::KeyPressEvent, Keypress);
//      //Keypress->Delete();
//      this->ErrorCallback->SetKWApplication(this->GetApplication());
        //if(!this->ViewMenu)
        //{
        //      this->ViewMenu = vtkKWMenu::New();
        //}
        //this->ViewMenu->SetParent(this->GetMenu());
        //this->ViewMenu->Create();
        //this->GetMenu()->AddCascade("Display", this->ViewMenu);
        //this->ViewMenu->AddCommand("Background Color", this, 
        //      "ViewWindowProperties");

        //this->ViewMenu->AddCommand("Window Snapshot", this, 
        //      "CaptureWindowSnapshot");
        //
        //this->ViewMenu->AddSeparator();
        //this->ViewMenu->AddCommand("View +X-axis", this, 
        //      "UpdateViewXaxis");
        //this->ViewMenu->AddCommand("View -X-axis", this, 
        //      "UpdateViewXaxisNeg");
        //this->ViewMenu->AddCommand("View +Y-axis", this, 
        //      "UpdateViewYaxis");
        //this->ViewMenu->AddCommand("View -Y-axis", this, 
        //      "UpdateViewYaxisNeg");
        //this->ViewMenu->AddCommand("View +Z-axis", this, 
        //      "UpdateViewZaxis");
        //this->ViewMenu->AddCommand("View -Z-axis", this, 
        //      "UpdateViewZaxisNeg");
        //      
        //      

        // for do and undo tree
      if(!this->DoUndoTree)
              this->DoUndoTree = vtkLinkedListWrapperTree::New();

//      if(!this->ViewProperties)
//      {
//              this->ViewProperties = vtkKWMimxViewProperties::New();
////            this->ViewProperties->SetDataType(2);
//              this->ViewProperties->SetParent(this->GetMainPanelFrame());
//              this->ViewProperties->SetMimxMainWindow(this);
//      //      this->ViewProperties->SetObjectList(this->BBoxList);
//              this->ViewProperties->Create();
//              this->ViewProperties->SetBorderWidth(3);
//              this->ViewProperties->SetReliefToGroove();
//              this->ViewProperties->SetDoUndoTree(this->DoUndoTree);
//      }
//
//      this->GetApplication()->Script("pack %s -side top -anchor nw -expand n -padx 2 -pady 5 -fill x", 
//              this->ViewProperties->GetMainFrame()->GetWidgetName());

        // frame with scroll bar to place all tabs 
/*      if(!this->MainNoteBookFrameScrollbar)
        {
                this->MainNoteBookFrameScrollbar = vtkKWFrameWithScrollbar::New();
        }
        this->MainNoteBookFrameScrollbar->HorizontalScrollbarVisibilityOn();
        this->MainNoteBookFrameScrollbar->VerticalScrollbarVisibilityOff();
        this->MainNoteBookFrameScrollbar->SetParent(this->GetMainPanelFrame());

        this->MainNoteBookFrameScrollbar->Create();
        this->GetApplication()->Script("pack %s -side top -expand n -fill x -padx 2 -pady 2",
                this->MainNoteBookFrameScrollbar->GetWidgetName()); */
        //      this->MainNoteBookFrameScrollbar->SetHeight(300);
        // add tabs

//      if(!this->MainUserInterfacePanel)
//      {
//              this->MainUserInterfacePanel = vtkKWMimxMainUserInterfacePanel::New();
//              this->MainUserInterfacePanel->SetMimxMainWindow(this);
//              this->MainUserInterfacePanel->SetDoUndoTree(this->DoUndoTree);
//              this->MainUserInterfacePanel->SetMultiColumnList(
//                      this->ViewProperties->GetMultiColumnList());
//      }
//      this->MainUserInterfacePanel->SetParent(this->GetMainPanelFrame());
//      this->MainUserInterfacePanel->SetApplication(this->GetApplication());
//      this->MainUserInterfacePanel->Create();
//      this->MainUserInterfacePanel->SetBorderWidth(3);
//      this->MainUserInterfacePanel->SetReliefToGroove();
//      this->MainUserInterfacePanel->GetMainFrame()->ExpandFrame();
//      //      this->MainNotebook->SetWidth(200);
//      this->GetApplication()->Script(
//              "pack %s -side top -anchor nw -expand n -fill y -pady 2 -fill x", 
//              this->MainUserInterfacePanel->GetWidgetName());
//      this->ViewProperties->SetViewPropertiesGroup(
//              this->MainUserInterfacePanel->GetViewPropertiesGroup());

//      if(!this->MimxMainNotebook)
//      {
//              this->MimxMainNotebook = vtkKWMimxMainNotebook::New();
//              this->MimxMainNotebook->SetMimxMainWindow(this);
//              this->MimxMainNotebook->SetDoUndoTree(this->DoUndoTree);
//      }
//      vtkKWFrame *frame = this->GetMainPanelFrame();
//      this->MimxMainNotebook->SetParent(this->GetMainPanelFrame());
//      this->MimxMainNotebook->SetApplication(this->GetApplication());
//      this->MimxMainNotebook->Create();
//      this->MimxMainNotebook->SetBorderWidth(3);
//      this->MimxMainNotebook->SetReliefToGroove();
////    this->MainNotebook->SetWidth(200);
//      this->GetApplication()->Script(
//              "pack %s -side top -anchor nw -expand n -fill y -pady 2 -fill x", 
//              this->MimxMainNotebook->GetWidgetName());
        
        // Custom Configuration of the Toolbar Menu
//      vtkKWMenu* fileMenu = this->GetFileMenu();
//  fileMenu->DeleteItem (0);
//  
//  vtkKWMenu* viewMenu = this->GetViewMenu();
//  viewMenu->AddCommand("View Settings", this, "DisplayPropertyCallback");
//  viewMenu->AddSeparator( );
//  
//  vtkKWMenu *fontSizeMenu = vtkKWMenu::New();
//  fontSizeMenu->SetParent(viewMenu);
//  fontSizeMenu->Create();
//  fontSizeMenu->AddRadioButton("small", this, "SetApplicationFontSize small");
//  fontSizeMenu->AddRadioButton("medium", this, "SetApplicationFontSize medium");
//  fontSizeMenu->AddRadioButton("large", this, "SetApplicationFontSize large");
//  fontSizeMenu->AddRadioButton("largest", this, "SetApplicationFontSize largest");
//  viewMenu->AddCascade("Font Size", fontSizeMenu);
//  
//  vtkKWMenu *fontTypeMenu = vtkKWMenu::New();
//  fontTypeMenu->SetParent(viewMenu);
//  fontTypeMenu->Create();
//  fontTypeMenu->AddRadioButton("Arial", this, "SetApplicationFontFamily Arial");
//  fontTypeMenu->AddRadioButton("Helvetica", this, "SetApplicationFontFamily Helvetica");
//  fontTypeMenu->AddRadioButton("Verdana", this, "SetApplicationFontFamily Verdana");
//  viewMenu->AddCascade("Font Family", fontTypeMenu);
//  
  
  
 /* This can be used to add Icons to the Top Toolbar */
 /*
  vtkKWMenu *toolbar = this->GetMenu();
  int curItem = toolbar->AddCommand( "XYZ" );
  vtkKWTkUtilities::UpdatePhoto( this->GetApplication()->GetMainInterp(),
                "mimxSnapShotImage",
                image_mimxSnapshot, 
      image_mimxSnapshot_width, 
      image_mimxSnapshot_height, 
      image_mimxSnapshot_pixel_size );
                
  toolbar->SetItemImage(curItem, "mimxSnapShotImage");   
  */                  
}


//----------------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::ViewWindowProperties()
//{
//      if(!this->ChangeColorButton)
//      {
//              this->ChangeColorButton = vtkKWChangeColorButton::New();
//              //              this->ChangeColorButton->SetCommand(this, "ChangeBackgroundColor");
//              this->ChangeColorButton->SetApplication(this->GetApplication());
//              this->ChangeColorButton->Create();
//      }
//      this->ChangeColorButton->SetColor(this->RenderWidget->GetRenderer()->GetBackground());
//      this->ChangeColorButton->ButtonPressCallback();
//      this->ChangeColorButton->ButtonReleaseCallback();
//      if(this->ChangeColorButton)
//      {
//              this->RenderWidget->GetRenderer()->SetBackground(this->ChangeColorButton->GetColor());
//              this->RenderWidget->Render();
//      }
//
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::CaptureWindowSnapshot()
//{
//      if(!this->LoadSaveDialog)
//      {
//              this->LoadSaveDialog = vtkKWLoadSaveDialog::New() ;
//              this->LoadSaveDialog->SetApplication(this->GetApplication());
//              this->LoadSaveDialog->SaveDialogOn();
//              this->LoadSaveDialog->Create();
//      }
//      this->LoadSaveDialog->SetTitle ("Save Window snap shot");
////    this->LoadSaveDialog->SetFileTypes ("{{JPEG Image} {.jpeg}} {{BMP Image} {.bmp}} {{PNG Image} {.png}} {{TIFF Image} {.tif}} {{PostScript Image} {.ps}}");
//      this->LoadSaveDialog->SetFileTypes ("{{JPEG Image} {.jpeg}}");
//      this->LoadSaveDialog->SetDefaultExtension (".jpeg");
//      this->LoadSaveDialog->RetrieveLastPathFromRegistry("LastPath");
//      this->LoadSaveDialog->Invoke();
//
//      if(this->LoadSaveDialog->GetStatus() == vtkKWDialog::StatusOK)
//      {
//              if(this->LoadSaveDialog->GetFileName())
//              {
//                      const char *filename = this->LoadSaveDialog->GetFileName();
//                      vtkWindowToImageFilter *imageout = vtkWindowToImageFilter::New();
//                      imageout->SetInput(this->RenderWidget->GetRenderWindow());
//                      vtkJPEGWriter *imagewriter = vtkJPEGWriter::New();
//                      imagewriter->SetInput(imageout->GetOutput());
//                      imagewriter->SetFileName(filename);
//                      imagewriter->Write();
//                      imagewriter->Delete();
//                      imageout->Delete();
//              }
//      }
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewXaxis()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,1,0);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(1,0,0);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewXaxisNeg()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,1,0);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(-1,0,0);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewYaxis()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,0,1);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(0,1,0);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewYaxisNeg()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,0,1);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(0,-1,0);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewZaxis()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,1,0);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(0,0,1);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
////----------------------------------------------------------------------------------------------
//void vtkKWMimxMainWindow::UpdateViewZaxisNeg()
//{
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetViewUp(0,1,0);
//  this->RenderWidget->GetNthRenderer(0)->GetActiveCamera()->SetPosition(0,0,-1);
//  this->RenderWidget->GetNthRenderer(0)->ResetCamera();
//  this->RenderWidget->GetRenderWindowInteractor()->Render();
//}
//----------------------------------------------------------------------------------------------
void updateAxis(vtkObject* caller, unsigned long , void* arg, void* ) {
        double    cPos[3], cFoc[3], aFoc[3];

        // set the axis camera according to the main renderer.
        vtkKWMimxMainWindow *MimxMainWindow = (vtkKWMimxMainWindow *)arg;
        MimxMainWindow->RenderWidget->GetRenderer()->GetActiveCamera()->GetPosition(cPos);
        MimxMainWindow->RenderWidget->GetRenderer()->GetActiveCamera()->GetFocalPoint(cFoc);
        MimxMainWindow->AxesRenderer->GetActiveCamera()->GetFocalPoint(aFoc);
        MimxMainWindow->AxesRenderer->GetActiveCamera()->SetViewUp(
                MimxMainWindow->RenderWidget->GetRenderer()->GetActiveCamera()->GetViewUp());
        MimxMainWindow->AxesRenderer->GetActiveCamera()->SetPosition(cPos[0] - cFoc[0] +
                aFoc[0],\
                cPos[1] - cFoc[1] + aFoc[1],\
                cPos[2] - cFoc[2] + aFoc[2]);
        MimxMainWindow->AxesRenderer->ResetCamera();
}
//----------------------------------------------------------------------------------------------
void vtkKWMimxMainWindow::DisplayPropertyCallback()
{
        if (!this->DisplayPropertyDialog)
        {
          this->DisplayPropertyDialog = vtkKWMimxDisplayPropertiesGroup::New();
          this->DisplayPropertyDialog->SetMimxMainWindow(this);
          this->DisplayPropertyDialog->SetApplication(this->GetApplication());
          this->DisplayPropertyDialog->Create();
        }
        this->DisplayPropertyDialog->Display();
        
}
//----------------------------------------------------------------------------------------------
void vtkKWMimxMainWindow::SetTextColor(double color[3])
{
  TextColor[0] = color[0];
  TextColor[1] = color[1];
  TextColor[2] = color[2];
}

//----------------------------------------------------------------------------------------------
double *vtkKWMimxMainWindow::GetTextColor( )
{
  return TextColor;
}

//----------------------------------------------------------------------------------------------
void vtkKWMimxMainWindow::SetBackgroundColor(double color[3])
{
  BackgroundColor[0] = color[0];
  BackgroundColor[1] = color[1];
  BackgroundColor[2] = color[2];
}

//----------------------------------------------------------------------------------------------
double *vtkKWMimxMainWindow::GetBackgroundColor( )
{
  return BackgroundColor;
}

//----------------------------------------------------------------------------------------------
void vtkKWMimxMainWindow::SetApplicationFontFamily( const char *font )
{
  vtkKWMimxApplication *app = vtkKWMimxApplication::SafeDownCast(this->GetApplication());
  app->SetApplicationFontFamily( font );
}

//----------------------------------------------------------------------------------------------
void vtkKWMimxMainWindow::SetApplicationFontSize( const char *size )
{
  vtkKWMimxApplication *app = vtkKWMimxApplication::SafeDownCast(this->GetApplication());
  app->SetApplicationFontSize( size );
}

//----------------------------------------------------------------------------------------------
