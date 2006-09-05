/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerApplicationGUI.cxx,v $
  Date:      $Date: 2006/01/08 04:48:05 $
  Version:   $Revision: 1.45 $

=========================================================================auto=*/

#include <sstream>
#include <string>
#include <vtksys/SystemTools.hxx> 
#include <itksys/SystemTools.hxx> 

#include "vtkCommand.h"
#include "vtkCornerAnnotation.h"
#include "vtkObjectFactory.h"
#include "vtkToolkits.h"

// things for temporary MainViewer display.
#include "vtkCubeSource.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkCamera.h"
#include "vtkPolyDataMapper.h"
#include "vtkRenderWindow.h"
#include "vtkImplicitPlaneWidget.h"

#include "vtkKWApplication.h"
#include "vtkKWTclInteractor.h"
#include "vtkKWWidget.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWLabel.h"
#include "vtkKWNotebook.h"
#include "vtkKWPushButton.h"
#include "vtkKWRenderWidget.h"
#include "vtkKWScale.h"
#include "vtkKWUserInterfacePanel.h"
#include "vtkKWResourceUtilities.h"

#include "vtkKWSplitFrame.h"
#include "vtkKWUserInterfaceManagerNotebook.h"

#include "vtkSlicerWindow.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkSlicerGUILayout.h"
#include "vtkSlicerTheme.h"
#include "vtkSlicerColor.h"
#include "vtkSlicerLogoIcons.h"
#include "vtkSlicerModuleNavigationIcons.h"
#include "vtkSlicerMRMLSaveDataWidget.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerApplicationGUI);
vtkCxxRevisionMacro(vtkSlicerApplicationGUI, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerApplicationGUI::vtkSlicerApplicationGUI (  )
{
    //---  
    // widgets used in the Slice module
    //---

    //--- slicer main window
    this->MainSlicerWin = vtkSlicerWindow::New ( );

    this->ApplicationToolbar = vtkSlicerToolbarGUI::New ( );
    this->ViewControlGUI = vtkSlicerViewControlGUI::New ( );
    
    //--- slicer icons
    this->SlicerLogoIcons = vtkSlicerLogoIcons::New ();
    this->SlicerModuleNavigationIcons = vtkSlicerModuleNavigationIcons::New ();

    //--- logo widgets to which icons are assigned.
    this->SlicerLogoLabel = vtkKWLabel::New();

    // Control frames that comprise the Main Slicer GUI
    this->LogoFrame = vtkKWFrame::New();
    this->ModuleChooseFrame = vtkKWFrame::New();
    this->SliceControlFrame = vtkKWFrame::New();    
    this->ViewControlFrame = vtkKWFrame::New();    

    //--- ui for the ModuleChooseFrame,
    this->ModulesMenuButton = vtkKWMenuButton::New();
    this->ModulesLabel = vtkKWLabel::New();
    this->ModulesPrev = vtkKWPushButton::New ( );
    this->ModulesNext = vtkKWPushButton::New ( );
    this->ModulesHistory = vtkKWPushButton::New ( );
    this->ModulesRefresh = vtkKWPushButton::New ( );
    
    //--- ui for the SliceControlframe.
    this->ToggleAnnotationButton = vtkKWPushButton::New ( );
    this->ToggleFgBgButton = vtkKWPushButton::New ( );
    this->SliceFadeScale = vtkKWScale::New ( );
    this->SliceOpacityScale = vtkKWScale::New ( );
    
    //--- main viewer and 3 main slice views
    this->ViewerWidget = NULL;
    this->MainSliceGUI0 = NULL;
    this->MainSliceGUI1 = NULL;
    this->MainSliceGUI2 = NULL;

    //--- save the main slice logic in these.
    this->MainSliceLogic0 = NULL;
    this->MainSliceLogic1 = NULL;
    this->MainSliceLogic2 = NULL;
    this->SliceGUICollection = NULL;
    this->PlaneWidget = NULL;
    this->LightboxFrame = NULL;

    this->LoadSceneDialog = vtkKWLoadSaveDialog::New();
    this->SaveSceneDialog = vtkKWLoadSaveDialog::New();   

    //--- unique tag used to mark all view notebook pages
    //--- so that they can be identified and deleted when 
    //--- viewer is reformatted.
    this->ViewerPageTag = 1999;

    this->SaveDataWidget = NULL;
    this->SaveDataDialog = NULL;      

}



//---------------------------------------------------------------------------
vtkSlicerApplicationGUI::~vtkSlicerApplicationGUI ( )
{


    if ( this->SlicerLogoIcons ) {
        this->SlicerLogoIcons->Delete ( );
        this->SlicerLogoIcons = NULL;
    }

    this->ViewControlGUI->Delete ( );
    
    if ( this->SlicerModuleNavigationIcons ) {
        this->SlicerModuleNavigationIcons->Delete ( );
        this->SlicerModuleNavigationIcons = NULL;
    }

    this->DeleteGUIPanelWidgets ( );

    this->ApplicationToolbar->Delete ( );

    if ( this->SliceGUICollection )
      {
        this->SliceGUICollection->RemoveAllItems();
        this->SetSliceGUICollection ( NULL );
      }

    this->DestroyMain3DViewer ( );
    this->DestroyMainSliceViewers ( );
    this->DeleteFrames ( );

    if ( this->LoadSceneDialog ) {
      this->LoadSceneDialog->SetParent ( NULL );
        this->LoadSceneDialog->Delete();
        this->LoadSceneDialog = NULL;
    }
    if ( this->SaveSceneDialog ) {
      this->SaveSceneDialog->SetParent ( NULL );
        this->SaveSceneDialog->Delete();
        this->SaveSceneDialog = NULL;
    }
    if ( this->MainSlicerWin ) {
      this->MainSlicerWin->SetParent ( NULL );
        this->MainSlicerWin->Delete ( );
        this->MainSlicerWin = NULL;
    }
    this->MainSliceLogic0 = NULL;
    this->MainSliceLogic1 = NULL;
    this->MainSliceLogic2 = NULL;

    if (this->SaveDataWidget)
      {
        this->SaveDataWidget->SetParent ( NULL );
      this->SaveDataWidget->Delete();
      }
    if (this->SaveDataDialog)
      {
        this->SaveDataDialog->SetParent ( NULL );
      this->SaveDataDialog->Delete();
      }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "SlicerApplicationGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "MainSlicerWin: " << this->GetMainSlicerWin ( ) << "\n";
    // print widgets?
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessLoadSceneCommand()
{
    this->LoadSceneDialog->RetrieveLastPathFromRegistry(
      "OpenPath");

    this->LoadSceneDialog->Invoke();
    // If a file has been selected for loading...
    char *fileName = this->LoadSceneDialog->GetFileName();
    if ( fileName ) 
      {
        if (this->GetMRMLScene()) 
          {
          this->GetMRMLScene()->SetURL(fileName);
          this->GetMRMLScene()->Connect();
          this->LoadSceneDialog->SaveLastPathToRegistry("OpenPath");
          }
      }
    return;
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessImportSceneCommand()
{
    this->LoadSceneDialog->RetrieveLastPathFromRegistry(
      "OpenPath");

    this->LoadSceneDialog->Invoke();
    // If a file has been selected for loading...
    char *fileName = this->LoadSceneDialog->GetFileName();
    if ( fileName ) 
      {
        if (this->GetMRMLScene()) 
          {
          this->GetMRMLScene()->SetURL(fileName);
          this->GetMRMLScene()->Import();
          this->LoadSceneDialog->SaveLastPathToRegistry("OpenPath");
          }
      }
    return;
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessSaveSceneAsCommand()
{
    this->SaveSceneDialog->RetrieveLastPathFromRegistry(
      "OpenPath");

    
   
    this->SaveSceneDialog->Invoke();



    // If a file has been selected for saving...
    char *fileName = this->SaveSceneDialog->GetFileName();
    if ( fileName ) 
      {
      this->SaveDataDialog = vtkKWDialog::New();
      this->SaveDataDialog->SetParent ( this->MainSlicerWin );
      this->SaveDataDialog->SetTitle("Save Unsaved Data");
      this->SaveDataDialog->SetSize(400, 100);
      this->SaveDataDialog->Create ( );
      //this->Script ( "pack %s -fill both -expand true",
      //            this->SaveDataDialog->GetWidgetName());
      
      this->SaveDataWidget = vtkSlicerMRMLSaveDataWidget::New();
      this->SaveDataWidget->SetParent ( this->SaveDataDialog);
      this->SaveDataWidget->SetAndObserveMRMLScene(this->GetMRMLScene());

      vtksys_stl::string dir =  vtksys::SystemTools::GetParentDirectory(fileName);   
      dir = dir + vtksys_stl::string("/");
      this->SaveDataWidget->SetFileDirectoryName(dir.c_str());

      this->SaveDataWidget->Create();
      this->SaveDataWidget->AddObserver ( vtkSlicerMRMLSaveDataWidget::DataSavedEvent,  (vtkCommand *)this->GUICallbackCommand );

      // TODO: make update event driven so that we don't have to call this
      int nrows = this->SaveDataWidget->UpdateFromMRML();

      if (nrows > 0) 
      {
        
        this->Script("pack %s -side top -anchor w -padx 2 -pady 4", 
                  this->SaveDataWidget->GetWidgetName());
        this->SaveDataDialog->Invoke ( );
      }

      this->SaveDataWidget->RemoveObservers ( vtkSlicerMRMLSaveDataWidget::DataSavedEvent,  (vtkCommand *)this->GUICallbackCommand );
      this->SaveDataWidget->SetParent(NULL);
      this->SaveDataDialog->SetParent(NULL);    
      this->SaveDataWidget->Delete();
      this->SaveDataDialog->Delete();      
      this->SaveDataWidget=NULL;
      this->SaveDataDialog=NULL;      

      if (this->GetMRMLScene()) 
        {
        // convert absolute paths to relative
        this->MRMLScene->InitTraversal();

        vtkMRMLNode *node;
        while ( (node = this->MRMLScene->GetNextNodeByClass("vtkMRMLStorageNode") ) != NULL)
          {
          vtkMRMLStorageNode *snode = vtkMRMLStorageNode::SafeDownCast(node);
          if (!this->MRMLScene->IsFilePathRelative(snode->GetFileName()))
            {
            vtksys_stl::string directory = vtksys::SystemTools::GetParentDirectory(fileName);   
            directory = directory + vtksys_stl::string("/");

            itksys_stl::string relPath = itksys::SystemTools::RelativePath((const char*)directory.c_str(), snode->GetFileName());
            snode->SetFileName(relPath.c_str());
            }
          }

        this->GetMRMLScene()->SetURL(fileName);
        this->GetMRMLScene()->Commit();  
        this->SaveSceneDialog->SaveLastPathToRegistry("OpenPath");
        }
      }
    return;
}    

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::AddGUIObservers ( )
{

    // add observer onto the menubutton in the SlicerControl frame
  this->ModulesMenuButton->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    
  this->GetApplicationToolbar()->AddGUIObservers ( );
  
    this->GetMainSlicerWin()->GetFileMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    
    this->LoadSceneDialog->AddObserver ( vtkCommand::ModifiedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SaveSceneDialog->AddObserver ( vtkCommand::ModifiedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->SliceFadeScale->AddObserver ( vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SliceFadeScale->AddObserver ( vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );

    this->ToggleFgBgButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->SliceOpacityScale->AddObserver ( vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SliceOpacityScale->AddObserver ( vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::RemoveGUIObservers ( )
{
  this->ModulesMenuButton->GetMenu()->RemoveObservers (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->GetApplicationToolbar()->RemoveGUIObservers ( );
  
    this->LoadSceneDialog->RemoveObservers ( vtkCommand::ModifiedEvent, (vtkCommand *) this->GUICallbackCommand );
    this->SaveSceneDialog->RemoveObservers ( vtkCommand::ModifiedEvent, (vtkCommand *) this->GUICallbackCommand );
    this->GetMainSlicerWin()->GetFileMenu()->RemoveObservers ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SliceFadeScale->RemoveObservers ( vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SliceFadeScale->RemoveObservers ( vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );    
    this->SliceOpacityScale->RemoveObservers ( vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
    this->SliceOpacityScale->RemoveObservers ( vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );    
    this->ToggleFgBgButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->RemoveMainSliceViewerObservers ( );

}





//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessGUIEvents ( vtkObject *caller,
                                                 unsigned long event, void *callData )
{
    
    // This code is just a placeholder until the logic is set up to use properly:
    // For now, the GUI controls the GUI instead of going thru the logic...
    // TODO:
    // Actually, these events want to set "activeModule" in the logic;
    // using this->Logic->SetActiveModule ( ) which is currently commented out.
    // Observers on that logic should raise and lower the appropriate page.
    // So for now, the GUI is controlling the GUI instead of going thru the logic.
    //---
    vtkSlicerModuleGUI * m;
    const char *mName;
    vtkKWPushButton *pushb = vtkKWPushButton::SafeDownCast (caller );
    vtkKWMenuButton *menub = vtkKWMenuButton::SafeDownCast (caller );
    vtkKWMenu *menu = vtkKWMenu::SafeDownCast (caller );
    vtkKWLoadSaveDialog *filebrowse = vtkKWLoadSaveDialog::SafeDownCast(caller);
    vtkKWScale *scale = vtkKWScale::SafeDownCast(caller);

    vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast( this->GetApplication() );
    vtkSlicerGUILayout *layout = app->GetMainLayout ( );
        
    if (menu == this->GetMainSlicerWin()->GetFileMenu() && event == vtkKWMenu::MenuItemInvokedEvent)
    {
      int index = (int) (*((int *)callData));
      if (index == 2)
        {
          // use command directly instead of this
          //this->ProcessLoadSceneCommand()
        }
      else if (index == 3)
        {
          // use command directly instead of this
          //this->ProcessSaveSceneCommand()
        }
    }

    //--- Process events from menubutton
    //--- TODO: change the Logic's "active module" and raise the appropriate UIPanel.
    //    if ( menub == this->ModulesMenuButton && event == vtkCommand::ModifiedEvent )
    if ( menu == this->ModulesMenuButton->GetMenu() && event == vtkKWMenu::MenuItemInvokedEvent )
        {
            if ( app->GetModuleGUICollection ( ) != NULL )
                {
                    app->GetModuleGUICollection( )->InitTraversal( );
                    m = vtkSlicerModuleGUI::SafeDownCast( app->GetModuleGUICollection( )->GetNextItemAsObject( ) );
                    while (m != NULL )
                        {
                            mName = m->GetUIPanel()->GetName();
                            if ( !strcmp (this->ModulesMenuButton->GetValue(), mName) ) {
                                m->GetUIPanel()->Raise();
                                break;
                            }
                            m = vtkSlicerModuleGUI::SafeDownCast( app->GetModuleGUICollection( )->GetNextItemAsObject( ) );
                        }
                    //this->ModulesMenuButton->SetValue ( "Modules" );
                }
        }

    // Process the Fade scale and button
    // -- set save state when manipulation starts
    // -- toggle the value if needed
    // -- adjust the Opacity of every composite node on every event
    if ( scale == this->SliceFadeScale && event == vtkKWScale::ScaleValueStartChangingEvent ||
         pushb == this->ToggleFgBgButton && event == vtkKWPushButton::InvokedEvent )
      {
      if (this->GetMRMLScene()) 
        {
        this->GetMRMLScene()->SaveStateForUndo();
        }
      }

    if ( scale == this->SliceFadeScale && event == vtkKWScale::ScaleValueChangingEvent ||
         pushb == this->ToggleFgBgButton && event == vtkKWPushButton::InvokedEvent )
      {

      if ( pushb == this->ToggleFgBgButton && event == vtkKWPushButton::InvokedEvent ) 
        {
        this->SliceFadeScale->SetValue( 1.0 - this->SliceFadeScale->GetValue() );
        }

      int i, nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode");
      vtkMRMLSliceCompositeNode *cnode;
      for (i = 0; i < nnodes; i++)
        {
        cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                this->MRMLScene->GetNthNodeByClass( i, "vtkMRMLSliceCompositeNode" ) );
        cnode->SetForegroundOpacity( this->SliceFadeScale->GetValue() );
        }
      }

    // Process the label Opacity scale 
    // -- set save state when manipulation starts
    // -- adjust the Opacity of every composite node on every event
    if ( scale == this->SliceOpacityScale && event == vtkKWScale::ScaleValueStartChangingEvent )
      {
      if (this->GetMRMLScene()) 
        {
        this->GetMRMLScene()->SaveStateForUndo();
        }
      }

    if ( scale == this->SliceOpacityScale && event == vtkKWScale::ScaleValueChangingEvent )
      {

      int i, nnodes = this->MRMLScene->GetNumberOfNodesByClass("vtkMRMLSliceCompositeNode");
      vtkMRMLSliceCompositeNode *cnode;
      for (i = 0; i < nnodes; i++)
        {
        cnode = vtkMRMLSliceCompositeNode::SafeDownCast (
                this->MRMLScene->GetNthNodeByClass( i, "vtkMRMLSliceCompositeNode" ) );
        cnode->SetLabelOpacity( this->SliceOpacityScale->GetValue() );
        }
      }

   vtkSlicerMRMLSaveDataWidget *saveDataWidget = vtkSlicerMRMLSaveDataWidget::SafeDownCast(caller);
   if (saveDataWidget == this->SaveDataWidget && event == vtkSlicerMRMLSaveDataWidget::DataSavedEvent)
    {
    this->SaveDataDialog->OK();
    }
  }


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessLogicEvents ( vtkObject *caller,
                                                   unsigned long event, void *callData )
{
    // Fill in
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ProcessMRMLEvents ( vtkObject *caller,
                                                  unsigned long event, void *callData )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::Enter ( )
{
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::Exit ( )
{
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildGUI ( )
{
    int i;
    
    // Set up the conventional window: 3Dviewer, slice widgets, UI panel for now.
    if ( this->GetApplication() != NULL ) {
        vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
        vtkSlicerGUILayout *layout = app->GetMainLayout ( );
        
        // Set a pointer to the MainSlicerWin in vtkSlicerGUILayout, and
        // Set default sizes for all main frames (UIpanel and viewers) in GUI
        layout->SetMainSlicerWin ( this->MainSlicerWin );
        layout->InitializeLayoutDimensions ( );

        if ( this->MainSlicerWin != NULL ) {

            // set up Slicer's main window
            this->MainSlicerWin->SecondaryPanelVisibilityOn ( );
            this->MainSlicerWin->MainPanelVisibilityOn ( );
            app->AddWindow ( this->MainSlicerWin );

            // Create the console before the window
            // - this will make the console independent of the main window
            //   so it can be raised/lowered independently
            this->MainSlicerWin->GetTclInteractor()->SetApplication(app);
            this->MainSlicerWin->GetTclInteractor()->Create();

            // TODO: it would be nice to make this a menu option on the tkcon itself,
            // but for now just up the font size
            this->MainSlicerWin->Script(".vtkKWTkcon0.tab1 configure -font {Courier 12}");

            this->MainSlicerWin->Create ( );        

            // configure initial GUI layout
            layout->InitializeMainSlicerWindowSize ( );
            layout->ConfigureMainSlicerWindowPanels ( );

            // Build main GUI and components
            vtkSlicerToolbarGUI *appTB = this->GetApplicationToolbar ( );
            appTB->SetApplicationGUI ( this );
            appTB->SetApplication ( app );
            appTB->BuildGUI ( );

            this->BuildGUIPanel ( );
            this->BuildLogoGUIPanel ( );
            this->BuildModuleChooseGUIPanel ( );
            this->BuildSliceControlGUIPanel ( );

            vtkSlicerViewControlGUI *vcGUI = this->GetViewControlGUI ( );
            vcGUI->SetApplicationGUI ( this );
            vcGUI->SetApplication ( app );
            vcGUI->BuildGUI ( this->ViewControlFrame );

            // Turn off the tabs for pages in the ModuleControlGUI
            this->MainSlicerWin->GetMainNotebook()->ShowIconsOff ( );

            this->MainSlicerWin->GetMainNotebook()->SetReliefToFlat();
            this->MainSlicerWin->GetMainNotebook()->SetBorderWidth ( 0 );
            this->MainSlicerWin->GetMainNotebook()->SetHighlightThickness ( 0 );

            //this->MainSlicerWin->GetMainNotebook()->SetAlwaysShowTabs ( 0 );
            this->MainSlicerWin->GetMainNotebook()->SetUseFrameWithScrollbars ( 1 );
            // Build 3DViewer and Slice Viewers

            this->RemoveMainSliceViewersFromCollection ( );            
            this->BuildMainViewer ( vtkSlicerGUILayout::SlicerLayoutDefaultView );
            this->AddMainSliceViewersToCollection ( );            

            // Construct menu bar and set up global key bindings
            // 
            // File Menu
            //
            this->GetMainSlicerWin()->GetFileMenu()->InsertCommand (
                      this->GetMainSlicerWin()->GetFileMenuInsertPosition(),
                                      "Load Scene...", this, "ProcessLoadSceneCommand");

            this->GetMainSlicerWin()->GetFileMenu()->InsertCommand (
                      this->GetMainSlicerWin()->GetFileMenuInsertPosition(),
                                      "Import Scene...", this, "ProcessImportSceneCommand");

            this->GetMainSlicerWin()->GetFileMenu()->InsertCommand (this->GetMainSlicerWin()->GetFileMenuInsertPosition(),
                                               "Save Scene As...", this, "ProcessSaveSceneAsCommand");

            this->GetMainSlicerWin()->GetFileMenu()->InsertSeparator (
                this->GetMainSlicerWin()->GetFileMenuInsertPosition());

            //
            // Edit Menu
            //
            i = this->MainSlicerWin->GetEditMenu()->AddCommand ("Set Home", NULL, NULL);
            this->MainSlicerWin->GetEditMenu()->SetItemAccelerator ( i, "Ctrl+H");
            i = this->MainSlicerWin->GetEditMenu()->AddCommand ( "Undo", NULL, "$::slicer3::MRMLScene Undo" );
            this->MainSlicerWin->GetEditMenu()->SetItemAccelerator ( i, "Ctrl+Z");
            i = this->MainSlicerWin->GetEditMenu()->AddCommand ( "Redo", NULL, "$::slicer3::MRMLScene Redo" );
            this->MainSlicerWin->GetEditMenu()->SetItemAccelerator ( i, "Ctrl+Y");

            //
            // View Menu
            //
            this->GetMainSlicerWin()->GetViewMenu()->InsertCommand (
                      this->GetMainSlicerWin()->GetViewMenuInsertPosition(),
                                      "Single Slice", NULL, "$::slicer3::ApplicationGUI UnpackMainSliceViewerFrames ; $::slicer3::ApplicationGUI PackFirstSliceViewerFrame ");
            this->GetMainSlicerWin()->GetViewMenu()->InsertCommand (
                      this->GetMainSlicerWin()->GetViewMenuInsertPosition(),
                                      "Three Slices", NULL, "$::slicer3::ApplicationGUI UnpackMainSliceViewerFrames ; $::slicer3::ApplicationGUI PackFirstSliceViewerFrame ");


            //i = this->MainSlicerWin->GetWindowMenu()->AddCommand ( ? );
            //i = this->MainSlicerWin->GetHelpMenu()->AddCommand ( ? );

            this->LoadSceneDialog->SetParent ( this->MainSlicerWin );
            this->LoadSceneDialog->Create ( );
            this->LoadSceneDialog->SetFileTypes("{ {MRML Scene} {*.mrml} }");
            this->LoadSceneDialog->RetrieveLastPathFromRegistry("OpenPath");

            this->SaveSceneDialog->SetParent ( this->MainSlicerWin );
            this->SaveSceneDialog->Create ( );
            this->SaveSceneDialog->SetFileTypes("{ {MRML Scene} {*.mrml} }");
            this->SaveSceneDialog->SaveDialogOn();
            this->SaveSceneDialog->RetrieveLastPathFromRegistry("OpenPath");
        }

        //
        // influence the theme of the ApplicationGUI
        //
        // toolbar color
        // GUI Panel
        // Logo GUI panel
        // Module choose GUI Panel
        // Slice Control
        // View Control

    }
}




//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DestroyMainSliceViewers ( )
{

  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
      //
      // Destroy 3 main slice viewers
      //
      if ( this->MainSliceGUI0 )
        {
          this->MainSliceGUI0->SetAndObserveMRMLScene (NULL );
          this->MainSliceGUI0->SetAndObserveModuleLogic ( NULL );
          this->MainSliceGUI0->RemoveGUIObservers ( );
          this->MainSliceGUI0->SetApplicationLogic ( NULL );
          if ( layout->GetCurrentViewArrangement() == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->MainSliceGUI0->UngridGUI ( );
            }
          else
            {
              this->MainSliceGUI0->UnpackGUI ( );
            }
          this->MainSliceGUI0->Delete () ;
          this->MainSliceGUI0 = NULL;
        }

      if ( this->MainSliceGUI1 )
        {
          this->MainSliceGUI1->SetAndObserveMRMLScene (NULL );
          this->MainSliceGUI1->SetAndObserveModuleLogic ( NULL );
          this->MainSliceGUI1->RemoveGUIObservers ( );
          this->MainSliceGUI1->SetApplicationLogic ( NULL );
          if ( layout->GetCurrentViewArrangement() == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->MainSliceGUI1->UngridGUI ( );
            }
          else
            {
              this->MainSliceGUI1->UnpackGUI ( );
            }
          this->MainSliceGUI1->Delete () ;
          this->MainSliceGUI1 = NULL;
        }

      if ( this->MainSliceGUI2 )
        {
          this->MainSliceGUI2->SetAndObserveMRMLScene (NULL );
          this->MainSliceGUI2->SetAndObserveModuleLogic ( NULL );
          this->MainSliceGUI2->RemoveGUIObservers ( );
          this->MainSliceGUI2->SetApplicationLogic ( NULL );
          if ( layout->GetCurrentViewArrangement() == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->MainSliceGUI2->UngridGUI ( );
            }
          else
            {
              this->MainSliceGUI2->UnpackGUI ( );
            }
          this->MainSliceGUI2->Delete () ;
          this->MainSliceGUI2 = NULL;
        }
      if ( this->LightboxFrame )
        {
          app->Script ("pack forget %s ", this->LightboxFrame->GetWidgetName ( ) );
          this->LightboxFrame->Delete ( );
          this->LightboxFrame = NULL;
        }
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DestroyMain3DViewer ( )
{
  //

  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
    
      // Destroy main 3D viewer
      //
      if ( this->ViewerWidget )
        {
          if ( this->PlaneWidget )
            {
              this->PlaneWidget->SetInteractor( NULL );
              this->PlaneWidget->Delete ( );
              this->PlaneWidget = NULL;
            }
          this->ViewerWidget->RemoveMRMLObservers ( );
          if ( layout->GetCurrentViewArrangement() == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->ViewerWidget->UngridWidget ( );
            }
          else
            {
              this->ViewerWidget->UnpackWidget ( );
            }
          this->ViewerWidget->SetParent ( NULL );
          this->ViewerWidget->Delete ( );
          this->ViewerWidget = NULL;
        }
    }
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayMainSlicerWindow ( )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    this->MainSlicerWin->Display ( );
    int w = this->MainSlicerWin->GetWidth ( );
    int h = this->MainSlicerWin->GetHeight ( );
    int vh = app->GetMainLayout()->GetDefault3DViewerHeight();
    int sh = app->GetMainLayout()->GetDefaultSliceGUIFrameHeight();
    int sfh = this->MainSlicerWin->GetSecondarySplitFrame()->GetFrame1Size();
    int sf2h = this->MainSlicerWin->GetSecondarySplitFrame()->GetFrame2Size();
    }
}

    


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DeleteGUIPanelWidgets ( )
{
    //--- widgets from the ModuleChooseFrame
    if ( this->ModulesMenuButton ) {
      this->ModulesMenuButton->SetParent ( NULL );
        this->ModulesMenuButton->Delete();
        this->ModulesMenuButton = NULL;
    }
    if ( this->ModulesLabel ) {
      this->ModulesLabel->SetParent ( NULL );
        this->ModulesLabel->Delete ( );
        this->ModulesLabel = NULL;
    }
    if ( this->ModulesPrev ) {
      this->ModulesPrev->SetParent ( NULL );
        this->ModulesPrev->Delete ( );
        this->ModulesPrev = NULL;
    }
    if ( this->ModulesNext ) {
      this->ModulesNext->SetParent ( NULL );
        this->ModulesNext->Delete ( );
        this->ModulesNext = NULL;
    }
    if ( this->ModulesHistory) {
      this->ModulesHistory->SetParent ( NULL );
        this->ModulesHistory->Delete ( );
        this->ModulesHistory = NULL;
    }
    if ( this->ModulesRefresh) {
      this->ModulesRefresh->SetParent ( NULL );
        this->ModulesRefresh->Delete ( );
        this->ModulesRefresh = NULL;
    }

    //--- widgets from LogoFrame
    if (this->SlicerLogoLabel ) {
      this->SlicerLogoLabel->SetParent ( NULL );
        this->SlicerLogoLabel->Delete();
        this->SlicerLogoLabel = NULL;
    }

    //--- widgets from the SliceControlFrame
    if ( this->ToggleAnnotationButton ) {
      this->ToggleAnnotationButton->SetParent ( NULL );
        this->ToggleAnnotationButton->Delete ( );
        this->ToggleAnnotationButton = NULL;
    }
    if ( this->ToggleFgBgButton ) {
      this->ToggleFgBgButton->SetParent ( NULL );
        this->ToggleFgBgButton->Delete ( );
        this->ToggleFgBgButton = NULL;
    }
    if ( this->SliceFadeScale ) {
      this->SliceFadeScale->SetParent ( NULL );
        this->SliceFadeScale->Delete ( );
        this->SliceFadeScale = NULL;
    }

    if ( this->SliceOpacityScale ) {
      this->SliceOpacityScale->SetParent ( NULL );
        this->SliceOpacityScale->Delete ( );
        this->SliceOpacityScale = NULL;
    }
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DeleteFrames ( )
{
    if ( this->LogoFrame ) {
      this->LogoFrame->SetParent ( NULL );
        this->LogoFrame->Delete ();
        this->LogoFrame = NULL;
    }
    if ( this->ModuleChooseFrame ) {
      this->ModuleChooseFrame->SetParent ( NULL );
        this->ModuleChooseFrame->Delete ();
        this->ModuleChooseFrame = NULL;
    }
    if ( this->SliceControlFrame ) {
      this->SliceControlFrame->SetParent ( NULL );
        this->SliceControlFrame->Delete ( );
        this->SliceControlFrame = NULL;
    }
    if ( this->ViewControlFrame ) {
      this->ViewControlFrame->SetParent ( NULL );
        this->ViewControlFrame->Delete ( );
        this->ViewControlFrame = NULL;
    }
}




//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildMainViewer ( int arrangementType)
{

  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
      vtkSlicerWindow *win = this->MainSlicerWin;
        
      // If Main viewer and 3 main sliceGUIs already exist, destroy them.
      this->Save3DViewConfig ( );
      this->DestroyMain3DViewer ( );
      this->DestroyMainSliceViewers ( );

      this->CreateMainSliceViewers ( arrangementType );
      this->CreateMain3DViewer (arrangementType );
      this->Restore3DViewConfig ( );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::CreateMainSliceViewers ( int arrangementType )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      //
      // 3 Slice Viewers
      //
      this->MainSliceGUI0 = vtkSlicerSliceGUI::New ( );
      this->MainSliceGUI0->SetApplication ( app );
      this->MainSliceGUI0->SetApplicationLogic ( this->ApplicationLogic );

      this->MainSliceGUI1 = vtkSlicerSliceGUI::New ( );
      this->MainSliceGUI1->SetApplication ( app );
      this->MainSliceGUI1->SetApplicationLogic ( this->ApplicationLogic );

      this->MainSliceGUI2 = vtkSlicerSliceGUI::New ( );
      this->MainSliceGUI2->SetApplication ( app );
      this->MainSliceGUI2->SetApplicationLogic ( this->ApplicationLogic );
      
      // TO DO: move this into CreateMainSliceViewers?
      // add observers on GUI, MRML 
      // add observers on Logic
      if ( this->MainSliceLogic0 )
        {
          this->MainSliceGUI0->SetAndObserveModuleLogic ( this->MainSliceLogic0 );
        }
      if (this->MainSliceLogic1 )
        {
          this->MainSliceGUI1->SetAndObserveModuleLogic ( this->MainSliceLogic1 );
        }
      if (this->MainSliceLogic2 )
        {
          this->MainSliceGUI2->SetAndObserveModuleLogic ( this->MainSliceLogic2 );
        }


      if ( this->MainSliceGUI0 )
        {
          this->MainSliceGUI0->AddGUIObservers ( );
          this->MainSliceGUI0->SetAndObserveMRMLScene ( this->MRMLScene );
        }
      if ( this->MainSliceGUI1 )
        {
          this->MainSliceGUI1->AddGUIObservers ( );
          this->MainSliceGUI1->SetAndObserveMRMLScene ( this->MRMLScene );
        }
      if ( this->MainSliceGUI2 )
        {
          this->MainSliceGUI2->AddGUIObservers ( );
          this->MainSliceGUI2->SetAndObserveMRMLScene ( this->MRMLScene );
        }

      // parent the sliceGUI  based on selected view arrangement & build
      switch ( arrangementType )
        {
        case vtkSlicerGUILayout::SlicerLayoutInitialView:
          this->DisplayConventionalView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutDefaultView:
          this->DisplayConventionalView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutFourUpView:
          this->DisplayFourUpView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutOneUp3DView:
          this->DisplayOneUp3DView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutOneUpSliceView:
          this->DisplayOneUpSliceView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutTabbed3DView:
          this->DisplayTabbed3DViewSliceViewers ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutTabbedSliceView:
          this->DisplayTabbedSliceView ( );
          break;
        case vtkSlicerGUILayout::SlicerLayoutLightboxView:
          this->DisplayLightboxView ( );
          break;
        default:
          break;
        }

    }
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::CreateMain3DViewer ( int arrangementType )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
      //
      // 3D Viewer
      //
      // only re-create the 3D view when we need it...
      if ( (arrangementType != vtkSlicerGUILayout::SlicerLayoutOneUpSliceView) &&
           (arrangementType != vtkSlicerGUILayout::SlicerLayoutLightboxView ) &&
           (arrangementType != vtkSlicerGUILayout::SlicerLayoutTabbedSliceView) )
        {
          this->MainSlicerWin->GetViewNotebook()->RemovePagesMatchingTag(this->ViewerPageTag );      
          this->ViewerWidget = vtkSlicerViewerWidget::New ( );
          this->ViewerWidget->SetApplication( app );
          if ( arrangementType == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->ViewerWidget->SetParent ( this->GetLightboxFrame ( ) );
            }
          else
            {
              this->ViewerWidget->SetParent(this->MainSlicerWin->GetViewFrame());
            }
          this->ViewerWidget->SetMRMLScene(this->MRMLScene);
          this->ViewerWidget->Create();
          this->ViewerWidget->GetMainViewer()->SetRendererBackgroundColor (app->GetSlicerTheme()->GetSlicerColors()->ViewerBlue );
          this->ViewerWidget->UpdateFromMRML();
          if ( arrangementType == vtkSlicerGUILayout::SlicerLayoutFourUpView )
            {
              this->ViewerWidget->GridWidget ( 0, 1 );
            }
          else
            {
              this->ViewerWidget->PackWidget();
            }

          // TODO: this requires a change to KWWidgets
          this->PlaneWidget = vtkImplicitPlaneWidget::New();
          this->PlaneWidget->SetInteractor( this->GetRenderWindowInteractor() );
          this->PlaneWidget->PlaceWidget();
          this->PlaneWidget->On();
        }
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::Save3DViewConfig ( )
{
  if ( this->ViewerWidget )
    {
      // TODO: Save the ViewerWidget's Camera Node
      this->ViewerWidget->GetMainViewer()->GetRenderer()->ComputeVisiblePropBounds ( this->MainRendererBBox );
    }
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::Restore3DViewConfig ( )
{
  if ( this->ViewerWidget )
    {
      // TODO: Restore the ViewerWidget's Camera Node
      this->ViewerWidget->GetMainViewer()->GetRenderer()->ResetCamera ( );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayConventionalView ( )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );

      // Expose the main panel frame and secondary panel frame.
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 1 );
      
      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIRed );
      this->MainSliceGUI0->PackGUI ( );      
      // Yellow slice viewer
      this->MainSliceGUI1->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIYellow );
      this->MainSliceGUI1->PackGUI ( );
      // Green slice viewer          
      this->MainSliceGUI2->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIGreen );
      this->MainSliceGUI2->PackGUI ( );

      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );
      //      layout->ConfigureMainSlicerWindow ( );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutDefaultView );
      this->MainSlicerWin->GetSecondarySplitFrame()->SetFrame1Size ( layout->GetDefaultSliceGUIFrameHeight () );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayOneUp3DView ( )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );

      // Expose the main panel frame only
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 0 );
      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );      

      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIRed );
      this->MainSliceGUI0->PackGUI ( );
      // Yellow slice viewer
      this->MainSliceGUI1->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIYellow );
      this->MainSliceGUI1->PackGUI ( );
      // Green slice viewer          
      this->MainSliceGUI2->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIGreen );
      this->MainSliceGUI2->PackGUI ( );
      
      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutOneUp3DView );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayOneUpSliceView ( )
{
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );

      // Expose the main panel frame only
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 0 );
      
      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->MainSlicerWin->GetViewFrame ( ), color->SliceGUIRed );
      this->MainSliceGUI0->PackGUI ( );
      // Yellow slice viewer
      this->MainSliceGUI1->BuildGUI ( NULL, color->SliceGUIYellow );
      this->MainSliceGUI1->PackGUI ( );
      // Green slice viewer
      this->MainSliceGUI2->BuildGUI ( NULL, color->SliceGUIGreen );
      this->MainSliceGUI2->PackGUI ( );

      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutOneUpSliceView );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayFourUpView ( )
{

  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );

      // Expose both the main panel frame and secondary panel frame
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 0 );
      this->MainSlicerWin->GetSecondarySplitFrame()->SetFrame1Size ( 0 );
      
      // Use this frame in MainSlicerWin's ViewFrame to grid in the various viewers.
      this->LightboxFrame = vtkKWFrame::New ( );
      this->LightboxFrame->SetParent ( this->MainSlicerWin->GetViewFrame ( ) );
      this->LightboxFrame->Create ( );
      this->Script ( "pack %s -side top -fill both -expand y -padx 0 -pady 0 ", this->LightboxFrame->GetWidgetName ( ) );
      this->Script ("grid rowconfigure %s 0 -weight 1", this->LightboxFrame->GetWidgetName() );
      this->Script ("grid rowconfigure %s 1 -weight 1", this->LightboxFrame->GetWidgetName() );
      this->Script ("grid columnconfigure %s 0 -weight 1", this->LightboxFrame->GetWidgetName() );
      this->Script ("grid columnconfigure %s 1 -weight 1", this->LightboxFrame->GetWidgetName() );
      
      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->GetLightboxFrame ( ), color->SliceGUIRed );
       this->MainSliceGUI0->GridGUI ( 0, 0 );
      // Yellow slice viewer
      this->MainSliceGUI1->BuildGUI ( this->GetLightboxFrame ( ), color->SliceGUIYellow );
      this->MainSliceGUI1->GridGUI ( 1, 0 );
      // Green slice viewer          
      this->MainSliceGUI2->BuildGUI ( this->GetLightboxFrame ( ), color->SliceGUIGreen );
      this->MainSliceGUI2->GridGUI ( 1, 1 );

      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutFourUpView );
    }
}




//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayTabbed3DViewSliceViewers ( )
{

  // TODO: implement multi-tabbed ViewerWidgets
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
      this->MainSlicerWin->GetSecondarySplitFrame()->SetFrame1Size ( layout->GetDefaultSliceGUIFrameHeight() );      
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 0 );

      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIRed );
      this->MainSliceGUI0->PackGUI ( );
      // Yellow slice viewer
      this->MainSliceGUI1->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIYellow );
      this->MainSliceGUI1->PackGUI ( );
      // Green slice viewer
      this->MainSliceGUI2->BuildGUI ( this->MainSlicerWin->GetSecondaryPanelFrame ( ), color->SliceGUIGreen );
      this->MainSliceGUI2->PackGUI ( );
      // Tab the 3D view
      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 1 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutTabbed3DView );
    }

}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayTabbedSliceView ( )
{
  // TODO: implement this and add an icon on the toolbar for it
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );
      this->MainSlicerWin->SetMainPanelVisibility ( 1 );
      this->MainSlicerWin->SetSecondaryPanelVisibility ( 0 );

      // Red slice viewer
      this->MainSliceGUI0->BuildGUI ( this->MainSlicerWin->GetViewFrame ( ), color->SliceGUIRed );
      this->MainSliceGUI0->PackGUI ( );
      // Yellow slice viewer
      this->MainSlicerWin->GetViewNotebook()->AddPage("yellow slice", NULL, NULL, this->ViewerPageTag );
      this->MainSliceGUI1->BuildGUI ( this->MainSlicerWin->GetViewFrame ( ), color->SliceGUIYellow );
      this->MainSliceGUI1->PackGUI ( );
      // Green slice viewer          
      this->MainSlicerWin->GetViewNotebook()->AddPage("green slice", NULL, NULL, this->ViewerPageTag );
      this->MainSliceGUI2->BuildGUI ( this->MainSlicerWin->GetViewFrame ( ), color->SliceGUIGreen );
      this->MainSliceGUI2->PackGUI ( );      
      // Tab the Slice views
      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 1 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutTabbedSliceView );
    }
}

//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::DisplayLightboxView ( )
{
  /*
  // TO DO implement this.
  if ( this->GetApplication() != NULL )
    {
      vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
      vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
      vtkSlicerGUILayout *layout = app->GetMainLayout ( );

      this->MainSlicerWin->GetViewNotebook()->SetAlwaysShowTabs ( 0 );
      layout->SetCurrentViewArrangement ( vtkSlicerGUILayout::SlicerLayoutLightboxView );
    }
  */
}




//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::AddMainSliceViewersToCollection ( )
{
  if ( this->SliceGUICollection != NULL )
    {
      if ( this->MainSliceGUI0 )
        {
          this->AddSliceGUIToCollection ( this->MainSliceGUI0 );
        }
      if ( this->MainSliceGUI1 )
        {
          this->AddSliceGUIToCollection ( this->MainSliceGUI1 );
        }
      if ( this->MainSliceGUI2 )
        {
          this->AddSliceGUIToCollection ( this->MainSliceGUI2 );
        }
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::RemoveMainSliceViewersFromCollection ( )
{
  if ( this->SliceGUICollection != NULL )
    {
      if ( this->MainSliceGUI0 )
        {
          this->RemoveSliceGUIFromCollection ( this->MainSliceGUI0 );
        }
      if ( this->MainSliceGUI1 )
        {
          this->RemoveSliceGUIFromCollection ( this->MainSliceGUI1 );
        }
      if ( this->MainSliceGUI2 )
        {
          this->RemoveSliceGUIFromCollection ( this->MainSliceGUI2 );
        }
    }
}




//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::AddSliceGUIToCollection ( vtkSlicerSliceGUI *s)
{
  
    if ( ( this->SliceGUICollection != NULL) && (s != NULL ) ) {
      this->SliceGUICollection->AddItem ( s );
    }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::RemoveSliceGUIFromCollection ( vtkSlicerSliceGUI *s )
{
    if ( (this->SliceGUICollection != NULL) && (s != NULL))
      {
           this->SliceGUICollection->InitTraversal ( );
            vtkSlicerSliceGUI *g = vtkSlicerSliceGUI::SafeDownCast ( this->SliceGUICollection->GetNextItemAsObject ( ) );
            while ( g != NULL ) {
                if ( g == s )
                    {
                        this->SliceGUICollection->RemoveItem ( g );
                        break;
                    }
                g = vtkSlicerSliceGUI::SafeDownCast (this->SliceGUICollection->GetNextItemAsObject ( ) );
            }
      }
}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::ConfigureMainSliceViewers ( )
{
  if ( this->MainSliceGUI0 && this->MainSliceGUI1 && this->MainSliceGUI2 )
    {
      this->GetMainSliceGUI0()->GetSliceController()->GetSliceNode()->SetOrientationToAxial();
      this->GetMainSliceGUI1()->GetSliceController()->GetSliceNode()->SetOrientationToSagittal();
      this->GetMainSliceGUI2()->GetSliceController()->GetSliceNode()->SetOrientationToCoronal();
    }
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::AddMainSliceViewerObservers ( )
{
  if ( this->MainSliceGUI0 && this->MainSliceGUI1 && this->MainSliceGUI2 )
    {
      this->GetMainSliceGUI0()->AddGUIObservers () ;
      this->GetMainSliceGUI1()->AddGUIObservers ();
      this->GetMainSliceGUI2()->AddGUIObservers ();
    }
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::RemoveMainSliceViewerObservers ( )
{
  if ( this->MainSliceGUI0 && this->MainSliceGUI1 && this->MainSliceGUI2 )
    {
      this->GetMainSliceGUI0()->RemoveGUIObservers () ;
      this->GetMainSliceGUI1()->RemoveGUIObservers ();
      this->GetMainSliceGUI2()->RemoveGUIObservers ();
    }
}


  
//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::SetAndObserveMainSliceLogic ( vtkSlicerSliceLogic *l0,
                                                            vtkSlicerSliceLogic *l1,
                                                            vtkSlicerSliceLogic *l2 )
{

  if ( this->MainSliceGUI0 && this->MainSliceGUI1 && this->MainSliceGUI2 )
    {
      this->GetMainSliceGUI0()->SetAndObserveModuleLogic ( l0 );
      this->GetMainSliceGUI1()->SetAndObserveModuleLogic ( l1 );
      this->GetMainSliceGUI2()->SetAndObserveModuleLogic ( l2 );
    }
  // Set and register the main slice logic here to reassign when
  // viewers are destroyed and recreated during view layout changes.
  this->SetMainSliceLogic0 ( l0 );
  this->SetMainSliceLogic1 ( l1 );
  this->SetMainSliceLogic2 ( l2 );

}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildLogoGUIPanel ( )
{
    if ( this->GetApplication( )  != NULL ) {
        vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast ( this->GetApplication () );
        this->SlicerLogoLabel->SetParent ( this->LogoFrame );
        this->SlicerLogoLabel->Create();
        this->SlicerLogoLabel->SetImageToIcon ( this->SlicerLogoIcons->GetSlicerLogo() );
        this->SlicerLogoLabel->SetBalloonHelpString ("placeholder logo");
        app->Script ( "pack %s -side top -anchor w -padx 2 -pady 0", this->SlicerLogoLabel->GetWidgetName( ) );        
    }
    
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::PopulateModuleChooseList ( )
{
    const char* mName;
    vtkSlicerModuleGUI *m;

    if ( this->GetApplication( )  != NULL ) {
        vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast( this->GetApplication() );
        //--- ALL modules pull-down menu 
        if ( app->GetModuleGUICollection ( ) != NULL ) {
            app->GetModuleGUICollection( )->InitTraversal( );
            m = vtkSlicerModuleGUI::SafeDownCast( app->GetModuleGUICollection( )->GetNextItemAsObject( ));
            while ( m != NULL ) {
                mName = m->GetUIPanel( )->GetName( );
                this->ModulesMenuButton->GetMenu( )->AddRadioButton( mName );
                m = vtkSlicerModuleGUI::SafeDownCast( app->GetModuleGUICollection( )->GetNextItemAsObject( ));
            }
        }
        //--- TODO: make the initial value be module user sets as "home"
        this->ModulesMenuButton->SetValue ("Volumes");
    }

}


//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildModuleChooseGUIPanel ( )
{
    
    if ( this->GetApplication( )  != NULL ) {
        vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast( this->GetApplication() );
        
        //--- ALL modules menu button label
        this->ModulesLabel->SetParent ( this->ModuleChooseFrame );
        this->ModulesLabel->Create ( );

        this->ModulesLabel->SetText ( "Modules:");
        this->ModulesLabel->SetAnchorToWest ( );
        this->ModulesLabel->SetWidth ( 7 );

        //--- All modules menu button
        this->ModulesMenuButton->SetParent ( this->ModuleChooseFrame );
        this->ModulesMenuButton->Create ( );
        this->ModulesMenuButton->SetWidth ( 28 );
        this->ModulesMenuButton->IndicatorVisibilityOn ( );
        this->ModulesMenuButton->SetBalloonHelpString ("Select a Slicer module.");

        //--- Next and previous module button
        this->ModulesNext->SetParent ( this->ModuleChooseFrame );
        this->ModulesNext->Create ( );
        this->ModulesNext->SetBorderWidth ( 0 );
        this->ModulesNext->SetImageToIcon ( this->SlicerModuleNavigationIcons->GetModuleNextIcon() );
        this->ModulesNext->SetBalloonHelpString ("Navigate to the next module in your use history.");

        this->ModulesPrev->SetParent ( this->ModuleChooseFrame );
        this->ModulesPrev->Create ( );
        this->ModulesPrev->SetBorderWidth ( 0 );
        this->ModulesPrev->SetImageToIcon ( this->SlicerModuleNavigationIcons->GetModulePrevIcon() );
        this->ModulesPrev->SetBalloonHelpString ("Navigate to the previous module in your use history.");
        
        this->ModulesHistory->SetParent ( this->ModuleChooseFrame );
        this->ModulesHistory->Create ( );
        this->ModulesHistory->SetBorderWidth ( 0 );
        this->ModulesHistory->SetImageToIcon ( this->SlicerModuleNavigationIcons->GetModuleHistoryIcon() );
        this->ModulesHistory->SetBalloonHelpString ("Pop up a window showing your module use history.");

        this->ModulesRefresh->SetParent ( this->ModuleChooseFrame );
        this->ModulesRefresh->Create ( );
        this->ModulesRefresh->SetBorderWidth ( 0 );
        this->ModulesRefresh->SetImageToIcon ( this->SlicerModuleNavigationIcons->GetModuleRefreshIcon() );
        this->ModulesRefresh->SetBalloonHelpString ("Refresh the list of available modules.");
        
        //--- pack everything up.
        app->Script ( "pack %s -side left -anchor n -padx 1 -ipadx 1 -pady 3", this->ModulesLabel->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor n -padx 1 -ipady 0 -pady 2", this->ModulesMenuButton->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor c -padx 2 -pady 2", this->ModulesPrev->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor c -padx 2 -pady 2", this->ModulesNext->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor c -padx 2 -pady 2", this->ModulesHistory->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor c -padx 2 -pady 2", this->ModulesRefresh->GetWidgetName( ) );
    }
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildSliceControlGUIPanel ( )
{

    //--- Populate the Slice Control Frame

    if ( this->GetApplication( )  != NULL ) {
        vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast( this->GetApplication() );
        this->SliceControlFrame->SetReliefToGroove();
        
        //--- create frames
        vtkKWFrame *f1 = vtkKWFrame::New ( );
        f1->SetParent ( this->SliceControlFrame );
        f1->Create ( );
        vtkKWFrame *f2 = vtkKWFrame::New ( );
        f2->SetParent ( this->SliceControlFrame );
        f2->Create ( );
        vtkKWFrame *f3 = vtkKWFrame::New ( );
        f3->SetParent ( this->SliceControlFrame );
        f3->Create ( );
        
        //--- pack everything up: buttons, labels, scales
        app->Script ( "pack %s -side left -anchor n -padx 0 -pady 5", f1->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor n -padx 0 -pady 5", f2->GetWidgetName( ) );
        app->Script ( "pack %s -side left -anchor n -padx 0 -pady 5", f3->GetWidgetName( ) );

        //--- make buttons for toggling Bg/Fg and annotations
        this->ToggleFgBgButton->SetParent ( f1 );
        this->ToggleFgBgButton->Create ( );
        this->ToggleFgBgButton->SetWidth ( 16 );
        this->ToggleFgBgButton->SetText ( "Toggle Bg/Fg" );
        this->ToggleAnnotationButton->SetParent ( f1 );
        this->ToggleAnnotationButton->Create ( );
        this->ToggleAnnotationButton->SetWidth ( 16 );
        this->ToggleAnnotationButton->SetText ( "Toggle Annotation" );
    
        app->Script ( "pack %s -side top -anchor w -padx 1 -pady 1", this->ToggleFgBgButton->GetWidgetName( ) );
        app->Script ( "pack %s -side top -anchor w -padx 1 -pady 1", this->ToggleAnnotationButton->GetWidgetName( ) );

        //--- make labels (can't reposition the Scale's labels, so
        //--- supressing those and using a new set.)
        vtkKWLabel *fadeLabel = vtkKWLabel::New ( );
        vtkKWLabel *opacityLabel = vtkKWLabel::New ( );
        fadeLabel->SetParent ( f2 );
        fadeLabel->Create ( );
        fadeLabel->SetWidth ( 14 );
        fadeLabel->SetAnchorToEast ( );
        fadeLabel->SetText ( "Fade (Bg/Fg):");
        opacityLabel->SetParent ( f2 );
        opacityLabel->Create ( );
        opacityLabel->SetWidth ( 14 );
        opacityLabel->SetAnchorToEast ( );
        opacityLabel->SetText ( "Label Opacity:");
        app->Script ( "pack %s -side top -anchor e -padx 1 -pady 1", fadeLabel->GetWidgetName( ) );
        app->Script ( "pack %s -side top -anchor e -padx 1 -pady 2", opacityLabel->GetWidgetName( ) );
        
        //--- make scales for sliding slice visibility in the SliceViewers
        //--- and for sliding slice opacity in the 3D Viewer.
        this->SliceFadeScale->SetParent ( f3 );
        this->SliceFadeScale->Create ( );
        this->SliceFadeScale->SetRange (0.0, 1.0);
        this->SliceFadeScale->SetResolution ( 0.01 );
        this->SliceFadeScale->SetValue ( 0.0 );
        this->SliceFadeScale->SetLength ( 120 );
        this->SliceFadeScale->SetOrientationToHorizontal ( );
        this->SliceFadeScale->ValueVisibilityOff ( );
        this->SliceFadeScale->SetBalloonHelpString ( "Scale fades between Bg and Fg Slice Layers" );

        this->SliceOpacityScale->SetParent ( f3 );
        this->SliceOpacityScale->Create ( );
        this->SliceOpacityScale->SetRange ( 0.0, 1.0 );
        this->SliceOpacityScale->SetResolution ( 0.01 );
        this->SliceOpacityScale->SetValue ( 1.0 );
        this->SliceOpacityScale->SetLength ( 120 );
        this->SliceOpacityScale->SetOrientationToHorizontal ( );
        this->SliceOpacityScale->ValueVisibilityOff ( );
        this->SliceOpacityScale->SetBalloonHelpString ( "Scale sets the opacity label overlay" );

        app->Script ( "pack %s -side top -anchor w -padx 0 -pady 1", this->SliceFadeScale->GetWidgetName( ) );
        app->Script ( "pack %s -side top -anchor w -padx 0 -pady 0", this->SliceOpacityScale->GetWidgetName( ) );

        fadeLabel->Delete ( );
        opacityLabel->Delete ( );
        f1->Delete ( );
        f2->Delete ( );
        f3->Delete ( );
    }
}





//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::PackFirstSliceViewerFrame ( )
{
  /*
  this->Script ("pack %s -side left  -expand y -fill both -padx 0 -pady 0", 
    this->DefaultSlice0Frame->GetWidgetName( ) );
  */
}



//---------------------------------------------------------------------------
void vtkSlicerApplicationGUI::BuildGUIPanel ( )
{


    if ( this->GetApplication() != NULL ) {
        // pointers for convenience
        vtkSlicerApplication *app = vtkSlicerApplication::SafeDownCast(this->GetApplication() );
        vtkSlicerGUILayout *layout = app->GetMainLayout ( );
        
        if ( this->MainSlicerWin != NULL ) {

            this->MainSlicerWin->GetMainPanelFrame()->SetWidth ( layout->GetDefaultGUIPanelWidth() );
            this->MainSlicerWin->GetMainPanelFrame()->SetHeight ( layout->GetDefaultGUIPanelHeight() );
            this->MainSlicerWin->GetMainPanelFrame()->SetReliefToSunken();

            this->LogoFrame->SetParent ( this->MainSlicerWin->GetMainPanelFrame ( ) );
            this->LogoFrame->Create( );
            this->LogoFrame->SetHeight ( layout->GetDefaultLogoFrameHeight ( ) );

            this->ModuleChooseFrame->SetParent ( this->MainSlicerWin->GetMainPanelFrame ( ) );
            this->ModuleChooseFrame->Create( );
            this->ModuleChooseFrame->SetHeight ( layout->GetDefaultModuleChooseFrameHeight ( ) );

            this->SliceControlFrame->SetParent ( this->MainSlicerWin->GetMainPanelFrame ( ) );
            this->SliceControlFrame->Create( );
            this->SliceControlFrame->SetHeight ( layout->GetDefaultSliceControlFrameHeight ( ) );
            
            this->ViewControlFrame->SetParent ( this->MainSlicerWin->GetMainPanelFrame ( ) );
            this->ViewControlFrame->Create( );
            this->ViewControlFrame->SetHeight (layout->GetDefaultViewControlFrameHeight ( ) );
            
            // pack logo and slicer control frames
            this->Script ( "pack %s -side top -fill x -padx 1 -pady 1", this->LogoFrame->GetWidgetName() );
            app->Script ( "pack %s -side top -fill x -padx 1 -pady 10", this->ModuleChooseFrame->GetWidgetName() );

            // pack slice and view control frames
            app->Script ( "pack %s -side bottom -expand n -fill x -padx 1 -pady 10", this->ViewControlFrame->GetWidgetName() );
            app->Script ( "pack %s -side bottom -expand n -fill x -padx 1 -pady 10", this->SliceControlFrame->GetWidgetName() );

        }
    }

}


