/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/EndoTracking/Modules/EndoNav/vtkEndoNavGUI.cxx $
  Date:      $Date: 2009-01-14 13:52:57 -0500 (Wed, 14 Jan 2009) $
  Version:   $Revision: 8387 $

==========================================================================*/

#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkEndoNavGUI.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerSliceGUI.h"
#include "vtkSlicerSlicesGUI.h"

#include "vtkSlicerColor.h"
#include "vtkSlicerTheme.h"

#include "vtkSlicerFiducialsGUI.h"
#include "vtkSlicerFiducialsLogic.h"

#include "vtkKWRenderWidget.h"
#include "vtkKWWidget.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"
#include "vtkKWRadioButton.h"
#include "vtkKWRadioButtonSet.h"
#include "vtkKWPushButton.h"
#include "vtkKWPushButtonSet.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWMenu.h"
#include "vtkKWLabel.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMessageDialog.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkKWEvent.h"
#include "vtkKWOptions.h"
#include "vtkKWComboBox.h"

#include "vtkKWTreeWithScrollbars.h"
#include "vtkKWTree.h"

#include "vtkKWTkUtilities.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkCornerAnnotation.h"

#include "vtkStringArray.h"


// for Realtime Image
#include "vtkImageChangeInformation.h"
#include "vtkSlicerColorLogic.h"

#include "vtkMRMLLinearTransformNode.h"

#include "vtkIGTLConnector.h"

#include <vector>
#include <sstream>

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkEndoNavGUI );
vtkCxxRevisionMacro ( vtkEndoNavGUI, "$Revision: 8387 $");
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// Labels
const char *vtkEndoNavGUI::ConnectorTypeStr[vtkIGTLConnector::NUM_TYPE] = 
{
  "?", // TYPE_NOT_DEFINED
  "S", // TYPE_SERVER
  "C", // TYPE_CLIENT
};

const char *vtkEndoNavGUI::ConnectorStatusStr[vtkIGTLConnector::NUM_STATE] = 
{
  "OFF",       // OFF
  "WAIT",      // WAIT_CONNECTION
  "ON",        // CONNECTED
};

//---------------------------------------------------------------------------
vtkEndoNavGUI::vtkEndoNavGUI ( )
{

  //----------------------------------------------------------------
  // Logic values
  
  this->Logic = NULL;
  this->DataManager = vtkIGTDataManager::New();
  this->Pat2ImgReg = vtkIGTPat2ImgRegistration::New();
  
  this->DataCallbackCommand = vtkCallbackCommand::New();
  this->DataCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
  this->DataCallbackCommand->SetCallback(vtkEndoNavGUI::DataCallback);
  
  this->SliceNode0 = NULL; 
  this->SliceNode1 = NULL; 
  this->SliceNode2 = NULL; 
  
  //----------------------------------------------------------------
  // Workphase Frame
  
  //----------------------------------------------------------------
  // Connector List Frame

  this->ConnectorList = NULL;
  this->AddConnectorButton = NULL;
  this->DeleteConnectorButton = NULL;
  this->ConnectorNameEntry = NULL;
  this->ConnectorTypeButtonSet = NULL;
  this->ConnectorStatusCheckButton = NULL;
  this->ConnectorAddressEntry = NULL;
  this->ConnectorPortEntry = NULL;

  //----------------------------------------------------------------
  // Data I/O Configuration frame

  this->IOConfigTree = NULL;
  this->EnableAdvancedSettingButton = NULL;
  this->IOConfigContextMenu = NULL;

  this->CurrentMrmlNodeListIndex = -1;
  this->CurrentNodeListAvailable.clear();
  this->CurrentNodeListSelected.clear();


  //----------------------------------------------------------------
  // Visualization Control Frame
  
  this->FreezeImageCheckButton  = NULL;
  this->ObliqueCheckButton      = NULL;
  this->SetLocatorModeButton    = NULL;
  this->SetUserModeButton       = NULL;
  this->RedSliceMenu            = NULL;
  this->YellowSliceMenu         = NULL;
  this->GreenSliceMenu          = NULL;
  this->LocatorCheckButton      = NULL;
  this->LocatorSourceMenu       = NULL;
  this->RealTimeImageSourceMenu = NULL;
  this->ImagingMenu             = NULL;


  //----------------------------------------------------------------
  // Locator  (MRML)
  this->CloseScene              = false;
  this->TimerFlag = 0;
  this->TimerInterval = 100;  // 100 ms

  this->ConnectorIDList.clear();
  this->IOConfigTreeConnectorList.clear();
  this->IOConfigTreeIOList.clear();
  this->IOConfigTreeNodeList.clear();

  this->ImageConverter = NULL;
  this->TransformConverter = NULL;
  this->ImageViewerUS = NULL;
  this->ImageViewerCT = NULL;


}

//---------------------------------------------------------------------------
vtkEndoNavGUI::~vtkEndoNavGUI ( )
{

  if (this->DataManager)
    {
    // If we don't set the scene to NULL for DataManager,
    // Slicer will report a lot leak when it is closed.
    this->DataManager->SetMRMLScene(NULL);
    this->DataManager->Delete();
    }
  if (this->Pat2ImgReg)
    {
    this->Pat2ImgReg->Delete();
    }
  if (this->DataCallbackCommand)
    {
    this->DataCallbackCommand->Delete();
    }

  // If Logic is NULL, then we only instatiated the class and never used
  // it, e.g. --ignore_module
  if (this->Logic)
    {
    this->RemoveGUIObservers();
    }

  this->SetModuleLogic ( NULL );


  //----------------------------------------------------------------
  // Visualization Control Frame

  if (this->EnableAdvancedSettingButton)
    {
    this->EnableAdvancedSettingButton->SetParent(NULL );
    this->EnableAdvancedSettingButton->Delete ( );
    }
  if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->SetParent(NULL );
    this->FreezeImageCheckButton->Delete ( );
    }
  if (this->ObliqueCheckButton)
    {
    this->ObliqueCheckButton->SetParent(NULL );
    this->ObliqueCheckButton->Delete ( );
    }


  if (this->SetLocatorModeButton)
    {
    this->SetLocatorModeButton->SetParent(NULL);
    this->SetLocatorModeButton->Delete();
    }
  
  if (this->SetUserModeButton)
    {
    this->SetUserModeButton->SetParent(NULL);
    this->SetUserModeButton->Delete();
    }

  if (this->RedSliceMenu)
    {
    this->RedSliceMenu->SetParent(NULL );
    this->RedSliceMenu->Delete ( );
    }

  if (this->YellowSliceMenu)
    {
    this->YellowSliceMenu->SetParent(NULL );
    this->YellowSliceMenu->Delete ( );
    }

  if (this->GreenSliceMenu)
    {
    this->GreenSliceMenu->SetParent(NULL );
    this->GreenSliceMenu->Delete ( );
    }

  if ( this->ImagingMenu )
    {
    this->ImagingMenu->SetParent(NULL);
    this->ImagingMenu->Delete();
    }


  this->IsSliceOrientationAdded = false;

  //----------------------------------------------------------------
  // Connector List Frame

  if ( this->ConnectorList )
    {
    this->ConnectorList->SetParent(NULL);
    this->ConnectorList->Delete();
    }

  if ( this->ConnectorTypeButtonSet )
    {
    this->ConnectorTypeButtonSet->SetParent(NULL);
    this->ConnectorTypeButtonSet->Delete();
    }

  if ( this->ConnectorNameEntry )
    {
    this->ConnectorNameEntry->SetParent(NULL);
    this->ConnectorNameEntry->Delete();
    }

  if ( this->AddConnectorButton )
    {
    this->AddConnectorButton->SetParent(NULL);
    this->AddConnectorButton->Delete();
    }

  if ( this->DeleteConnectorButton )
    {
    this->DeleteConnectorButton->SetParent(NULL);
    this->DeleteConnectorButton->Delete();
    }

  if ( this->ConnectorStatusCheckButton )
    {
    this->ConnectorStatusCheckButton->SetParent(NULL);
    this->ConnectorStatusCheckButton->Delete();
    }

  if ( this->ConnectorAddressEntry )
    {
    this->ConnectorAddressEntry->SetParent(NULL);
    this->ConnectorAddressEntry->Delete();
    }

  if ( this->ConnectorPortEntry )
    {
    this->ConnectorPortEntry->SetParent(NULL);
    this->ConnectorPortEntry->Delete();
    }

  if ( this->SleepTimeEntry )
    {
    this->SleepTimeEntry->SetParent(NULL);
    this->SleepTimeEntry->Delete();
    }

  //----------------------------------------------------------------
  // Etc Frame

  if (this->LocatorSourceMenu)
    {
    this->LocatorSourceMenu->SetParent(NULL );
    this->LocatorSourceMenu->Delete ( );
    }

  if (this->LocatorCheckButton)
    {
    this->LocatorCheckButton->SetParent(NULL );
    this->LocatorCheckButton->Delete ( );
    }

  if (this->RealTimeImageSourceMenu)
    {
    this->RealTimeImageSourceMenu->SetParent(NULL );
    this->RealTimeImageSourceMenu->Delete ( );
    }

  if (this->IOConfigContextMenu)
    {
    this->IOConfigContextMenu->SetParent(NULL);
    this->IOConfigContextMenu->Delete();
    }

  if (this->IOConfigTree)
    {
    this->IOConfigTree->SetParent(NULL);
    this->IOConfigTree->Delete();
    }
  if (this->ImageConverter)
    {
    this->ImageConverter->Delete();
    }
  if (this->TransformConverter)
    {
    this->TransformConverter->Delete();
    }
  if (this->ImageViewerUS)
    {
    this->ImageViewerUS->Delete();
    }
  if (this->ImageViewerCT)
    {
    this->ImageViewerCT->Delete();
    }
}



//---------------------------------------------------------------------------
void vtkEndoNavGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
  this->vtkObject::PrintSelf ( os, indent );

  os << indent << "EndoNav: " << this->GetClassName ( ) << "\n";
  os << indent << "Logic: " << this->GetLogic ( ) << "\n";
   
  // print widgets?
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::RemoveGUIObservers ( )
{
  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  
  if ( appGUI && appGUI->GetMainSliceGUI("Red") )
    {
    appGUI->GetMainSliceGUI("Red")->GetSliceViewer()->GetRenderWidget()
      ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
  if ( appGUI && appGUI->GetMainSliceGUI("Yellow") )
    {
    appGUI->GetMainSliceGUI("Yellow")->GetSliceViewer()->GetRenderWidget()
      ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
  if ( appGUI && appGUI->GetMainSliceGUI("Yellow") )
    {
    appGUI->GetMainSliceGUI("Green")->GetSliceViewer()->GetRenderWidget()
      ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  //----------------------------------------------------------------
  // Connector Browser Frame

  if (this->ConnectorList && this->ConnectorList->GetWidget())
    {
    this->ConnectorList->GetWidget()
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  if (this->AddConnectorButton)
    {
    this->AddConnectorButton
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  if (this->DeleteConnectorButton)
    {
    this->DeleteConnectorButton
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }


  if (this->ConnectorNameEntry)
    {
    this->ConnectorNameEntry
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
      
  if (this->ConnectorTypeButtonSet)
    {
    this->ConnectorTypeButtonSet->GetWidget(0)
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    this->ConnectorTypeButtonSet->GetWidget(1)
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
  
  if (this->ConnectorStatusCheckButton)
    {
    this->ConnectorStatusCheckButton
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }

  if (this->ConnectorAddressEntry)
    {
    this->ConnectorAddressEntry
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  if (this->ConnectorPortEntry)
    {
    this->ConnectorPortEntry
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  if (this->SleepTimeEntry)
    {
    this->SleepTimeEntry
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  //----------------------------------------------------------------
  // Data I/O Configuration frame

  if (this->IOConfigTree)
    {
    this->IOConfigTree->GetWidget()
      ->RemoveObservers(vtkKWTree::SelectionChangedEvent, 
                        (vtkCommand *)this->GUICallbackCommand);  
    this->IOConfigTree->GetWidget()
      ->RemoveObservers(vtkKWTree::RightClickOnNodeEvent, 
                        (vtkCommand *)this->GUICallbackCommand);  
    }

  if (this->EnableAdvancedSettingButton)
    {
    this->EnableAdvancedSettingButton
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }

  //----------------------------------------------------------------
  // Visualization Control Frame

  if (this->LocatorCheckButton)
    {
    this->LocatorCheckButton->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }

  if (this->LocatorSourceMenu)
    {
    this->LocatorSourceMenu->GetMenu()
      ->RemoveObserver((vtkCommand*)this->GUICallbackCommand);
    }

  if (this->RealTimeImageSourceMenu)
    {
    this->RealTimeImageSourceMenu->GetMenu()
      ->RemoveObserver((vtkCommand*)this->GUICallbackCommand);
    }

  if (this->RedSliceMenu)
    {
    this->RedSliceMenu->GetMenu()
      ->RemoveObserver((vtkCommand*)this->GUICallbackCommand);
    }
  if (this->YellowSliceMenu)
    {
    this->YellowSliceMenu->GetMenu()
      ->RemoveObserver((vtkCommand*)this->GUICallbackCommand);
    }
  if (this->GreenSliceMenu)
    {
    this->GreenSliceMenu->GetMenu()
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
  if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }
  if (this->ObliqueCheckButton)
    {
    this->ObliqueCheckButton->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SetLocatorModeButton)
    {
    this->SetLocatorModeButton->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SetUserModeButton)
    {
    this->SetUserModeButton->RemoveObserver((vtkCommand *)this->GUICallbackCommand );
    }

  if (this->ImagingMenu)
    {
    this->ImagingMenu->GetMenu()
      ->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    }
  
  this->RemoveLogicObservers();
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::RemoveLogicObservers ( )
{
  //vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  if (this->GetLogic())
    {
    this->GetLogic()->RemoveObservers(vtkCommand::ModifiedEvent,
                                      (vtkCommand *)this->LogicCallbackCommand);
    }
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::AddGUIObservers ( )
{
  this->RemoveGUIObservers();

  // make a user interactor style to process our events
  // look at the InteractorStyle to get our events

  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();

  //----------------------------------------------------------------
  // MRML

  vtkIntArray* events = vtkIntArray::New();
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  events->InsertNextValue(vtkMRMLScene::SceneCloseEvent);
  
  if (this->GetMRMLScene() != NULL)
    {
    this->SetAndObserveMRMLSceneEvents(this->GetMRMLScene(), events);
    }
  events->Delete();

  
  //----------------------------------------------------------------
  // Main Slice GUI

  appGUI->GetMainSliceGUI("Red")->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI("Yellow")->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI("Green")->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  

  //----------------------------------------------------------------
  // Connector Browser Frame

  //vtkKWMultiColumnListWithScrollbars* ConnectorList;
  
  this->ConnectorList->GetWidget()
    ->AddObserver(vtkKWMultiColumnList::SelectionChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->AddConnectorButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->DeleteConnectorButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ConnectorNameEntry
    ->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ConnectorTypeButtonSet->GetWidget(0)
    ->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ConnectorTypeButtonSet->GetWidget(1)
    ->AddObserver(vtkKWRadioButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->ConnectorStatusCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);

  this->ConnectorAddressEntry
    ->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ConnectorPortEntry
    ->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->SleepTimeEntry
    ->AddObserver(vtkKWEntry::EntryValueChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  //----------------------------------------------------------------
  // Data I/O Configuration frame
  this->IOConfigTree->GetWidget()
    ->AddObserver(vtkKWTree::SelectionChangedEvent, 
                  (vtkCommand *)this->GUICallbackCommand );

  this->IOConfigTree->GetWidget()
    ->AddObserver(vtkKWTree::RightClickOnNodeEvent, 
                  (vtkCommand *)this->GUICallbackCommand);
  
  this->EnableAdvancedSettingButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);

  //----------------------------------------------------------------
  // Visualization Control Frame

  this->LocatorCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);

  this->LocatorSourceMenu->GetMenu()
    ->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);

  this->RealTimeImageSourceMenu->GetMenu()
    ->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);

  this->RedSliceMenu->GetMenu()
    ->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
  this->YellowSliceMenu->GetMenu()
    ->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand*)this->GUICallbackCommand);
  this->GreenSliceMenu->GetMenu()
    ->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->FreezeImageCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);
  this->ObliqueCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);

  this->SetLocatorModeButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->SetUserModeButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);


  //----------------------------------------------------------------
  // Etc Frame

  // observer load volume button

  this->AddLogicObservers();
  
  
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::AddLogicObservers ( )
{
  this->RemoveLogicObservers();  

  if (this->GetLogic())
    {
    /*
      this->GetLogic()->AddObserver(vtkEndoNavLogic::LocatorUpdateEvent,
      (vtkCommand *)this->LogicCallbackCommand);
    */
    /*
      this->GetLogic()->AddObserver(vtkEndoNavLogic::SliceUpdateEvent,
      (vtkCommand *)this->LogicCallbackCommand);
    */
    this->GetLogic()->AddObserver(vtkEndoNavLogic::StatusUpdateEvent,
                                  (vtkCommand *)this->LogicCallbackCommand);
    }
}

//---------------------------------------------------------------------------
void vtkEndoNavGUI::HandleMouseEvent(vtkSlicerInteractorStyle *style)
{

  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  vtkSlicerInteractorStyle *istyle0 
    = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI("Red")->GetSliceViewer()
                                             ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
  vtkSlicerInteractorStyle *istyle1 
    = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI("Yellow")->GetSliceViewer()
                                             ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
  vtkSlicerInteractorStyle *istyle2 
    = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI("Green")->GetSliceViewer()
                                             ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
  
  vtkCornerAnnotation *anno = NULL;
  if (style == istyle0)
    {
    anno = appGUI->GetMainSliceGUI("Red")->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
  else if (style == istyle1)
    {
    anno = appGUI->GetMainSliceGUI("Yellow")->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
  else if (style == istyle2)
    {
    anno = appGUI->GetMainSliceGUI("Green")->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
  /*
  if (anno)
    {
    const char *rasText = anno->GetText(1);
    if ( rasText != NULL )
      {
      std::string ras = std::string(rasText);
        
      // remove "R:," "A:," and "S:" from the string
      unsigned int loc = ras.find("R:", 0);
      if ( loc != std::string::npos ) 
        {
        ras = ras.replace(loc, 2, "");
        }
      loc = ras.find("A:", 0);
      if ( loc != std::string::npos ) 
        {
        ras = ras.replace(loc, 2, "");
        }
      loc = ras.find("S:", 0);
      if ( loc != std::string::npos ) 
        {
        ras = ras.replace(loc, 2, "");
        }
      
      // remove "\n" from the string
      unsigned int found = ras.find("\n", 0);
      while ( found != std::string::npos )
        {
        ras = ras.replace(found, 1, " ");
        found = ras.find("\n", 0);
        }
      
      }
    }
  */
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::ProcessGUIEvents(vtkObject *caller,
                                         unsigned long event, void *callData)
{

  const char *eventName = vtkCommand::GetStringFromEventId(event);

  if (strcmp(eventName, "LeftButtonPressEvent") == 0)
    {
    vtkSlicerInteractorStyle *style = vtkSlicerInteractorStyle::SafeDownCast(caller);
    HandleMouseEvent(style);
    return;
    }

  //----------------------------------------------------------------
  // Connector Browser Frame

  else if (this->ConnectorList->GetWidget() == vtkKWMultiColumnList::SafeDownCast(caller)
           && event == vtkKWMultiColumnList::SelectionChangedEvent)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    UpdateConnectorPropertyFrame(selected);
    UpdateIOConfigTree();
    }

  else if (this->AddConnectorButton == vtkKWPushButton::SafeDownCast(caller)
           && event == vtkKWPushButton::InvokedEvent)
    {
    this->GetLogic()->AddConnector();
    UpdateConnectorList(UPDATE_ALL);
    int select = this->ConnectorList->GetWidget()->GetNumberOfRows() - 1;
    this->ConnectorList->GetWidget()->SelectSingleRow(select);
    UpdateConnectorPropertyFrame(select);
    UpdateIOConfigTree();
    }

  else if (this->DeleteConnectorButton == vtkKWPushButton::SafeDownCast(caller)
           && event == vtkKWPushButton::InvokedEvent)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    this->GetLogic()->DeleteConnector(id);
    UpdateConnectorList(UPDATE_ALL);
    int nrow = this->ConnectorList->GetWidget()->GetNumberOfRows();
    if (selected >= nrow)
      {
      selected = nrow - 1;
      }
    this->ConnectorList->GetWidget()->SelectSingleRow(selected);
    UpdateConnectorList(UPDATE_ALL);
    UpdateConnectorPropertyFrame(selected);
    UpdateIOConfigTree();
    }
  
  else if (this->ConnectorNameEntry == vtkKWEntry::SafeDownCast(caller)
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      connector->SetName(this->ConnectorNameEntry->GetValue());
      UpdateConnectorList(UPDATE_SELECTED_ONLY);
      }
    }

  else if (this->ConnectorTypeButtonSet->GetWidget(0) == vtkKWRadioButton::SafeDownCast(caller)
           && event == vtkKWRadioButton::SelectedStateChangedEvent
           && this->ConnectorTypeButtonSet->GetWidget(0)->GetSelectedState() == 1)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      connector->SetType(vtkIGTLConnector::TYPE_SERVER);
      UpdateConnectorList(UPDATE_SELECTED_ONLY);
      UpdateConnectorPropertyFrame(selected);
      UpdateIOConfigTree();
      }
    }

  else if (this->ConnectorTypeButtonSet->GetWidget(1) == vtkKWRadioButton::SafeDownCast(caller)
           && event == vtkKWRadioButton::SelectedStateChangedEvent
           && this->ConnectorTypeButtonSet->GetWidget(1)->GetSelectedState() == 1)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      connector->SetType(vtkIGTLConnector::TYPE_CLIENT);
      UpdateConnectorList(UPDATE_SELECTED_ONLY);
      UpdateConnectorPropertyFrame(selected);
      UpdateIOConfigTree();
      }
    }

  else if (this->ConnectorStatusCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      if (this->ConnectorStatusCheckButton->GetSelectedState()) // Activated
        {
        //vtkErrorMacro("Starting Connector...........");
        connector->Start();
        UpdateConnectorList(UPDATE_SELECTED_ONLY);
        UpdateConnectorPropertyFrame(selected);
        UpdateIOConfigTree();
        }
      else  // Deactivated
        {
        connector->Stop();
        //vtkErrorMacro("Connector Stopped...........");
        UpdateConnectorList(UPDATE_SELECTED_ONLY);
        UpdateConnectorPropertyFrame(selected);
        UpdateIOConfigTree();
        }
      }
    }

  else if (this->ConnectorAddressEntry == vtkKWEntry::SafeDownCast(caller)
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      connector->SetServerHostname(this->ConnectorAddressEntry->GetValue());
      }
    UpdateConnectorList(UPDATE_SELECTED_ONLY);
    }

  else if (this->ConnectorPortEntry == vtkKWEntry::SafeDownCast(caller)
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
    int id = -1;
    if (selected >= 0 && selected < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[selected];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      connector->SetServerPort(this->ConnectorPortEntry->GetValueAsInt());
      }
    UpdateConnectorList(UPDATE_SELECTED_ONLY);
    }
  else if (this->SleepTimeEntry == vtkKWEntry::SafeDownCast(caller)
           && event == vtkKWEntry::EntryValueChangedEvent)
    {
    this->SetTimerInterval(SleepTimeEntry->GetValueAsInt());

    UpdateConnectorList(UPDATE_SELECTED_ONLY);
    }
  else if (this->EnableAdvancedSettingButton == vtkKWCheckButton::SafeDownCast(caller)
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    if (this->EnableAdvancedSettingButton->GetSelectedState())
      {
      this->GetLogic()->SetRestrictDeviceName(1);
      }
    else
      {
      this->GetLogic()->SetRestrictDeviceName(0);
      }
    }

  else if (event == vtkKWTree::RightClickOnNodeEvent)
    {
    IOConfigTreeContextMenu((const char*)callData);
    }


  //----------------------------------------------------------------
  // Visualization Control Frame

  // -- this should be moved to Logic?
  else if (this->LocatorCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    int checked = this->LocatorCheckButton->GetSelectedState(); 
    if (!this->CloseScene)
      {
      if (checked)
        {
        //this->GetLogic()->SetVisibilityOfLocatorModel("IGTLocator", 1);
        this->GetLogic()->EnableLocatorDriver(1);
        }
      else
        {
        //this->GetLogic()->SetVisibilityOfLocatorModel("IGTLocator", 0);
        this->GetLogic()->EnableLocatorDriver(0);
        }
      }
    else
      {
      this->CloseScene = false;
      }
    }

  else if (this->LocatorSourceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    int selected = this->LocatorSourceMenu->GetMenu()->GetIndexOfSelectedItem();
    this->GetLogic()->SetLocatorDriver(this->LocatorSourceList[selected].nodeID.c_str());
    }

  else if (this->RealTimeImageSourceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    int selected = this->RealTimeImageSourceMenu->GetMenu()->GetIndexOfSelectedItem();
    this->GetLogic()->SetRealTimeImageSource(this->RealTimeImageSourceList[selected].nodeID.c_str());
    }

  else if (this->RedSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent)
   {
    const char* selected = this->RedSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_RED, selected);
    }

  else if (this->YellowSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    const char* selected = this->YellowSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_YELLOW, selected);
    }

  else if (this->GreenSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    const char* selected = this->GreenSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_GREEN, selected);
    }

  // -- "Locator" button 
  else if (this->SetLocatorModeButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_RED, "Locator");
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_YELLOW, "Locator");
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_GREEN, "Locator");
    }
  
  // -- "User" button 
  else if (this->SetUserModeButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_RED, "User");
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_YELLOW, "User");
    ChangeSlicePlaneDriver(vtkEndoNavGUI::SLICE_PLANE_GREEN, "User");
    }
  
  // -- "Freeze Image Position" check button 
  else if (this->FreezeImageCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    if (this->FreezeImageCheckButton->GetSelectedState() == 1)
      {
      this->GetLogic()->SetFreezePlane(true);
      }
    else
      {
      this->GetLogic()->SetFreezePlane(false);
      }
    }
  else if (this->ObliqueCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    if (this->ObliqueCheckButton->GetSelectedState() == 1)
      {
      this->GetLogic()->SetEnableOblique(true);
      }
    else
      {
      this->GetLogic()->SetEnableOblique(false);
      }
    }

  //----------------------------------------------------------------
  // Etc Frame

          

} 


void vtkEndoNavGUI::Init()
{
  this->DataManager->SetMRMLScene(this->GetMRMLScene());
}



void vtkEndoNavGUI::DataCallback(vtkObject *caller, 
                                     unsigned long eid, void *clientData, void *callData)
{
  vtkEndoNavGUI *self = reinterpret_cast<vtkEndoNavGUI *>(clientData);
  vtkDebugWithObjectMacro(self, "In vtkEndoNavGUI DataCallback");

  self->UpdateAll();
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::ProcessLogicEvents ( vtkObject *caller,
                                             unsigned long event, void *callData )
{

  if (this->GetLogic() == vtkEndoNavLogic::SafeDownCast(caller))
    {
    if (event == vtkEndoNavLogic::StatusUpdateEvent)
      {
      //this->UpdateDeviceStatus();
      }
    }
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event, void *callData )
{
  // Fill in

  if (event == vtkMRMLScene::NodeAddedEvent|| event == vtkMRMLScene::NodeRemovedEvent)
    {
    UpdateLocatorSourceMenu();
    UpdateRealTimeImageSourceMenu();
    }
  if (event == vtkMRMLScene::SceneCloseEvent)
    {
    if (this->LocatorCheckButton != NULL && this->LocatorCheckButton->GetSelectedState())
      {
      this->CloseScene = true;
      this->LocatorCheckButton->SelectedStateOff();
      }
    }
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::ProcessTimerEvents()
{
  if (this->TimerFlag)
    {
    // -----------------------------------------
    // Check connector status

    if (this->GetLogic()->CheckConnectorsStatusUpdates())
      {
      
      //--- update connector list
      int selected = this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow();
      UpdateConnectorList(UPDATE_STATUS_ALL);
      UpdateConnectorPropertyFrame(selected);

      UpdateIOConfigTree();
      }


    // -----------------------------------------
    // Check incomming new data
    this->GetLogic()->ImportFromCircularBuffers();
    vtkKWTkUtilities::CreateTimerHandler(vtkKWApplication::GetMainInterp(), 
                                         this->TimerInterval,
                                         this, "ProcessTimerEvents");        
    }
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::Enter()
{
  // Fill in
  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  
  this->SliceNode0 = appGUI->GetMainSliceGUI("Red")->GetLogic()->GetSliceNode();
  this->SliceNode1 = appGUI->GetMainSliceGUI("Yellow")->GetLogic()->GetSliceNode();
  this->SliceNode2 = appGUI->GetMainSliceGUI("Green")->GetLogic()->GetSliceNode();
  
  //this->GetLogic()->AddRealtimeVolumeNode("Realtime");
  //this->Logic0->GetForegroundLayer()->SetUseReslice(0);
  
  if (this->TimerFlag == 0)
    {
    this->TimerFlag = 1;
    ProcessTimerEvents();
    }

  this->GetLogic()->Initialize();
  this->UpdateConnectorList(UPDATE_ALL);

  /*
  //----------------------------------------------------------------
  // Hack SlicerSlicesControlGUI -- Add "InPlane", "InPlane90", "Perp"

  if (this->IsSliceOrientationAdded == false && this->GetApplication() )
    {
    vtkSlicerSliceGUI* sgui;
    vtkSlicerSlicesGUI* ssgui;
    vtkSlicerApplication* app = vtkSlicerApplication::SafeDownCast (this->GetApplication());
    ssgui = vtkSlicerSlicesGUI::SafeDownCast(app->GetModuleGUIByName ("Slices") );

    if (ssgui != NULL)
      {
      ssgui->GetSliceGUICollection()->InitTraversal();
      sgui = vtkSlicerSliceGUI::SafeDownCast(ssgui->GetSliceGUICollection()->GetNextItemAsObject());
      while ( sgui != NULL )
        {
        vtkSlicerSliceControllerWidget* sscw = sgui->GetSliceController();
        vtkKWMenuButtonWithSpinButtonsWithLabel* oriSel = sscw->GetOrientationSelector();
        vtkKWMenuButton *mb = oriSel->GetWidget()->GetWidget();
        mb->GetMenu()->AddRadioButton("InPlane");
        mb->GetMenu()->AddRadioButton("InPlane90");
        mb->GetMenu()->AddRadioButton("Perp");
        sgui = vtkSlicerSliceGUI::SafeDownCast ( ssgui->GetSliceGUICollection()->GetNextItemAsObject() );
        }
      }
    this->IsSliceOrientationAdded = true;
    }
  */

}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::Exit ( )
{
  // Fill in
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::BuildGUI ( )
{

  // ---
  // MODULE GUI FRAME 
  // create a page
  this->UIPanel->AddPage ( "EndoNav", "EndoNav", NULL );

  BuildGUIForHelpFrame();
  BuildGUIForConnectorBrowserFrame();
  BuildGUIForIOConfig();
  BuildGUIForVisualizationControlFrame();

  UpdateConnectorPropertyFrame(-1);
  //UpdateIOConfigTree();

  this->ImageViewerUS = vtkImageViewer::New();
  this->ImageViewerUS->SetColorWindow( 256);
  this->ImageViewerUS->SetColorLevel (127.5);

  this->ImageViewerCT = vtkImageViewer::New();
  this->ImageViewerCT->SetColorWindow( 350);
  this->ImageViewerCT->SetColorLevel (40);


  this->ImageConverter = vtkIGTLToViewerImage::New();
  this->ImageConverter->SetImageViewerUS(this->ImageViewerUS);

 //this->ImageConverter->SetSliceViewer(this->GetApplicationGUI()->GetMainSliceGUI("Red")->GetSliceViewer());
  //this->ImageConverter->SetSliceGUI(this->GetApplicationGUI()->GetMainSliceGUI("Red"));

  this->TransformConverter = vtkIGTLToViewerTransform::New();
  this->TransformConverter->SetImageViewerCT(this->ImageViewerCT);
  this->TransformConverter->SetViewer(this->GetApplicationGUI()->GetNthViewerWidget(0));

  this->Logic->RegisterMessageConverter(this->ImageConverter);
  this->Logic->RegisterMessageConverter(this->TransformConverter);
}


void vtkEndoNavGUI::BuildGUIForHelpFrame ()
{

  // ----------------------------------------------------------------
  // HELP FRAME         
  // ----------------------------------------------------------------

  // Define your help text here.
  const char *help = 
    "**The EndoNav Interface Module** helps you to manage EndoNav connections:"
    "<a>http://www.slicer.org/slicerWiki/index.php/Modules:EndoNav</a> for details.";
  const char *about =
    "The module is designed and implemented by Alex Yarmarkovich (ISOMIC)."
    "It is partially based on OpernIGTLink IF Slicer3 module.";

  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "EndoNav" );
  this->BuildHelpAndAboutFrame (page, help, about);

}

//---------------------------------------------------------------------------
void vtkEndoNavGUI::BuildGUIForConnectorBrowserFrame ()
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkKWWidget *page = this->UIPanel->GetPageWidget ("EndoNav");
  
  vtkSlicerModuleCollapsibleFrame *conBrowsFrame = vtkSlicerModuleCollapsibleFrame::New();

  conBrowsFrame->SetParent(page);
  conBrowsFrame->Create();
  conBrowsFrame->SetLabelText("Connectors");
  conBrowsFrame->ExpandFrame();
  app->Script ("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
               conBrowsFrame->GetWidgetName(), page->GetWidgetName());

  conBrowsFrame->Delete();

  // -----------------------------------------
  // Connector List Frame
  
  vtkKWFrameWithLabel *listFrame = vtkKWFrameWithLabel::New();
  listFrame->SetParent(conBrowsFrame->GetFrame());
  listFrame->Create();
  listFrame->SetLabelText ("Connectors");
  app->Script ( "pack %s -fill both -expand true",  
                listFrame->GetWidgetName());

  listFrame->Delete();


  this->ConnectorList = vtkKWMultiColumnListWithScrollbars::New();
  this->ConnectorList->SetParent(listFrame->GetFrame());
  this->ConnectorList->Create();
  this->ConnectorList->SetHeight(1);
  this->ConnectorList->GetWidget()->SetSelectionTypeToRow();
  this->ConnectorList->GetWidget()->SetSelectionModeToSingle();
  this->ConnectorList->GetWidget()->MovableRowsOff();
  this->ConnectorList->GetWidget()->MovableColumnsOff();

  const char* labels[] =
    { "Name", "Type", "Status", "Destination"};
  const int widths[] = 
    { 10, 5, 7, 20 };

  for (int col = 0; col < 4; col ++)
    {
    this->ConnectorList->GetWidget()->AddColumn(labels[col]);
    this->ConnectorList->GetWidget()->SetColumnWidth(col, widths[col]);
    this->ConnectorList->GetWidget()->SetColumnAlignmentToLeft(col);
    this->ConnectorList->GetWidget()->ColumnEditableOff(col);
    //this->ConnectorList->GetWidget()->ColumnEditableOn(col);
    this->ConnectorList->GetWidget()->SetColumnEditWindowToSpinBox(col);
    }
  this->ConnectorList->GetWidget()->SetColumnEditWindowToCheckButton(0);
  //this->ConnectorList->GetWidget()->SetCellUpdatedCommand(this, "OnConnectorListUpdate");
  //this->ConnectorList->GetWidget()->SetSelectionChangedCommand(this, "OnConnectorListSelectionChanged");
  
  vtkKWFrame *listButtonsFrame = vtkKWFrame::New();
  listButtonsFrame->SetParent(listFrame->GetFrame());
  listButtonsFrame->Create();

  listButtonsFrame->Delete();

  app->Script ("pack %s %s -fill both -expand true",  
               //app->Script( "pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2",
               this->ConnectorList->GetWidgetName(), listButtonsFrame->GetWidgetName());

  this->AddConnectorButton = vtkKWPushButton::New();
  this->AddConnectorButton->SetParent(listButtonsFrame);
  this->AddConnectorButton->Create();
  this->AddConnectorButton->SetText( "Add" );
  this->AddConnectorButton->SetWidth (6);

  this->DeleteConnectorButton = vtkKWPushButton::New();
  this->DeleteConnectorButton->SetParent(listButtonsFrame);
  this->DeleteConnectorButton->Create();
  this->DeleteConnectorButton->SetText( "Delete" );
  this->DeleteConnectorButton->SetWidth (6);

  app->Script( "pack %s %s -side left -anchor nw -expand n -padx 2 -pady 2",
               this->AddConnectorButton->GetWidgetName(), this->DeleteConnectorButton->GetWidgetName());

  // -----------------------------------------
  // Connector Property frame

  vtkKWFrameWithLabel *controlFrame = vtkKWFrameWithLabel::New();
  controlFrame->SetParent(conBrowsFrame->GetFrame());
  controlFrame->Create();
  controlFrame->SetLabelText ("Connector Property");
  app->Script ( "pack %s -fill both -expand true",  
                controlFrame->GetWidgetName());


  // Connector Property -- Connector name
  vtkKWFrame *nameFrame = vtkKWFrame::New();
  nameFrame->SetParent(controlFrame->GetFrame());
  nameFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                nameFrame->GetWidgetName());

  vtkKWLabel *nameLabel = vtkKWLabel::New();
  nameLabel->SetParent(nameFrame);
  nameLabel->Create();
  nameLabel->SetWidth(8);
  nameLabel->SetText("Name: ");

  this->ConnectorNameEntry = vtkKWEntry::New();
  this->ConnectorNameEntry->SetParent(nameFrame);
  this->ConnectorNameEntry->Create();
  this->ConnectorNameEntry->SetWidth(18);


  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              nameLabel->GetWidgetName() , this->ConnectorNameEntry->GetWidgetName());
  
  nameLabel->Delete();
  nameFrame->Delete();

  // Connector Property -- Connector type (server or client)
  vtkKWFrame *typeFrame = vtkKWFrame::New();
  typeFrame->SetParent(controlFrame->GetFrame());
  typeFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                typeFrame->GetWidgetName());

  vtkKWLabel *typeLabel = vtkKWLabel::New();
  typeLabel->SetParent(typeFrame);
  typeLabel->Create();
  typeLabel->SetWidth(8);
  typeLabel->SetText("Type: ");

  this->ConnectorTypeButtonSet = vtkKWRadioButtonSet::New();
  this->ConnectorTypeButtonSet->SetParent(typeFrame);
  this->ConnectorTypeButtonSet->Create();
  this->ConnectorTypeButtonSet->PackHorizontallyOn();
  this->ConnectorTypeButtonSet->SetMaximumNumberOfWidgetsInPackingDirection(2);
  this->ConnectorTypeButtonSet->UniformColumnsOn();
  this->ConnectorTypeButtonSet->UniformRowsOn();

  this->ConnectorTypeButtonSet->AddWidget(0);
  this->ConnectorTypeButtonSet->GetWidget(0)->SetText("Server");
  this->ConnectorTypeButtonSet->AddWidget(1);
  this->ConnectorTypeButtonSet->GetWidget(1)->SetText("Client");
  
  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              typeLabel->GetWidgetName() , this->ConnectorTypeButtonSet->GetWidgetName());

  typeFrame->Delete();

  // Connector Property -- Connector type (server or client)
  vtkKWFrame *statusFrame = vtkKWFrame::New();
  statusFrame->SetParent(controlFrame->GetFrame());
  statusFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                statusFrame->GetWidgetName());
  
  vtkKWLabel *statusLabel = vtkKWLabel::New();
  statusLabel->SetParent(statusFrame);
  statusLabel->Create();
  statusLabel->SetWidth(8);
  statusLabel->SetText("Status: ");

  this->ConnectorStatusCheckButton = vtkKWCheckButton::New();
  this->ConnectorStatusCheckButton->SetParent(statusFrame);
  this->ConnectorStatusCheckButton->Create();
  this->ConnectorStatusCheckButton->SelectedStateOff();
  this->ConnectorStatusCheckButton->SetText("Active");
  
  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              statusLabel->GetWidgetName() , this->ConnectorStatusCheckButton->GetWidgetName());

  statusFrame->Delete();
  statusLabel->Delete();

  vtkKWFrame *addressFrame = vtkKWFrame::New();
  addressFrame->SetParent(controlFrame->GetFrame());
  addressFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                addressFrame->GetWidgetName());

  vtkKWLabel *addressLabel = vtkKWLabel::New();
  addressLabel->SetParent(addressFrame);
  addressLabel->Create();
  addressLabel->SetWidth(8);
  addressLabel->SetText("Addr: ");

  this->ConnectorAddressEntry = vtkKWEntry::New();
  this->ConnectorAddressEntry->SetParent(addressFrame);
  this->ConnectorAddressEntry->Create();
  this->ConnectorAddressEntry->SetWidth(18);

  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              addressLabel->GetWidgetName() , this->ConnectorAddressEntry->GetWidgetName());
  
  addressFrame->Delete();
  addressLabel->Delete();

  vtkKWFrame *portFrame = vtkKWFrame::New();
  portFrame->SetParent(controlFrame->GetFrame());
  portFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                portFrame->GetWidgetName());

  vtkKWLabel *portLabel = vtkKWLabel::New();
  portLabel->SetParent(portFrame);
  portLabel->Create();
  portLabel->SetWidth(8);
  portLabel->SetText("Port: ");

  this->ConnectorPortEntry = vtkKWEntry::New();
  this->ConnectorPortEntry->SetParent(portFrame);
  this->ConnectorPortEntry->SetRestrictValueToInteger();
  this->ConnectorPortEntry->Create();
  this->ConnectorPortEntry->SetWidth(8);

  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              portLabel->GetWidgetName() , this->ConnectorPortEntry->GetWidgetName());

  portFrame->Delete();
  portLabel->Delete();

  vtkKWFrame *waitFrame = vtkKWFrame::New();
  waitFrame->SetParent(controlFrame->GetFrame());
  waitFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                waitFrame->GetWidgetName());

  vtkKWLabel *waitLabel = vtkKWLabel::New();
  waitLabel->SetParent(waitFrame);
  waitLabel->Create();
  waitLabel->SetWidth(8);
  waitLabel->SetText("Sleep(ms) ");

  this->SleepTimeEntry = vtkKWEntry::New();
  this->SleepTimeEntry->SetParent(waitFrame);
  this->SleepTimeEntry->SetRestrictValueToInteger();
  this->SleepTimeEntry->Create();
  this->SleepTimeEntry->SetValueAsInt(1);
  this->SleepTimeEntry->SetWidth(8);

  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              waitLabel->GetWidgetName() , this->SleepTimeEntry->GetWidgetName());

  controlFrame->ExpandFrame();

  waitFrame->Delete();
  waitLabel->Delete();

  typeLabel->Delete();
  controlFrame->Delete();
}

//---------------------------------------------------------------------------
void vtkEndoNavGUI::BuildGUIForIOConfig()
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkKWWidget *page = this->UIPanel->GetPageWidget ("EndoNav");
  
  vtkSlicerModuleCollapsibleFrame *ioConfigFrame = vtkSlicerModuleCollapsibleFrame::New();

  ioConfigFrame->SetParent(page);
  ioConfigFrame->Create();
  ioConfigFrame->SetLabelText("Data I/O Configurations");
  ioConfigFrame->CollapseFrame();
  //app->Script ("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
  //             ioConfigFrame->GetWidgetName(), page->GetWidgetName());

  ioConfigFrame->Delete();

  // -----------------------------------------
  // Data I/O Configurations

  vtkKWFrameWithLabel *advancedSettingFrame = vtkKWFrameWithLabel::New();
  advancedSettingFrame->SetParent(ioConfigFrame->GetFrame());
  advancedSettingFrame->Create();
  advancedSettingFrame->SetLabelText ("Data I/O Settings");
  app->Script ( "pack %s -fill both -expand true",  
                advancedSettingFrame->GetWidgetName());

  vtkKWFrame *treeFrame = vtkKWFrame::New();
  treeFrame->SetParent(advancedSettingFrame->GetFrame());
  treeFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                treeFrame  ->GetWidgetName());

  this->IOConfigTree = vtkKWTreeWithScrollbars::New() ;
  this->IOConfigTree->SetParent ( treeFrame );
  this->IOConfigTree->VerticalScrollbarVisibilityOn();
  this->IOConfigTree->HorizontalScrollbarVisibilityOff();
  //this->IOConfigTree->ResizeButtonsVisibilityOn();
  this->IOConfigTree->Create ( );
  this->IOConfigTree->SetBalloonHelpString("MRML Tree");
  ///  this->IOConfigTree->SetBorderWidth(2);
  //this->IOConfigTree->SetReliefToGroove();
  this->Script ( "pack %s -side top -anchor nw -expand y -fill both -padx 2 -pady 2",
                 this->IOConfigTree->GetWidgetName());

  vtkKWTree *tree = this->IOConfigTree->GetWidget();
  tree->SelectionFillOn();
  tree->SetSelectionModeToMultiple ();
/*  tree->SetNodeParentChangedCommand(this, "NodeParentChangedCallback");*/
  tree->EnableReparentingOn();
  tree->SetSelectionModeToSingle();
  tree->SetHeight(11);

  this->UpdateIOConfigTree();


  vtkKWFrame *enableASFrame = vtkKWFrame::New();
  enableASFrame->SetParent(advancedSettingFrame->GetFrame());
  enableASFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                enableASFrame->GetWidgetName());
  
  vtkKWLabel *enableASLabel = vtkKWLabel::New();
  enableASLabel->SetParent(enableASFrame);
  enableASLabel->Create();
  enableASLabel->SetWidth(20);
  enableASLabel->SetText("Restrict Device Name: ");

  this->EnableAdvancedSettingButton = vtkKWCheckButton::New();
  this->EnableAdvancedSettingButton->SetParent(enableASFrame);
  this->EnableAdvancedSettingButton->Create();
  this->EnableAdvancedSettingButton->SelectedStateOff();
  this->EnableAdvancedSettingButton->SetText("Enabled");
  
  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              enableASLabel->GetWidgetName() , this->EnableAdvancedSettingButton->GetWidgetName());

  enableASFrame->Delete();
  enableASLabel->Delete();
  advancedSettingFrame->Delete();
  treeFrame->Delete();
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::BuildGUIForVisualizationControlFrame ()
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkKWWidget *page = this->UIPanel->GetPageWidget ("EndoNav");
  
  vtkSlicerModuleCollapsibleFrame *visCtrlFrame = vtkSlicerModuleCollapsibleFrame::New();
  visCtrlFrame->SetParent(page);
  visCtrlFrame->Create();
  visCtrlFrame->SetLabelText("Visualization / Slice Control");
  visCtrlFrame->CollapseFrame();
  app->Script ("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
               visCtrlFrame->GetWidgetName(), page->GetWidgetName());

  // -----------------------------------------
  // Locator Display frame: Options to locator display 

  vtkKWFrameWithLabel *displayFrame = vtkKWFrameWithLabel::New ( );
  displayFrame->SetParent(visCtrlFrame->GetFrame());
  displayFrame->Create();
  displayFrame->SetLabelText("Locator Display");
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               displayFrame->GetWidgetName());

  vtkKWFrame *nodeFrame = vtkKWFrame::New();
  nodeFrame->SetParent(displayFrame->GetFrame());
  nodeFrame->Create();
  app->Script ( "pack %s -fill both -expand true",  
                nodeFrame->GetWidgetName());

  vtkKWLabel *nodeLabel = vtkKWLabel::New();
  nodeLabel->SetParent(nodeFrame);
  nodeLabel->Create();
  //nodeLabel->SetWidth(20);
  nodeLabel->SetText("Locator source: ");

  this->LocatorSourceMenu = vtkKWMenuButton::New();
  this->LocatorSourceMenu->SetParent(nodeFrame);
  this->LocatorSourceMenu->Create();
  this->LocatorSourceMenu->SetWidth(20);
  UpdateLocatorSourceMenu();

  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              nodeLabel->GetWidgetName() , this->LocatorSourceMenu->GetWidgetName());
  
  this->LocatorCheckButton = vtkKWCheckButton::New();
  this->LocatorCheckButton->SetParent(displayFrame->GetFrame());
  this->LocatorCheckButton->Create();
  this->LocatorCheckButton->SelectedStateOff();
  this->LocatorCheckButton->SetText("Show Locator");
  
  this->Script("pack %s -side left -anchor w -padx 2 -pady 2", 
               this->LocatorCheckButton->GetWidgetName());


  // -----------------------------------------
  // Driver frame: Locator can drive slices 

  vtkKWFrameWithLabel *driverFrame = vtkKWFrameWithLabel::New();
  driverFrame->SetParent(visCtrlFrame->GetFrame());
  driverFrame->Create();
  driverFrame->SetLabelText ("Driver");
  //this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
  //             driverFrame->GetWidgetName());
  
  // Source frame
  vtkKWFrame *imageSourceFrame = vtkKWFrame::New();
  imageSourceFrame->SetParent(driverFrame->GetFrame());
  imageSourceFrame->Create();
  app->Script("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
              imageSourceFrame->GetWidgetName(),
              driverFrame->GetFrame()->GetWidgetName());
  
  vtkKWLabel *imageSourceLabel = vtkKWLabel::New();
  imageSourceLabel->SetParent(imageSourceFrame);
  imageSourceLabel->Create();
  //nodeLabel->SetWidth(20);
  imageSourceLabel->SetText("RT image source: ");

  this->RealTimeImageSourceMenu = vtkKWMenuButton::New();
  this->RealTimeImageSourceMenu->SetParent(imageSourceFrame);
  this->RealTimeImageSourceMenu->Create();
  this->RealTimeImageSourceMenu->SetWidth(20);
  UpdateRealTimeImageSourceMenu();

  app->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
              imageSourceLabel->GetWidgetName() , this->RealTimeImageSourceMenu->GetWidgetName());

  // slice frame
  vtkKWFrame *sliceFrame = vtkKWFrame::New();
  sliceFrame->SetParent(driverFrame->GetFrame());
  sliceFrame->Create();
  app->Script("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
              sliceFrame->GetWidgetName(),
              driverFrame->GetFrame()->GetWidgetName());
  

  // Contents in slice frame 
  vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
  
  this->RedSliceMenu = vtkKWMenuButton::New();
  this->RedSliceMenu->SetParent(sliceFrame);
  this->RedSliceMenu->Create();
  this->RedSliceMenu->SetWidth(10);
  this->RedSliceMenu->SetBackgroundColor(color->SliceGUIRed);
  this->RedSliceMenu->SetActiveBackgroundColor(color->SliceGUIRed);
  this->RedSliceMenu->GetMenu()->AddRadioButton ("User");
  this->RedSliceMenu->GetMenu()->AddRadioButton ("Locator");
  this->RedSliceMenu->GetMenu()->AddRadioButton ("RT Image");
  this->RedSliceMenu->SetValue ("User");
  
  this->YellowSliceMenu = vtkKWMenuButton::New();
  this->YellowSliceMenu->SetParent(sliceFrame);
  this->YellowSliceMenu->Create();
  this->YellowSliceMenu->SetWidth(10);
  this->YellowSliceMenu->SetBackgroundColor(color->SliceGUIYellow);
  this->YellowSliceMenu->SetActiveBackgroundColor(color->SliceGUIYellow);
  this->YellowSliceMenu->GetMenu()->AddRadioButton ("User");
  this->YellowSliceMenu->GetMenu()->AddRadioButton ("Locator");
  this->YellowSliceMenu->GetMenu()->AddRadioButton ("RT Image");
  this->YellowSliceMenu->SetValue ("User");
  
  this->GreenSliceMenu = vtkKWMenuButton::New();
  this->GreenSliceMenu->SetParent(sliceFrame);
  this->GreenSliceMenu->Create();
  this->GreenSliceMenu->SetWidth(10);
  this->GreenSliceMenu->SetBackgroundColor(color->SliceGUIGreen);
  this->GreenSliceMenu->SetActiveBackgroundColor(color->SliceGUIGreen);
  this->GreenSliceMenu->GetMenu()->AddRadioButton ("User");
  this->GreenSliceMenu->GetMenu()->AddRadioButton ("Locator");
  this->GreenSliceMenu->GetMenu()->AddRadioButton ("RT Image");
  this->GreenSliceMenu->SetValue ("User");
  
  this->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
               this->RedSliceMenu->GetWidgetName(),
               this->YellowSliceMenu->GetWidgetName(),
               this->GreenSliceMenu->GetWidgetName());
  
  // Mode frame
  vtkKWFrame *modeFrame = vtkKWFrame::New();
  modeFrame->SetParent ( driverFrame->GetFrame() );
  modeFrame->Create ( );
  app->Script ("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
               modeFrame->GetWidgetName(),
               driverFrame->GetFrame()->GetWidgetName());
  
  // "Locator All" button
  this->SetLocatorModeButton = vtkKWPushButton::New ( );
  this->SetLocatorModeButton->SetParent ( modeFrame );
  this->SetLocatorModeButton->Create ( );
  this->SetLocatorModeButton->SetText ("Locator All");
  this->SetLocatorModeButton->SetWidth (12);
  
  // "User All" button
  this->SetUserModeButton = vtkKWPushButton::New ( );
  this->SetUserModeButton->SetParent ( modeFrame );
  this->SetUserModeButton->Create ( );
  this->SetUserModeButton->SetText ("User All");
  this->SetUserModeButton->SetWidth (12);
  
  
  // "Freeze" check button
  this->FreezeImageCheckButton = vtkKWCheckButton::New();
  this->FreezeImageCheckButton->SetParent(modeFrame);
  this->FreezeImageCheckButton->Create();
  this->FreezeImageCheckButton->SelectedStateOff();
  this->FreezeImageCheckButton->SetText("Freeze");
  
  this->ObliqueCheckButton = vtkKWCheckButton::New();
  this->ObliqueCheckButton->SetParent(modeFrame);
  this->ObliqueCheckButton->Create();
  this->ObliqueCheckButton->SelectedStateOff();
  this->ObliqueCheckButton->SetText("Orient");
/**
  this->Script("pack %s %s %s %s -side left -anchor w -padx 2 -pady 2", 
               this->SetLocatorModeButton->GetWidgetName(),
               this->SetUserModeButton->GetWidgetName(),
               this->FreezeImageCheckButton->GetWidgetName(),
               this->ObliqueCheckButton->GetWidgetName());
**/

  visCtrlFrame->ExpandFrame();

  nodeLabel->Delete();
  nodeFrame->Delete();
  modeFrame->Delete();
  sliceFrame->Delete();
  visCtrlFrame->Delete();
  driverFrame->Delete();
  imageSourceFrame->Delete();
  imageSourceLabel->Delete();
  displayFrame->Delete();
}


//----------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateAll()
{

  /*
  if (this->LocatorCheckButton->GetSelectedState())
    {
    }
  */

}


//----------------------------------------------------------------------------
void vtkEndoNavGUI::IOConfigTreeContextMenu(const char *callData)
{

  if (!this->IOConfigContextMenu)
    {
    this->IOConfigContextMenu = vtkKWMenu::New();
    }
  if (!this->IOConfigContextMenu->IsCreated())
    {
    this->IOConfigContextMenu->SetParent(this->IOConfigTree->GetWidget());
    this->IOConfigContextMenu->Create();
    }
  this->IOConfigContextMenu->DeleteAllItems();
  
  int px, py;
  vtkKWTkUtilities::GetMousePointerCoordinates(this->IOConfigTree->GetWidget(), &px, &py);
  
  int conID;
  int devID;
  int io;

  int type = this->IsIOConfigTreeLeafSelected(callData, &conID, &devID, &io);
  if (type != 0)
    {
    this->IOConfigTree->GetWidget()->ClearSelection();
    this->IOConfigTree->GetWidget()->SelectNode((const char *)callData);

    AddIOConfigContextMenuItem(type, conID, devID, io);
    }

  this->IOConfigContextMenu->PopUp(px, py);

}


//----------------------------------------------------------------------------
int vtkEndoNavGUI::IsIOConfigTreeLeafSelected(const char* callData, int* conID, int* devID, int* io)
  // 0: none                       : do nothing
  // NODE_CONNECTOR: connector     : do nothing
  // NODE_IO:        I/O           : show add node menu
  // NODE_DEVICE:    device node   : show delete menu
{
  IOConfigNodeInfoListType::iterator iter;

  // search connector node list
  for (iter = this->IOConfigTreeConnectorList.begin(); iter != this->IOConfigTreeConnectorList.end(); iter ++)
    {
    if (iter->nodeName == callData)
      {
      *conID = iter->connectorID;
      return NODE_CONNECTOR;
      }
    }

  // search connector i/o node list
  for (iter = this->IOConfigTreeIOList.begin(); iter != this->IOConfigTreeIOList.end(); iter ++)
    {
    if (iter->nodeName == callData)
      {
      *conID = iter->connectorID;
      *io = iter->io;
      return NODE_IO;
      }
    }

  // search device node list
  for (iter = this->IOConfigTreeNodeList.begin(); iter != this->IOConfigTreeNodeList.end(); iter ++)
    {
    if (iter->nodeName == callData)
      {
      *conID = iter->connectorID;
      *devID = iter->deviceID;
      *io    = iter->io;
      return NODE_DEVICE;
      }
    }
  return 0;
}

//----------------------------------------------------------------------------
void vtkEndoNavGUI::AddIOConfigContextMenuItem(int type, int conID, int devID, int io)
{
  char command[125];
  char label[125];

  if (type == NODE_IO)
    {
    this->GetLogic()->GetDeviceNamesFromMrml(this->CurrentNodeListAvailable);
    vtkEndoNavLogic::IGTLMrmlNodeListType::iterator iter;
    for (iter = this->CurrentNodeListAvailable.begin(); iter != this->CurrentNodeListAvailable.end(); iter ++)
      {
      sprintf(command, "AddNodeCallback %d %d {%s} {%s}", conID, io, iter->name.c_str(), iter->type.c_str());
      sprintf(label, "Add %s (%s)", iter->name.c_str(), iter->type.c_str());
      this->IOConfigContextMenu->AddCommand(label, this, command);
      }
    }
  else if (type == NODE_DEVICE)
    {
    sprintf(command, "DeleteNodeCallback %d %d %d", conID, io, devID);
    this->IOConfigContextMenu->AddCommand("Delete this node", this, command);
    }

}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::AddNodeCallback(int conID, int io, const char* name, const char* type)
{
  this->GetLogic()->AddDeviceToConnector(conID, name, type, io);
  UpdateIOConfigTree();
}


//----------------------------------------------------------------------------
void vtkEndoNavGUI::DeleteNodeCallback(int  conID, int io, int devID)
{
  this->GetLogic()->DeleteDeviceFromConnector(conID, devID, io);
  UpdateIOConfigTree();
}


//----------------------------------------------------------------------------
void vtkEndoNavGUI::ChangeSlicePlaneDriver(int slice, const char* driver)
{

  if (slice == vtkEndoNavGUI::SLICE_PLANE_RED)
    {
    this->RedSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode0->SetOrientationToAxial();
      //this->SliceDriver0 = vtkEndoNavGUI::SLICE_DRIVER_USER;
      //this->GetLogic()->SetSliceDriver0(vtkEndoNavLogic::SLICE_DRIVER_USER);
      this->GetLogic()->SetSliceDriver(0, vtkEndoNavLogic::SLICE_DRIVER_USER);
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      //this->SliceDriver0 = vtkEndoNavGUI::SLICE_DRIVER_LOCATOR;
      //this->GetLogic()->SetSliceDriver0(vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      this->GetLogic()->SetSliceDriver(0, vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      //this->SliceDriver0 = vtkEndoNavGUI::SLICE_DRIVER_RTIMAGE;
      //this->GetLogic()->SetSliceDriver0(vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      this->GetLogic()->SetSliceDriver(0, vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      }
    }
  else if (slice == vtkEndoNavGUI::SLICE_PLANE_YELLOW)
    {
    this->YellowSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode1->SetOrientationToSagittal();
      //this->SliceDriver1 = vtkEndoNavGUI::SLICE_DRIVER_USER;
      //this->GetLogic()->SetSliceDriver1(vtkEndoNavLogic::SLICE_DRIVER_USER);
      this->GetLogic()->SetSliceDriver(1, vtkEndoNavLogic::SLICE_DRIVER_USER);
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      //this->SliceDriver1 = vtkEndoNavGUI::SLICE_DRIVER_LOCATOR;
      //this->GetLogic()->SetSliceDriver1(vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      this->GetLogic()->SetSliceDriver(1, vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      //this->SliceDriver1 = vtkEndoNavGUI::SLICE_DRIVER_RTIMAGE;
      //this->GetLogic()->SetSliceDriver1(vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      this->GetLogic()->SetSliceDriver(1, vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      }
    }
  else //if ( slice == vtkEndoNavGUI::SLICE_PLANE_GREEN )
    {
    this->GreenSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode2->SetOrientationToCoronal();
      //this->SliceDriver2 = vtkEndoNavGUI::SLICE_DRIVER_USER;
      //this->GetLogic()->SetSliceDriver2(vtkEndoNavLogic::SLICE_DRIVER_USER);
      this->GetLogic()->SetSliceDriver(2, vtkEndoNavLogic::SLICE_DRIVER_USER);
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      //this->SliceDriver2 = vtkEndoNavGUI::SLICE_DRIVER_LOCATOR;
      //this->GetLogic()->SetSliceDriver2(vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      this->GetLogic()->SetSliceDriver(2, vtkEndoNavLogic::SLICE_DRIVER_LOCATOR);
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      //this->SliceDriver2 = vtkEndoNavGUI::SLICE_DRIVER_RTIMAGE;
      //this->GetLogic()->SetSliceDriver2(vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      this->GetLogic()->SetSliceDriver(2, vtkEndoNavLogic::SLICE_DRIVER_RTIMAGE);
      }
    }

}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateLocatorSourceMenu()
{

  if (!this->LocatorSourceMenu)
    {
    return;
    }
  
  this->GetLogic()->GetDeviceNamesFromMrml(this->LocatorSourceList, "LinearTransform");
  vtkEndoNavLogic::IGTLMrmlNodeListType::iterator iter;
  this->LocatorSourceMenu->GetMenu()->DeleteAllItems();
  for (iter = this->LocatorSourceList.begin();
       iter != this->LocatorSourceList.end();
       iter ++)
    {
    char str[256];
    sprintf(str, "%s (%s)", iter->name.c_str(), iter->type.c_str());
    this->LocatorSourceMenu->GetMenu()->AddRadioButton(str);
    }
  
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateRealTimeImageSourceMenu()
{

  if (!this->RealTimeImageSourceMenu)
    {
    return;
    }
  this->GetLogic()->GetDeviceNamesFromMrml(this->RealTimeImageSourceList, "Volume");
  vtkEndoNavLogic::IGTLMrmlNodeListType::iterator iter;
  this->RealTimeImageSourceMenu->GetMenu()->DeleteAllItems();
  for (iter = this->RealTimeImageSourceList.begin();
       iter != this->RealTimeImageSourceList.end();
       iter ++)
    {
    char str[256];
    sprintf(str, "%s (%s)", iter->name.c_str(), iter->type.c_str());
    this->RealTimeImageSourceMenu->GetMenu()->AddRadioButton(str);
    }
  
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateIOConfigTree()
{

  vtksys_stl::string selected_node(this->IOConfigTree->GetWidget()->GetSelection());
  this->IOConfigTree->GetWidget()->DeleteAllNodes();

  //vtkMRMLScene *scene = this->GetMRMLScene();
  //vtkMRMLNode *node = NULL;

  // create Root node
  vtkKWTree *tree = this->IOConfigTree->GetWidget();
  tree->AddNode(NULL, "EndoNav", "EndoNav Interface");
  const char* rootNode = "EndoNav";

  this->IOConfigTreeConnectorList.clear();
  this->IOConfigTreeIOList.clear();
  this->IOConfigTreeNodeList.clear();

  vtkEndoNavLogic::ConnectorMapType* conMap = this->GetLogic()->GetConnectorMap();
  vtkEndoNavLogic::ConnectorMapType::iterator iter;
  for (iter = conMap->begin(); iter != conMap->end(); iter ++)
    {
    vtkIGTLConnector* con = iter->second;
    if (con)
      {
      char conNode[32];
      char conInNode[32];
      char conOutNode[32];
      char conDeviceNode[128];
      char conDeviceNodeName[128];
      IOConfigNodeInfoType nodeInfo;

      int id = iter->first;
      sprintf(conNode, "con%d", id);
      tree->AddNode(rootNode, conNode, con->GetName());
      nodeInfo.nodeName = conNode;
      nodeInfo.deviceID = -1;
      nodeInfo.connectorID = id;
      nodeInfo.io = vtkIGTLConnector::IO_UNSPECIFIED;
      this->IOConfigTreeConnectorList.push_back(nodeInfo);

      sprintf(conInNode, "con%d/in", id);
      tree->AddNode(conNode,  conInNode, "IN");
      nodeInfo.nodeName = conInNode;
      nodeInfo.deviceID = -1;
      nodeInfo.connectorID = id;
      nodeInfo.io = vtkIGTLConnector::IO_INCOMING;
      this->IOConfigTreeIOList.push_back(nodeInfo);

      sprintf(conOutNode, "con%d/out", id);
      tree->AddNode(conNode,  conOutNode, "OUT");
      nodeInfo.nodeName = conOutNode;
      nodeInfo.deviceID = -1;
      nodeInfo.connectorID = id;
      nodeInfo.io = vtkIGTLConnector::IO_OUTGOING;
      this->IOConfigTreeIOList.push_back(nodeInfo);

      // Incoming devices
      vtkIGTLConnector::DeviceIDSetType* inDeviceSet = con->GetIncomingDevice();
      vtkIGTLConnector::DeviceIDSetType::iterator iter_in;
      for (iter_in = inDeviceSet->begin(); iter_in != inDeviceSet->end(); iter_in ++)
        {
        vtkIGTLConnector::DeviceInfoType* info = con->GetDeviceInfo(*iter_in);
        sprintf(conDeviceNode, "con%d/in/%s_%s", id, info->name.c_str(), info->type.c_str());
        sprintf(conDeviceNodeName, "%s (%s)", info->name.c_str(), info->type.c_str());
        tree->AddNode(conInNode, conDeviceNode, conDeviceNodeName);

        nodeInfo.nodeName = conDeviceNode;
        nodeInfo.deviceID = *iter_in;
        nodeInfo.connectorID = id;
        nodeInfo.io = vtkIGTLConnector::IO_INCOMING;
        this->IOConfigTreeNodeList.push_back(nodeInfo);
        }

      // Outgoing Devices
      vtkIGTLConnector::DeviceIDSetType* outDeviceSet = con->GetOutgoingDevice();
      vtkIGTLConnector::DeviceIDSetType::iterator iter_out;
      for (iter_out = outDeviceSet->begin(); iter_out != outDeviceSet->end(); iter_out ++)
        {
        vtkIGTLConnector::DeviceInfoType* info = con->GetDeviceInfo(*iter_out);
        sprintf(conDeviceNode, "con%d/out/%s_%s", id, info->name.c_str(), info->type.c_str());
        sprintf(conDeviceNodeName, "%s (%s)", info->name.c_str(), info->type.c_str());
        tree->AddNode(conOutNode, conDeviceNode, conDeviceNodeName);

        nodeInfo.nodeName = conDeviceNode;
        nodeInfo.deviceID = *iter_out;
        nodeInfo.connectorID = id;
        nodeInfo.io = vtkIGTLConnector::IO_OUTGOING;
        this->IOConfigTreeNodeList.push_back(nodeInfo);
        }

      }
    }

  // check that the selected node is still in the tree
  if (this->IOConfigTree->GetWidget()->HasNode(selected_node.c_str()))
    {
    this->IOConfigTree->GetWidget()->SelectNode(selected_node.c_str());
    }
  else
    {
    if (selected_node != "")
      {
      vtkWarningMacro("Selected node no longer in tree: " << selected_node.c_str());
      }
    }
  // At this point you probably want to reset the MRML node inspector fields
  // in case nothing in the tree is selected anymore (here, we delete all nodes
  // each time, so nothing will be selected, but it's not a bad thing to 
  // try to save the old selection, or just update the tree in a smarter
  // way).
  
  this->IOConfigTree->GetWidget()->OpenFirstNode ();

  //this->UpdateNodeInspector(this->GetSelectedNodeInTree());

}

//---------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateConnectorList(int updateLevel)
{
  // Update Connector List in "Connectors"
  if (this->ConnectorList == NULL)
    {
    return;
    }

  //----------------------------------------------------------------
  // Change number of rows (UPDATE_ALL only)
  if (updateLevel >= UPDATE_ALL)
    {
    // Adjust number of rows
    int numRows = this->ConnectorList->GetWidget()->GetNumberOfRows();
    int numConnectors = this->GetLogic()->GetNumberOfConnectors();

    if (numRows < numConnectors)
      {
      this->ConnectorList->GetWidget()->AddRows(numConnectors-numRows);
      }
    else if (numRows > numConnectors)
      {
      int ndel = numRows-numConnectors;
      for (int i = 0; i < ndel; i ++)
        {
        this->ConnectorList->GetWidget()->DeleteRow(numConnectors);
        }
      }

    // Create connector id list
    this->ConnectorIDList.clear();
    vtkEndoNavLogic::ConnectorMapType* conMap = this->GetLogic()->GetConnectorMap();
    vtkEndoNavLogic::ConnectorMapType::iterator iter;
    for (iter = conMap->begin(); iter != conMap->end(); iter ++)
      {
      this->ConnectorIDList.push_back(iter->first);
      }
    }

  int numItems = this->GetLogic()->GetNumberOfConnectors();

  //----------------------------------------------------------------
  // Update rows (UPDATE_ALL, UPDATE_PROPERTIES_ALL and UPDATE_SELECTED_ONLY)

  // Generate lisft of rows to update
  std::vector<int> updateRows;
  updateRows.clear();
  if (updateLevel != UPDATE_STATUS_ALL)
    {
    if (updateLevel >= UPDATE_PROPERTY_ALL)
      {
      for (int i = 0; i < numItems; i ++)
        {
        updateRows.push_back(i);
        }
      }
    else if (updateLevel >= UPDATE_SELECTED_ONLY)
      {
      updateRows.push_back(this->ConnectorList->GetWidget()->GetIndexOfFirstSelectedRow());
      }
    }
      
  std::vector<int>::iterator iter;
  for (iter = updateRows.begin(); iter != updateRows.end(); iter ++)
    {
    int i = *iter;
    int id = -1;
    if (i >= 0 && i < (int)this->ConnectorIDList.size())
      {
      id = this->ConnectorIDList[i];
      }

    vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
    if (connector)
      {
      // Connector Name
      this->ConnectorList->GetWidget()
        ->SetCellText(i,0, connector->GetName());
          
      // Connector Type
      this->ConnectorList->GetWidget()
        ->SetCellText(i,1, vtkEndoNavGUI::ConnectorTypeStr[connector->GetType()]);

      // Connector Status
      this->ConnectorList->GetWidget()
        ->SetCellText(i,2, vtkEndoNavGUI::ConnectorStatusStr[connector->GetState()]);
          
      // Server and port information
      std::ostringstream ss;
      if (connector->GetType() == vtkIGTLConnector::TYPE_SERVER)
        {
        ss << "--.--.--.--" << ":" << connector->GetServerPort();
        }
      else if (connector->GetType() == vtkIGTLConnector::TYPE_CLIENT)
        {
        ss << connector->GetServerHostname() << ":" << connector->GetServerPort();
        }
      else
        {
        ss << "--.--.--.--" << ":" << "--";
        }
      this->ConnectorList->GetWidget()->SetCellText(i, 3, ss.str().c_str());
      }

    }

  //----------------------------------------------------------------
  // UPDATE_ALL, UPDATE_PROPERTY_ALL and UPDATE_STATUS_ALL
  if (updateLevel == UPDATE_STATUS_ALL)
    {
    for (int i = 0; i < numItems; i ++)
      {
      int id = -1;
      if (i >= 0 && i < (int)this->ConnectorIDList.size())
        {
        id = this->ConnectorIDList[i];
        }

      vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);
      if (connector)
        {
        this->ConnectorList->GetWidget()
          ->SetCellText(i,2, ConnectorStatusStr[connector->GetState()]);
        }
      }
    }
}


//---------------------------------------------------------------------------
void vtkEndoNavGUI::UpdateConnectorPropertyFrame(int i)
{
  // if i < 0, all fields are deactivated.
  
  int numRows = this->ConnectorList->GetWidget()->GetNumberOfRows();
  if (i >= 0 && i >= numRows)
    {
    return;
    }

  //----------------------------------------------------------------
  // No connector is registered
  if (i < 0 || numRows <= 0)
    {
    // Deactivate everything
    // Connector Name entry
    this->ConnectorNameEntry->SetValue("");
    this->ConnectorNameEntry->EnabledOff();
    this->ConnectorNameEntry->UpdateEnableState();

    // Connector Type
    this->ConnectorTypeButtonSet->GetWidget(0)->SelectedStateOff();
    this->ConnectorTypeButtonSet->GetWidget(1)->SelectedStateOff();
    this->ConnectorTypeButtonSet->EnabledOff();
    this->ConnectorTypeButtonSet->UpdateEnableState();

    // Connector Status
    this->ConnectorStatusCheckButton->SelectedStateOff();
    this->ConnectorStatusCheckButton->EnabledOff();

    // Server Address
    this->ConnectorAddressEntry->SetValue("");
    this->ConnectorAddressEntry->EnabledOff();
    this->ConnectorAddressEntry->UpdateEnableState();

    // Server Port
    this->ConnectorPortEntry->SetValueAsInt(0);
    this->ConnectorPortEntry->EnabledOff();
    this->ConnectorPortEntry->UpdateEnableState();

    return;
    }

  //----------------------------------------------------------------
  // A connector is selected on the list

  int id = -1;
  if (i >= 0 && i < (int)this->ConnectorIDList.size())
    {
    id = this->ConnectorIDList[i];
    }
  else
    {
    return;
    }

  vtkIGTLConnector* connector = this->GetLogic()->GetConnector(id);

  // Check if the connector is active
  bool activeFlag = (connector->GetState() != vtkIGTLConnector::STATE_OFF);

  // Connection Name entry
  this->ConnectorNameEntry->SetValue(connector->GetName());
  if (activeFlag)
    {
    this->ConnectorNameEntry->EnabledOff();
    }
  else
    {
    this->ConnectorNameEntry->EnabledOn();
    }
  this->ConnectorNameEntry->UpdateEnableState();

  // Connection Type (server or client)
  if (connector->GetType() == vtkIGTLConnector::TYPE_SERVER)
    {
    this->ConnectorTypeButtonSet->GetWidget(0)->SelectedStateOn();
    this->ConnectorTypeButtonSet->GetWidget(1)->SelectedStateOff();
    }
  else if (connector->GetType() == vtkIGTLConnector::TYPE_CLIENT)
    {
    this->ConnectorTypeButtonSet->GetWidget(0)->SelectedStateOff();
    this->ConnectorTypeButtonSet->GetWidget(1)->SelectedStateOn();
    }
  else // if (connector->GetType == TYPE_NOT_DEFINED)
    {
    this->ConnectorTypeButtonSet->GetWidget(0)->SelectedStateOff();
    this->ConnectorTypeButtonSet->GetWidget(1)->SelectedStateOff();
    }
  if (activeFlag)
    {
    this->ConnectorTypeButtonSet->EnabledOff();
    }
  else
    {
    this->ConnectorTypeButtonSet->EnabledOn();
    }

  this->ConnectorTypeButtonSet->UpdateEnableState();

  // Connection Status
  if (connector->GetState() == vtkIGTLConnector::STATE_OFF)
    {
    this->ConnectorStatusCheckButton->SelectedStateOff();
    }
  else
    {
    this->ConnectorStatusCheckButton->SelectedStateOn();
    }

  if (connector->GetType() == vtkIGTLConnector::TYPE_NOT_DEFINED)  
    {
    this->ConnectorStatusCheckButton->EnabledOff();
    }
  else
    {
    this->ConnectorStatusCheckButton->EnabledOn();
    }
  this->ConnectorStatusCheckButton->UpdateEnableState();


  // Connection Server Address entry
  if (connector->GetType() == vtkIGTLConnector::TYPE_SERVER)
    {
    this->ConnectorAddressEntry->SetValue("--.--.--.--");
    this->ConnectorAddressEntry->EnabledOff();
    }
  else
    {
    this->ConnectorAddressEntry->SetValue(connector->GetServerHostname());
    if (activeFlag)
      {
      this->ConnectorAddressEntry->EnabledOff();
      }
    else
      {
      this->ConnectorAddressEntry->EnabledOn();
      }
    }
  this->ConnectorAddressEntry->UpdateEnableState();

  // Connection Port entry
  this->ConnectorPortEntry->SetValueAsInt(connector->GetServerPort());
  if (activeFlag)
    {
    this->ConnectorPortEntry->EnabledOff();
    }
  else
    {
    this->ConnectorPortEntry->EnabledOn();
    }
  this->ConnectorPortEntry->UpdateEnableState();

}



//---------------------------------------------------------------------------
int vtkEndoNavGUI::OnMrmlNodeListChanged(int row, int col, const char* item)
{
  if (this->CurrentMrmlNodeListIndex < 0 ||
      this->CurrentMrmlNodeListIndex >= (int)this->ConnectorIDList.size())
    {
    return 0;
    }
    
  int conID = this->ConnectorIDList[this->CurrentMrmlNodeListIndex];
  vtkIGTLConnector* connector = this->GetLogic()
    ->GetConnector(conID);
  if (connector == NULL)
    {
    // failed to get connector class
    return 0;
    }

  // -----------------------------------------
  // Get original node info at (row, col)

  //vtkIGTLConnector::DeviceNameList* list;
  std::string& origName = this->CurrentNodeListSelected[row].name;
  std::string& origType = this->CurrentNodeListSelected[row].type;
  int origIo            = this->CurrentNodeListSelected[row].io;

  // -----------------------------------------
  // Update the outgoing / incoming node list

  if (col == 0) // Name (Type) column has been interacted
    {
    // Get current node info at (row, col)
    int index = this->MrmlNodeList->GetWidget()->GetCellWindowAsComboBox(row, 0)->GetValueIndex(item);
    if (index < 0 || index >= (int)this->CurrentNodeListAvailable.size())
      {
      // invalid index
      return 0;
      }
    std::string& currName = this->CurrentNodeListAvailable[index].name;
    std::string& currType = this->CurrentNodeListAvailable[index].type;
    
    if (origName != currName || origType != currType)
      {
      this->GetLogic()->DeleteDeviceFromConnector(conID, origName.c_str(), origType.c_str(), origIo);
      this->GetLogic()->AddDeviceToConnector(conID, currName.c_str(), currType.c_str(), origIo);
      this->CurrentNodeListSelected[row].name = currName;
      this->CurrentNodeListSelected[row].type = currType;
      }
    }
  else // IO column has been interacted
    {
    // Get curretn IO (in integer)
    const char* iostr[] = {"--", "IN", "OUT"}; // refer vtkEndoNavLogic::DEVICE_* 
    int currIo = 0;
    for (currIo = 0; currIo < 3; currIo ++)
      {
      if (strcmp(iostr[currIo], item) == 0)
        {
        break;
        }
      }

    if (currIo != origIo)
      {
      this->GetLogic()->DeleteDeviceFromConnector(conID,
                                                  origName.c_str(), origType.c_str(), origIo);
      this->GetLogic()->AddDeviceToConnector(conID,
                                             origName.c_str(), origType.c_str(), currIo);
      this->CurrentNodeListSelected[row].io = currIo;
      }
    }
    
  return 1;
}
