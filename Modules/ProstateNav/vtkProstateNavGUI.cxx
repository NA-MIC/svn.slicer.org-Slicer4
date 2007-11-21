/*=auto=========================================================================

  Portions (c) Copyright 2007 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkProstateNavGUI.h"
#include "BRPTPRInterface.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerColor.h"
#include "vtkSlicerTheme.h"

#include "vtkKWWizardWidget.h"
#include "vtkKWWizardWorkflow.h"
#include "vtkProstateNavStep.h"
#include "vtkProstateNavConfigurationStep.h"
#include "vtkProstateNavScanControlStep.h"
#include "vtkProstateNavCalibrationStep.h"
#include "vtkProstateNavTargetingStep.h"
#include "vtkProstateNavManualControlStep.h"

#include "vtkKWRenderWidget.h"
#include "vtkKWWidget.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"
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

//#if defined(OT_VERSION_20) || defined(OT_VERSION_13)

#ifdef USE_NAVITRACK
//#include <OpenTracker/input/SlicerNTModule.h>
//#include <OpenTracker/OpenTracker.h>
//#include <OpenTracker/input/SPLModules.h>
#endif //USE_NAVITRACK
//#endif

#include "vtkKWTkUtilities.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkCylinderSource.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkCornerAnnotation.h"

// for Realtime Image
#include "vtkImageChangeInformation.h"
#include "vtkSlicerColorLogic.h"
#include "vtkSlicerVolumesGUI.h"


#include "vtkIGTDataStream.h"
#include "vtkCylinderSource.h"
#include "vtkMRMLLinearTransformNode.h"


#include <vector>

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkProstateNavGUI );
vtkCxxRevisionMacro ( vtkProstateNavGUI, "$Revision: 1.0 $");
//---------------------------------------------------------------------------

vtkProstateNavGUI::vtkProstateNavGUI ( )
{

  this->Logic = NULL;
  
  //----------------------------------------------------------------
  // Workphase Frame
  
  this->WorkPhaseButtonSet = NULL;
  

  //----------------------------------------------------------------  
  // Wizard Frame
  
  this->WizardWidget = vtkKWWizardWidget::New();
  this->WizardSteps = new vtkProstateNavStep*[vtkProstateNavLogic::NumPhases];
  for (int i = 0; i < vtkProstateNavLogic::NumPhases; i ++)
    {
    this->WizardSteps[i] = NULL;
    }
  

  //----------------------------------------------------------------
  // Visualization Control Frame
  
  this->FreezeImageCheckButton = NULL;
  this->SetLocatorModeButton   = NULL;
  this->SetUserModeButton      = NULL;
  this->RedSliceMenu           = NULL;
  this->YellowSliceMenu        = NULL;
  this->GreenSliceMenu         = NULL;

  this->StartScanButton        = NULL;
  this->StopScanButton         = NULL;

  this->FreezeImageCheckButton = NULL;




  this->LocatorCheckButton = NULL;

    this->NREntry = NULL;
    this->NAEntry = NULL;

    this->NSEntry = NULL;
    this->TREntry = NULL;
    this->TAEntry = NULL;
    this->TSEntry = NULL;
    this->PREntry = NULL;
    this->PAEntry = NULL;
    this->PSEntry = NULL;
    this->O4Entry = NULL;



#ifdef USE_NAVITRACK
  
    this->LoadConfigButtonNT = NULL;
    this->ConfigFileEntry = NULL;
    this->ScannerStatusLabelDisp = NULL;
    this->SoftwareStatusLabelDisp = NULL;
#endif

    this->AddCoordsandOrientTarget = NULL;
    this->SetOrientButton = NULL;
   

    this->PointPairMultiColumnList = NULL;
    this->TargetListColumnList = NULL;

    this->DeleteTargetPushButton = NULL;
    this->DeleteAllTargetPushButton = NULL;
    this->MoveBWPushButton = NULL;
    this->MoveFWPushButton = NULL;
  
    this->DataManager = vtkIGTDataManager::New();
    this->Pat2ImgReg = vtkIGTPat2ImgRegistration::New();

    this->DataCallbackCommand = vtkCallbackCommand::New();
    this->DataCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
    this->DataCallbackCommand->SetCallback(vtkProstateNavGUI::DataCallback);

    this->Logic0 = NULL; 
    this->Logic1 = NULL; 
    this->Logic2 = NULL; 
    this->SliceNode0 = NULL; 
    this->SliceNode1 = NULL; 
    this->SliceNode2 = NULL; 
    this->Control0 = NULL; 
    this->Control1 = NULL; 
    this->Control2 = NULL; 

    //this->VolumesLogic = NULL;

    this->NeedOrientationUpdate0 = 0;
    this->NeedOrientationUpdate1 = 0;
    this->NeedOrientationUpdate2 = 0;

    this->NeedRealtimeImageUpdate = 0;
    this->FreezeOrientationUpdate = 0;

    this->SliceDriver0 = 0;
    this->SliceDriver1 = 0;
    this->SliceDriver2 = 0;

}

//---------------------------------------------------------------------------
vtkProstateNavGUI::~vtkProstateNavGUI ( )
{

  if (this->DataManager)
    {
    // If we don't set the scence to NULL for DataManager,
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

  this->RemoveGUIObservers();


  //----------------------------------------------------------------
  // Workphase Frame

  if (this->WorkPhaseButtonSet)
    {
    this->WorkPhaseButtonSet->SetParent(NULL);
    this->WorkPhaseButtonSet->Delete();
    }


  //----------------------------------------------------------------
  // Wizard Frame

  if (this->WizardWidget)
    {
    this->WizardWidget->Delete();
    this->WizardWidget = NULL;
    }

  this->SetModuleLogic ( NULL );


  //----------------------------------------------------------------
  // Visualization Control Frame

  if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->SetParent(NULL );
    this->FreezeImageCheckButton->Delete ( );
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

  if (this->StartScanButton)
    {
    this->StartScanButton->SetParent(NULL);
    this->StartScanButton->Delete();
    }

  if (this->StopScanButton)
    {
    this->StopScanButton->SetParent(NULL);
    this->StartScanButton->Delete();
    }


  //----------------------------------------------------------------
  // Etc Frame



  if (this->NREntry)
  {
      this->NREntry->SetParent(NULL );
      this->NREntry->Delete ( );
  }

  if (this->NAEntry)
  {
      this->NAEntry->SetParent(NULL );
      this->NAEntry->Delete ( );
  }
  if (this->NSEntry)
  {
      this->NSEntry->SetParent(NULL );
      this->NSEntry->Delete ( );
  }
  if (this->TREntry)
  {
      this->TREntry->SetParent(NULL );
      this->TREntry->Delete ( );
  }
  if (this->TAEntry)
  {
      this->TAEntry->SetParent(NULL );
      this->TAEntry->Delete ( );
  }
  if (this->TSEntry)
  {
      this->TSEntry->SetParent(NULL );
      this->TSEntry->Delete ( );
  }
  if (this->PREntry)
  {
      this->PREntry->SetParent(NULL );
      this->PREntry->Delete ( );
  }
  if (this->PAEntry)
  {
      this->PAEntry->SetParent(NULL );
      this->PAEntry->Delete ( );
  }
  if (this->PSEntry)
  {
      this->PSEntry->SetParent(NULL );
      this->PSEntry->Delete ( );
  }
  if (this->O4Entry)
  {
      this->O4Entry->SetParent(NULL );
      this->O4Entry->Delete ( );
  }
  /*
  if (this->RedColorScale)
  {
  this->RedColorScale->SetParent(NULL );
  this->RedColorScale->Delete ( );
  }
  if (this->GreenColorScale)
  {
  this->GreenColorScale->SetParent(NULL );
  this->GreenColorScale->Delete ( );
  }
  if (this->BlueColorScale)
  {
  this->BlueColorScale->SetParent(NULL );
  this->BlueColorScale->Delete ( );
  }
  */
  
  if (this->LocatorCheckButton)
  {
  this->LocatorCheckButton->SetParent(NULL );
  this->LocatorCheckButton->Delete ( );
  }

#ifdef USE_NAVITRACK
  if (this->LoadConfigButtonNT)
  {
  this->LoadConfigButtonNT->SetParent(NULL );
  this->LoadConfigButtonNT->Delete ( );
  }
  if (this->ConfigFileEntry)
  {
  this->ConfigFileEntry->SetParent(NULL );
  this->ConfigFileEntry->Delete ( );
  }
#endif

  if (this->SetOrientButton)
  {
  this->SetOrientButton->SetParent(NULL );
  this->SetOrientButton->Delete ( );
  }

  if (this->AddCoordsandOrientTarget)
  {
  this->AddCoordsandOrientTarget->SetParent(NULL );
  this->AddCoordsandOrientTarget->Delete ( );
  }

  if (this->PointPairMultiColumnList)
  {
  this->PointPairMultiColumnList->SetParent(NULL );
  this->PointPairMultiColumnList->Delete ( );
  }
  if (this->TargetListColumnList)
  {
  this->TargetListColumnList->SetParent(NULL );
  this->TargetListColumnList->Delete ( );
  }

  if (this->DeleteTargetPushButton)
  {
  this->DeleteTargetPushButton->SetParent(NULL );
  this->DeleteTargetPushButton->Delete ( );
  }
  if (this->DeleteAllTargetPushButton)
  {
  this->DeleteAllTargetPushButton->SetParent(NULL );
  this->DeleteAllTargetPushButton->Delete ( );
  }   
  if (this->MoveBWPushButton)
  {
  this->MoveBWPushButton->SetParent(NULL );
  this->MoveBWPushButton->Delete ( );
  }
  if (this->MoveFWPushButton)
  {
  this->MoveFWPushButton->SetParent(NULL );
  this->MoveFWPushButton->Delete ( );
  }

}



//---------------------------------------------------------------------------
void vtkProstateNavGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );
    



    os << indent << "ProstateNavGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";
   
    // print widgets?
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::RemoveGUIObservers ( )
{
  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  
  appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
  
  //----------------------------------------------------------------
  // Workphase Frame

  if (this->WorkPhaseButtonSet)
    {
    for (int i = 0; i < this->WorkPhaseButtonSet->GetNumberOfWidgets(); i ++)
      {
      this->WorkPhaseButtonSet->GetWidget(i)->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
      }
    }
    

  //----------------------------------------------------------------
  // Wizard Frame

  this->WizardWidget->GetWizardWorkflow()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);

  if (this->SetOrientButton)
    {
    this->SetOrientButton->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                           (vtkCommand *)this->GUICallbackCommand );
    }


  //----------------------------------------------------------------
  // Visualization Control Frame

  if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent,
                                                  (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SetLocatorModeButton)
    {
    this->SetLocatorModeButton->RemoveObservers (vtkKWPushButton::InvokedEvent,
                                                  (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->SetUserModeButton)
    {
    this->SetUserModeButton->RemoveObservers (vtkKWPushButton::InvokedEvent,
                                              (vtkCommand *)this->GUICallbackCommand );
    }
  
  


  if (this->AddCoordsandOrientTarget)
    {
    this->AddCoordsandOrientTarget->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                                    (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->DeleteTargetPushButton)
    {
    this->DeleteTargetPushButton->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                                  (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->MoveFWPushButton)
    {
    this->MoveFWPushButton->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                            (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->MoveBWPushButton)
    {
    this->MoveBWPushButton->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                             (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->DeleteAllTargetPushButton)
    {
    this->DeleteAllTargetPushButton->RemoveObservers(vtkKWPushButton::InvokedEvent,
                                                     (vtkCommand *)this->GUICallbackCommand );
    }
  if (this->LocatorCheckButton)
    {
    this->LocatorCheckButton->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent,
                                              (vtkCommand *)this->GUICallbackCommand );
    }
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::AddGUIObservers ( )
{
  this->RemoveGUIObservers();

  // make a user interactor style to process our events
  // look at the InteractorStyle to get our events
  
  vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
  
  appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()
    ->GetRenderWindowInteractor()->GetInteractorStyle()
    ->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
  
  //----------------------------------------------------------------
  // Workphase Frame

  for (int i = 0; i < this->WorkPhaseButtonSet->GetNumberOfWidgets(); i ++)
    {
    this->WorkPhaseButtonSet->GetWidget(i)
      ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
    }
  
  
  //----------------------------------------------------------------
  // Wizard Frame

  this->WizardWidget->GetWizardWorkflow()->AddObserver(vtkKWWizardWorkflow::CurrentStateChangedEvent,
                                                       (vtkCommand *)this->GUICallbackCommand);


  //----------------------------------------------------------------
  // Visualization Control Frame

  this->FreezeImageCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand*)this->GUICallbackCommand);
  this->SetLocatorModeButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->SetUserModeButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->StartScanButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->StopScanButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );


  //----------------------------------------------------------------
  // Etc Frame

  // observer load volume button
  
  //this->SetOrientButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
  
  this->AddCoordsandOrientTarget
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->DeleteTargetPushButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->DeleteAllTargetPushButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  
  this->MoveBWPushButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->MoveFWPushButton
    ->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->LocatorCheckButton
    ->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);

}



void vtkProstateNavGUI::HandleMouseEvent(vtkSlicerInteractorStyle *style)
{
    vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
    vtkSlicerInteractorStyle *istyle0 
      = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI0()->GetSliceViewer()
                                               ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
    vtkSlicerInteractorStyle *istyle1 
      = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI1()->GetSliceViewer()
                                               ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
    vtkSlicerInteractorStyle *istyle2 
      = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI2()->GetSliceViewer()
                                               ->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());

    vtkCornerAnnotation *anno = NULL;
    if (style == istyle0)
    {
        anno = appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
    else if (style == istyle1)
    {
        anno = appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
    else if (style == istyle2)
    {
        anno = appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()->GetCornerAnnotation();
    }
    if (anno)
    {
        const char *rasText = anno->GetText(1);
        if ( rasText != NULL )
        {
            std::string ras = std::string(rasText);
            
            // remove "R:," "A:," and "S:" from the string
            int loc = ras.find("R:", 0);
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
            int found = ras.find("\n", 0);
            while ( found != std::string::npos )
            {
                ras = ras.replace(found, 1, " ");
                found = ras.find("\n", 0);
            }

        }
    }
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::ProcessGUIEvents(vtkObject *caller,
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
  // Check Work Phase Transition Buttons

  if ( event == vtkKWPushButton::InvokedEvent)
    {
    int phase;
    for (phase = 0; phase < this->WorkPhaseButtonSet->GetNumberOfWidgets(); phase ++)
      {
      if (this->WorkPhaseButtonSet->GetWidget(phase) == vtkKWPushButton::SafeDownCast(caller))
        {
        break;
        }
      }
    if (phase < vtkProstateNavLogic::NumPhases) // if pressed one of them
      {
      ChangeWorkPhase(phase, 1);
      }
    }


  //----------------------------------------------------------------
  // Wizard Frame

  if (this->WizardWidget->GetWizardWorkflow() == vtkKWWizardWorkflow::SafeDownCast(caller) &&
      event == vtkKWWizardWorkflow::CurrentStateChangedEvent)
    {
          
    int phase = vtkProstateNavLogic::Emergency;
    vtkKWWizardStep* step =  this->WizardWidget->GetWizardWorkflow()->GetCurrentStep();

    for (int i = 0; i < vtkProstateNavLogic::NumPhases-1; i ++)
      {
      if (step == vtkKWWizardStep::SafeDownCast(this->WizardSteps[i]))
        {
        phase = i;
        }
      }
    
    ChangeWorkPhase(phase);
    }


  //----------------------------------------------------------------
  // Visualization Control Frame
  
  else if (this->RedSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
            && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    const char* selected = this->RedSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_RED, selected);
    }

  else if (this->YellowSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
            && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    const char* selected = this->YellowSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_YELLOW, selected);
    }

  else if (this->GreenSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
          && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    const char* selected = this->GreenSliceMenu->GetValue();
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_GREEN, selected);
    }

  // -- "Locator" button 
  else if (this->SetLocatorModeButton == vtkKWPushButton::SafeDownCast(caller) 
            && event == vtkKWPushButton::InvokedEvent)
    {
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_RED, "Locator");
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_YELLOW, "Locator");
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_GREEN, "Locator");
    }
  
  // -- "User" button 
  else if (this->SetUserModeButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_RED, "User");
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_YELLOW, "User");
    ChangeSlicePlaneDriver(vtkProstateNavGUI::SLICE_PLANE_GREEN, "User");
    }
  
  // -- "Freeze Image Position" check button 
  else if (this->FreezeImageCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
           && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    if (this->FreezeImageCheckButton->GetSelectedState() == 1)
      {
      this->FreezeOrientationUpdate = 1;
      /*
        this->OrgNeedOrientationUpdate0  = this->NeedOrientationUpdate0;
        this->OrgNeedOrientationUpdate1  = this->NeedOrientationUpdate1;
        this->OrgNeedOrientationUpdate2  = this->NeedOrientationUpdate2;
        this->OrgNeedRealtimeImageUpdate0 = this->NeedRealtimeImageUpdate0;
        this->OrgNeedRealtimeImageUpdate1 = this->NeedRealtimeImageUpdate1;
        this->OrgNeedRealtimeImageUpdate2 = this->NeedRealtimeImageUpdate2;
        
        this->NeedOrientationUpdate0 = 0;
        this->NeedOrientationUpdate1 = 0;
        this->NeedOrientationUpdate2 = 0;
        this->NeedRealtimeImageUpdate0 = 0;
        this->NeedRealtimeImageUpdate1 = 0;
        this->NeedRealtimeImageUpdate2 = 0;
      */
      }
    else
      {
      this->FreezeOrientationUpdate = 0;
      /*
        this->NeedOrientationUpdate0  = this->OrgNeedOrientationUpdate0;
        this->NeedOrientationUpdate1  = this->OrgNeedOrientationUpdate1;
        this->NeedOrientationUpdate2  = this->OrgNeedOrientationUpdate2;
        this->NeedRealtimeImageUpdate0 = this->OrgNeedRealtimeImageUpdate0;
        this->NeedRealtimeImageUpdate1 = this->OrgNeedRealtimeImageUpdate1;
        this->NeedRealtimeImageUpdate2 = this->OrgNeedRealtimeImageUpdate2;
      */
      }
    }
  else if (this->ImagingMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
           && event == vtkKWMenu::MenuItemInvokedEvent )
    {
      
    const char* selected = this->ImagingMenu->GetValue();
    if (strcmp(selected, "Perpendicular") == 0)
      {
      this->RealtimeImageOrient = vtkProstateNavGUI::SLICE_RTIMAGE_PERP;
      }
    else if (strcmp(selected, "In-plane 90") == 0)
      {
      this->RealtimeImageOrient = vtkProstateNavGUI::SLICE_RTIMAGE_INPLANE90;
      }
    else //if ( strcmp(selected, "In-plane") == 0 )
      {
      this->RealtimeImageOrient = vtkProstateNavGUI::SLICE_RTIMAGE_INPLANE;
      }
    
    std::cerr << "ImagingMenu =======> " << selected << "  :  " << this->RealtimeImageOrient << std::endl;
    
    }

  else if (this->StartScanButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    this->Logic->ScanStart();
    }
  else if (this->StopScanButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    this->Logic->ScanStop();
    }
  


  //----------------------------------------------------------------
  // Etc Frame

          
#ifdef USE_NAVITRACK
  else if (this->LoadConfigButtonNT->GetWidget() == vtkKWLoadSaveButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent )
    {
    const char * filename = this->LoadConfigButtonNT->GetWidget()->GetFileName();
    if (filename)
      {
      const vtksys_stl::string fname(filename);
      this->ConfigFileEntry->SetValue(fname.c_str());
      }
    else
      {
      this->ConfigFileEntry->SetValue("");
      }
    //this->LoadConfigButtonNT->GetWidget()->SetText ("Browse Config File");
    }
#endif

  else if (this->AddCoordsandOrientTarget == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    
    {
    int row = this->TargetListColumnList->GetWidget()->GetNumberOfRows();
    int rownumber = row + 1; 
    
    char xcoordsrobot[12];
    char ycoordsrobot[12];
    char zcoordsrobot[12];
    char o1coordsrobot[12];
    char o2coordsrobot[12];
    char o3coordsrobot[12];
    char o4coordsrobot[12];
    
    strncpy(xcoordsrobot, this->NREntry->GetWidget()->GetValue(), 12);
    strncpy(ycoordsrobot, this->NAEntry->GetWidget()->GetValue(), 12);
    strncpy(zcoordsrobot, this->NSEntry->GetWidget()->GetValue(), 12);
    
    strncpy(o1coordsrobot, this->PREntry->GetWidget()->GetValue(), 12);
    strncpy(o2coordsrobot, this->PAEntry->GetWidget()->GetValue(), 12);
    strncpy(o3coordsrobot, this->PSEntry->GetWidget()->GetValue(), 12);
    strncpy(o4coordsrobot, this->O4Entry->GetWidget()->GetValue(), 12);
    
    //merge coordinates of the same type in on vector
    float xcoordsrobotforsend = atof(this->NREntry->GetWidget()->GetValue());
    xsendrobotcoords.push_back(xcoordsrobotforsend );
    float ycoordsrobotforsend = atof(this->NAEntry->GetWidget()->GetValue());
    ysendrobotcoords.push_back(ycoordsrobotforsend );
    float zcoordsrobotforsend = atof(this->NSEntry->GetWidget()->GetValue());
    zsendrobotcoords.push_back(zcoordsrobotforsend );
    
    float o1coordsrobotforsend = atof(this->PREntry->GetWidget()->GetValue());
    osendrobotcoords.push_back(o1coordsrobotforsend );
    float o2coordsrobotforsend = atof(this->PAEntry->GetWidget()->GetValue());
    osendrobotcoords.push_back(o2coordsrobotforsend );
    float o3coordsrobotforsend = atof(this->PSEntry->GetWidget()->GetValue());
    osendrobotcoords.push_back(o3coordsrobotforsend );
    float o4coordsrobotforsend = atof(this->O4Entry->GetWidget()->GetValue());
    osendrobotcoords.push_back(o4coordsrobotforsend );
    
    
    sendrobotcoordsvector.push_back(osendrobotcoords);   
    
    osendrobotcoords.clear();
    
    char coordsxyz[512]; 
    sprintf(coordsxyz, "%s, %s, %s", xcoordsrobot, ycoordsrobot, zcoordsrobot);
    char orientsxyz[512]; 
    sprintf(orientsxyz, "%s, %s, %s, %s", o1coordsrobot, o2coordsrobot, o3coordsrobot, o4coordsrobot);
    
    /*
      int CountTarget;   
      CountTarget = 1;
      char DispCountTarget[512];
      sprintf(DispCountTarget,"%s", CountTarget); 
    */
    this->TargetListColumnList->GetWidget()->AddRow();
    // this->TargetListColumnList->GetWidget()->SetCellText(row, 0,DispCountTarget);
    this->TargetListColumnList->GetWidget()->SetCellText(row, 1,coordsxyz);
    this->TargetListColumnList->GetWidget()->SetCellText(row, 2,orientsxyz);
    
    }

  else if (this->SetOrientButton == vtkKWPushButton::SafeDownCast(caller) 
               && event == vtkKWPushButton::InvokedEvent)
    {      
    vtkSlicerApplication::GetInstance()->ErrorMessage("xsendrobotcoords[sendindex]"); 
          
    std::string robotcommandkey;
    std::string robotcommandvalue;  
    robotcommandkey = "command";
    robotcommandvalue = BRPTPR_TARGET;
    
    
    int sendindex = this->TargetListColumnList->GetWidget()->GetIndexOfFirstSelectedRow();
    
    }
  else if (this->DeleteTargetPushButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    int numOfRows = this->TargetListColumnList->GetWidget()->GetNumberOfSelectedRows();
    if (numOfRows == 1)
      {
      int index[2];
      this->TargetListColumnList->GetWidget()->GetSelectedRows(index);
      this->TargetListColumnList->GetWidget()->DeleteRow(index[0]);
      }
    }
  else if (this->DeleteAllTargetPushButton == vtkKWPushButton::SafeDownCast(caller) 
           && event == vtkKWPushButton::InvokedEvent)
    {
    this->TargetListColumnList->GetWidget()->DeleteAllRows();
    }
  
  // Process Wizard GUI (Active step only)
  else
    {
    int phase = this->Logic->GetCurrentPhase();
    this->WizardSteps[phase]->ProcessGUIEvents(caller, event, callData);
    }

} 


void vtkProstateNavGUI::Init()
{
    this->DataManager->SetMRMLScene(this->GetMRMLScene());
    //   this->LocatorModelID = std::string(this->DataManager->RegisterStream(0));
    this->LocatorModelID_new = std::string(this->DataManager->RegisterStream_new(0));
    
}



void vtkProstateNavGUI::DataCallback(vtkObject *caller, 
        unsigned long eid, void *clientData, void *callData)
{
    vtkProstateNavGUI *self = reinterpret_cast<vtkProstateNavGUI *>(clientData);
    vtkDebugWithObjectMacro(self, "In vtkProstateNavGUI DataCallback");

    self->UpdateAll();
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::ProcessLogicEvents ( vtkObject *caller,
    unsigned long event, void *callData )
{

    // Fill in
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::ProcessMRMLEvents ( vtkObject *caller,
    unsigned long event, void *callData )
{
    // Fill in
}



//---------------------------------------------------------------------------
void vtkProstateNavGUI::Enter ( )
{
    // Fill in
    vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();

    this->Logic0 = appGUI->GetMainSliceGUI0()->GetLogic();
    this->Logic1 = appGUI->GetMainSliceGUI1()->GetLogic();
    this->Logic2 = appGUI->GetMainSliceGUI2()->GetLogic();
    this->SliceNode0 = appGUI->GetMainSliceGUI0()->GetLogic()->GetSliceNode();
  
    this->SliceNode1 = appGUI->GetMainSliceGUI1()->GetLogic()->GetSliceNode();
    this->SliceNode2 = appGUI->GetMainSliceGUI2()->GetLogic()->GetSliceNode();
    this->Control0 = appGUI->GetMainSliceGUI0()->GetSliceController();
    this->Control1 = appGUI->GetMainSliceGUI1()->GetSliceController();
    this->Control2 = appGUI->GetMainSliceGUI2()->GetSliceController();

    vtkSlicerApplication  *app          = (vtkSlicerApplication *)this->GetApplication();
    vtkSlicerVolumesGUI   *volGui       = (vtkSlicerVolumesGUI*)app->GetModuleGUIByName("Volumes");
    vtkSlicerVolumesLogic *VolumesLogic = (vtkSlicerVolumesLogic*)(volGui->GetLogic());

    this->GetLogic()->AddRealtimeVolumeNode(VolumesLogic, "Realtime");

    // neccessary?
    //this->Logic0->GetForegroundLayer()->SetUseReslice(0);

}

//---------------------------------------------------------------------------
void vtkProstateNavGUI::Exit ( )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::BuildGUI ( )
{

    // ---
    // MODULE GUI FRAME 
    // create a page
    this->UIPanel->AddPage ( "ProstateNav", "ProstateNav", NULL );

    BuildGUIForHelpFrame();
    BuildGUIForWorkPhaseFrame ();
    BuildGUIForWizardFrame();
    BuildGUIForVisualizationControlFrame();
    BuildGUIForDeviceFrame();

}


//---------------------------------------------------------------------------
void vtkProstateNavGUI::BuildGUIForWizardFrame()
{
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ProstateNav" );
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    

    // ----------------------------------------------------------------
    // WIZARD FRAME         
    // ----------------------------------------------------------------

    vtkSlicerModuleCollapsibleFrame *wizardFrame = 
      vtkSlicerModuleCollapsibleFrame::New();
    wizardFrame->SetParent(page);
    wizardFrame->Create();
    wizardFrame->SetLabelText("Wizard");
    wizardFrame->ExpandFrame();

    app->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                wizardFrame->GetWidgetName(), 
                page->GetWidgetName());
   
    this->WizardWidget->SetParent(wizardFrame->GetFrame());
    this->WizardWidget->Create();
    this->WizardWidget->GetSubTitleLabel()->SetHeight(1);
    this->WizardWidget->SetClientAreaMinimumHeight(200);
    //this->WizardWidget->SetButtonsPositionToTop();
    this->WizardWidget->NextButtonVisibilityOn();
    this->WizardWidget->BackButtonVisibilityOn();
    this->WizardWidget->OKButtonVisibilityOff();
    this->WizardWidget->CancelButtonVisibilityOff();
    this->WizardWidget->FinishButtonVisibilityOff();
    this->WizardWidget->HelpButtonVisibilityOn();

    app->Script("pack %s -side top -anchor nw -fill both -expand y",
                this->WizardWidget->GetWidgetName());

    wizardFrame->Delete();

    // -----------------------------------------------------------------
    // Add the steps to the workflow

    vtkKWWizardWorkflow *wizard_workflow = 
      this->WizardWidget->GetWizardWorkflow();

    // -----------------------------------------------------------------
    // Config File step

    if (!this->WizardSteps[vtkProstateNavLogic::StartUp])
      {
      this->WizardSteps[vtkProstateNavLogic::StartUp] = vtkProstateNavConfigurationStep::New();
      }

    // -----------------------------------------------------------------
    // Scan Control step

    if (!this->WizardSteps[vtkProstateNavLogic::Planning])
      {
      this->WizardSteps[vtkProstateNavLogic::Planning] = vtkProstateNavScanControlStep::New();
      }

    // -----------------------------------------------------------------
    // Calibration step

    if (!this->WizardSteps[vtkProstateNavLogic::Calibration])
      {
      this->WizardSteps[vtkProstateNavLogic::Calibration] = vtkProstateNavCalibrationStep::New();
      }

    // -----------------------------------------------------------------
    // Targeting step

    if (!this->WizardSteps[vtkProstateNavLogic::Targeting])
      {
      this->WizardSteps[vtkProstateNavLogic::Targeting] = vtkProstateNavTargetingStep::New();
      }

    // -----------------------------------------------------------------
    // ManualControl step

    if (!this->WizardSteps[vtkProstateNavLogic::Manual])
      {
      this->WizardSteps[vtkProstateNavLogic::Manual] = vtkProstateNavManualControlStep::New();
      }


    // -----------------------------------------------------------------
    // Set GUI/Logic to each step and add to workflow

    for (int i = 0; i < vtkProstateNavLogic::NumPhases-1; i ++)
      {
      this->WizardSteps[i]->SetGUI(this);
      this->WizardSteps[i]->SetLogic(this->Logic);
      wizard_workflow->AddNextStep(this->WizardSteps[i]);
      }


    // -----------------------------------------------------------------
    // Initial and finish step

    //wizard_workflow->SetFinishStep(this->ManualControlStep);
    wizard_workflow->SetFinishStep(this->WizardSteps[vtkProstateNavLogic::Manual]);
    wizard_workflow->CreateGoToTransitionsToFinishStep();
    //wizard_workflow->SetInitialStep(this->ConfigurationStep);
    wizard_workflow->SetInitialStep(this->WizardSteps[vtkProstateNavLogic::StartUp]);

    // -----------------------------------------------------------------
    // Show the user interface

    this->WizardWidget->GetWizardWorkflow()->
      GetCurrentStep()->ShowUserInterface();

}


void vtkProstateNavGUI::BuildGUIForHelpFrame ()
{

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();

    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ProstateNav" );

    // Define your help text here.
    const char *help = 
      "The **ProstateNav Module** helps you to do prostate Biopsy and Treatment by:"
      " getting Realtime Images from MR-Scanner into Slicer3, control Scanner with Slicer 3,"
      " determin fiducial detection and control the Robot."
      " Module and Logic mainly coded by Junichi Tokuda, David Gobbi and Philip Mewes"; 

    // ----------------------------------------------------------------
    // HELP FRAME         
    // ----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *ProstateNavHelpFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    ProstateNavHelpFrame->SetParent ( page );
    ProstateNavHelpFrame->Create ( );
    ProstateNavHelpFrame->CollapseFrame ( );
    ProstateNavHelpFrame->SetLabelText ("Help");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        ProstateNavHelpFrame->GetWidgetName(), page->GetWidgetName());

    // configure the parent classes help text widget
    this->HelpText->SetParent ( ProstateNavHelpFrame->GetFrame() );
    this->HelpText->Create ( );
    this->HelpText->SetHorizontalScrollbarVisibility ( 0 );
    this->HelpText->SetVerticalScrollbarVisibility ( 1 );
    this->HelpText->GetWidget()->SetText ( help );
    this->HelpText->GetWidget()->SetReliefToFlat ( );
    this->HelpText->GetWidget()->SetWrapToWord ( );
    this->HelpText->GetWidget()->ReadOnlyOn ( );
    this->HelpText->GetWidget()->QuickFormattingOn ( );
    this->HelpText->GetWidget()->SetBalloonHelpString ( "" );
    app->Script ( "pack %s -side top -fill x -expand y -anchor w -padx 2 -pady 4",
        this->HelpText->GetWidgetName ( ) );

    ProstateNavHelpFrame->Delete();

}


void vtkProstateNavGUI::BuildGUIForWorkPhaseFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ProstateNav" );

    //----------------------------------------------------------------
    // WORKPHASE FRAME         
    //----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *workphaseFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    workphaseFrame->SetParent(page);
    workphaseFrame->Create();
    workphaseFrame->SetLabelText("Workphase Frame");
    workphaseFrame->ExpandFrame();
    app->Script("pack %s -side top -anchor center -fill x -padx 2 -pady 2 -in %s",
    workphaseFrame->GetWidgetName(), page->GetWidgetName());

    vtkKWFrame *buttonFrame = vtkKWFrame::New();
    buttonFrame->SetParent( workphaseFrame->GetFrame());
    buttonFrame->Create();

    vtkKWFrame *workphaseStatusFrame = vtkKWFrame::New ( );
    workphaseStatusFrame->SetParent ( workphaseFrame->GetFrame() );
    workphaseStatusFrame->Create ( );
    app->Script ( "pack %s %s -side top -anchor center -fill x -padx 2 -pady 1",
                  buttonFrame->GetWidgetName(),
                  workphaseStatusFrame->GetWidgetName());

    //
    // Work Phase Transition Buttons
    //

    this->WorkPhaseButtonSet = vtkKWPushButtonSet::New();
    this->WorkPhaseButtonSet->SetParent(buttonFrame);
    this->WorkPhaseButtonSet->Create();
    this->WorkPhaseButtonSet->PackHorizontallyOn();
    this->WorkPhaseButtonSet->SetMaximumNumberOfWidgetsInPackingDirection(3);
    this->WorkPhaseButtonSet->SetWidgetsPadX(2);
    this->WorkPhaseButtonSet->SetWidgetsPadY(2);
    this->WorkPhaseButtonSet->UniformColumnsOn();
    this->WorkPhaseButtonSet->UniformRowsOn();

    for (int i = 0; i < vtkProstateNavLogic::NumPhases; i ++)
    {
        this->WorkPhaseButtonSet->AddWidget(i);
        this->WorkPhaseButtonSet->GetWidget(i)->SetWidth(16);
    }

    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::StartUp    )->SetText("Start Up");
    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::Planning   )->SetText("Planning");
    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::Calibration)->SetText("Calibration");
    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::Targeting  )->SetText("Targeting");
    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::Manual     )->SetText("Manual");
    this->WorkPhaseButtonSet->GetWidget(vtkProstateNavLogic::Emergency  )->SetText("Emergency");

    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->WorkPhaseButtonSet->GetWidgetName()
                 );

                 
    //    vtkKWFrameWithLabel *filterFrame = vtkKWFrameWithLabel::New ( );
    //    filterFrame->SetParent ( workphaseFrame->GetFrame() );
    //    filterFrame->Create ( );
    //    filterFrame->SetLabelText ("Connection to server and Needle-Display");
    //    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
    //           filterFrame->GetWidgetName() );

    vtkKWLabel *SoftwareStatusLabel = vtkKWLabel::New();
    SoftwareStatusLabel->SetParent(workphaseStatusFrame);
    SoftwareStatusLabel->Create();
    SoftwareStatusLabel->SetWidth(15);
    SoftwareStatusLabel->SetText("Software:");

    this->SoftwareStatusLabelDisp = vtkKWEntry::New();
    this->SoftwareStatusLabelDisp->SetParent(workphaseStatusFrame);
    this->SoftwareStatusLabelDisp->Create();
    this->SoftwareStatusLabelDisp->SetWidth(10);
    this->SoftwareStatusLabelDisp->SetValue ( "" );

    vtkKWLabel *ScannerStatusLabel = vtkKWLabel::New();
    ScannerStatusLabel->SetParent(workphaseStatusFrame);
    ScannerStatusLabel->Create();
    ScannerStatusLabel->SetWidth(15);
    ScannerStatusLabel->SetText("Scanner:");
    
    this->ScannerStatusLabelDisp = vtkKWEntry::New();
    this->ScannerStatusLabelDisp->SetParent(workphaseStatusFrame);
    this->ScannerStatusLabelDisp->Create();
    this->ScannerStatusLabelDisp->SetWidth(10);
    this->ScannerStatusLabelDisp->SetValue ( "" );

    vtkKWLabel *RobotStatusLabel = vtkKWLabel::New();
    RobotStatusLabel->SetParent(workphaseStatusFrame);
    RobotStatusLabel->Create();
    RobotStatusLabel->SetWidth(15);
    RobotStatusLabel->SetText("Robot:");
    
    this->RobotStatusLabelDisp = vtkKWEntry::New();
    this->RobotStatusLabelDisp->SetParent(workphaseStatusFrame);
    this->RobotStatusLabelDisp->Create();
    this->RobotStatusLabelDisp->SetWidth(10);
    this->RobotStatusLabelDisp->SetValue ( "" );

    this->Script("pack %s %s %s %s %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 SoftwareStatusLabel->GetWidgetName(),
                 SoftwareStatusLabelDisp->GetWidgetName(),
                 ScannerStatusLabel->GetWidgetName(),
                 ScannerStatusLabelDisp->GetWidgetName(),
                 RobotStatusLabel->GetWidgetName(),
                 RobotStatusLabelDisp->GetWidgetName()
                 );

    workphaseFrame->Delete ();
    buttonFrame->Delete ();
    workphaseStatusFrame->Delete ();
  
}


void vtkProstateNavGUI::BuildGUIForDeviceFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ProstateNav" );

    // ----------------------------------------------------------------
    // ROBOT DEVICE FRAME           
    // ----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *deviceFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    deviceFrame->SetParent ( page );
    deviceFrame->Create ( );
    deviceFrame->SetLabelText ("Robot Controll (Coordinates, Speed, Feeding)");
    deviceFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        deviceFrame->GetWidgetName(), page->GetWidgetName());

    /////////////////////////////////////////////////////////////////////
    /// Robot Controll frame 
    /////////////////////////////////////////////////////////////////////

    
    vtkKWFrameWithLabel *controllrobotFrame = vtkKWFrameWithLabel::New();
    controllrobotFrame->SetParent ( deviceFrame->GetFrame() );
    controllrobotFrame->Create ( );
    controllrobotFrame->CollapseFrame ( );
    controllrobotFrame->SetLabelText ("Type and Send to Robot");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
          controllrobotFrame->GetWidgetName());
     

    // Header 1 frame
    vtkKWFrame *header1robotFrame = vtkKWFrame::New();
    header1robotFrame->SetParent ( controllrobotFrame->GetFrame() );
    header1robotFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 header1robotFrame->GetWidgetName(),
                 controllrobotFrame->GetFrame()->GetWidgetName());
    
    
    // Coordinates frame
    vtkKWFrame *coordinatesrobotFrame = vtkKWFrame::New();
    coordinatesrobotFrame->SetParent ( controllrobotFrame->GetFrame() );
    coordinatesrobotFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 coordinatesrobotFrame->GetWidgetName(),
                 controllrobotFrame->GetFrame()->GetWidgetName());
    
    // Header 2 frame
    vtkKWFrame *header2robotFrame = vtkKWFrame::New();
    header2robotFrame->SetParent ( controllrobotFrame->GetFrame() );
    header2robotFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 header2robotFrame->GetWidgetName(),
                 controllrobotFrame->GetFrame()->GetWidgetName());
    // Orientations frame
    vtkKWFrame *orientationsrobotFrame = vtkKWFrame::New();
    orientationsrobotFrame->SetParent ( controllrobotFrame->GetFrame() );
    orientationsrobotFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 orientationsrobotFrame->GetWidgetName(),
                 controllrobotFrame->GetFrame()->GetWidgetName());

    vtkKWFrame *orientationsrobotADDFrame = vtkKWFrame::New();
    orientationsrobotADDFrame->SetParent ( controllrobotFrame->GetFrame() );
    orientationsrobotADDFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 orientationsrobotADDFrame->GetWidgetName(),
                 controllrobotFrame->GetFrame()->GetWidgetName());

    
    // Contents in header 1 frame
    vtkKWLabel *empty1Label = vtkKWLabel::New();
    empty1Label->SetParent(header1robotFrame);
    empty1Label->Create();
    empty1Label->SetWidth(5);
    empty1Label->SetText("");

    vtkKWLabel *xLabel = vtkKWLabel::New();
    xLabel->SetParent(header1robotFrame);
    xLabel->Create();
    xLabel->SetWidth(5);
    xLabel->SetText("X");

    vtkKWLabel *yLabel = vtkKWLabel::New();
    yLabel->SetParent(header1robotFrame);
    yLabel->Create();
    yLabel->SetWidth(5);
    yLabel->SetText("Y");

    vtkKWLabel *zLabel = vtkKWLabel::New();
    zLabel->SetParent(header1robotFrame);
    zLabel->Create();
    zLabel->SetWidth(5);
    zLabel->SetText("Z");
  
    this->Script("pack %s %s %s %s -side left -anchor w -padx 2 -pady 2", 
                empty1Label->GetWidgetName(),
                  xLabel->GetWidgetName(),
                yLabel->GetWidgetName(),
                 zLabel->GetWidgetName());
            
    
        // Contents in C frame 
    vtkKWLabel *nLabel = vtkKWLabel::New();
    nLabel->SetParent(coordinatesrobotFrame);
    nLabel->Create();
    nLabel->SetWidth(5);
    nLabel->SetText("C:");
   
    this->NREntry = vtkKWEntryWithLabel::New();
    this->NREntry->SetParent(coordinatesrobotFrame);
    this->NREntry->Create();
    this->NREntry->GetWidget()->SetWidth(5);
    this->NREntry->GetWidget()->SetValue("0");

    this->NAEntry = vtkKWEntryWithLabel::New();
    this->NAEntry->SetParent(coordinatesrobotFrame);
    this->NAEntry->Create();
    this->NAEntry->GetWidget()->SetWidth(5);
    this->NAEntry->GetWidget()->SetValue("0");

    this->NSEntry = vtkKWEntryWithLabel::New();
    this->NSEntry->SetParent(coordinatesrobotFrame);
    this->NSEntry->Create();
    this->NSEntry->GetWidget()->SetWidth(5);
    this->NSEntry->GetWidget()->SetValue("0");
  
    this->Script("pack %s %s %s %s -side left -anchor w -padx 2 -pady 2", 
                nLabel->GetWidgetName(),
                this->NREntry->GetWidgetName(),
                this->NAEntry->GetWidgetName(),
                this->NSEntry->GetWidgetName());

    // Contents in header 1 frame
    vtkKWLabel *empty2Label = vtkKWLabel::New();
    empty2Label->SetParent(header2robotFrame);
    empty2Label->Create();
    empty2Label->SetWidth(5);
    empty2Label->SetText("");

    vtkKWLabel *o1Label = vtkKWLabel::New();
    o1Label->SetParent(header2robotFrame);
    o1Label->Create();
    o1Label->SetWidth(5);
    o1Label->SetText("O-1");

    vtkKWLabel *o2Label = vtkKWLabel::New();
    o2Label->SetParent(header2robotFrame);
    o2Label->Create();
    o2Label->SetWidth(5);
    o2Label->SetText("O-2");

    vtkKWLabel *o3Label = vtkKWLabel::New();
    o3Label->SetParent(header2robotFrame);
    o3Label->Create();
    o3Label->SetWidth(5);
    o3Label->SetText("O-3");
    
    vtkKWLabel *o4Label = vtkKWLabel::New();
    o4Label->SetParent(header2robotFrame);
    o4Label->Create();
    o4Label->SetWidth(5);
    o4Label->SetText("O-4");

    this->Script("pack %s %s %s %s %s  -side left -anchor w -padx 2 -pady 2", 
                empty2Label->GetWidgetName(),
               o1Label->GetWidgetName(),
               o2Label->GetWidgetName(),
               o3Label->GetWidgetName(),
               o4Label->GetWidgetName());
   

    // Contents in P frame
    vtkKWLabel *oLabel = vtkKWLabel::New();
    oLabel->SetParent(orientationsrobotFrame);
    oLabel->Create();
    oLabel->SetWidth(5);
    oLabel->SetText("O:");
   
  

    this->PREntry = vtkKWEntryWithLabel::New();
    this->PREntry->SetParent(orientationsrobotFrame);
    this->PREntry->Create();
    this->PREntry->GetWidget()->SetWidth(5);
    this->PREntry->GetWidget()->SetValue("0");

    this->PAEntry = vtkKWEntryWithLabel::New();
    this->PAEntry->SetParent(orientationsrobotFrame);
    this->PAEntry->Create();
    this->PAEntry->GetWidget()->SetWidth(5);
    this->PAEntry->GetWidget()->SetValue("0");

    this->PSEntry = vtkKWEntryWithLabel::New();
    this->PSEntry->SetParent(orientationsrobotFrame);
    this->PSEntry->Create();
    this->PSEntry->GetWidget()->SetWidth(5);
    this->PSEntry->GetWidget()->SetValue("0");

    this->O4Entry = vtkKWEntryWithLabel::New();
    this->O4Entry->SetParent(orientationsrobotFrame);
    this->O4Entry->Create();
    this->O4Entry->GetWidget()->SetWidth(5);
    this->O4Entry->GetWidget()->SetValue("0");

    this->Script("pack %s %s %s %s %s -side left -anchor w -padx 2 -pady 2", 
               oLabel->GetWidgetName(),
               this->PREntry->GetWidgetName(),
               this->PAEntry->GetWidgetName(),
               this->PSEntry->GetWidgetName(),
              this->O4Entry->GetWidgetName());

  

    this->AddCoordsandOrientTarget = vtkKWPushButton::New();
    this->AddCoordsandOrientTarget->SetParent(orientationsrobotADDFrame);
    this->AddCoordsandOrientTarget->Create();
    this->AddCoordsandOrientTarget->SetText( "OK" );
    this->AddCoordsandOrientTarget->SetWidth ( 12 );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                  this->AddCoordsandOrientTarget->GetWidgetName());


    vtkKWFrameWithLabel *targetlistFrame = vtkKWFrameWithLabel::New();
    targetlistFrame->SetParent ( deviceFrame->GetFrame() );
    targetlistFrame->Create ( );
    targetlistFrame->CollapseFrame ( );
    targetlistFrame->SetLabelText ("Defined Target Points");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  targetlistFrame->GetWidgetName());

    
    // add the multicolumn list to show the points
    this->TargetListColumnList = vtkKWMultiColumnListWithScrollbars::New ( );
    this->TargetListColumnList->SetParent ( targetlistFrame->GetFrame() );
    this->TargetListColumnList->Create ( );
    this->TargetListColumnList->SetHeight(1);
    this->TargetListColumnList->GetWidget()->SetSelectionTypeToRow();
    this->TargetListColumnList->GetWidget()->MovableRowsOff();
    this->TargetListColumnList->GetWidget()->MovableColumnsOff();
    // set up the columns of data for each point
    // refer to the header file for order
      this->TargetListColumnList->GetWidget()->AddColumn("T#");
    this->TargetListColumnList->GetWidget()->AddColumn("Target Coords. (x,y,z)");
    this->TargetListColumnList->GetWidget()->AddColumn("Target Orient. ()");
  
    
    // now set the attributes that are equal across the columns
    for (int col = 0; col < 3; col++)
    {
      if(col==0)
   {
        this->TargetListColumnList->GetWidget()->SetColumnWidth(col, 7);
   }
      else
   {
   this->TargetListColumnList->GetWidget()->SetColumnWidth(col, 22);
   }

        this->TargetListColumnList->GetWidget()->SetColumnAlignmentToLeft(col);
        this->TargetListColumnList->GetWidget()->ColumnEditableOff(col);
    }

    app->Script ( "pack %s -fill both -expand true",
                  this->TargetListColumnList->GetWidgetName());

    
    vtkKWFrame *targetbuttonFrame = vtkKWFrame::New();
    targetbuttonFrame->SetParent ( targetlistFrame->GetFrame() );
    targetbuttonFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 targetbuttonFrame->GetWidgetName(),
                 targetlistFrame->GetFrame()->GetWidgetName());

    // add a delete button 
    this->DeleteTargetPushButton = vtkKWPushButton::New ( );
    this->DeleteTargetPushButton->SetParent ( targetbuttonFrame );
    this->DeleteTargetPushButton->Create ( );
    this->DeleteTargetPushButton->SetText ("Delete Target");
    this->DeleteTargetPushButton->SetWidth (12);
    this->DeleteTargetPushButton->SetBalloonHelpString("Delete the selected Target.");

    // add a delete button 
    this->DeleteAllTargetPushButton = vtkKWPushButton::New ( );
    this->DeleteAllTargetPushButton->SetParent ( targetbuttonFrame );
    this->DeleteAllTargetPushButton->Create ( );
    this->DeleteAllTargetPushButton->SetText ("Delete All Targets");
    this->DeleteAllTargetPushButton->SetWidth (12);
    this->DeleteAllTargetPushButton->SetBalloonHelpString("Delete all Target Points.");

    app->Script("pack %s %s -side left -anchor w -padx 2 -pady 2", 
                this->DeleteTargetPushButton->GetWidgetName(),
                this->DeleteAllTargetPushButton->GetWidgetName());

    vtkKWFrameWithLabel *SetOrientandMoveFrame = vtkKWFrameWithLabel::New();
    SetOrientandMoveFrame->SetParent ( deviceFrame->GetFrame() );
    SetOrientandMoveFrame->Create ( );
    SetOrientandMoveFrame->CollapseFrame ( );
    SetOrientandMoveFrame->SetLabelText ("Command Frame (Speed, Orientation, Feed)");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  SetOrientandMoveFrame->GetWidgetName());

    vtkKWFrame *RobotSpeedFrame = vtkKWFrame::New();
    RobotSpeedFrame->SetParent ( SetOrientandMoveFrame->GetFrame() );
    RobotSpeedFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                  RobotSpeedFrame->GetWidgetName(),
                  SetOrientandMoveFrame->GetFrame()->GetWidgetName());
    

    vtkKWFrame *OrientMoveFrame = vtkKWFrame::New();
    OrientMoveFrame->SetParent ( SetOrientandMoveFrame->GetFrame() );
    OrientMoveFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                  OrientMoveFrame->GetWidgetName(),
                  SetOrientandMoveFrame->GetFrame()->GetWidgetName());
    
    
    // BW button 
    this->MoveFWPushButton = vtkKWPushButton::New ( );
    this->MoveFWPushButton->SetParent ( OrientMoveFrame );
    this->MoveFWPushButton->Create ( );
    this->MoveFWPushButton->SetText ("<--Move(BW)");
    this->MoveFWPushButton->SetWidth (12);
    this->MoveFWPushButton->SetBalloonHelpString("Delete the selected Target.");
    
    // FW  button 
    this->MoveBWPushButton = vtkKWPushButton::New ( );
    this->MoveBWPushButton->SetParent ( OrientMoveFrame );
    this->MoveBWPushButton->Create ( );
    this->MoveBWPushButton->SetText ("Move(FW)-->");
    this->MoveBWPushButton->SetWidth (12);
    this->MoveBWPushButton->SetBalloonHelpString("Delete all Target Points.");
    
    // Set Orientation button 
    this->SetOrientButton = vtkKWPushButton::New ( );
    this->SetOrientButton->SetParent ( OrientMoveFrame );
    this->SetOrientButton->Create ( );
    this->SetOrientButton->SetText ("Set Orientation");
    this->SetOrientButton->SetWidth (17);
    
    app->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
                this->MoveFWPushButton->GetWidgetName(),
                this->MoveBWPushButton->GetWidgetName(),     
                this->SetOrientButton->GetWidgetName());

    

    vtkKWFrameWithLabel *DisplayRobotCoordsLabelFrame = vtkKWFrameWithLabel::New();
    DisplayRobotCoordsLabelFrame->SetParent ( deviceFrame->GetFrame() );
    DisplayRobotCoordsLabelFrame->Create ( );
    DisplayRobotCoordsLabelFrame->CollapseFrame ( );
    DisplayRobotCoordsLabelFrame->SetLabelText ("Show Robot Coords. and Orient.");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  DisplayRobotCoordsLabelFrame->GetWidgetName());


    vtkKWFrame *DisplayRobotCoordsFrame = vtkKWFrame::New();
    DisplayRobotCoordsFrame->SetParent (DisplayRobotCoordsLabelFrame->GetFrame() );
    DisplayRobotCoordsFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                  DisplayRobotCoordsFrame->GetWidgetName(),
                  DisplayRobotCoordsLabelFrame->GetFrame()->GetWidgetName());

    vtkKWFrame *DisplayRobotOrientFrame = vtkKWFrame::New();
    DisplayRobotOrientFrame->SetParent (DisplayRobotCoordsLabelFrame->GetFrame() );
    DisplayRobotOrientFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                  DisplayRobotOrientFrame->GetWidgetName(),
                  DisplayRobotCoordsLabelFrame->GetFrame()->GetWidgetName());

    
   
    vtkKWLabel *RobotPositionLabel = vtkKWLabel::New();
    RobotPositionLabel->SetParent(DisplayRobotCoordsFrame);
    RobotPositionLabel->Create();
    RobotPositionLabel->SetWidth(20);
    RobotPositionLabel->SetText("Position: ");

    this->PositionEntry = vtkKWEntryWithLabel::New();
    this->PositionEntry->SetParent(DisplayRobotCoordsFrame);
    this->PositionEntry->Create();
    this->PositionEntry->GetWidget()->SetWidth(23);
    this->PositionEntry->GetWidget()->SetValue("0.0, 0.0 ,0.0");
    
    this->Script("pack %s %s -side left -anchor w -padx 2 -pady 2", 
                RobotPositionLabel->GetWidgetName(),
                this->PositionEntry->GetWidgetName());


    vtkKWLabel *RobotOrientLabel = vtkKWLabel::New();
    RobotOrientLabel->SetParent(DisplayRobotOrientFrame);
    RobotOrientLabel->Create();
    RobotOrientLabel->SetWidth(20);
    RobotOrientLabel->SetText("Normal / Transnormal: ");

    this->OrientEntry = vtkKWEntryWithLabel::New();
    this->OrientEntry->SetParent(DisplayRobotOrientFrame);
    this->OrientEntry->Create();
    this->OrientEntry->GetWidget()->SetWidth(40);
    this->OrientEntry->GetWidget()->SetValue("0.0, 0.0, 0.0");
    
    this->Script("pack %s %s -side left -anchor w -padx 2 -pady 2", 
                RobotOrientLabel->GetWidgetName(),
                this->OrientEntry->GetWidgetName());

    //---------------------------------------------------------------------------------------------------------------------------
     empty1Label->Delete();
     empty2Label->Delete();
     xLabel->Delete();
     yLabel->Delete();
     zLabel->Delete();

     o1Label->Delete();
     o2Label->Delete();
     o3Label->Delete();
     
     oLabel->Delete();
     nLabel->Delete();

     deviceFrame->Delete();
     OrientMoveFrame->Delete();
     targetbuttonFrame->Delete();
     targetlistFrame->Delete();
     controllrobotFrame->Delete();
     header1robotFrame->Delete();
     header2robotFrame->Delete();
     coordinatesrobotFrame->Delete();
     orientationsrobotFrame->Delete();

}


/*
void vtkProstateNavGUI::BuildGUIForTrackingFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "ProstateNav" );


    // ----------------------------------------------------------------
    // Navigation FRAME           notin use
    // ----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *trackingFrame = vtkSlicerModuleCollapsibleFrame::New ( );      
    trackingFrame->SetParent ( page );
    trackingFrame->Create ( );
    trackingFrame->SetLabelText ("Orientation Controll");
    //trackingFrame->ExpandFrame ( );
    trackingFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        trackingFrame->GetWidgetName(), page->GetWidgetName());



    // Display frame: Options to locator display 
    // -----------------------------------------
    vtkKWFrameWithLabel *displayFrame = vtkKWFrameWithLabel::New ( );
    displayFrame->SetParent ( trackingFrame->GetFrame() );
    displayFrame->Create ( );
    displayFrame->SetLabelText ("Locator Display");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           displayFrame->GetWidgetName() );

    this->LocatorCheckButton = vtkKWCheckButton::New();
    this->LocatorCheckButton->SetParent(displayFrame->GetFrame());
    this->LocatorCheckButton->Create();
    this->LocatorCheckButton->SelectedStateOff();
    this->LocatorCheckButton->SetText("Show Locator");
    this->Script("pack %s -side left -anchor w -padx 2 -pady 2", 
        this->LocatorCheckButton->GetWidgetName());


    // Driver frame: Locator can drive slices 
    // -----------------------------------------
    vtkKWFrameWithLabel *driverFrame = vtkKWFrameWithLabel::New ( );
    driverFrame->SetParent ( trackingFrame->GetFrame() );
    driverFrame->Create ( );
    driverFrame->SetLabelText ("Driver");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           driverFrame->GetWidgetName() );

    // Mode frame
    vtkKWFrame *modeFrame = vtkKWFrame::New();
    modeFrame->SetParent ( driverFrame->GetFrame() );
    modeFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
         modeFrame->GetWidgetName(),
         driverFrame->GetFrame()->GetWidgetName());


    this->LocatorModeCheckButton = vtkKWCheckButton::New();
    this->LocatorModeCheckButton->SetParent(modeFrame);
    this->LocatorModeCheckButton->Create();
    this->LocatorModeCheckButton->SelectedStateOff();
    this->LocatorModeCheckButton->SetText("Locator");

    this->UserModeCheckButton = vtkKWCheckButton::New();
    this->UserModeCheckButton->SetParent(modeFrame);
    this->UserModeCheckButton->Create();
    this->UserModeCheckButton->SelectedStateOn();
    this->UserModeCheckButton->SetText("User");

    this->FreezeImageCheckButton = vtkKWCheckButton::New();
    this->FreezeImageCheckButton->SetParent(modeFrame);
    this->FreezeImageCheckButton->Create();
    this->FreezeImageCheckButton->SelectedStateOff();
    this->FreezeImageCheckButton->SetText("Freeze Image Position");


    this->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
        this->LocatorModeCheckButton->GetWidgetName(),
        this->FreezeImageCheckButton->GetWidgetName(),
        this->UserModeCheckButton->GetWidgetName());


    // slice frame
    vtkKWFrame *sliceFrame = vtkKWFrame::New();
    sliceFrame->SetParent ( driverFrame->GetFrame() );
    sliceFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
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
    this->RedSliceMenu->SetValue ("User");

    this->YellowSliceMenu = vtkKWMenuButton::New();
    this->YellowSliceMenu->SetParent(sliceFrame);
    this->YellowSliceMenu->Create();
    this->YellowSliceMenu->SetWidth(10);
    this->YellowSliceMenu->SetBackgroundColor(color->SliceGUIYellow);
    this->YellowSliceMenu->SetActiveBackgroundColor(color->SliceGUIYellow);
    this->YellowSliceMenu->GetMenu()->AddRadioButton ("User");
    this->YellowSliceMenu->GetMenu()->AddRadioButton ("Locator");
    this->YellowSliceMenu->SetValue ("User");

    this->GreenSliceMenu = vtkKWMenuButton::New();
    this->GreenSliceMenu->SetParent(sliceFrame);
    this->GreenSliceMenu->Create();
    this->GreenSliceMenu->SetWidth(10);
    this->GreenSliceMenu->SetBackgroundColor(color->SliceGUIGreen);
    this->GreenSliceMenu->SetActiveBackgroundColor(color->SliceGUIGreen);
    this->GreenSliceMenu->GetMenu()->AddRadioButton ("User");
    this->GreenSliceMenu->GetMenu()->AddRadioButton ("Locator");
    this->GreenSliceMenu->SetValue ("User");

    this->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
                 this->RedSliceMenu->GetWidgetName(),
                 this->YellowSliceMenu->GetWidgetName(),
                 this->GreenSliceMenu->GetWidgetName());

    trackingFrame->Delete();
    displayFrame->Delete();
    driverFrame->Delete();
    modeFrame->Delete();
    sliceFrame->Delete();
}
*/


//---------------------------------------------------------------------------
void vtkProstateNavGUI::BuildGUIForVisualizationControlFrame ()
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
  vtkKWWidget *page = this->UIPanel->GetPageWidget ("ProstateNav");
  
  vtkSlicerModuleCollapsibleFrame *visCtrlFrame = vtkSlicerModuleCollapsibleFrame::New();
  visCtrlFrame->SetParent(page);
  visCtrlFrame->Create();
  visCtrlFrame->SetLabelText("Visualization / Scanner Control");
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
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               driverFrame->GetWidgetName());
  
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
  this->FreezeImageCheckButton->SetText("Freeze Image Position");
  this->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
               this->SetLocatorModeButton->GetWidgetName(),
               this->SetUserModeButton->GetWidgetName(),
               this->FreezeImageCheckButton->GetWidgetName());
  
  
  // -----------------------------------------
  // Real-time imaging: Scanner controled

  vtkKWFrameWithLabel *rtImageFrame = vtkKWFrameWithLabel::New ( );
  rtImageFrame->SetParent(visCtrlFrame->GetFrame());
  rtImageFrame->Create();
  rtImageFrame->SetLabelText("Real-time Imaging");
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               rtImageFrame->GetWidgetName());

  // Scan start/stop frame
  vtkKWFrame *scanFrame = vtkKWFrame::New();
  scanFrame->SetParent (rtImageFrame->GetFrame());
  scanFrame->Create();
  app->Script("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
              scanFrame->GetWidgetName(),
              rtImageFrame->GetFrame()->GetWidgetName());
  
  this->StartScanButton = vtkKWPushButton::New();
  this->StartScanButton->SetParent(scanFrame);
  this->StartScanButton->Create();
  this->StartScanButton->SetText("Start Scan");
  this->StartScanButton->SetWidth(12);
  
  this->StopScanButton = vtkKWPushButton::New();
  this->StopScanButton->SetParent(scanFrame);
  this->StopScanButton->Create();
  this->StopScanButton->SetText("Stop Scan");
  this->StopScanButton->SetWidth(12);

  this->Script("pack %s %s -side left -anchor w -padx 2 -pady 2", 
               StartScanButton->GetWidgetName(),
               StopScanButton->GetWidgetName());


  // Orientation control frame
  vtkKWFrame *orientationFrame = vtkKWFrame::New();
  modeFrame->SetParent(rtImageFrame);
  modeFrame->Create();
  app->Script("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
              orientationFrame->GetWidgetName(),
              rtImageFrame->GetFrame()->GetWidgetName());

  this->ImagingControlCheckButton = vtkKWCheckButton::New();
  this->ImagingControlCheckButton->SetParent(orientationFrame);
  this->ImagingControlCheckButton->Create();
  this->ImagingControlCheckButton->SelectedStateOff();
  this->ImagingControlCheckButton->SetText("Imaging Orientation Control:");
  
  this->ImagingMenu = vtkKWMenuButton::New();
  this->ImagingMenu->SetParent(orientationFrame);
  this->ImagingMenu->Create();
  this->ImagingMenu->SetWidth(10);
  this->ImagingMenu->GetMenu()->AddRadioButton ("Perpendicular");
  this->ImagingMenu->GetMenu()->AddRadioButton ("In-plane 90");
  this->ImagingMenu->GetMenu()->AddRadioButton ("In-plane");
  this->ImagingMenu->SetValue("Perpendicular");
  
  this->Script( "pack %s %s -side left -anchor w -padx 2 -pady 2", 
                this->ImagingControlCheckButton->GetWidgetName(),
                this->ImagingMenu->GetWidgetName());
  
  displayFrame->Delete();
  driverFrame->Delete();
  modeFrame->Delete();
  sliceFrame->Delete();
  visCtrlFrame->Delete();
}


//----------------------------------------------------------------------------
int vtkProstateNavGUI::ChangeWorkPhase(int phase, int fChangeWizard)
{

  cerr << "ChangeWorkPhase: started" << endl;
    if (!this->Logic->SwitchWorkPhase(phase)) // Set next phase
    {
      cerr << "ChangeWorkPhase: Cannot make transition!" << endl;
        return 0;
    }
  
    for (int i = 0; i < vtkProstateNavLogic::NumPhases; i ++)
    {
        vtkKWPushButton *pb = this->WorkPhaseButtonSet->GetWidget(i);
        if (i == this->Logic->GetCurrentPhase())
        {
            pb->SetReliefToSunken();
        }
        else if (this->Logic->IsPhaseTransitable(i))
        {
            pb->SetReliefToGroove();
            pb->SetStateToNormal();
        }
        else
        {
            pb->SetReliefToGroove();
            pb->SetStateToDisabled();
        }
    }

    // Switch Wizard Frame
    // 11/09/2007 Junichi Tokuda -- This part looks ugly. Will be fixed later.
    if (fChangeWizard)
    {
        vtkKWWizardWorkflow *wizard = 
          this->WizardWidget->GetWizardWorkflow();
        
        int step_from;
        int step_to;
        
        step_to = this->Logic->GetCurrentPhase();
        step_from = this->Logic->GetPrevPhase();
        if (step_to == vtkProstateNavLogic::Emergency)
        {
            step_to = vtkProstateNavLogic::Manual;
        }
        if (step_from == vtkProstateNavLogic::Emergency)
        {
            step_from = vtkProstateNavLogic::Manual;
        }
        
        int steps =  step_to - step_from;
        if (steps > 0)
        {
            for (int i = 0; i < steps; i ++) 
            {
                wizard->AttemptToGoToNextStep();
            }
        }
        else
        {
            steps = -steps;
            for (int i = 0; i < steps; i ++)
            {
                wizard->AttemptToGoToPreviousStep();
            }
        }
        wizard->GetCurrentStep()->ShowUserInterface();
    }

    return 1;
}



void vtkProstateNavGUI::UpdateAll()
{
    std::string received_scanner_status;
    std::string received_error_status;
    std::string received_robot_status;

    //Philip Mewes 17.07.2007: defining and sending te workphase (WP) commands depending of requestet WP

    // received_robot_status = NULL;
    
    if(received_robot_status == BRPTPR_Initializing && RequestedWorkphase==1)
    {
        RobotStatusLabelDisp->SetValue ( "Initializing" );
    }

    if(received_robot_status == BRPTPR_Uncalibrated && RequestedWorkphase==1)
    {
        RobotStatusLabelDisp->SetValue ( "Uncalibrated" );
    }
    else if(RequestedWorkphase==2)
    {
        RobotStatusLabelDisp->SetValue ( "Planning" );
    }
    else if(received_robot_status == BRPTPR_Ready && RequestedWorkphase==3)
    {
        RobotStatusLabelDisp->SetValue ( "Ready (Calb.)" );
    }
    // Philip Mewes 17.07.2007: Receiving Status for the Beginning
    // of the Positioning (Moving) Process
    else if(received_robot_status == BRPTPR_Moving && RequestedWorkphase==4)
    {
        RobotStatusLabelDisp->SetValue ( "Moving" );
    }
    // Philip Mewes 17.07.2007: Receiving Status when Robot
    //is in the right position
    else if(received_robot_status == BRPTPR_Ready  && RequestedWorkphase==4)
    {
        RobotStatusLabelDisp->SetValue ( "Ready (Pos.)" );
    }
    // Philip Mewes 17.07.2007: After sending the Manual WP-Command
    //and the breaking of the robot axes
    else if(received_robot_status == BRPTPR_Manual && RequestedWorkphase==5)
    {
        RobotStatusLabelDisp->SetValue ( "MANUAL" );
    }
    else if(received_robot_status == BRPTPR_EStop && RequestedWorkphase==6)
      {
        RobotStatusLabelDisp->SetValue ( "911" );
    }
    else if(BRPTPR_Error)
    {
        RobotStatusLabelDisp->SetValue ( "Err: ");
    }
    
    else
    {
        RobotStatusLabelDisp->SetValue ( "" );
    }
    
}


void vtkProstateNavGUI::UpdateLocator(vtkTransform *transform, vtkTransform *transform_cb2)
{

    //vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->LocatorModelID_new.c_str())); 
    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID("vtkMRMLModelNode1")); 
    if (model != NULL)
    {
        if (transform)
        {
            vtkMRMLLinearTransformNode *lnode = (vtkMRMLLinearTransformNode *)model->GetParentTransformNode();
            lnode->SetAndObserveMatrixTransformToParent(transform->GetMatrix());
            this->GetMRMLScene()->Modified();
        }
        if (transform_cb2)
        {
            vtkMRMLLinearTransformNode *lnode = (vtkMRMLLinearTransformNode *)model->GetParentTransformNode();
            lnode->SetAndObserveMatrixTransformToParent(transform_cb2->GetMatrix());
            this->GetMRMLScene()->Modified();
        }
    }

}


void vtkProstateNavGUI::UpdateSliceDisplay(float nx, float ny, float nz, 
                    float tx, float ty, float tz, 
                    float px, float py, float pz)
{

  //std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() is called." << std::endl;

    // Reslice -- Perpendicular
    if ( this->SliceDriver0 == vtkProstateNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver0 == vtkProstateNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode0->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 0);
        this->Logic0->UpdatePipeline ();
    }
    else if ( this->SliceDriver0 == vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->Logic->GetNeedRealtimeImageUpdate0())
        {
          //            std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode0->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 0);
            this->Logic0->UpdatePipeline ();
        }
    }


    // Reslice -- In-plane 90
    if ( this->SliceDriver1 == vtkProstateNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver1 == vtkProstateNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode1->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 1);
        this->Logic1->UpdatePipeline ();
    }
    else if ( this->SliceDriver1 == vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->Logic->GetNeedRealtimeImageUpdate1())
        {
          //            std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode1->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 1);
            this->Logic1->UpdatePipeline ();
        }
    }


    // Reslice -- In-plane
    if ( this->SliceDriver2 == vtkProstateNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver2 == vtkProstateNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode2->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 2);
        this->Logic2->UpdatePipeline ();
    }
    else if ( this->SliceDriver2 == vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->Logic->GetNeedRealtimeImageUpdate2())
        {
          //            std::cerr << "vtkProstateNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode2->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 2);
            this->Logic2->UpdatePipeline ();
        }
    }
}


void vtkProstateNavGUI::ChangeSlicePlaneDriver(int slice, const char* driver)
{
  std::cerr << "ChangeSlicePlaneDriver -- Slice: " << slice << ", Driver: " << driver << std::endl;
  
  if (slice == vtkProstateNavGUI::SLICE_PLANE_RED)
    {
    this->RedSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode0->SetOrientationToAxial();
      this->SliceDriver0 = vtkProstateNavGUI::SLICE_DRIVER_USER;
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      this->SliceDriver0 = vtkProstateNavGUI::SLICE_DRIVER_LOCATOR;
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      this->SliceDriver0 = vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE;
      }
    }
  else if (slice == vtkProstateNavGUI::SLICE_PLANE_YELLOW)
    {
    this->YellowSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode1->SetOrientationToSagittal();
      this->SliceDriver1 = vtkProstateNavGUI::SLICE_DRIVER_USER;
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      this->SliceDriver1 = vtkProstateNavGUI::SLICE_DRIVER_LOCATOR;
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      this->SliceDriver1 = vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE;
      }
    }
  else //if ( slice == vtkProstateNavGUI::SLICE_PLANE_GREEN )
    {
    this->GreenSliceMenu->SetValue(driver);
    if (strcmp(driver, "User") == 0)
      {
      this->SliceNode2->SetOrientationToCoronal();
      this->SliceDriver2 = vtkProstateNavGUI::SLICE_DRIVER_USER;
      }
    else if (strcmp(driver, "Locator") == 0)
      {
      this->SliceDriver2 = vtkProstateNavGUI::SLICE_DRIVER_LOCATOR;
      }
    else if (strcmp(driver, "RT Image") == 0)
      {
      this->SliceDriver2 = vtkProstateNavGUI::SLICE_DRIVER_RTIMAGE;
      }
    }
}

