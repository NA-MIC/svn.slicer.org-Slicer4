



#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkBrpNavGUI.h"
#include "BRPTPRInterface.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleCollapsibleFrame.h"
#include "vtkSlicerSliceControllerWidget.h"
#include "vtkSlicerColor.h"
#include "vtkSlicerTheme.h"

#include "vtkKWRenderWidget.h"
#include "vtkKWWidget.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"
#include "vtkKWPushButton.h"
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

#if defined(OT_VERSION_20) || defined(OT_VERSION_13)

#include <OpenTracker/input/SlicerNTModule.h>
#include <OpenTracker/OpenTracker.h>
#include <OpenTracker/input/SPLModules.h>

#endif

#include "vtkKWTkUtilities.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkCylinderSource.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkCornerAnnotation.h"

//RI
#include "vtkImageChangeInformation.h"
#include "vtkSlicerColorLogic.h"
#include "vtkSlicerVolumesGUI.h"


#include "vtkIGTDataStream.h"

#include "vtkCylinderSource.h"
#include "vtkMRMLLinearTransformNode.h"

#include <vector>

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkBrpNavGUI );
vtkCxxRevisionMacro ( vtkBrpNavGUI, "$Revision: 1.0 $");
//---------------------------------------------------------------------------

vtkBrpNavGUI::vtkBrpNavGUI ( )
{
 

    this->Logic = NULL;

    this->NormalOffsetEntry = NULL; 
    this->TransOffsetEntry = NULL;
    this->NXTOffsetEntry = NULL;

    this->NormalSizeEntry = NULL;
    this->TransSizeEntry = NULL;
    this->RadiusEntry = NULL;

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

    this->ExtraFrame = NULL;

/*
    this->RedColorScale = NULL;
    this->GreenColorScale = NULL;
    this->BlueColorScale = NULL;
    */

 
   
    this->ConnectCheckButtonRI = NULL;
    this->NeedleCheckButton = NULL;
    this->ConnectCheckButtonNT = NULL;
    this->ConnectCheckButtonSEND = NULL;
    this->ConnectCheckButtonPASSROBOTCOORDS = NULL;
    this->ConnectCheckButtonStartScanner = NULL;
    this->ConnectCheckButtonStopScanner = NULL;
    this->ConnectCheckButtonsetprotocol = NULL;
    this->ConnectCheckButtonprepScanner = NULL;
    this->ConnectCheckButtonpauseScanner = NULL;
    this->ConnectCheckButtonresumeScanner = NULL;
    this->ConnectCheckButtonnewexam = NULL;

     this->LocatorCheckButton = NULL;
     this->FreezeImageCheckButton = NULL;

     this->WorkPhaseStartUpButton = NULL;
     this->WorkPhasePlanningButton = NULL;
     this->WorkPhaseCalibarationButton = NULL;
     this->WorkPhaseTargetingButton = NULL;
     this->WorkPhaseManualButton = NULL;
     this->WorkPhaseEmergencyButton = NULL;
      this->ClearWorkPhasecontrollButton = NULL;


    this->NeedleCheckButton = NULL;
    this->HandleCheckButton = NULL;
    this->GuideCheckButton = NULL;

    this->LocatorModeCheckButton = NULL;
    this->UserModeCheckButton = NULL;

    this->RedSliceMenu = NULL;
    this->YellowSliceMenu = NULL;
    this->GreenSliceMenu = NULL;

#ifdef USE_OPENTRACKER
  
     this->LoadConfigButtonNT = NULL;
     this->ConfigFileEntry = NULL;
     this->ScannerStatusLabelDisp = NULL;
     this->SoftwareStatusLabelDisp = NULL;
#endif
#ifdef USE_IGSTK
    this->DeviceMenuButton = NULL;
    this->PortNumberMenuButton = NULL;
    this->BaudRateMenuButton = NULL;
    this->DataBitsMenuButton = NULL;
    this->ParityTypeMenuButton = NULL;
    this->StopBitsMenuButton = NULL;
    this->HandShakeMenuButton = NULL;
#endif

   

    this->UpdateRateEntry = NULL;
    this->GetImageSize = NULL;
 
    this->AddCoordsandOrientTarget = NULL;
    this->SetOrientButton = NULL;
   

    this->PointPairMultiColumnList = NULL;
    this->TargetListColumnList = NULL;

    // this->LoadPointPairPushButton = NULL;
    // this->SavePointPairPushButton = NULL;
  
  
    this->DeleteTargetPushButton = NULL;
    this->DeleteAllTargetPushButton = NULL;
    this->MoveBWPushButton = NULL;
    this->MoveFWPushButton = NULL;
  
    this->LocatorMatrix = NULL;
    this->LocatorMatrix_cb2 = NULL;

    this->LocatorModelDisplayNode = NULL;

    this->DataManager = vtkIGTDataManager::New();
    this->Pat2ImgReg = vtkIGTPat2ImgRegistration::New();

    this->DataCallbackCommand = vtkCallbackCommand::New();
    this->DataCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
    this->DataCallbackCommand->SetCallback(vtkBrpNavGUI::DataCallback);

    this->Logic0 = NULL; 
    this->Logic1 = NULL; 
    this->Logic2 = NULL; 
    this->SliceNode0 = NULL; 
    this->SliceNode1 = NULL; 
    this->SliceNode2 = NULL; 
    this->Control0 = NULL; 
    this->Control1 = NULL; 
    this->Control2 = NULL; 

    this->VolumesLogic = NULL;
    this->RealtimeVolumeNode = NULL;

    this->NeedOrientationUpdate0 = 0;
    this->NeedOrientationUpdate1 = 0;
    this->NeedOrientationUpdate2 = 0;
    
    this->RealtimeXsize = 0;
    this->RealtimeYsize = 0;
    this->RealtimeImageSerial = 0;

    this->NeedRealtimeImageUpdate = 0;
    

#ifdef USE_OPENTRACKER
    this->OpenTrackerStream = vtkIGTOpenTrackerStream::New();
#endif
#ifdef USE_IGSTK
    this->IGSTKStream = vtkIGTIGSTKStream::New();
#endif



}

//---------------------------------------------------------------------------
vtkBrpNavGUI::~vtkBrpNavGUI ( )
{
#ifdef USE_OPENTRACKER
    if (this->OpenTrackerStream)
    {
         this->OpenTrackerStream->Delete();
    }
#endif
#ifdef USE_IGSTK
    if (this->IGSTKStream)
    {
    this->IGSTKStream->Delete();
    }
#endif


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

    if (this->NormalOffsetEntry)
    {
    this->NormalOffsetEntry->SetParent(NULL );
    this->NormalOffsetEntry->Delete ( );
    }
    if (this->TransOffsetEntry)
    {
    this->TransOffsetEntry->SetParent(NULL );
    this->TransOffsetEntry->Delete ( );
    }
    if (this->NXTOffsetEntry)
    {
    this->NXTOffsetEntry->SetParent(NULL );
    this->NXTOffsetEntry->Delete ( );
    }
    if (this->NormalSizeEntry)
    {
    this->NormalSizeEntry->SetParent(NULL );
    this->NormalSizeEntry->Delete ( );
    }
    if (this->TransSizeEntry)
    {
    this->TransSizeEntry->SetParent(NULL );
    this->TransSizeEntry->Delete ( );
    }
    if (this->RadiusEntry)
    {
    this->RadiusEntry->SetParent(NULL );
    this->RadiusEntry->Delete ( );
    }
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
    

  
    if (this->ConnectCheckButtonRI)
    {
    this->ConnectCheckButtonRI->SetParent(NULL );
    this->ConnectCheckButtonRI->Delete ( );
    }

    if (this->NeedleCheckButton)
    {
    this->NeedleCheckButton->SetParent(NULL );
    this->NeedleCheckButton->Delete ( );
    }
     if (this->ConnectCheckButtonNT)
    {
    this->ConnectCheckButtonNT->SetParent(NULL );
    this->ConnectCheckButtonNT->Delete ( );
    }

       if (this->ConnectCheckButtonSEND)
    {
    this->ConnectCheckButtonSEND->SetParent(NULL );
    this->ConnectCheckButtonSEND->Delete ( );
    }
       
      if (this->ConnectCheckButtonPASSROBOTCOORDS)
    {
    this->ConnectCheckButtonPASSROBOTCOORDS->SetParent(NULL );
    this->ConnectCheckButtonPASSROBOTCOORDS->Delete ( );
    }

 
    if (this->ConnectCheckButtonStopScanner)
    {
    this->ConnectCheckButtonStopScanner->SetParent(NULL );
    this->ConnectCheckButtonStopScanner->Delete ( );
    }

     if (this->ConnectCheckButtonStartScanner)
    {
    this->ConnectCheckButtonStartScanner->SetParent(NULL );
    this->ConnectCheckButtonStartScanner->Delete ( );
    }
      
      if (this->ConnectCheckButtonsetprotocol)
    {
    this->ConnectCheckButtonsetprotocol->SetParent(NULL );
    this->ConnectCheckButtonsetprotocol->Delete ( );
    }
       if (this->ConnectCheckButtonprepScanner)
    {
    this->ConnectCheckButtonprepScanner->SetParent(NULL );
    this->ConnectCheckButtonprepScanner->Delete ( );
    }
        if (this->ConnectCheckButtonpauseScanner)
    {
    this->ConnectCheckButtonpauseScanner->SetParent(NULL );
    this->ConnectCheckButtonpauseScanner->Delete ( );
    }
         if (this->ConnectCheckButtonresumeScanner)
    {
    this->ConnectCheckButtonresumeScanner->SetParent(NULL );
    this->ConnectCheckButtonresumeScanner->Delete ( );
    }

          if (this->ConnectCheckButtonnewexam)
    {
    this->ConnectCheckButtonnewexam->SetParent(NULL );
    this->ConnectCheckButtonnewexam->Delete ( );
    }

    if (this->LocatorCheckButton)
    {
    this->LocatorCheckButton->SetParent(NULL );
    this->LocatorCheckButton->Delete ( );
    }

    if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->SetParent(NULL );
    this->FreezeImageCheckButton->Delete ( );
    }


    if (this->WorkPhaseStartUpButton)
    {
    this->WorkPhaseStartUpButton->SetParent(NULL );
    this->WorkPhaseStartUpButton->Delete ( );
    }

if (this->WorkPhasePlanningButton)
    {
    this->WorkPhasePlanningButton->SetParent(NULL );
    this->WorkPhasePlanningButton->Delete ( );
    }
if (this->WorkPhaseCalibarationButton)
    {
    this->WorkPhaseCalibarationButton->SetParent(NULL );
    this->WorkPhaseCalibarationButton->Delete ( );
    }
if (this->WorkPhaseTargetingButton)
    {
    this->WorkPhaseTargetingButton->SetParent(NULL );
    this->WorkPhaseTargetingButton->Delete ( );
    }
if (this->WorkPhaseManualButton)
    {
    this->WorkPhaseManualButton->SetParent(NULL );
    this->WorkPhaseManualButton->Delete ( );
    }
if (this->WorkPhaseEmergencyButton)
    {
    this->WorkPhaseEmergencyButton->SetParent(NULL );
    this->WorkPhaseEmergencyButton->Delete ( );
    }

 if (this->ClearWorkPhasecontrollButton)
    {
    this->ClearWorkPhasecontrollButton->SetParent(NULL );
    this->ClearWorkPhasecontrollButton->Delete ( );
    }



    if (this->NeedleCheckButton)
    {
    this->NeedleCheckButton->SetParent(NULL );
    this->NeedleCheckButton->Delete ( );
    }
    if (this->HandleCheckButton)
    {
    this->HandleCheckButton->SetParent(NULL );
    this->HandleCheckButton->Delete ( );
    }
    if (this->GuideCheckButton)
    {
    this->GuideCheckButton->SetParent(NULL );
    this->GuideCheckButton->Delete ( );
    }

    if (this->LocatorModeCheckButton)
    {
    this->LocatorModeCheckButton->SetParent(NULL );
    this->LocatorModeCheckButton->Delete ( );
    }
    if (this->UserModeCheckButton)
    {
    this->UserModeCheckButton->SetParent(NULL );
    this->UserModeCheckButton->Delete ( );
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

#ifdef USE_OPENTRACKER
   
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
#ifdef USE_IGSTK
    if (this->DeviceMenuButton) 
    {
    this->DeviceMenuButton->SetParent(NULL);
    this->DeviceMenuButton->Delete();
    }
    if (this->PortNumberMenuButton) 
    {
    this->PortNumberMenuButton->SetParent(NULL);
    this->PortNumberMenuButton->Delete();
    }
    if (this->BaudRateMenuButton) 
    {
    this->BaudRateMenuButton->SetParent(NULL);
    this->BaudRateMenuButton->Delete();
    }
    if (this->DataBitsMenuButton) 
    {
    this->DataBitsMenuButton->SetParent(NULL);
    this->DataBitsMenuButton->Delete();
    }
    if (this->ParityTypeMenuButton) 
    {
    this->ParityTypeMenuButton->SetParent(NULL);
    this->ParityTypeMenuButton->Delete();
    }
    if (this->StopBitsMenuButton) 
    {
    this->StopBitsMenuButton->SetParent(NULL);
    this->StopBitsMenuButton->Delete();
    }
    if (this->HandShakeMenuButton) 
    {
    this->HandShakeMenuButton->SetParent(NULL);
    this->HandShakeMenuButton->Delete();
    }

#endif


    if (this->UpdateRateEntry)
    {
    this->UpdateRateEntry->SetParent(NULL );
    this->UpdateRateEntry->Delete ( );
    }

    if (this->GetImageSize)
    {
    this->GetImageSize->SetParent(NULL );
    this->GetImageSize->Delete ( );
    }
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

    /*
    if (this->LoadPointPairPushButton)
    {
    this->LoadPointPairPushButton->SetParent(NULL );
    this->LoadPointPairPushButton->Delete ( );
    }
    if (this->SavePointPairPushButton)
    {
    this->SavePointPairPushButton->SetParent(NULL );
    this->SavePointPairPushButton->Delete ( );
    }
    */
  
   
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

    this->SetModuleLogic ( NULL );


    if (this->ExtraFrame)
    {
    this->ExtraFrame->Delete ( );
    }
}



//---------------------------------------------------------------------------
void vtkBrpNavGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );
    



    os << indent << "BrpNavGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "Logic: " << this->GetLogic ( ) << "\n";
   
    // print widgets?
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::RemoveGUIObservers ( )
{
    vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();

    appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);
    appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->RemoveObserver((vtkCommand *)this->GUICallbackCommand);


#ifdef USE_OPENTRACKER
    this->OpenTrackerStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
   
    this->LoadConfigButtonNT->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
 #endif
#ifdef USE_IGSTK
    this->IGSTKStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
    this->DeviceMenuButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
#endif


    
  
    if (this->ConnectCheckButtonRI)
    {
    this->ConnectCheckButtonRI->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->NeedleCheckButton)
    {
    this->NeedleCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
     if (this->ConnectCheckButtonnewexam)
    {
    this->ConnectCheckButtonnewexam->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,     (vtkCommand *)this->GUICallbackCommand );
    }
     
    if (this->ConnectCheckButtonNT)
    {
    this->ConnectCheckButtonNT->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

     if (this->ConnectCheckButtonSEND)
    {
    this->ConnectCheckButtonSEND->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
      if (this->ConnectCheckButtonPASSROBOTCOORDS)
    {
    this->ConnectCheckButtonPASSROBOTCOORDS->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }  

      if (this->ConnectCheckButtonStartScanner)
    {
    this->ConnectCheckButtonStartScanner->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
     
      if (this->ConnectCheckButtonStopScanner)
    {
    this->ConnectCheckButtonStopScanner->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

       if (this->ConnectCheckButtonsetprotocol)
    {
    this->ConnectCheckButtonsetprotocol->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->ConnectCheckButtonprepScanner)
    {
    this->ConnectCheckButtonprepScanner->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

     if (this->ConnectCheckButtonpauseScanner)
    {
    this->ConnectCheckButtonpauseScanner->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

      if (this->ConnectCheckButtonresumeScanner)
    {
    this->ConnectCheckButtonresumeScanner->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
   
    if (this->SetOrientButton)
    {
    this->SetOrientButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
    
    if (this->AddCoordsandOrientTarget)
    {
    this->AddCoordsandOrientTarget->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
 
    if (this->DeleteTargetPushButton)
    {
    this->DeleteTargetPushButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->MoveFWPushButton)
    {
    this->MoveFWPushButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->MoveBWPushButton)
    {
    this->MoveBWPushButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->DeleteAllTargetPushButton)
    {
    this->DeleteAllTargetPushButton->RemoveObservers ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
  
    if (this->LocatorCheckButton)
    {
    this->LocatorCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->FreezeImageCheckButton)
    {
    this->FreezeImageCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->WorkPhaseStartUpButton)
    {
    this->WorkPhaseStartUpButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->WorkPhasePlanningButton)
    {
    this->WorkPhasePlanningButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->WorkPhaseCalibarationButton)
    {
    this->WorkPhaseCalibarationButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->WorkPhaseTargetingButton)
    {
    this->WorkPhaseTargetingButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->WorkPhaseManualButton)
    {
    this->WorkPhaseManualButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

    if (this->WorkPhaseEmergencyButton)
    {
    this->WorkPhaseEmergencyButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
        if (this->ClearWorkPhasecontrollButton)
     {
    this->ClearWorkPhasecontrollButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
     }
    
    if (this->NeedleCheckButton)
    {
    this->NeedleCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->LocatorModeCheckButton)
    {
    this->LocatorModeCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->UserModeCheckButton)
    {
    this->UserModeCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::AddGUIObservers ( )
{
    this->RemoveGUIObservers();

    // make a user interactor style to process our events
    // look at the InteractorStyle to get our events

    vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();

    appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
    appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);
    appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle()->AddObserver(vtkCommand::LeftButtonPressEvent, (vtkCommand *)this->GUICallbackCommand);


    // Fill in
    // observer load volume button
   
 


  
    this->ConnectCheckButtonRI->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NeedleCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NeedleCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    
this->ConnectCheckButtonNT->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConnectCheckButtonSEND->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConnectCheckButtonPASSROBOTCOORDS->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
   this->ConnectCheckButtonStartScanner->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConnectCheckButtonStopScanner->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

   this->ConnectCheckButtonsetprotocol->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

   this->ConnectCheckButtonprepScanner->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

   this->ConnectCheckButtonpauseScanner->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

   this->ConnectCheckButtonresumeScanner->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

   this->ConnectCheckButtonnewexam->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->SetOrientButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    
this->AddCoordsandOrientTarget->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    
   
 
    this->DeleteTargetPushButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->DeleteAllTargetPushButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    
    

    this->MoveBWPushButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->MoveFWPushButton->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    this->LocatorCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->FreezeImageCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhaseStartUpButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhasePlanningButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhaseCalibarationButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhaseTargetingButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhaseManualButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->WorkPhaseEmergencyButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
     this->ClearWorkPhasecontrollButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    

   
    this->NeedleCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->LocatorModeCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->UserModeCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );


#ifdef USE_OPENTRACKER
    this->OpenTrackerStream->AddObserver( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
   
     this->LoadConfigButtonNT->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
   #endif
#ifdef USE_IGSTK
    this->IGSTKStream->AddObserver( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
    this->DeviceMenuButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
#endif

}



void vtkBrpNavGUI::HandleMouseEvent(vtkSlicerInteractorStyle *style)
{
    vtkSlicerApplicationGUI *appGUI = this->GetApplicationGUI();
    vtkSlicerInteractorStyle *istyle0 = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI0()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
    vtkSlicerInteractorStyle *istyle1 = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI1()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());
    vtkSlicerInteractorStyle *istyle2 = vtkSlicerInteractorStyle::SafeDownCast(appGUI->GetMainSliceGUI2()->GetSliceViewer()->GetRenderWidget()->GetRenderWindowInteractor()->GetInteractorStyle());


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

      //      this->SlicerCoordinatesEntry->GetWidget()->SetValue(ras.c_str());
      }
    }
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::ProcessGUIEvents ( vtkObject *caller,
    unsigned long event, void *callData )
{
    const char *eventName = vtkCommand::GetStringFromEventId(event);
    if (strcmp(eventName, "LeftButtonPressEvent") == 0)
    {
    vtkSlicerInteractorStyle *style = vtkSlicerInteractorStyle::SafeDownCast(caller);
    HandleMouseEvent(style);
    }
    else
    {
     
    if (this->ConnectCheckButtonRI == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
#ifdef USE_OPENTRACKER
        UpdateRealtimeImg();
#endif
#ifdef USE_IGSTK
        SetIGSTKConnectionParameters();
#endif
    }
    
     if (this->ConnectCheckButtonNT == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
      
#ifdef USE_OPENTRACKER
        SetOpenTrackerConnectionParameters();
#endif
      
#ifdef USE_IGSTK
        SetIGSTKConnectionParameters();
#endif
      
    }

     
     // GET RI 

     if (this->ConnectCheckButtonRI == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {      
      
      /*
      this->OpenTrackerStream->GetSizeforRealtimeImaging(&xsizevalueRI, &ysizevalueRI);
      this->OpenTrackerStream->GetImageDataforRealtimeImaging(&ImageDataRI);
      */
     }


if (this->ConnectCheckButtonSEND == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent  )
    {
      
#ifdef USE_OPENTRACKER

       SetOpenTrackerConnectionCoordandOrient();
#endif
      
#ifdef USE_IGSTK
        SetIGSTKConnectionParameters();
#endif
      
    }

 if ((this->ConnectCheckButtonStartScanner == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     
     ||(
     this->ConnectCheckButtonStopScanner == vtkKWCheckButton::SafeDownCast(caller) 
     && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
   this->ConnectCheckButtonsetprotocol == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
      this->ConnectCheckButtonprepScanner == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->ConnectCheckButtonpauseScanner == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->ConnectCheckButtonresumeScanner == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->ConnectCheckButtonnewexam == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     
     )
    {
      
#ifdef USE_OPENTRACKER

       SetOpenTrackerforScannerControll();
#endif
      
#ifdef USE_IGSTK
        SetIGSTKConnectionParameters();
#endif
      
   }
 
 if (this->ClearWorkPhasecontrollButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
    {
     
      StateTransitionDiagramControll();
      this->SoftwareStatusLabelDisp->SetValue("CLEARED");
    }

 
 if ((this->WorkPhaseStartUpButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->WorkPhasePlanningButton == vtkKWCheckButton::SafeDownCast(caller) 
     && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
   this->WorkPhaseCalibarationButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
      this->WorkPhaseTargetingButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->WorkPhaseManualButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent)
     ||(
     this->WorkPhaseEmergencyButton == vtkKWCheckButton::SafeDownCast(caller) 
      && event == vtkKWCheckButton::SelectedStateChangedEvent))

    {
      
#ifdef USE_OPENTRACKER
     
      
   
      this->WorkPhaseStartUpButton->SetStateToNormal();
      this->WorkPhasePlanningButton->SetStateToNormal();
      this->WorkPhaseCalibarationButton->SetStateToNormal();
      this->WorkPhaseTargetingButton->SetStateToNormal();
      this->WorkPhaseManualButton->SetStateToNormal();
      this->WorkPhaseEmergencyButton->SetStateToNormal();
         
      WorkFlowProcessStart = 1;

      int checkedPhase1 = this->WorkPhaseStartUpButton->GetSelectedState();
      int checkedPhase2 = this->WorkPhasePlanningButton->GetSelectedState();
      int checkedPhase3 = this->WorkPhaseCalibarationButton->GetSelectedState();
      int checkedPhase4 = this->WorkPhaseTargetingButton->GetSelectedState();
      int checkedPhase5 = this->WorkPhaseManualButton->GetSelectedState();
      int checkedPhase6 = this->WorkPhaseEmergencyButton->GetSelectedState();
      
      cout<<"Button pressed"<<endl;
  
      if(checkedPhase1)
      RequestedWorkphase=1; 
      else if(checkedPhase2) 
      RequestedWorkphase=2;
      else if(checkedPhase3)
      RequestedWorkphase=3;
      else if(checkedPhase4)
      RequestedWorkphase=4;
      else if(checkedPhase5)
       RequestedWorkphase=5;
      else if(checkedPhase6)
      RequestedWorkphase=6;

      if (checkedPhase1 || checkedPhase2 || checkedPhase3 || checkedPhase4 || checkedPhase5 || checkedPhase6)
        {
        StateTransitionDiagramControll();
        }
      else
        {
        this->SoftwareStatusLabelDisp->SetValue("");
        WorkFlowProcessStart = 0;
        RequestedWorkphase=0;
        }
#endif
            
        }


#ifdef USE_OPENTRACKER
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
        this->LoadConfigButtonNT->GetWidget()->SetText ("Browse Config File");
    }
#endif

     else if (this->AddCoordsandOrientTarget == vtkKWPushButton::SafeDownCast(caller) 
                && event == vtkKWPushButton::InvokedEvent)
       
    {
         int row = this->TargetListColumnList->GetWidget()->GetNumberOfRows();
         int rownumber = row + 1; 
         
         

     
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
               
        this->OpenTrackerStream->SetOrientationforRobot(xsendrobotcoords[sendindex],ysendrobotcoords[sendindex],zsendrobotcoords[sendindex], sendrobotcoordsvector[sendindex], robotcommandvalue,robotcommandkey);
      
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

 /*   
#ifdef USE_OPENTRACKER
        this->OpenTrackerStream->SetRegMatrix(this->Pat2ImgReg->GetLandmarkTransformMatrix());
#endif
#ifdef USE_IGSTK
        this->IGSTKStream->SetRegMatrix(this->Pat2ImgReg->GetLandmarkTransformMatrix());
#endif
        }
    }
 */

    else if (this->LocatorCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
        int checked = this->LocatorCheckButton->GetSelectedState(); 

        vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->LocatorModelID_new.c_str())); 
        if (model != NULL)
        {
        vtkMRMLModelDisplayNode *disp = model->GetDisplayNode();

        vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
        vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
        disp->SetColor(color->SliceGUIGreen);
        disp->SetVisibility(checked);
        }

    }
 /*
 else if (this->NeedleCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
    
        int checked = this->NeedleCheckButton->GetSelectedState(); 

        vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->LocatorModelID.c_str())); 
        if (model != NULL)
        {
        vtkMRMLModelDisplayNode *disp = model->GetDisplayNode();

        vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
        vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
        disp->SetColor(color->SliceGUIGreen);
        disp->SetVisibility(checked);
        }

    }
 */
    else if (this->LocatorModeCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
        int checked = this->LocatorModeCheckButton->GetSelectedState(); 
        std::string val("Locator");

        if (checked)
        {
        this->UserModeCheckButton->SelectedStateOff();
        }
        else
        {
        this->UserModeCheckButton->SelectedStateOn();
        this->SliceNode0->SetOrientationToAxial();
        this->SliceNode1->SetOrientationToSagittal();
        this->SliceNode2->SetOrientationToCoronal();
        this->NeedOrientationUpdate0 = 0;
        this->NeedOrientationUpdate1 = 0;
        this->NeedOrientationUpdate2 = 0;

        this->NeedRealtimeImageUpdate = 0;

        val = "User";
        }
        this->RedSliceMenu->SetValue(val.c_str());
        this->YellowSliceMenu->SetValue(val.c_str());
        this->GreenSliceMenu->SetValue(val.c_str());
    }
    else if (this->UserModeCheckButton == vtkKWCheckButton::SafeDownCast(caller) 
        && event == vtkKWCheckButton::SelectedStateChangedEvent )
    {
        int checked = this->UserModeCheckButton->GetSelectedState(); 
        std::string val("User");

        if (checked)
        {
        this->LocatorModeCheckButton->SelectedStateOff();
        this->SliceNode0->SetOrientationToAxial();
        this->SliceNode1->SetOrientationToSagittal();
        this->SliceNode2->SetOrientationToCoronal();

    
        this->NeedOrientationUpdate0 = 0;
        this->NeedOrientationUpdate1 = 0;
        this->NeedOrientationUpdate2 = 0;

        this->NeedRealtimeImageUpdate = 0;
        }
        else
        {
        this->LocatorModeCheckButton->SelectedStateOn();
        val = "Locator";
        }
        this->RedSliceMenu->SetValue(val.c_str());
        this->YellowSliceMenu->SetValue(val.c_str());
        this->GreenSliceMenu->SetValue(val.c_str());
      }
    }
} 

void vtkBrpNavGUI::Init()
{
    this->DataManager->SetMRMLScene(this->GetMRMLScene());
    //   this->LocatorModelID = std::string(this->DataManager->RegisterStream(0));
    this->LocatorModelID_new = std::string(this->DataManager->RegisterStream_new(0));
    
}



void vtkBrpNavGUI::DataCallback(vtkObject *caller, 
        unsigned long eid, void *clientData, void *callData)
{
    vtkBrpNavGUI *self = reinterpret_cast<vtkBrpNavGUI *>(clientData);
    vtkDebugWithObjectMacro(self, "In vtkBrpNavGUI DataCallback");

    self->UpdateAll();
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::ProcessLogicEvents ( vtkObject *caller,
    unsigned long event, void *callData )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::ProcessMRMLEvents ( vtkObject *caller,
    unsigned long event, void *callData )
{
    // Fill in
}



//---------------------------------------------------------------------------
void vtkBrpNavGUI::Enter ( )
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

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkSlicerVolumesGUI *volGui = (vtkSlicerVolumesGUI*)app->GetModuleGUIByName("Volumes");
    this->VolumesLogic = (vtkSlicerVolumesLogic*)(volGui->GetLogic());
      
  if (this->RealtimeVolumeNode == NULL)
    this->RealtimeVolumeNode = AddVolumeNode(this->VolumesLogic, "Realtime");
    
    //Set to 1, philip 21/06/2007
    this->Logic0->GetForegroundLayer()->SetUseReslice(0);



}

//---------------------------------------------------------------------------
void vtkBrpNavGUI::Exit ( )
{
    // Fill in
}


//---------------------------------------------------------------------------
void vtkBrpNavGUI::BuildGUI ( )
{

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    // Define your help text here.
    const char *help = "The **BrpNav Module** helps you to do prostate Biopsy and Treatment by: getting Realtime Images from MR-Scanner into Slicer3, control Scanner with Slicer 3, determinate fiducial detection and control the Robot. Module and Logic mainly coded by Junichi Tokuda and Philip Mewes"; 

    // ---
    // MODULE GUI FRAME 
    // create a page
    this->UIPanel->AddPage ( "BrpNav", "BrpNav", NULL );

    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );


    // ----------------------------------------------------------------
    // HELP FRAME         
    // ----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *BrpNavHelpFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    BrpNavHelpFrame->SetParent ( page );
    BrpNavHelpFrame->Create ( );
    BrpNavHelpFrame->CollapseFrame ( );
    BrpNavHelpFrame->SetLabelText ("Help");
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        BrpNavHelpFrame->GetWidgetName(), page->GetWidgetName());

    // configure the parent classes help text widget
    this->HelpText->SetParent ( BrpNavHelpFrame->GetFrame() );
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

    BrpNavHelpFrame->Delete();
 
    BuildGUIForWorkPhaseFrame ();
    BuildGUIForDeviceFrame ();
     BuildGUIForTrackingFrame ();
    BuildGUIForscancontrollFrame ();
     BuildGUIForRealtimeacqFrame ();


    //    BuildGUIForHandPieceFrame ();
}




void vtkBrpNavGUI::BuildGUIForWorkPhaseFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );

    //----------------------------------------------------------------
    // WORKPHASE FRAME         
    //----------------------------------------------------------------
    vtkSlicerModuleCollapsibleFrame *workphaseFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    workphaseFrame->SetParent ( page );
    workphaseFrame->Create ( );
    workphaseFrame->SetLabelText ("Workphase Frame");
    workphaseFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
    workphaseFrame->GetWidgetName(), page->GetWidgetName());


    vtkKWFrameWithLabel *filterFrame = vtkKWFrameWithLabel::New ( );
    filterFrame->SetParent ( workphaseFrame->GetFrame() );
    filterFrame->Create ( );
    filterFrame->SetLabelText ("Connection to server and Needle-Display");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           filterFrame->GetWidgetName() );

    vtkKWFrame *filter2Frame = vtkKWFrame::New ( );
    filter2Frame->SetParent ( filterFrame->GetFrame() );
    filter2Frame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   filter2Frame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());

    vtkKWFrame *filter3Frame = vtkKWFrame::New ( );
    filter3Frame->SetParent ( filterFrame->GetFrame() );
    filter3Frame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   filter3Frame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());


      vtkKWFrame *workphasestatusFrame = vtkKWFrame::New ( );
    workphasestatusFrame->SetParent ( filterFrame->GetFrame() );
    workphasestatusFrame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   workphasestatusFrame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());

      vtkKWFrame *workphasestatus2Frame = vtkKWFrame::New ( );
    workphasestatus2Frame->SetParent ( filterFrame->GetFrame() );
    workphasestatus2Frame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   workphasestatus2Frame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());

      vtkKWFrame *workphasestatus3Frame = vtkKWFrame::New ( );
    workphasestatus3Frame->SetParent ( filterFrame->GetFrame() );
    workphasestatus3Frame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   workphasestatus3Frame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());
      
      vtkKWFrame *ClearWorkphaseButtonFrame = vtkKWFrame::New ( );
    ClearWorkphaseButtonFrame->SetParent ( filterFrame->GetFrame() );
    ClearWorkphaseButtonFrame->Create ( );
      app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   ClearWorkphaseButtonFrame->GetWidgetName(),
                   filterFrame->GetFrame()->GetWidgetName());

      
      


    this->WorkPhaseStartUpButton = vtkKWCheckButton::New();
    this->WorkPhaseStartUpButton->SetParent(filter2Frame);
    this->WorkPhaseStartUpButton->Create();
    this->WorkPhaseStartUpButton->SelectedStateOff();
    this->WorkPhaseStartUpButton->SetText("Start Up Phase");
 
    this->WorkPhasePlanningButton = vtkKWCheckButton::New();
    this->WorkPhasePlanningButton->SetParent(filter2Frame);
    this->WorkPhasePlanningButton->Create();
    this->WorkPhasePlanningButton->SelectedStateOff();
    this->WorkPhasePlanningButton->SetText("Planning Phase");
    
    this->WorkPhaseCalibarationButton = vtkKWCheckButton::New();
    this->WorkPhaseCalibarationButton->SetParent(filter2Frame);
    this->WorkPhaseCalibarationButton->Create();
    this->WorkPhaseCalibarationButton->SelectedStateOff();
    this->WorkPhaseCalibarationButton->SetText("Calibration Phase");
    
    this->Script("pack %s %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
        this->WorkPhaseStartUpButton->GetWidgetName(),
        this->WorkPhasePlanningButton->GetWidgetName(),
        this->WorkPhaseCalibarationButton->GetWidgetName());

    this->WorkPhaseTargetingButton = vtkKWCheckButton::New();
    this->WorkPhaseTargetingButton->SetParent(filter3Frame);
    this->WorkPhaseTargetingButton->Create();
    this->WorkPhaseTargetingButton->SelectedStateOff();
    this->WorkPhaseTargetingButton->SetText("Targeting Phase");
 
    this->WorkPhaseManualButton = vtkKWCheckButton::New();
    this->WorkPhaseManualButton->SetParent(filter3Frame);
    this->WorkPhaseManualButton->Create();
    this->WorkPhaseManualButton->SelectedStateOff();
    this->WorkPhaseManualButton->SetText("Manual Controll Phase");
    
    this->WorkPhaseEmergencyButton = vtkKWCheckButton::New();
    this->WorkPhaseEmergencyButton->SetParent(filter3Frame);
    this->WorkPhaseEmergencyButton->Create();
    this->WorkPhaseEmergencyButton->SelectedStateOff();
    this->WorkPhaseEmergencyButton->SetText("EMERGENCY");
    
    this->Script("pack %s %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
        this->WorkPhaseTargetingButton->GetWidgetName(),
        this->WorkPhaseManualButton->GetWidgetName(),
        this->WorkPhaseEmergencyButton->GetWidgetName());



    vtkKWLabel *SoftwareStatusLabel = vtkKWLabel::New();
    SoftwareStatusLabel->SetParent(workphasestatusFrame);
    SoftwareStatusLabel->Create();
    SoftwareStatusLabel->SetWidth(15);
    SoftwareStatusLabel->SetText("Software Status:");


    this->SoftwareStatusLabelDisp = vtkKWEntry::New();
    this->SoftwareStatusLabelDisp->SetParent(workphasestatusFrame);
    this->SoftwareStatusLabelDisp->Create();
    this->SoftwareStatusLabelDisp->SetWidth(10);
    this->SoftwareStatusLabelDisp->SetValue ( "" );

    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 SoftwareStatusLabel->GetWidgetName(),
                 SoftwareStatusLabelDisp->GetWidgetName());


    vtkKWLabel *ScannerStatusLabel = vtkKWLabel::New();
    ScannerStatusLabel->SetParent(workphasestatus2Frame);
    ScannerStatusLabel->Create();
    ScannerStatusLabel->SetWidth(15);
    ScannerStatusLabel->SetText("Scanner Status:");


    
    this->ScannerStatusLabelDisp = vtkKWEntry::New();
    this->ScannerStatusLabelDisp->SetParent(workphasestatus2Frame);
    this->ScannerStatusLabelDisp->Create();
    this->ScannerStatusLabelDisp->SetWidth(10);
    this->ScannerStatusLabelDisp->SetValue ( "" );

    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 ScannerStatusLabel->GetWidgetName(),
                 ScannerStatusLabelDisp->GetWidgetName()
                 );

    vtkKWLabel *RobotStatusLabel = vtkKWLabel::New();
    RobotStatusLabel->SetParent(workphasestatus3Frame);
    RobotStatusLabel->Create();
    RobotStatusLabel->SetWidth(15);
    RobotStatusLabel->SetText("Robot Status:");


    
    this->RobotStatusLabelDisp = vtkKWEntry::New();
    this->RobotStatusLabelDisp->SetParent(workphasestatus3Frame);
    this->RobotStatusLabelDisp->Create();
    this->RobotStatusLabelDisp->SetWidth(10);
    this->RobotStatusLabelDisp->SetValue ( "" );

    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 RobotStatusLabel->GetWidgetName(),
                 RobotStatusLabelDisp->GetWidgetName()
                 );

    
    this->ClearWorkPhasecontrollButton = vtkKWCheckButton::New();
    this->ClearWorkPhasecontrollButton->SetParent(ClearWorkphaseButtonFrame);
    this->ClearWorkPhasecontrollButton->Create();
    this->ClearWorkPhasecontrollButton->SelectedStateOff();
    this->ClearWorkPhasecontrollButton->SetText("Clear WorkphaseControll");
    
    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
              this->ClearWorkPhasecontrollButton->GetWidgetName());
    
    
    

    
    workphaseFrame->Delete ();
    filterFrame->Delete ();
    
  
}


void vtkBrpNavGUI::BuildGUIForDeviceFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );

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
    
    this->setSpeedEntry = vtkKWEntryWithLabel::New();
    this->setSpeedEntry->SetParent(RobotSpeedFrame);
    this->setSpeedEntry->Create();
    this->setSpeedEntry->SetWidth(5);
    this->setSpeedEntry->SetLabelWidth(25);
    this->setSpeedEntry->SetLabelText("Set Speed in mm/s: ");
    this->setSpeedEntry->GetWidget()->SetValue ("0");
    
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->setSpeedEntry->GetWidgetName());


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


    


    /*
     this->NREntry->GetWidget()->SetWidth(5);
    this->NREntry->GetWidget()->SetValue("0");
    */

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



void vtkBrpNavGUI::BuildGUIForTrackingFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );


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

/*
    this->HandleCheckButton = vtkKWCheckButton::New();
    this->HandleCheckButton->SetParent(displayFrame->GetFrame());
    this->HandleCheckButton->Create();
    this->HandleCheckButton->SelectedStateOff();
    this->HandleCheckButton->SetText("Show Handle");

    this->GuideCheckButton = vtkKWCheckButton::New();
    this->GuideCheckButton->SetParent(displayFrame->GetFrame());
    this->GuideCheckButton->Create();
    this->GuideCheckButton->SelectedStateOff();
    this->GuideCheckButton->SetText("Show Guide");


    this->Script("pack %s %s %s -side left -anchor w -padx 2 -pady 2", 
        this->LocatorCheckButton->GetWidgetName(),
        this->HandleCheckButton->GetWidgetName(),
        this->GuideCheckButton->GetWidgetName());
*/


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



    // ----------------------------------------------------------------
    // Scanner Controll FRAME      START          
    // ----------------------------------------------------------------


void vtkBrpNavGUI::BuildGUIForscancontrollFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );

   
    vtkSlicerModuleCollapsibleFrame *scancontrollbrpFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    scancontrollbrpFrame->SetParent ( page );
    scancontrollbrpFrame->Create ( );
    scancontrollbrpFrame->SetLabelText ("Config-File + Scan Controll");

    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        scancontrollbrpFrame->GetWidgetName(), page->GetWidgetName());

    /////////////////////////////////////////////////////////////////////
    /// Main Controlle frame 
    /////////////////////////////////////////////////////////////////////
    
#ifdef USE_OPENTRACKER
    
    vtkKWFrameWithLabel *maincontroll = vtkKWFrameWithLabel::New();
    maincontroll->SetParent ( scancontrollbrpFrame->GetFrame() );
    maincontroll->Create ( );
    maincontroll->CollapseFrame ( );
    maincontroll->SetLabelText ("Main Controll Functions");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
          maincontroll->GetWidgetName());

       vtkKWFrame *maincontrollsetpatientNTFrame = vtkKWFrame::New();
    maincontrollsetpatientNTFrame->SetParent ( maincontroll->GetFrame() );
    maincontrollsetpatientNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          maincontrollsetpatientNTFrame->GetWidgetName());
  
    

     vtkKWFrame *maincontrollsetprotocolNTFrame = vtkKWFrame::New();
    maincontrollsetprotocolNTFrame->SetParent ( maincontroll->GetFrame() );
    maincontrollsetprotocolNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          maincontrollsetprotocolNTFrame->GetWidgetName());
    
     vtkKWFrame *maincontrollsetscannerNTFrame = vtkKWFrame::New();
    maincontrollsetscannerNTFrame->SetParent ( maincontroll->GetFrame() );
    maincontrollsetscannerNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          maincontrollsetscannerNTFrame->GetWidgetName());
    

    /////////////////////////NEW EXAM//////////////////////////////////////////
    

    
    this->positionbrppatientid = vtkKWEntryWithLabel::New();
    this->positionbrppatientid->SetParent(maincontrollsetpatientNTFrame);
    this->positionbrppatientid->Create();
    this->positionbrppatientid->SetWidth(5);
    this->positionbrppatientid->SetLabelWidth(25);
    this->positionbrppatientid->SetLabelText("Patient ID");
    this->positionbrppatientid->GetWidget()->SetValue ("");  


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrppatientid->GetWidgetName());
    
    this->positionbrppatientname = vtkKWEntryWithLabel::New();
    this->positionbrppatientname->SetParent(maincontrollsetpatientNTFrame);
    this->positionbrppatientname->Create();
    this->positionbrppatientname->SetWidth(5);
    this->positionbrppatientname->SetLabelWidth(25);
    this->positionbrppatientname->SetLabelText("Patient Name");
    this->positionbrppatientname->GetWidget()->SetValue ("");  


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrppatientname->GetWidgetName());

    this->positionbrppatientweight = vtkKWEntryWithLabel::New();
    this->positionbrppatientweight->SetParent(maincontrollsetpatientNTFrame);
    this->positionbrppatientweight->Create();
    this->positionbrppatientweight->SetWidth(5);
    this->positionbrppatientweight->SetLabelWidth(25);
    this->positionbrppatientweight->SetLabelText("Patient Weight (lbs.)");
    this->positionbrppatientweight->GetWidget()->SetValue ("");     


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrppatientweight->GetWidgetName());



    this->ConnectCheckButtonnewexam = vtkKWCheckButton::New();
    this->ConnectCheckButtonnewexam->SetParent(maincontrollsetpatientNTFrame);
    this->ConnectCheckButtonnewexam->Create();
    this->ConnectCheckButtonnewexam->SelectedStateOff();
    this->ConnectCheckButtonnewexam->SetText("New Exam         ----------------------------------------");

    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonnewexam->GetWidgetName());
    
    ////////////////////////////////SET PROTOCOL//////////////////////////////
    
    
    this->positionbrpsetprotocol = vtkKWEntryWithLabel::New();
    this->positionbrpsetprotocol->SetParent(maincontrollsetprotocolNTFrame);
    this->positionbrpsetprotocol->Create();
    this->positionbrpsetprotocol->SetWidth(5);
    this->positionbrpsetprotocol->SetLabelWidth(25);
    this->positionbrpsetprotocol->SetLabelText("Protocol Name");
    this->positionbrpsetprotocol->GetWidget()->SetValue ("");  


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrpsetprotocol->GetWidgetName());



    this->ConnectCheckButtonsetprotocol = vtkKWCheckButton::New();
    this->ConnectCheckButtonsetprotocol->SetParent(maincontrollsetprotocolNTFrame);
    this->ConnectCheckButtonsetprotocol->Create();
    this->ConnectCheckButtonsetprotocol->SelectedStateOff();
    this->ConnectCheckButtonsetprotocol->SetText("Set Protocol      ----------------------------------------");

    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonsetprotocol->GetWidgetName());
    
    



    /////////////////////////////////////main function Scanner/////////////////
    
    this->ConnectCheckButtonprepScanner = vtkKWCheckButton::New();
    this->ConnectCheckButtonprepScanner->SetParent(maincontrollsetscannerNTFrame);
    this->ConnectCheckButtonprepScanner->Create();
    this->ConnectCheckButtonprepScanner->SelectedStateOff();
    this->ConnectCheckButtonprepScanner->SetText("PreScan");

    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonprepScanner->GetWidgetName());
    


    this->ConnectCheckButtonStartScanner = vtkKWCheckButton::New();
    this->ConnectCheckButtonStartScanner->SetParent(maincontrollsetscannerNTFrame);
    this->ConnectCheckButtonStartScanner->Create();
    this->ConnectCheckButtonStartScanner->SelectedStateOff();
    this->ConnectCheckButtonStartScanner->SetText("Start Scanner");

    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonStartScanner->GetWidgetName());
    
    this->ConnectCheckButtonStopScanner = vtkKWCheckButton::New();
    this->ConnectCheckButtonStopScanner->SetParent(maincontrollsetscannerNTFrame);
    this->ConnectCheckButtonStopScanner->Create();
    this->ConnectCheckButtonStopScanner->SelectedStateOff();
    this->ConnectCheckButtonStopScanner->SetText("Stop Scanner");

    this->Script("pack %s -side right -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonStopScanner->GetWidgetName());
    
    
    this->ConnectCheckButtonpauseScanner = vtkKWCheckButton::New();
    this->ConnectCheckButtonpauseScanner->SetParent(maincontrollsetscannerNTFrame);
    this->ConnectCheckButtonpauseScanner->Create();
    this->ConnectCheckButtonpauseScanner->SelectedStateOff();
    this->ConnectCheckButtonpauseScanner->SetText("Pause Scanner");

    this->Script("pack %s -side right -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonpauseScanner->GetWidgetName());
    
    this->ConnectCheckButtonresumeScanner = vtkKWCheckButton::New();
    this->ConnectCheckButtonresumeScanner->SetParent(maincontrollsetscannerNTFrame);
    this->ConnectCheckButtonresumeScanner->Create();
    this->ConnectCheckButtonresumeScanner->SelectedStateOff();
    this->ConnectCheckButtonresumeScanner->SetText("Resume Scanner");

    this->Script("pack %s -side right -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonresumeScanner->GetWidgetName());
    


    /////////////////////////////////////////////////////////////////////
    /// Config file frame
    /////////////////////////////////////////////////////////////////////
   


    vtkKWFrameWithLabel *configcoordsorient = vtkKWFrameWithLabel::New();
    configcoordsorient->SetParent ( scancontrollbrpFrame->GetFrame() );
    configcoordsorient->Create ( );
    configcoordsorient->SetLabelText ("Config File + Connect");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
          configcoordsorient->GetWidgetName());
    
   
     vtkKWFrame *configNTFrame = vtkKWFrame::New();
    configNTFrame->SetParent ( configcoordsorient->GetFrame() );
    configNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          configNTFrame->GetWidgetName());
    
    vtkKWFrame *connectNTFrame = vtkKWFrame::New();
    connectNTFrame->SetParent ( configcoordsorient->GetFrame() );
    connectNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          connectNTFrame->GetWidgetName());

    
    this->ConfigFileEntry = vtkKWEntry::New();
    this->ConfigFileEntry->SetParent(configNTFrame);
    this->ConfigFileEntry->Create();
    this->ConfigFileEntry->SetWidth(50);
    this->ConfigFileEntry->SetValue ( "" );

    this->LoadConfigButtonNT = vtkKWLoadSaveButtonWithLabel::New ( );
    this->LoadConfigButtonNT->SetParent (configNTFrame);
    this->LoadConfigButtonNT->Create ( );
    this->LoadConfigButtonNT->SetWidth(15);
    this->LoadConfigButtonNT->GetWidget()->SetText ("Browse Config File");
    this->LoadConfigButtonNT->GetWidget()->GetLoadSaveDialog()->SetFileTypes(
                                  "{ {BrpNav} {*.xml} }");
    this->LoadConfigButtonNT->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry(
      "OpenPath");

    //const char *xmlpathfilename = this->LoadConfigButton->GetWidget()->GetFileName(); 
    //strncpy(xmlpathfilename, this->LoadConfigButton->GetWidget()->GetFileName(), 256); 

    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
        this->LoadConfigButtonNT->GetWidgetName(),
        this->ConfigFileEntry->GetWidgetName());


    /// Connnect button 
  
    this->ConnectCheckButtonNT = vtkKWCheckButton::New();
    this->ConnectCheckButtonNT->SetParent(connectNTFrame);
    this->ConnectCheckButtonNT->Create();
    this->ConnectCheckButtonNT->SelectedStateOff();
    this->ConnectCheckButtonNT->SetText("Connect");

    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
        this->ConnectCheckButtonNT->GetWidgetName());


     /////////////////////////////////////////////////////////////////////
    /// Send Orientation and Coords frame 
    /////////////////////////////////////////////////////////////////////
   
 
 vtkKWFrameWithLabel *coordsbrpFrame = vtkKWFrameWithLabel::New();
    coordsbrpFrame->SetParent ( scancontrollbrpFrame->GetFrame() );
    coordsbrpFrame->Create ( );
    coordsbrpFrame->CollapseFrame ( );
    coordsbrpFrame->SetLabelText ("Type and Send");
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
          coordsbrpFrame->GetWidgetName());
    
   
     vtkKWFrame *typecoordsorientNTFrame = vtkKWFrame::New();
    typecoordsorientNTFrame->SetParent ( coordsbrpFrame->GetFrame() );
    typecoordsorientNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          typecoordsorientNTFrame->GetWidgetName());
    
    vtkKWFrame *sendNTFrame = vtkKWFrame::New();
    sendNTFrame->SetParent ( coordsbrpFrame->GetFrame() );
    sendNTFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          sendNTFrame->GetWidgetName());




    this->positionbrpy = vtkKWEntryWithLabel::New();
    this->positionbrpy->SetParent(typecoordsorientNTFrame);
    this->positionbrpy->Create();
    this->positionbrpy->SetWidth(5);
    this->positionbrpy->SetLabelWidth(25);
    this->positionbrpy->SetLabelText("Y-Position: ");
    this->positionbrpy->GetWidget()->SetValue ("");  


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrpy->GetWidgetName());

this->positionbrpx = vtkKWEntryWithLabel::New();
    this->positionbrpx->SetParent(typecoordsorientNTFrame);
    this->positionbrpx->Create();
    this->positionbrpx->SetWidth(5);
    this->positionbrpx->SetLabelWidth(25);
    this->positionbrpx->SetLabelText("X-Position: ");
    this->positionbrpx->GetWidget()->SetValue ("");
   


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrpx->GetWidgetName());
 
this->positionbrpz = vtkKWEntryWithLabel::New();
    this->positionbrpz->SetParent(typecoordsorientNTFrame);
    this->positionbrpz->Create();
    this->positionbrpz->SetWidth(5);
    this->positionbrpz->SetLabelWidth(25);
    this->positionbrpz->SetLabelText("Z-Position: ");
    this->positionbrpz->GetWidget()->SetValue ("");
   


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->positionbrpz->GetWidgetName());


 //----------Orientation-----------------------------------------------------------------------------------------------------------------
    /*
  vtkKWFrame *orientationbrpFrame = vtkKWFrame::New();
    orientationbrpFrame->SetParent ( scancontrollbrpFrame->GetFrame() );
    orientationbrpFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          orientationbrpFrame->GetWidgetName());
    */
    this->orientationbrpo1 = vtkKWEntryWithLabel::New();
    this->orientationbrpo1->SetParent(typecoordsorientNTFrame);
    this->orientationbrpo1->Create();
    this->orientationbrpo1->SetWidth(5);
    this->orientationbrpo1->SetLabelWidth(25);
    this->orientationbrpo1->SetLabelText("O1-Orientation: ");
    this->orientationbrpo1->GetWidget()->SetValue ("");     


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->orientationbrpo1->GetWidgetName());


    this->orientationbrpo2 = vtkKWEntryWithLabel::New();
    this->orientationbrpo2->SetParent(typecoordsorientNTFrame);
    this->orientationbrpo2->Create();
    this->orientationbrpo2->SetWidth(5);
    this->orientationbrpo2->SetLabelWidth(25);
    this->orientationbrpo2->SetLabelText("O2-Orientation: ");
    this->orientationbrpo2->GetWidget()->SetValue ("");     


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->orientationbrpo2->GetWidgetName());


    this->orientationbrpo3 = vtkKWEntryWithLabel::New();
    this->orientationbrpo3->SetParent(typecoordsorientNTFrame);
    this->orientationbrpo3->Create();
    this->orientationbrpo3->SetWidth(5);
    this->orientationbrpo3->SetLabelWidth(25);
    this->orientationbrpo3->SetLabelText("O3-Orientation: ");
    this->orientationbrpo3->GetWidget()->SetValue ("");     


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->orientationbrpo3->GetWidgetName());

    this->orientationbrpo4 = vtkKWEntryWithLabel::New();
    this->orientationbrpo4->SetParent(typecoordsorientNTFrame);
    this->orientationbrpo4->Create();
    this->orientationbrpo4->SetWidth(5);
    this->orientationbrpo4->SetLabelWidth(25);
    this->orientationbrpo4->SetLabelText("O4-Orientation: ");
    this->orientationbrpo4->GetWidget()->SetValue ("");     


    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->orientationbrpo4->GetWidgetName());

    this->ConnectCheckButtonSEND = vtkKWCheckButton::New();
    this->ConnectCheckButtonSEND->SetParent(sendNTFrame);
    this->ConnectCheckButtonSEND->Create();
    this->ConnectCheckButtonSEND->SelectedStateOff();
    this->ConnectCheckButtonSEND->SetText("Send");
    
    this->ConnectCheckButtonPASSROBOTCOORDS = vtkKWCheckButton::New();
    this->ConnectCheckButtonPASSROBOTCOORDS->SetParent(sendNTFrame);
    this->ConnectCheckButtonPASSROBOTCOORDS->Create();
    this->ConnectCheckButtonPASSROBOTCOORDS->SelectedStateOff();
    this->ConnectCheckButtonPASSROBOTCOORDS->SetText("Switch (Manual / Robot)");



    this->Script("pack %s %s -side top -anchor w -padx 2 -pady 2", 
                 this->ConnectCheckButtonSEND->GetWidgetName(),
                 this->ConnectCheckButtonPASSROBOTCOORDS->GetWidgetName());
    
   



    //---------------------------------------------------------------------------------------------------------------------------
   
     maincontroll->Delete();
     connectNTFrame->Delete();
     configNTFrame->Delete();
     sendNTFrame->Delete();
     typecoordsorientNTFrame->Delete();
     configcoordsorient->Delete();
    coordsbrpFrame->Delete ();    
    scancontrollbrpFrame->Delete ();
    //    orientationbrpFrame->Delete ();
#endif
}

    // ----------------------------------------------------------------
    // Scanner Controll FRAME  END        
    // ----------------------------------------------------------------





void vtkBrpNavGUI::BuildGUIForRealtimeacqFrame ()
{
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );

    // ----------------------------------------------------------------
    // REALTIME FRAME         
    // ----------------------------------------------------------------
    
    
    vtkSlicerModuleCollapsibleFrame *realtimeacqFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    realtimeacqFrame->SetParent ( page );
    realtimeacqFrame->Create ( );
    realtimeacqFrame->SetLabelText ("Realtime Imaging");
    realtimeacqFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
        realtimeacqFrame->GetWidgetName(), page->GetWidgetName());

#ifdef USE_OPENTRACKER 
   //------------------------------------Content--------------*
    
    
    //------------------------------- SERVER FRAME-------------*        
    
    /*
    // Active server frame: Server options 
    // -----------------------------------------



    vtkKWFrame *activeServerFrame = vtkKWFrame::New();
    activeServerFrame->SetParent ( realtimeacqFrame->GetFrame() );
    activeServerFrame->Create ( );
    app->Script ("pack %s -side top -anchor nw -fill x -pady 1 -in %s",
           activeServerFrame->GetWidgetName(),
         realtimeacqFrame->GetFrame()->GetWidgetName());


      

    // active server 
    this->ServerMenu = vtkKWMenuButtonWithLabel::New();
    this->ServerMenu->SetParent(activeServerFrame);
    this->ServerMenu->Create();
    this->ServerMenu->SetWidth(25);
    this->ServerMenu->SetLabelWidth(12);
    this->ServerMenu->SetLabelText("Active Server:");
    // this->ServerMenu->GetWidget()->GetMenu()->AddRadioButton ( "None");
    this->ServerMenu->GetWidget()->GetMenu()->AddRadioButton ( "SPL Open Tracker");
    this->ServerMenu->GetWidget()->SetValue ( "SPL Open Tracker" );
    this->Script(
        "pack %s -side top -anchor nw -expand n -padx 2 -pady 2", 
        this->ServerMenu->GetWidgetName());
    */
     
    // Setup frame: Config file and update rate 
    // -----------------------------------------
    vtkKWFrameWithLabel *setupFrame = vtkKWFrameWithLabel::New ( );
    setupFrame->SetParent ( realtimeacqFrame->GetFrame() );
    setupFrame->Create ( );
    setupFrame->SetLabelText ("Setup");
    setupFrame->CollapseFrame ( );
    app->Script ("pack %s -side top -anchor nw -fill x -padx 2 -pady 1 -in %s",
         setupFrame->GetWidgetName(),
         realtimeacqFrame->GetFrame()->GetWidgetName());

    
    
    // add a file browser 
    vtkKWFrame *fileFrame = vtkKWFrame::New();
    fileFrame->SetParent ( realtimeacqFrame->GetFrame() );
   
    fileFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          fileFrame->GetWidgetName());

    /* 
    this->PAEntry = vtkKWEntry::New();
    PAEntry->SetParent(setupFrame);
    PAEntry->Create();
    PAEntry->SetWidth(10);
    PAEntry->SetValue("");

    */
       // update rate 
    vtkKWFrame *rateFrame = vtkKWFrame::New();
    rateFrame->SetParent ( setupFrame->GetFrame() );
    rateFrame->Create ( );
   
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
          rateFrame->GetWidgetName());

    this->UpdateRateEntry = vtkKWEntryWithLabel::New();
    this->UpdateRateEntry->SetParent(rateFrame);
    this->UpdateRateEntry->Create();
    this->UpdateRateEntry->SetWidth(25);
    this->UpdateRateEntry->SetLabelWidth(15);
    this->UpdateRateEntry->SetLabelText("Update Rate (ms):");
    this->UpdateRateEntry->GetWidget()->SetValue ( "200" );
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->UpdateRateEntry->GetWidgetName());
    

    this->GetImageSize = vtkKWEntryWithLabel::New();
    this->GetImageSize->SetParent(rateFrame);
    this->GetImageSize->Create();
    this->GetImageSize->SetWidth(25);
    this->GetImageSize->SetLabelWidth(15);
    this->GetImageSize->SetLabelText("Size");
    this->GetImageSize->GetWidget()->SetValue ("");
    this->Script(
      "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
      this->GetImageSize->GetWidgetName());
    

   
    // Connect frame: Connects to server 
    // -----------------------------------------
    vtkKWFrameWithLabel *connectFrame = vtkKWFrameWithLabel::New ( );
    connectFrame->SetParent ( realtimeacqFrame->GetFrame() );
    connectFrame->Create ( );
    connectFrame->SetLabelText ("Connection to server and Needle-Display");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           connectFrame->GetWidgetName() );
    
    this->ConnectCheckButtonRI = vtkKWCheckButton::New();
    this->ConnectCheckButtonRI->SetParent(connectFrame->GetFrame());
    this->ConnectCheckButtonRI->Create();
    this->ConnectCheckButtonRI->SelectedStateOff();
    this->ConnectCheckButtonRI->SetText("Connect");
 


    this->NeedleCheckButton = vtkKWCheckButton::New();
    this->NeedleCheckButton->SetParent(connectFrame->GetFrame());
    this->NeedleCheckButton->Create();
    this->NeedleCheckButton->SelectedStateOff();
    this->NeedleCheckButton->SetText("Show Needle");
    

    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->ConnectCheckButtonRI->GetWidgetName(),
        this->NeedleCheckButton->GetWidgetName());


    vtkKWFrameWithLabel *imageorientationFrame = vtkKWFrameWithLabel::New ( );
    imageorientationFrame->SetParent ( realtimeacqFrame->GetFrame() );
    imageorientationFrame->Create ( );
    imageorientationFrame->SetLabelText ("Image Orientation");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           imageorientationFrame->GetWidgetName() );
    /*   
    this->LocatorModeCheckButton = vtkKWCheckButton::New();
    this->LocatorModeCheckButton->SetParent(imageorientationFrame->GetFrame());
    this->LocatorModeCheckButton->Create();
    this->LocatorModeCheckButton->SelectedStateOff();
    this->LocatorModeCheckButton->SetText("Lock");
    

    this->Script("pack %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->LocatorModeCheckButton->GetWidgetName());
       
    */





    /* 
vtkKWFrameWithLabel *tempFrame = vtkKWFrameWithLabel::New ( );
    connectFrame->SetParent ( realtimeacqFrame->GetFrame() );
    connectFrame->Create ( );
    connectFrame->SetLabelText ("test");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
           connectFrame->GetWidgetName() );
    


    
    this->PAEntry = vtkKWEntry::New();
    PAEntry->SetParent(tempFrame);
    PAEntry->Create();
    PAEntry->SetWidth(13);
    PAEntry->SetValue("");
    */

    realtimeacqFrame->Delete ();
    //  activeServerFrame->Delete ();
     setupFrame->Delete ();
     fileFrame->Delete ();
     rateFrame->Delete ();
    connectFrame->Delete ();
    // tempFrame->Delete ();
   
#endif

}




void vtkBrpNavGUI::UpdateAll()
{
    this->LocatorMatrix = NULL;
  

#ifdef USE_OPENTRACKER
    this->LocatorMatrix = this->OpenTrackerStream->GetLocatorMatrix();
    //    this->OpenTrackerStream->GetSizeforRealtimeImaging(&xsizevalueRI, &ysizevalueRI);
       

#endif
        

    //Philip Mewes 17.07.2007: defining and sending te workphase (WP) commands depending of requestet WP
    //
        
    // received_robot_status = NULL;
    
    this->OpenTrackerStream->GetDevicesStatus(received_robot_status, received_scanner_status, received_error_status);
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
        else  if(received_robot_status == BRPTPR_EStop && RequestedWorkphase==6)
          {
            RobotStatusLabelDisp->SetValue ( "911" );
          }
        else  if(BRPTPR_Error)
          {
           
            RobotStatusLabelDisp->SetValue ( "Err: ");
          }
        
        else
           RobotStatusLabelDisp->SetValue ( "" );

        //timer for resend workphase request
        /*        
        if(!received_robot_status) //|| (var_status_robot < RequestedWorkphase))
          {
            int i=0;
               for (i=0; i<100; i++)
                 {
                   if(received_robot_status)
                   i = 100;
                 }
               //recall Requestfunction
               if(!received_robot_status)
                 {
                 ActualWorkPhase = ActualWorkPhase -1; 
                 StateTransitionDiagramControll();     
                 }
          }
        */

    
        
    int checkedpassrobotcoords = this->ConnectCheckButtonPASSROBOTCOORDS->GetSelectedState();
    if (checkedpassrobotcoords)
          {
        
              std::vector<float> pos;
              std::vector<float> quat;
       

                pos.resize(3);
                quat.resize(4);

 
             float OrientationForScanner0;
             float OrientationForScanner1;
             float OrientationForScanner2;
             float OrientationForScanner3;
             float PositionForScanner0;
             float PositionForScanner1;
             float PositionForScanner2;
  
             
            this->OpenTrackerStream->GetCoordsOrientforScanner(&OrientationForScanner0, &OrientationForScanner1, &OrientationForScanner2, &OrientationForScanner3, &PositionForScanner0, &PositionForScanner1, &PositionForScanner2);
  



           pos[0]= PositionForScanner0;
            pos[1]= PositionForScanner1;
            pos[2]= PositionForScanner2;
            quat[0]= OrientationForScanner0;
            quat[1]= OrientationForScanner1;
            quat[2]= OrientationForScanner2;
            quat[3]= OrientationForScanner3;

          
            this->OpenTrackerStream->SetTracker(pos,quat);
          }
      
    
  if (this->OpenTrackerStream)
    {
         vtkImageData* vid = NULL;
               if (this->RealtimeVolumeNode)
               vid = this->RealtimeVolumeNode->GetImageData();

        //  std::cerr << "vid = " << vid << std::endl;
                if (vid) {
                          //  std::cerr << "BrpNavGUI::UpdateAll(): update realtime image" << std::endl;
                          int orgSerial = this->RealtimeImageSerial;
                          this->OpenTrackerStream->GetRealtimeImage(&(this->RealtimeImageSerial), vid);
                              if (orgSerial != this->RealtimeImageSerial) {
                                          this->NeedRealtimeImageUpdate = 1;
                                           //this->RealtimeVolumeNode->UpdateScene(this->GetMRMLScene());
                                          //this->Logic0->UpdatePipeline ();
                                           this->RealtimeVolumeNode->SetAndObserveImageData(vid);
                                          
                                         }
                              } else {
          //std::cerr << "BrpNavGUI::UpdateAll(): no realtime image" << std::endl;
        }
 
}



  

#ifdef USE_IGSTK
    this->LocatorMatrix = this->IGSTKStream->GetLocatorMatrix();
#endif

   


    if (this->LocatorMatrix)
    {
    char Val[10];
    
    
    float px = this->LocatorMatrix->GetElement(0, 0);
    float py = this->LocatorMatrix->GetElement(1, 0);
    float pz = this->LocatorMatrix->GetElement(2, 0);
    float nx = this->LocatorMatrix->GetElement(0, 1);
    float ny = this->LocatorMatrix->GetElement(1, 1);
    float nz = this->LocatorMatrix->GetElement(2, 1);
    float tx = this->LocatorMatrix->GetElement(0, 2);
    float ty = this->LocatorMatrix->GetElement(1, 2);
    float tz = this->LocatorMatrix->GetElement(2, 2);
    

    /*
    
    float px = 0;
    float py = 0;
    float pz = 0;
    float nx = 0;
    float ny = 0;
    float nz = 0;
    float tx = 0.5;
    float ty = 0.5;
    float tz = 0.1;
    */

 /*
    sprintf(Val, "%6.2f", px);
    this->PREntry->SetValue(Val);
    sprintf(Val, "%6.2f", py);
    this->PAEntry->SetValue(Val);
    sprintf(Val, "%6.2f", pz);
    this->PSEntry->SetValue(Val);

    sprintf(Val, "%6.2f", nx);
    this->NREntry->SetValue(Val);
    sprintf(Val, "%6.2f", ny);
    this->NAEntry->SetValue(Val);
    sprintf(Val, "%6.2f", nz);
    this->NSEntry->SetValue(Val);

    sprintf(Val, "%6.2f", tx);
    this->TREntry->SetValue(Val);
    sprintf(Val, "%6.2f", ty);
    this->TAEntry->SetValue(Val);
    sprintf(Val, "UI6.2f", tz);
    this->TSEntry->SetValue(Val);
    */

    //Philip Mewes: For better debugging reasons and verification
    //in clinic workflow Needle tipp position, normal and transnormal vector
    //are going to be displayed here
    
    char coordsxyz[512];
    sprintf(coordsxyz, "%6.2f, %6.2f, %6.2f", px, py, pz);
    this->PositionEntry->GetWidget()->SetValue(coordsxyz);
    
    char orientxyz[512];
    sprintf(orientxyz, "(%6.2f, %6.2f, %6.2f) (%6.2f, %6.2f, %6.2f)", nx, ny, nz, tx, ty, tz);
    this->OrientEntry->GetWidget()->SetValue(orientxyz);
    

    // update the display of locator
    if (this->LocatorCheckButton->GetSelectedState()) this->UpdateLocator();
    if (this->NeedleCheckButton->GetSelectedState()) this->UpdateLocator();


    //  this->UpdateSliceDisplay(px, py, pz);     // RSierra 3/9/07: This line is redundant. If you remove it the slice views are still updated.
    this->UpdateSliceDisplay(nx, ny, nz, tx, ty, tz, px, py, pz);

    this->Logic0->UpdatePipeline ();
  
    }
}


void vtkBrpNavGUI::UpdateLocator()
{
 

    vtkTransform *transform = NULL;
    vtkTransform *transform_cb2 = NULL;

#ifdef USE_OPENTRACKER
    this->OpenTrackerStream->SetLocatorTransforms();
    transform = this->OpenTrackerStream->GetLocatorNormalTransform();

    this->OpenTrackerStream->SetLocatorTransforms();
    transform_cb2 = this->OpenTrackerStream->GetLocatorNormalTransform();
#endif
#ifdef USE_IGSTK
    this->IGSTKStream->SetLocatorTransforms();
    transform = this->IGSTKStream->GetLocatorNormalTransform(); 
#endif

    vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->LocatorModelID_new.c_str())); 
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


void vtkBrpNavGUI::UpdateRealtimeImg()
{
  /*
  int checkedRI = this->ConnectCheckButtonRI->GetSelectedState(); 
    if (checkedRI)
    {
#ifdef USE_OPENTRACKER
     
        
     this->OpenTrackerStream->GetSizeforRealtimeImaging(&xsizevalueRI, &ysizevalueRI);
     this->OpenTrackerStream->GetImageDataforRealtimeImaging(&ImageDataRI);
  

#endif
      // xsizevalue = 5;
      printf("GetRealTimeImage\n");
      
      cout << "xsixe:  ";
      cout<< xsizevalueRI << endl;
       
       cout << "ysixe:    ";
       cout<< xsizevalueRI << endl;
`
       
       ofstream fout("output.raw");
       fout.write((const char*)ImageDataRI.image_ptr, ImageDataRI.size());
       fout.close();
      
      
    }
*/
}




void vtkBrpNavGUI::UpdateSliceDisplay(float nx, float ny, float nz, 
                    float tx, float ty, float tz, 
                    float px, float py, float pz)
{

int checked = this->FreezeImageCheckButton->GetSelectedState();


     if (this->NeedOrientationUpdate0 ||
        this->NeedOrientationUpdate1 ||
         this->NeedOrientationUpdate2
         //     ||   this->NeedRealtimeImageUpdate
         )
    {
 

 vtkMatrix4x4* mat = vtkMatrix4x4::New();
 cout<<"UpdateSliceDisplay" <<endl;
      


      mat->SetElement(0, 0, nx);
      mat->SetElement(0, 1, ny);
      mat->SetElement(0, 2, nz);
      mat->SetElement(1, 0, tx);
      mat->SetElement(1, 1, ty);
      mat->SetElement(1, 2, tz);
      mat->SetElement(2, 0, ny*tz-nz*ty);
      mat->SetElement(2, 1, nz*tx-nx*tz);
      mat->SetElement(2, 2, nx*ty-ny*tx);


        mat->Invert();

         
      double ns[3];
      double ts[3];
      double cx = -128;
      double cy = -128;
      ns[0] = cx * nx;
      ns[1] = cx * ny;
      ns[2] = cx * nz;

      ts[0] = cy * tx;
      ts[1] = cy * ty;
      ts[2] = cy * tz;

      mat->SetElement(0, 3, px + ns[0] + ts[0]);
      mat->SetElement(1, 3, py + ns[1] + ts[1]);
      mat->SetElement(2, 3, pz + ns[2] + ts[2]);
      mat->SetElement(3, 3, 1.0);
        

     
      mat->SetElement(3, 3, 1.0);
      

      //Philip Mewes: Image can be frozen at the last updated
       if(!checked)
         {
       this->RealtimeVolumeNode->SetIJKToRASMatrix(mat);
         }
      


       /*
       char* order="AP";
       double spacing[3];
        spacing[0]=1.0;
        spacing[1]=1.0;
        spacing[2]=0.0;
        int dim[3];
         dim[0]=256;
         dim[1]=256;
         dim[2]=1;       
         bool centerImage=1;       
            this->RealtimeVolumeNode->ComputeIJKToRASFromScanOrder(order,spacing,dim,centerImage,mat);
       */
    

 

      mat->Delete();
    }

     
    // Axial
    if (strcmp(this->RedSliceMenu->GetValue(), "Locator"))
    {
      if (this->NeedOrientationUpdate0) 
      {  
        this->SliceNode0->SetOrientationToAxial();
        this->NeedOrientationUpdate0 = 0;
      }
    }
    else
    {
      if(!checked)
        {
      this->SliceNode0->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 2);
        }
      this->Control0->GetOffsetScale()->SetValue(pz);
      //this->Logic0->SetSliceOffset(pz);
      this->NeedOrientationUpdate0 = 1;

    }

    // Sagittal
    if (strcmp(this->YellowSliceMenu->GetValue(), "Locator"))
    {
      if (this->NeedOrientationUpdate1) 
      {
        this->SliceNode1->SetOrientationToSagittal();
        this->NeedOrientationUpdate1 = 0;
      }
    }
    else
    {
      this->SliceNode1->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 1);
      this->Control1->GetOffsetScale()->SetValue(px);
      //this->Logic1->SetSliceOffset(px);
      this->NeedOrientationUpdate1 = 1;
    }

    // Coronal
    if (strcmp(this->GreenSliceMenu->GetValue(), "Locator"))
    {
      if (this->NeedOrientationUpdate2) 
      {
        this->SliceNode2->SetOrientationToCoronal();
        this->NeedOrientationUpdate2 = 0;
      }
    }
    else
    {
      this->SliceNode2->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 3);
      this->Control2->GetOffsetScale()->SetValue(py);
      //this->Logic2->SetSliceOffset(py);
      this->NeedOrientationUpdate2 = 1;



    }
  
}

////////////////////////////////////////////////////////////////////
////////////////if for Checkbutton to XML+Coordinates send//////////////////////
///////////////////////////////////////////////////////////////////


#ifdef USE_OPENTRACKER
void vtkBrpNavGUI::SetOpenTrackerConnectionParameters()
{
    int checked = this->ConnectCheckButtonNT->GetSelectedState(); 
    if (checked)
    {
      printf("SetOpenTrackerConnectionParameters()\n");
    // connected
      strncpy(xmlpathfilename, this->LoadConfigButtonNT->GetWidget()->GetFileName(), 256);
    
      // if (! filename)
      //{

      vtkKWMessageDialog *dialog = vtkKWMessageDialog::New();
      dialog->SetParent ( this->ExtraFrame );
      dialog->SetStyleToMessage();
      std::string msg = std::string("Please input a valid configuration file (.xml).");
      

      char buf[128];
        sprintf(buf, "Connect to OpenTracker with %s file?", xmlpathfilename);

     
      //dialog->SetText(msg.c_str());
      dialog->SetText(buf);
      dialog->Create();
      dialog->Invoke();
      dialog->Delete();
      // this->ConnectCheckButtonNT->SetSelectedState(0);

   

      this->OpenTrackerStream->Init(xmlpathfilename); 
        
      cout << "=====================1GUI " << endl; 

      //int sp = atoi(this->UpdateRateEntry->GetWidget()->GetValue());
      int sp = 100;
      //float multi = atof(this->MultiFactorEntry->GetWidget()->GetValue());
        float multi = 1.0;
      cout << "=====================2GUI " << endl; 
      this->OpenTrackerStream->SetSpeed(sp);
      cout << "=====================3GUI " << endl; 
      this->OpenTrackerStream->SetMultiFactor(multi);
      cout << "=====================4GUI " << endl; 
      this->OpenTrackerStream->SetStartTimer(1);
      cout << "=====================5GUI " << endl; 
      this->OpenTrackerStream->ProcessTimerEvents();    
      cout << "=====================6GUI " << endl; 
       
    }
    else
     {
    this->OpenTrackerStream->StopPolling();
    }


      


}

void vtkBrpNavGUI::SetOpenTrackerConnectionCoordandOrient()
{

 int checkedsend = this->ConnectCheckButtonSEND->GetSelectedState();

 

                std::vector<float> pos;
                std::vector<float> quat;
       
             pos.resize(3);
             quat.resize(4);

         
      // float brptmp = atof(this->positionbrpy->GetWidget()->GetValue ());
      //  float brptmp = this->positionbrpy->GetWidget()->GetValue ();
     

        if (checkedsend)
          {
            
            
            cout << "Prepare to send orientations and postion so scanner (MANUAL)" <<endl; 
             
             pos[0]= atof(this->positionbrpy->GetWidget()->GetValue ());
            pos[1]= atof(this->positionbrpx->GetWidget()->GetValue ());
            pos[2]= atof(this->positionbrpz->GetWidget()->GetValue ());
            quat[0]= atof(this->orientationbrpo1->GetWidget()->GetValue ());
            quat[1]= atof(this->orientationbrpo2->GetWidget()->GetValue ());
            quat[2]= atof(this->orientationbrpo3->GetWidget()->GetValue ());
            quat[3]= atof(this->orientationbrpo4->GetWidget()->GetValue ());
            
  }
        /*
        if (checkedpassrobotcoords)
          {
            
            // std::vector<float> OrientationForScanner;
            //    std::vector<float> PositionForScanner;
  
                   float OrientationForScanner;
                  float PositionForScanner;
  

        this->OpenTrackerStream->GetCoordsOrientforScanner(&OrientationForScanner, &PositionForScanner);
            
        cout << "Prepare to send orientations and postion so scanner (TRANSFER ROBOT COORDS. and ORIENT." <<endl;
             cout <<OrientationForScanner <<endl;
             cout <<PositionForScanner <<endl;
             

              pos[0]= PositionForScanner;
            pos[1]= atof(this->positionbrpx->GetWidget()->GetValue ());
            pos[2]= atof(this->positionbrpz->GetWidget()->GetValue ());
            quat[0]= OrientationForScanner;
            quat[1]= atof(this->orientationbrpo2->GetWidget()->GetValue ());
            quat[2]= atof(this->orientationbrpo3->GetWidget()->GetValue ());
            quat[3]= atof(this->orientationbrpo4->GetWidget()->GetValue ());
                       
  


            this->OpenTrackerStream->SetTracker(pos,quat);
            
            cout << "Send orientations and postion so scanner, Mission accomplished :-)" <<endl;
        */
    
}
void vtkBrpNavGUI::StateTransitionDiagramControll()

{


   //
  // State transition diagram of the Workphase:
  //                |
  //                |START_UP (SU)
  //                v 
  //         +-------------+
  //         |             |
  //     +-->| PREP_PHASE  |
  //     |   |             |
  //     |   +-------------+
  //     |          |
  //     |EM        |PLANNING (PL)
  //     |          v
  //     |   +-------------+
  //     |   |             |
  //     +---|   PLANNING  |
  //     |   |(Seg,Targets |<---------+
  //     |   +-------------+          |
  //     |          |                 |
  //     |EM        |CALIBRATION (CL) |PL
  //     |          v                 |
  //     |   +-------------+          |
  //     |   |             |----------+
  //     +---| CALIBRATION |            
  //     |   | (Z-FRAME)   |<-------+            
  //     |   +-------------+        |            
  //     |          |               |            
  //     |EM        |TARGETING(TR)  |CL        
  //     |          v               |            
  //     |   +-------------+        |            
  //     |   |             |--------+            
  //     +---|   TARGETING |                   
  //     |   |(Positioning)|<----+            
  //     |   +-------------+     |            
  //     |          |            |  
  //     |EM        |MANUAL      |TR       
  //     |          v            | 
  //     |   +-------------+     |            
  //     |   |             |-----+            
  //     +---|    MANUAL   |                  
  //     |   | (INSERTION) |
  //     |   +-------------+        
  //     |          |               
  //     |          |EMERGENCY               
  //     |          v               
  //     |   +-------------+                 
  //     |   |             |            
  //     +---|    MANUAL   |                  
  //         | (INSERTION) |
  //         +-------------+
  // 
  //
  //actual (last active) workphase
  //
  //
  // | 1,1,0,0,0,1 | START_UP    |
  // | 0,0,1,0,0,1 | PLANNING    |
  // | 0,0,0,1,0,1 | CALIBRATION |
  // | 0,1,0,0,1,1 | TARGETING   |
  // | 0,0,0,1,1,1 | MANUAL      |
  // | 0,0,0,0,0,0 | EMERGENCY   |
  //
  //

  


  /*
    cout<<"robotstatusSD3): ";
    cout<<received_robot_status<<endl;
  */ 

  int WorkphaseClearanceSOFT = 0;
   int checkedClear = this->ClearWorkPhasecontrollButton->GetSelectedState();

    if (checkedClear)
   {
        ActualWorkPhase = 0;
     RequestedWorkphase = 0;
    }
    else
      
    {
      
  if (ActualWorkPhase==0)
    {
      ActualWorkPhase= ActualWorkPhase+1;
    }

      int WorkphaseCheckArrayD2[6][6] ={ {1,1,0,0,0,1}, {0,0,1,0,0,1}, {0,0,0,1,0,1}, {0,1,0,0,1,1}, {0,0,0,1,1,1}, {0,0,0,0,0,0} };
       ActualWorkPhase = ActualWorkPhase - 1;
       RequestedWorkphase = RequestedWorkphase -1;

 if (WorkphaseCheckArrayD2[ActualWorkPhase][RequestedWorkphase]==1)
       {
         WorkphaseClearanceSOFT = 1;
         ActualWorkPhase = RequestedWorkphase;
       }
     else
       {
          WorkphaseClearanceSOFT = 0;
          RequestedWorkphase =  ActualWorkPhase; 
       }
  
    ActualWorkPhase = ActualWorkPhase + 1;
    RequestedWorkphase = RequestedWorkphase + 1;


              if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==1)
                {
                this->SoftwareStatusLabelDisp->SetValue("READY");
                this->WorkPhasePlanningButton->SetStateToDisabled();
                this->WorkPhaseCalibarationButton->SetStateToDisabled();
                this->WorkPhaseTargetingButton->SetStateToDisabled();
                this->WorkPhaseManualButton->SetStateToDisabled();
                this->WorkPhaseEmergencyButton->SetStateToDisabled();
               
                } 

             else if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==2)
               {
              this->SoftwareStatusLabelDisp->SetValue("READY");
              this->WorkPhaseStartUpButton->SetStateToDisabled();
              this->WorkPhaseCalibarationButton->SetStateToDisabled();
              this->WorkPhaseTargetingButton->SetStateToDisabled();
              this->WorkPhaseManualButton->SetStateToDisabled();
              this->WorkPhaseEmergencyButton->SetStateToDisabled();
              }
       
             else if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==3)
               {
               this->SoftwareStatusLabelDisp->SetValue("READY");
               this->WorkPhaseStartUpButton->SetStateToDisabled();
               this->WorkPhasePlanningButton->SetStateToDisabled();
               this->WorkPhaseTargetingButton->SetStateToDisabled();
               this->WorkPhaseManualButton->SetStateToDisabled();
               this->WorkPhaseEmergencyButton->SetStateToDisabled();
               }
          
             else if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==4)
               {
               this->SoftwareStatusLabelDisp->SetValue("READY");
               this->WorkPhaseStartUpButton->SetStateToDisabled();
               this->WorkPhaseCalibarationButton->SetStateToDisabled();
               this->WorkPhasePlanningButton->SetStateToDisabled();
               this->WorkPhaseManualButton->SetStateToDisabled();
               this->WorkPhaseEmergencyButton->SetStateToDisabled();
               }
          else if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==5)
            {
          this->SoftwareStatusLabelDisp->SetValue("READY");
          this->WorkPhaseStartUpButton->SetStateToDisabled();
          this->WorkPhaseCalibarationButton->SetStateToDisabled();
          this->WorkPhaseTargetingButton->SetStateToDisabled();
          this->WorkPhasePlanningButton->SetStateToDisabled();
          this->WorkPhaseEmergencyButton->SetStateToDisabled();
          }
        else if(WorkphaseClearanceSOFT==1 && RequestedWorkphase==6)
              {
              this->SoftwareStatusLabelDisp->SetValue("EMER");
              this->WorkPhaseStartUpButton->SetStateToDisabled();
              this->WorkPhaseCalibarationButton->SetStateToDisabled();
              this->WorkPhaseTargetingButton->SetStateToDisabled();
              this->WorkPhaseManualButton->SetStateToDisabled();
              this->WorkPhasePlanningButton->SetStateToDisabled();
              }
          
      else if(!WorkphaseClearanceSOFT)
        {
         this->SoftwareStatusLabelDisp->SetValue("N/A");
         cout<<"requested workphase Change not allowed"<<endl;
        }

              //gives Clearance to ValveFiler (Workphases)
      
      if(WorkphaseClearanceSOFT)
        {
            SetOpenTrackerforBRPDataFlowValveFilter();
        }

             //give Clearance to process the Workphases
      /*
      if(    var_status_scanner == received_scanner_status
       &&  var_status_robot == received_robot_status
         &&  WorkphaseClearanceSOFT)
       {
        ProcessClearance = 1;
        }
      */
       }
}





void vtkBrpNavGUI::SetOpenTrackerforBRPDataFlowValveFilter()
{

              int checkedWorkPhaseStartUpButton = this->WorkPhaseStartUpButton->GetSelectedState();
              int checkedWorkPhasePlanningButton = this->WorkPhasePlanningButton->GetSelectedState() ;
              int checkedWorkPhaseCalibarationButton = this->WorkPhaseCalibarationButton->GetSelectedState();
              int checkedWorkPhaseTargetingButton = this->WorkPhaseTargetingButton->GetSelectedState();
              int checkedWorkPhaseManualButton = this->WorkPhaseManualButton->GetSelectedState();
              int checkedWorkPhaseEmergencyButton = this->WorkPhaseEmergencyButton->GetSelectedState();

               if (checkedWorkPhaseStartUpButton ||checkedWorkPhasePlanningButton
                   ||checkedWorkPhaseCalibarationButton ||checkedWorkPhaseTargetingButton
                   ||checkedWorkPhaseManualButton ||checkedWorkPhaseEmergencyButton )   
                      {        
                        
                        
                       std::vector<std::string> filtercommandkeys;
                       std::vector<std::string> filtercommandvalues;
                        filtercommandkeys.resize(1);
                        filtercommandvalues.resize(1);
  
 
                             if (checkedWorkPhaseStartUpButton){                
                                              filtercommandkeys[0] = "workphase";
                                              filtercommandvalues[0] = BRPTPR_START_UP;
                                                           }
       
                             if (checkedWorkPhasePlanningButton){filtercommandkeys[0] = "workphase";
                                               filtercommandvalues[0] = BRPTPR_PLANNING; }
        
                             if (checkedWorkPhaseCalibarationButton){filtercommandkeys[0] = "workphase";
                                               filtercommandvalues[0] = BRPTPR_CALIBRATION; }
  
                             if (checkedWorkPhaseTargetingButton){filtercommandkeys[0] = "workphase";
                                               filtercommandvalues[0] = BRPTPR_TARGETING;}
  
                             if (checkedWorkPhaseManualButton){filtercommandkeys[0] = "workphase";
                                               filtercommandvalues[0] = BRPTPR_MANUAL; }
  
                             if (checkedWorkPhaseEmergencyButton){filtercommandkeys[0] = "workphase";
                                               filtercommandvalues[0] = BRPTPR_EMERGENCY; }
       
                             this->OpenTrackerStream->SetOpenTrackerforBRPDataFlowValveFilter(filtercommandkeys, filtercommandvalues);

                         
                      }

               //08/02/2007 Philip Mewes TCL timer for resending Workphase command
               //defined in the Silcer<->robot Handshake Protokol. This is also used for
               //coordinates and orientation sending protokol
               
               
               if(
                  (received_robot_status==BRPTPR_Ready && checkedWorkPhaseStartUpButton) 
                  ||
                  (received_robot_status==BRPTPR_Uncalibrated && checkedWorkPhaseCalibarationButton) 

                  )
                 {

           cout<<"run TCL TIMER"<<endl;
           this->Script("after 5000 \"%s SetOpenTrackerforBRPDataFlowValveFilter\"", this->GetTclName());

                 }
  }





void vtkBrpNavGUI::SetOpenTrackerforScannerControll()
{



 int checkedsendstartScanner = this->ConnectCheckButtonStartScanner->GetSelectedState(); 
 int checkedsendstopScanner = this->ConnectCheckButtonStopScanner->GetSelectedState(); 
 int checkedsendsetprotocol = this->ConnectCheckButtonsetprotocol->GetSelectedState(); 
 int checkedsendprepScanner = this->ConnectCheckButtonprepScanner->GetSelectedState(); 
 int checkedsendpauseScanner = this->ConnectCheckButtonpauseScanner->GetSelectedState(); 
 int checkedsendresumeScanner = this->ConnectCheckButtonresumeScanner->GetSelectedState(); 
 int checkedsendnewexam = this->ConnectCheckButtonnewexam->GetSelectedState(); 
   



 if (checkedsendstartScanner ||checkedsendstopScanner
     ||checkedsendsetprotocol ||checkedsendprepScanner
     ||checkedsendpauseScanner ||checkedsendresumeScanner
     ||checkedsendnewexam )   
   {        
      std::vector<std::string> scancommandkeys;
      std::vector<std::string> scancommandvalues;
      
      
       if (checkedsendstartScanner)      {scancommandkeys.resize(1);
                      scancommandkeys[0] = "mrctrl_cmd";
                      scancommandvalues.resize(1);
                      scancommandvalues[0] = "START_SCAN"; }


       if (checkedsendstopScanner)      {scancommandkeys.resize(1);
                      scancommandkeys[0] = "mrctrl_cmd";
                      scancommandvalues.resize(1);
                      scancommandvalues[0] = "STOP_SCAN"; }



       if (checkedsendprepScanner)      {scancommandkeys.resize(1);
                      scancommandkeys[0] = "mrctrl_cmd";
                      scancommandvalues.resize(1);
                      scancommandvalues[0] = "PREP_SCAN"; }

       if (checkedsendpauseScanner)     {scancommandkeys.resize(1);
                      scancommandkeys[0] = "mrctrl_cmd";
                      scancommandvalues.resize(1);
                      scancommandvalues[0] = "PAUSE_SCAN"; }


       if (checkedsendresumeScanner)  {scancommandkeys.resize(1);
                      scancommandkeys[0] = "mrctrl_cmd";
                      scancommandvalues.resize(1);
                      scancommandvalues[0] = "RESUME_SCAN"; }

       if (checkedsendnewexam)     {scancommandkeys.resize(4);
                      scancommandvalues.resize(4);
                        
                      scancommandkeys[0]= "mrctrl_cmd";
                      scancommandkeys[1]= "patient_id";
                      scancommandkeys[2]= "patient_name";
                      scancommandkeys[3]= "patient_weight";
                      
                      scancommandvalues[0]= "NEW_EXAM";
                      scancommandvalues[1]= this->positionbrppatientid->GetWidget()->GetValue ();
                      scancommandvalues[2]= this->positionbrppatientname->GetWidget()->GetValue ();      
                      scancommandvalues[3]= this->positionbrppatientweight->GetWidget()->GetValue ();}

       if (checkedsendsetprotocol)     {scancommandkeys.resize(2);
                      scancommandvalues.resize(2);

                      scancommandkeys[0]= "mrctrl_cmd";
                      scancommandkeys[1]= "protocol_name";

                      scancommandvalues[0] = "LOAD_PROTOCOL"; 
                      scancommandvalues[1]= this->positionbrpsetprotocol->GetWidget()->GetValue ();}
      
      this->OpenTrackerStream->SetOpenTrackerforScannerControll(scancommandkeys, scancommandvalues);
    
    }


}






#endif




vtkMRMLVolumeNode* vtkBrpNavGUI::AddVolumeNode(vtkSlicerVolumesLogic* volLogic, const char* volumeNodeName)
{

  std::cerr << "AddVolumeNode(): called." << std::endl;

  vtkMRMLVolumeNode *volumeNode = NULL;

  if (volumeNode == NULL)  // if real-time volume node has not been created
    {

      vtkMRMLVolumeDisplayNode *displayNode = NULL;
      vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();

      vtkImageData* image = vtkImageData::New();

      //image->SetDimensions(RealtimeXsize, RealtimeYsize, 1);
      image->SetDimensions(256, 256, 1);
      //image->SetExtent( xmin, xmax, ymin, ymax, zmin, zmax );
      image->SetExtent(0, 255, 0, 255, 0, 0 );
      image->SetNumberOfScalarComponents( 1 );
      image->SetOrigin( 0, 0, 0 );
      image->SetSpacing( 1, 1, 0 );
      image->SetScalarTypeToShort();
      image->AllocateScalars();

      short* dest = (short*) image->GetScalarPointer();
      if (dest) {
       memset(dest, 0x01, 256*100*sizeof(short));
       image->Update();
      
      }
  
      /*
      vtkMRMLSliceNode *sliceorient = vtkMRMLSliceNode::New();
      sliceorient->SetSliceToRAS(LocatorMatrix);
      sliceorient->UpdateMatrices();
      */
      
      vtkSlicerSliceLayerLogic *reslice = vtkSlicerSliceLayerLogic::New();
      reslice->SetUseReslice(0);

      vtkImageChangeInformation *ici = vtkImageChangeInformation::New();
      ici->SetInput (image);
      ici->SetOutputSpacing( 1, 1, 1 );
      ici->SetOutputOrigin( 0, 0, 0 );
      ici->Update();
      scalarNode->SetAndObserveImageData (ici->GetOutput());

      vtkMatrix4x4* mat = vtkMatrix4x4::New();
      double space[3];
      int dim[3];
      space[0] = 1;
      space[1] = 1;
      space[2] = 10;
      dim[0]   = 256;
      dim[1]   = 256;
      dim[2]   = 1;
      scalarNode->ComputeIJKToRASFromScanOrder("IS",
                                               // possible is IS, PA, LR and inverse
                                              //image->GetSpacing(),
                                              space,
                                              //image->GetDimensions(),
                                              dim,
                                              true, mat);
      scalarNode->SetIJKToRASMatrix(mat);
      mat->Delete();
      image->Delete();

      ici->Delete();


      /* Based on the code in vtkSlicerVolumeLogic::AddHeaderVolume() */

      displayNode = vtkMRMLVolumeDisplayNode::New();
      scalarNode->SetLabelMap(0);
      volumeNode = scalarNode;

      if (volumeNode != NULL)
        {
          volumeNode->SetName(volumeNodeName);
          volLogic->GetMRMLScene()->SaveStateForUndo();

          vtkDebugMacro("Setting scene info");
          volumeNode->SetScene(volLogic->GetMRMLScene());
          displayNode->SetScene(volLogic->GetMRMLScene());

          //should we give the user the chance to modify this?.
          double range[2];
          vtkDebugMacro("Set basic display info");
          volumeNode->GetImageData()->GetScalarRange(range);
           range[0] = 0.0;
           range[1] = 256.0;
          displayNode->SetLowerThreshold(range[0]);
          displayNode->SetUpperThreshold(range[1]);
          displayNode->SetWindow(range[1] - range[0]);
          displayNode->SetLevel(0.5 * (range[1] - range[0]) );

          vtkDebugMacro("Adding node..");
          volLogic->GetMRMLScene()->AddNode(displayNode);

          //displayNode->SetDefaultColorMap();
          vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
          displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
          colorLogic->Delete();

          volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());

          vtkDebugMacro("Name vol node "<<volumeNode->GetClassName());
          vtkDebugMacro("Display node "<<displayNode->GetClassName());

          volLogic->GetMRMLScene()->AddNode(volumeNode);
          vtkDebugMacro("Node added to scene");

          volLogic->SetActiveVolumeNode(volumeNode);
          volLogic->Modified();
        }

      scalarNode->Delete();

      if (displayNode)
        {
          displayNode->Delete();
        }
    }

  return volumeNode;

}

/*
#ifdef USE_IGSTK
void vtkBrpNavGUI::SetIGSTKConnectionParameters()
{
    int checked = this->ConnectCheckButton->GetSelectedState(); 
    if (checked)
    {

    // Pulling rate for data
    int sp = atoi(this->UpdateRateEntry->GetWidget()->GetValue());
    this->IGSTKStream->SetSpeed(sp);

    // Conversion rate
    float multi = atof(this->MultiFactorEntry->GetWidget()->GetValue());
    this->IGSTKStream->SetMultiFactor(multi);


    // Device type 
    vtkKWMenuButton *mb = this->DeviceMenuButton->GetWidget();
    if (!strcmp (mb->GetValue(), "Polaris"))   
    {
        this->IGSTKStream->SetTrackerType(0);
    }
    else 
    {
        this->IGSTKStream->SetTrackerType(1);

    }


    // Port number
    int pn = atoi(this->PortNumberMenuButton->GetWidget()->GetValue());
    this->IGSTKStream->SetPortNumber((PortNumberT)pn);

    // Baud rate 
    int br = atoi(this->BaudRateMenuButton->GetWidget()->GetValue());
    this->IGSTKStream->SetBaudRate((BaudRateT)br);

    // Data bits 
    int db = atoi(this->DataBitsMenuButton->GetWidget()->GetValue());
    this->IGSTKStream->SetBaudRate((BaudRateT)db);

    // Parity 
    mb = this->ParityTypeMenuButton->GetWidget();
    if (!strcmp (mb->GetValue(), "No"))   

    {
        this->IGSTKStream->SetParity(igstk::SerialCommunication::NoParity);
    }
    else if     (!strcmp (mb->GetValue(), "Odd")) 
    {
        this->IGSTKStream->SetParity(igstk::SerialCommunication::OddParity);
    }
    else
    {

        this->IGSTKStream->SetParity(igstk::SerialCommunication::EvenParity);
    }

    // Stop bits 
    int sb = atoi(this->StopBitsMenuButton->GetWidget()->GetValue());
    this->IGSTKStream->SetStopBits((StopBitsT)sb);

    // Hand shake
    mb = this->HandShakeMenuButton->GetWidget();
    if (!strcmp (mb->GetValue(), "Off"))   
    {
        this->IGSTKStream->SetHandShake(igstk::SerialCommunication::HandshakeOff);
    }
    else
    {
        this->IGSTKStream->SetHandShake(igstk::SerialCommunication::HandshakeOn);
    }

    this->IGSTKStream->SetTracking(1);
    this->IGSTKStream->SetSpeed(sp);
    this->IGSTKStream->SetMultiFactor(multi);
    this->IGSTKStream->Init();
    this->IGSTKStream->ProcessTimerEvents();
    }
    else
    {
    this->IGSTKStream->SetTracking(0);
    }
}
#endif
*/
