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

#include "vtkKWTkUtilities.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkCylinderSource.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkActor.h"
#include "vtkProperty.h"
#include "vtkCornerAnnotation.h"

// Real-time image display
#include "vtkSlicerColorLogic.h"
#include "vtkSlicerVolumesGUI.h"


#include "vtkIGTDataStream.h"

#include "vtkCylinderSource.h"
#include "vtkMRMLLinearTransformNode.h"


// for DICOM read
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkGDCMImageIO.h"
#include "itkSpatialOrientationAdapter.h"


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
   
    //this->NeedleCheckButton = NULL;
    this->ConnectCheckButtonNT = NULL;
    this->ConnectCheckButtonSEND = NULL;
    //this->ConnectCheckButtonPASSROBOTCOORDS = NULL;
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


    //this->NeedleCheckButton = NULL;
    this->HandleCheckButton = NULL;
    this->GuideCheckButton = NULL;

    this->SetLocatorModeButton = NULL;
    this->SetUserModeButton    = NULL;

    this->RedSliceMenu = NULL;
    this->YellowSliceMenu = NULL;
    this->GreenSliceMenu = NULL;

    this->ImagingControlCheckButton = NULL;
    this->ImagingMenu = NULL;

#ifdef USE_NAVITRACK
  
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

    this->AddCoordsandOrientTarget = NULL;
    this->SetOrientButton = NULL;
   

    this->PointPairMultiColumnList = NULL;
    this->TargetListColumnList = NULL;

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
    //this->NeedOrientationUpdate0 = 0;
    //this->NeedOrientationUpdate1 = 0;
    //this->NeedOrientationUpdate2 = 0;

    this->SliceDriver0 = 0;
    this->SliceDriver1 = 0;
    this->SliceDriver2 = 0;

    this->NeedRealtimeImageUpdate0 = 0;
    this->NeedRealtimeImageUpdate1 = 0;
    this->NeedRealtimeImageUpdate2 = 0;

    this->FreezeOrientationUpdate  = 0;
    //this->OrgNeedOrientationUpdate0 = 0;
    //this->OrgNeedOrientationUpdate1 = 0;
    //this->OrgNeedOrientationUpdate2 = 0;
    //this->OrgNeedRealtimeImageUpdate0 = 0;
    //this->OrgNeedRealtimeImageUpdate1 = 0;
    //this->OrgNeedRealtimeImageUpdate2 = 0;

    this->RealtimeImageOrient = vtkBrpNavGUI::SLICE_RTIMAGE_PERP;

    // for Real-time image display
    this->RealtimeVolumeNode = NULL;
    this->RealtimeXsize = 0;
    this->RealtimeYsize = 0;
    this->RealtimeImageSerial = 0;


    // Widgets for Calibration Frame
    this->CalibImageFileEntry      = NULL;
    this->ReadCalibImageFileButton = NULL;
    this->ListCalibImageFileButton = NULL;
    

#ifdef USE_NAVITRACK
    this->OpenTrackerStream = vtkIGTOpenTrackerStream2::New();
#endif
#ifdef USE_IGSTK
    this->IGSTKStream = vtkIGTIGSTKStream::New();
#endif



}

//---------------------------------------------------------------------------
vtkBrpNavGUI::~vtkBrpNavGUI ( )
{
#ifdef USE_NAVITRACK
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

//    if (this->NeedleCheckButton)
//    {
//        this->NeedleCheckButton->SetParent(NULL );
//        this->NeedleCheckButton->Delete ( );
//    }

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
       
    //if (this->ConnectCheckButtonPASSROBOTCOORDS)
    //{
    //    this->ConnectCheckButtonPASSROBOTCOORDS->SetParent(NULL );
    //    this->ConnectCheckButtonPASSROBOTCOORDS->Delete ( );
    //}

 
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



//    if (this->NeedleCheckButton)
//    {
//    this->NeedleCheckButton->SetParent(NULL );
//    this->NeedleCheckButton->Delete ( );
//    }
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

    if ( this->ImagingControlCheckButton )
    {
        this->ImagingControlCheckButton->SetParent(NULL);
        this->ImagingControlCheckButton->Delete();
    }

    if ( this->ImagingMenu )
    {
        this->ImagingMenu->SetParent(NULL);
        this->ImagingMenu->Delete();
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


#ifdef USE_NAVITRACK
    this->OpenTrackerStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
   
    this->LoadConfigButtonNT->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
 #endif
#ifdef USE_IGSTK
    this->IGSTKStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
    this->DeviceMenuButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
#endif

  
//    if (this->NeedleCheckButton)
//    {
//    this->NeedleCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
//    }
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
//      if (this->ConnectCheckButtonPASSROBOTCOORDS)
//    {
//    this->ConnectCheckButtonPASSROBOTCOORDS->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
//    }  

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
    
//    if (this->NeedleCheckButton)
//    {
//    this->NeedleCheckButton->RemoveObservers ( vtkKWCheckButton::SelectedStateChangedEvent,  (vtkCommand *)this->GUICallbackCommand );
//    }
    if (this->SetLocatorModeButton)
    {
        this->SetLocatorModeButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,
                                                      (vtkCommand *)this->GUICallbackCommand );
    }
    if (this->SetUserModeButton)
    {
        this->SetUserModeButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,
                                                   (vtkCommand *)this->GUICallbackCommand );
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


    // Driver selections
    this->RedSliceMenu->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->YellowSliceMenu->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->GreenSliceMenu->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    // Real-time imaging
    this->ImagingControlCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ImagingMenu->GetMenu()->AddObserver ( vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    // Fill in
    // observer load volume button
    //this->NeedleCheckButton->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    
    this->ConnectCheckButtonNT->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->ConnectCheckButtonSEND->AddObserver ( vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
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
    
    
    
    this->SetLocatorModeButton->AddObserver ( vtkKWPushButton::InvokedEvent,
                                              (vtkCommand *)this->GUICallbackCommand );
    this->SetUserModeButton->AddObserver ( vtkKWPushButton::InvokedEvent,
                                           (vtkCommand *)this->GUICallbackCommand );
    this->ReadCalibImageFileButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);

#ifdef USE_NAVITRACK
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

            //this->SlicerCoordinatesEntry->GetWidget()->SetValue(ras.c_str());
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

        if (this->ConnectCheckButtonNT == vtkKWCheckButton::SafeDownCast(caller) 
            && event == vtkKWCheckButton::SelectedStateChangedEvent )
        {
      
#ifdef USE_NAVITRACK
            SetOpenTrackerConnectionParameters();
#endif
      
#ifdef USE_IGSTK
            SetIGSTKConnectionParameters();
#endif
      
        }

     
        if (this->ConnectCheckButtonSEND == vtkKWCheckButton::SafeDownCast(caller) 
            && event == vtkKWCheckButton::SelectedStateChangedEvent  )
        {
      
#ifdef USE_NAVITRACK
       
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
      
#ifdef USE_NAVITRACK
          
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
            
#ifdef USE_NAVITRACK
          
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
             #ifdef USE_NAVITRACK
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
            
            //vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->LocatorModelID_new.c_str())); 
            vtkMRMLModelNode *model = vtkMRMLModelNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID("vtkMRMLModelNode1")); 
            if (model != NULL)
            {
              //vtkMRMLModelDisplayNode *disp = vtkMRMLModelDisplayNode::SafeDownCast(model->GetDisplayNode());
              vtkMRMLModelDisplayNode *disp = model->GetModelDisplayNode();
                
                if (disp != NULL)
                {
                    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
                    vtkSlicerColor *color = app->GetSlicerTheme()->GetSlicerColors ( );
                    disp->SetColor(color->SliceGUIGreen);
                    disp->SetVisibility(checked);
                }
            }
        }
        
        //
        // Slice Plane Driver selection
        //

        else if ( this->RedSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
                  && event == vtkKWMenu::MenuItemInvokedEvent )
        {
            const char* selected = this->RedSliceMenu->GetValue();
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_RED, selected);
        }

        else if ( this->YellowSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
                  && event == vtkKWMenu::MenuItemInvokedEvent )
        {
            const char* selected = this->YellowSliceMenu->GetValue();
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_YELLOW, selected);
        }

        else if ( this->GreenSliceMenu->GetMenu() == vtkKWMenu::SafeDownCast(caller)
                  && event == vtkKWMenu::MenuItemInvokedEvent )
        {
          const char* selected = this->GreenSliceMenu->GetValue();
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_GREEN, selected);
        }

        // -- "Locator" button 
        else if ( this->SetLocatorModeButton == vtkKWPushButton::SafeDownCast(caller) 
                  && event == vtkKWPushButton::InvokedEvent )
        {
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_RED, "Locator");
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_YELLOW, "Locator");
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_GREEN, "Locator");
        }

        // -- "User" button 
        else if (this->SetUserModeButton == vtkKWPushButton::SafeDownCast(caller) 
                 && event == vtkKWPushButton::InvokedEvent )
        {
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_RED, "User");
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_YELLOW, "User");
            ChangeSlicePlaneDriver(vtkBrpNavGUI::SLICE_PLANE_GREEN, "User");
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
            if ( strcmp(selected, "Perpendicular") == 0 )
            {
                this->RealtimeImageOrient = vtkBrpNavGUI::SLICE_RTIMAGE_PERP;
            }
            else if ( strcmp(selected, "In-plane 90") == 0 )
            {
                this->RealtimeImageOrient = vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE90;
            }
            else //if ( strcmp(selected, "In-plane") == 0 )
            {
                this->RealtimeImageOrient = vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE;
            }

            std::cerr << "ImagingMenu =======> " << selected << "  :  " << this->RealtimeImageOrient << std::endl;

        }


        //
        // Z-frame calibration control
        //

        else if (this->ReadCalibImageFileButton == vtkKWPushButton::SafeDownCast(caller) 
                 && event == vtkKWPushButton::InvokedEvent)
        {
            const char * filename = this->ListCalibImageFileButton->GetWidget()->GetFileName();
            if (filename)
            {
                int w, h;
                std::vector<float> position;
                std::vector<float> orientation;
                position.resize(3, 0.0);
                orientation.resize(4, 0.0);
                
                Image *ZFrameImage = DicomRead(filename, &w, &h, position, orientation);
                this->OpenTrackerStream->SetZFrameTrackingData(ZFrameImage, w, h,
                                                               position, orientation);

            }
        }
        else if (this->ListCalibImageFileButton->GetWidget() == vtkKWLoadSaveButton::SafeDownCast(caller) 
              && event == vtkKWPushButton::InvokedEvent )
        {
            const char * filename = this->ListCalibImageFileButton->GetWidget()->GetFileName();
            if (filename)
            {
                const vtksys_stl::string fname(filename);
                this->CalibImageFileEntry->SetValue(fname.c_str());
            }
            else
            {
                this->CalibImageFileEntry->SetValue("");
            }
            this->ListCalibImageFileButton->GetWidget()->SetText ("Browse Image File");
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
    BuildGUIForCalibration();

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


    // Scanner: Scanner controled by the locator
    // -----------------------------------------
    vtkKWFrameWithLabel *imagingFrame = vtkKWFrameWithLabel::New ( );
    imagingFrame->SetParent ( trackingFrame->GetFrame() );
    imagingFrame->Create ( );
    imagingFrame->SetLabelText ("Real-time Imaging");
    this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                   imagingFrame->GetWidgetName() );
    
    this->ImagingControlCheckButton = vtkKWCheckButton::New();
    this->ImagingControlCheckButton->SetParent(imagingFrame->GetFrame());
    this->ImagingControlCheckButton->Create();
    this->ImagingControlCheckButton->SelectedStateOff();
    this->ImagingControlCheckButton->SetText("Imaging Orientation Control");

    this->ImagingMenu = vtkKWMenuButton::New();
    this->ImagingMenu->SetParent(imagingFrame->GetFrame());
    this->ImagingMenu->Create();
    this->ImagingMenu->SetWidth(10);
    this->ImagingMenu->GetMenu()->AddRadioButton ("Perpendicular");
    this->ImagingMenu->GetMenu()->AddRadioButton ("In-plane 90");
    this->ImagingMenu->GetMenu()->AddRadioButton ("In-plane");
    this->ImagingMenu->SetValue("Perpendicular");

    this->Script( "pack %s %s -side left -anchor w -padx 2 -pady 2", 
                  this->ImagingControlCheckButton->GetWidgetName(),
                  this->ImagingMenu->GetWidgetName() );

    trackingFrame->Delete();
    displayFrame->Delete();
    driverFrame->Delete();
    modeFrame->Delete();
    sliceFrame->Delete();
}



void vtkBrpNavGUI::BuildGUIForRealtimeacqFrame()
{

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
    
#ifdef USE_NAVITRACK
    
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
    
    //this->ConnectCheckButtonPASSROBOTCOORDS = vtkKWCheckButton::New();
    //this->ConnectCheckButtonPASSROBOTCOORDS->SetParent(sendNTFrame);
    //this->ConnectCheckButtonPASSROBOTCOORDS->Create();
    //this->ConnectCheckButtonPASSROBOTCOORDS->SelectedStateOff();
    //this->ConnectCheckButtonPASSROBOTCOORDS->SetText("Imaging orientation Control ");
    //
    //
    //this->Script("pack %s %s -side top -anchor w -padx 2 -pady 2", 
    //             this->ConnectCheckButtonSEND->GetWidgetName(),
    //             this->ConnectCheckButtonPASSROBOTCOORDS->GetWidgetName());
    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
                 this->ConnectCheckButtonSEND->GetWidgetName());

    
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


void vtkBrpNavGUI::BuildGUIForCalibration()
{

    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    vtkKWWidget *page = this->UIPanel->GetPageWidget ( "BrpNav" );

    // ----------------------------------------------------------------
    // CALIBRATION FRAME         
    // ----------------------------------------------------------------
    
    vtkSlicerModuleCollapsibleFrame *calibrationFrame = vtkSlicerModuleCollapsibleFrame::New ( );
    calibrationFrame->SetParent ( page );
    calibrationFrame->Create ( );
    calibrationFrame->SetLabelText ("Calibration");
    calibrationFrame->CollapseFrame ( );
    app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                  calibrationFrame->GetWidgetName(), page->GetWidgetName());

    vtkKWFrameWithLabel *calibImageFrame = vtkKWFrameWithLabel::New ( );
    calibImageFrame->SetParent ( calibrationFrame->GetFrame() );
    calibImageFrame->Create ( );
    calibImageFrame->SetLabelText ("Z Frame Calibration");
    //setupFrame->CollapseFrame ( );
    this->Script( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                  calibImageFrame->GetWidgetName());

    vtkKWFrame *calibImageFileFrame = vtkKWFrame::New();
    calibImageFileFrame->SetParent ( calibImageFrame->GetFrame() );
    calibImageFileFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                  calibImageFileFrame->GetWidgetName());
    
    vtkKWFrame *calibImageButtonsFrame = vtkKWFrame::New();
    calibImageButtonsFrame->SetParent ( calibImageFrame->GetFrame() );
    calibImageButtonsFrame->Create ( );
    this->Script( "pack %s -side top -anchor nw -expand n -padx 2 -pady 2",
                  calibImageButtonsFrame->GetWidgetName());

    this->CalibImageFileEntry = vtkKWEntry::New();
    this->CalibImageFileEntry->SetParent(calibImageFileFrame);
    this->CalibImageFileEntry->Create();
    this->CalibImageFileEntry->SetWidth(50);
    this->CalibImageFileEntry->SetValue ( "" );

    this->ListCalibImageFileButton = vtkKWLoadSaveButtonWithLabel::New ( );
    this->ListCalibImageFileButton->SetParent (calibImageFileFrame);
    this->ListCalibImageFileButton->Create ( );
    this->ListCalibImageFileButton->SetWidth(15);
    this->ListCalibImageFileButton->GetWidget()->SetText ("Browse Image File");
    this->ListCalibImageFileButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes(
                                  "{ {BrpNav} {*.*} }");
    this->ListCalibImageFileButton->GetWidget()->
      GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->Script("pack %s %s -side left -anchor w -fill x -padx 2 -pady 2", 
                 this->ListCalibImageFileButton->GetWidgetName(),
                 this->CalibImageFileEntry->GetWidgetName());
    
    this->ReadCalibImageFileButton = vtkKWPushButton::New ( );
    this->ReadCalibImageFileButton->SetParent (calibImageButtonsFrame);
    this->ReadCalibImageFileButton->Create ( );
    this->ReadCalibImageFileButton->SetText (" Read Z-Frame Image ");
    this->ReadCalibImageFileButton->SetBalloonHelpString(" Read Z-Frame Image ");
    
    this->Script("pack %s -side top -anchor w -padx 2 -pady 2", 
                this->ReadCalibImageFileButton->GetWidgetName());
    
    calibrationFrame->Delete ();
    calibImageFrame->Delete ();
    calibImageFileFrame->Delete ();
    calibImageButtonsFrame->Delete ();

}




void vtkBrpNavGUI::UpdateAll()
{

    // Position / orientation parameters:
    //   (px, py, pz) : position
    //   (nx, ny, nz) : normal vector
    //   (tx, ty, tz) : transverse vector
    //   (sx, sy, sz) : vector orthogonal to n and t ( n x t )

    float px, py, pz, nx, ny, nz, tx, ty, tz;
    float sx, sy, sz;

    this->LocatorMatrix = NULL;

#ifdef USE_NAVITRACK
    this->LocatorMatrix = this->OpenTrackerStream->GetLocatorMatrix();
    //    this->OpenTrackerStream->GetSizeforRealtimeImaging(&xsizevalueRI, &ysizevalueRI);
#endif
#ifdef USE_IGSTK
    this->LocatorMatrix = this->IGSTKStream->GetLocatorMatrix();
#endif

    if (this->LocatorMatrix)
    {
        px = this->LocatorMatrix->GetElement(0, 0);
        py = this->LocatorMatrix->GetElement(1, 0);
        pz = this->LocatorMatrix->GetElement(2, 0);

        nx = this->LocatorMatrix->GetElement(0, 1);
        ny = this->LocatorMatrix->GetElement(1, 1);
        nz = this->LocatorMatrix->GetElement(2, 1);

        tx = this->LocatorMatrix->GetElement(0, 2);
        ty = this->LocatorMatrix->GetElement(1, 2);
        tz = this->LocatorMatrix->GetElement(2, 2);

        sx = ny*tz-nz*ty;
        sy = nz*tx-nx*tz;
        sz = nx*ty-ny*tx;
    }
    else
    {
        px = 0.0;
        py = 0.0;
        pz = 0.0;

        nx = 0.0;
        ny = 0.0;
        nz = 1.0;

        tx = 1.0;
        ty = 0.0;
        tz = 0.0;

        sx = 0.0;
        sy = 1.0;
        sz = 0.0;
    }

    //std::cerr << "==== Locator position ====" << std::endl;
    //std::cerr << "  (px, py, pz) =  ( " << px << ", " << py << ", " << pz << " )" << std::endl;
    //std::cerr << "  (nx, ny, nz) =  ( " << nx << ", " << ny << ", " << nz << " )" << std::endl;
    //std::cerr << "  (tx, ty, tz) =  ( " << tx << ", " << ty << ", " << tz << " )" << std::endl;

    //Philip Mewes 17.07.2007: defining and sending te workphase (WP) commands depending of requestet WP
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


    // Get real-time image orientation
    int rtimgslice = this->RealtimeImageOrient;

    if (this->OpenTrackerStream)
    {

        // Junichi Tokuda 10/18/2007: Definition of scan plane (for scanner) and
        //  display (for Slicer) plane
        //
        //  Normal (N_l) and Transverse (T_l) vectors of locator are givien.
        //     M_p       : IJK to RAS matrix
        //     N_l x T_l : cross product of N_l and T_l
        //     M_s       : scan plane rotation matrix (transformation from axial plane to scan plane)
        //
        //   1) Perpendicular (Plane perpendicular to the locator)
        //
        //     M_p   = ( T_l, N_l x T_l, N_l )
        //
        //     #         / tx ty tz \  / 1  0  0 \       / tx ty tz \ 
        //     #        |            ||           |     |            |
        //     #M_s  =  |  sx sy sz  ||  0  1  0  |  =  |  sx sy sz  |
        //     #        |            ||           |     |            |
        //     #         \ nx ny nz /  \ 0  0  1 /       \ nx ny nz / 
        //
        //
        //              / tx sx nx \  / 1  0  0 \       / tx -sx -nx \ 
        //             |            ||           |     |              |
        //     M_s  =  |  ty sy ny  ||  0 -1  0  |  =  |  ty -sy -ny  |
        //             |            ||           |     |              |
        //              \ tz sz nz /  \ 0  0 -1 /       \ tz -sz -nz / 
        //
        //
        //   2) In-plane 90  (plane along the locator: perpendicular to In-plane)
        //
        //     M_p  = ( N_l x T_l, N_l, T_l )
        //
        //     #         / tx ty tz \  / 0  0  1 \       / ty tz tx \ 
        //     #        |            ||           |     |            |
        //     #M_s  =  |  sx sy sz  ||  1  0  0  |  =  |  sy sz sx  |
        //     #        |            ||           |     |            |
        //     #         \ nx ny nz /  \ 0  1  0 /       \ ny nz nx / 
        // 
        //
        //              / tx sx nx \  /  0  0 -1 \       / sx -nx -tx \ 
        //             |            ||            |     |              |
        //     M_s  =  |  ty sy ny  ||   1  0  0  |  =  |  sy -ny -ty  |
        //             |            ||            |     |              |
        //              \ tz sz nz /  \  0 -1  0 /       \ sz -nz -tz / 
        //
        // 
        //   3) In-Plane     (plane along the locator)
        //
        //     M_p  = ( N_l, T_l, N_l x T_l )
        //
        //     #         / tx ty tz \  / 0  1  0 \       / tz tx ty \ 
        //     #        |            ||           |     |            |
        //     #M_s  =  |  sx sy sz  ||  0  0  1  |  =  |  sz sx sy  |
        //     #        |            ||           |     |            |
        //     #         \ nx ny nz /  \ 1  0  0 /       \ nz nx ny / 
        //
        //
        //              / tx sx nx \  /  0 -1  0 \       / nx -tx -sx \ 
        //             |            ||            |     |              |
        //     M_s  =  |  ty sy ny  ||   0  0 -1  |  =  |  ny -ty -sy  |
        //             |            ||            |     |              |
        //              \ tz sz nz /  \  1  0  0 /       \ nz -tz -sz / 
        //
        //

        //
        // Real-time image display plane transformation 
        //
        
        // Junichi Tokuda 10/16/2007:
        // Since the position/orientation for the real-time image is not available,
        // the transformation is calculated based on the locator matrix.
        // This must be fixed, when the image information become available.

        vtkImageData* vid = NULL;
        if (this->RealtimeVolumeNode)
        {
            vid = this->RealtimeVolumeNode->GetImageData();
        }

        if (vid && !this->FreezeOrientationUpdate)
        {
            //std::cerr << "BrpNavGUI::UpdateAll(): update realtime image" << std::endl;

            int orgSerial = this->RealtimeImageSerial;
            this->OpenTrackerStream->GetRealtimeImage(&(this->RealtimeImageSerial), vid);
            if (orgSerial != this->RealtimeImageSerial)  // if new image has been arrived
            {

                vtkMatrix4x4* rtimgTransform = vtkMatrix4x4::New();

                //this->RealtimeVolumeNode->UpdateScene(this->GetMRMLScene());
                this->RealtimeVolumeNode->SetAndObserveImageData(vid);

                // One of NeedRealtimeImageUpdate0 - 2 is chosen based on the scan plane.
                
                if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_PERP)  /* Perpendicular */
                {
                    this->NeedRealtimeImageUpdate0 = 1;

                    rtimgTransform->SetElement(0, 0, tx);
                    rtimgTransform->SetElement(1, 0, ty);
                    rtimgTransform->SetElement(2, 0, tz);
                    
                    rtimgTransform->SetElement(0, 1, sx);
                    rtimgTransform->SetElement(1, 1, sy);
                    rtimgTransform->SetElement(2, 1, sz);
                    
                    rtimgTransform->SetElement(0, 2, nx);
                    rtimgTransform->SetElement(1, 2, ny);
                    rtimgTransform->SetElement(2, 2, nz);
                }
                else if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE90)  /* In-plane 90 */
                {
                    this->NeedRealtimeImageUpdate1 = 1;

                    rtimgTransform->SetElement(0, 0, sx);
                    rtimgTransform->SetElement(1, 0, sy);
                    rtimgTransform->SetElement(2, 0, sz);
                    
                    rtimgTransform->SetElement(0, 1, nx);
                    rtimgTransform->SetElement(1, 1, ny);
                    rtimgTransform->SetElement(2, 1, nz);
                    
                    rtimgTransform->SetElement(0, 2, tx);
                    rtimgTransform->SetElement(1, 2, ty);
                    rtimgTransform->SetElement(2, 2, tz);
                }
                else // if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE)   /* In-Plane */
                  {
                    this->NeedRealtimeImageUpdate2 = 1;

                    rtimgTransform->SetElement(0, 0, nx);
                    rtimgTransform->SetElement(1, 0, ny);
                    rtimgTransform->SetElement(2, 0, nz);
                    
                    rtimgTransform->SetElement(0, 1, tx);
                    rtimgTransform->SetElement(1, 1, ty);
                    rtimgTransform->SetElement(2, 1, tz);
                    
                    rtimgTransform->SetElement(0, 2, sx);
                    rtimgTransform->SetElement(1, 2, sy);
                    rtimgTransform->SetElement(2, 2, sz);
                }

                rtimgTransform->SetElement(0, 3, px);
                rtimgTransform->SetElement(1, 3, py);
                rtimgTransform->SetElement(2, 3, pz);
                rtimgTransform->SetElement(3, 3, 1.0);
                
                this->RealtimeVolumeNode->SetIJKToRASMatrix(rtimgTransform);

                this->RealtimeVolumeNode->UpdateScene(this->VolumesLogic->GetMRMLScene());
                this->VolumesLogic->SetActiveVolumeNode(this->RealtimeVolumeNode);

                this->VolumesLogic->Modified();
                rtimgTransform->Delete();
            }

            
        }
        else
        {
          //std::cerr << "BrpNavGUI::UpdateAll(): no realtime image" << std::endl;
        }

        //
        // Imaging plane transformation 
        //

        //if (this->ConnectCheckButtonPASSROBOTCOORDS->GetSelectedState())
        if (this->ImagingControlCheckButton->GetSelectedState())
        {
            std::vector<float> pos;
            std::vector<float> quat;
            pos.resize(3);
            quat.resize(4);

            float scanTrans[3][3];  // Rotation matrix from axial plane to scan plane
            
            /* Parpendicular */
            if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_PERP)
            {
                scanTrans[0][0] = tx;
                scanTrans[1][0] = ty;
                scanTrans[2][0] = tz;
                scanTrans[0][1] = -sx;
                scanTrans[1][1] = -sy;
                scanTrans[2][1] = -sz;
                scanTrans[0][2] = -nx;
                scanTrans[1][2] = -ny;
                scanTrans[2][2] = -nz;
            }
            /* In-plane 90 */
            else if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE90)
            {
                scanTrans[0][0] = sx;
                scanTrans[1][0] = sy;
                scanTrans[2][0] = sz;
                scanTrans[0][1] = -nx;
                scanTrans[1][1] = -ny;
                scanTrans[2][1] = -nz;
                scanTrans[0][2] = -tx;
                scanTrans[1][2] = -ty;
                scanTrans[2][2] = -tz;
            }
            /* In-Plane */
            else // if (rtimgslice == vtkBrpNavGUI::SLICE_RTIMAGE_INPLANE)
            {
                scanTrans[0][0] = nx;
                scanTrans[1][0] = ny;
                scanTrans[2][0] = nz;
                scanTrans[0][1] = -tx;
                scanTrans[1][1] = -ty;
                scanTrans[2][1] = -tz;
                scanTrans[0][2] = -sx;
                scanTrans[1][2] = -sy;
                scanTrans[2][2] = -sz;
            }

            MathUtils::matrixToQuaternion(scanTrans, quat);
            pos[0] = px;
            pos[1] = py;
            pos[2] = pz;
            
            // send coordinate to the scanner
            this->OpenTrackerStream->SetTracker(pos,quat);
        }

        // update the display of locator
        if (this->LocatorCheckButton->GetSelectedState())
        {
            this->UpdateLocator();
        }
        if (!this->FreezeOrientationUpdate)
        {
            this->UpdateSliceDisplay(nx, ny, nz, tx, ty, tz, px, py, pz);
        }
    }

    this->NeedRealtimeImageUpdate0 = 0;
    this->NeedRealtimeImageUpdate1 = 0;
    this->NeedRealtimeImageUpdate2 = 0;

}


void vtkBrpNavGUI::UpdateLocator()
{
 

    vtkTransform *transform = NULL;
    vtkTransform *transform_cb2 = NULL;

#ifdef USE_NAVITRACK
    this->OpenTrackerStream->SetLocatorTransforms();
    transform = this->OpenTrackerStream->GetLocatorNormalTransform();

    this->OpenTrackerStream->SetLocatorTransforms();
    transform_cb2 = this->OpenTrackerStream->GetLocatorNormalTransform();
#endif
#ifdef USE_IGSTK
    this->IGSTKStream->SetLocatorTransforms();
    transform = this->IGSTKStream->GetLocatorNormalTransform(); 
#endif

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


void vtkBrpNavGUI::UpdateSliceDisplay(float nx, float ny, float nz, 
                                      float tx, float ty, float tz, 
                                      float px, float py, float pz)
{
  //std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() is called." << std::endl;

    // Reslice -- Perpendicular
    if ( this->SliceDriver0 == vtkBrpNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver0 == vtkBrpNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode0->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 0);
        this->Logic0->UpdatePipeline ();
    }
    else if ( this->SliceDriver0 == vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->NeedRealtimeImageUpdate0)
        {
          //            std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : Perp: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode0->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 0);
            this->Logic0->UpdatePipeline ();
        }
    }


    // Reslice -- In-plane 90
    if ( this->SliceDriver1 == vtkBrpNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver1 == vtkBrpNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode1->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 1);
        this->Logic1->UpdatePipeline ();
    }
    else if ( this->SliceDriver1 == vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->NeedRealtimeImageUpdate1)
        {
          //            std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane 90: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode1->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 1);
            this->Logic1->UpdatePipeline ();
        }
    }


    // Reslice -- In-plane
    if ( this->SliceDriver2 == vtkBrpNavGUI::SLICE_DRIVER_USER )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_USER" << std::endl;
    }
    else if ( this->SliceDriver2 == vtkBrpNavGUI::SLICE_DRIVER_LOCATOR )
    {
      //        std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_LOCATOR" << std::endl;
        this->SliceNode2->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 2);
        this->Logic2->UpdatePipeline ();
    }
    else if ( this->SliceDriver2 == vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE )
    {
        if (this->NeedRealtimeImageUpdate2)
        {
          //            std::cerr << "vtkBrpNavGUI::UpdateSliceDisplay() : In-plane: SLICE_DRIVER_RTIMAGE" << std::endl;
            this->SliceNode2->SetSliceToRASByNTP( nx, ny, nz, tx, ty, tz, px, py, pz, 2);
            this->Logic2->UpdatePipeline ();
        }
    }
}


void vtkBrpNavGUI::ChangeSlicePlaneDriver(int slice, const char* driver)
{
  std::cerr << "ChangeSlicePlaneDriver -- Slice: " << slice << ", Driver: " << driver << std::endl;

    if ( slice == vtkBrpNavGUI::SLICE_PLANE_RED )
    {
        this->RedSliceMenu->SetValue(driver);
        if ( strcmp(driver, "User") == 0 )
        {
            this->SliceNode0->SetOrientationToAxial();
            this->SliceDriver0 = vtkBrpNavGUI::SLICE_DRIVER_USER;
        }
        else if ( strcmp(driver, "Locator") == 0 )
        {
            this->SliceDriver0 = vtkBrpNavGUI::SLICE_DRIVER_LOCATOR;
        }
        else if ( strcmp(driver, "RT Image") == 0 )
        {
            this->SliceDriver0 = vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE;
        }
    }
    else if ( slice == vtkBrpNavGUI::SLICE_PLANE_YELLOW )
    {
        this->YellowSliceMenu->SetValue(driver);
        if ( strcmp(driver, "User") == 0 )
        {
            this->SliceNode1->SetOrientationToSagittal();
            this->SliceDriver1 = vtkBrpNavGUI::SLICE_DRIVER_USER;
        }
        else if ( strcmp(driver, "Locator") == 0 )
        {
            this->SliceDriver1 = vtkBrpNavGUI::SLICE_DRIVER_LOCATOR;
        }
        else if ( strcmp(driver, "RT Image") == 0 )
        {
            this->SliceDriver1 = vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE;
        }
    }
    else //if ( slice == vtkBrpNavGUI::SLICE_PLANE_GREEN )
    {
        this->GreenSliceMenu->SetValue(driver);
        if ( strcmp(driver, "User") == 0 )
        {
            this->SliceNode2->SetOrientationToCoronal();
            this->SliceDriver2 = vtkBrpNavGUI::SLICE_DRIVER_USER;
        }
        else if ( strcmp(driver, "Locator") == 0 )
        {
            this->SliceDriver2 = vtkBrpNavGUI::SLICE_DRIVER_LOCATOR;
        }
        else if ( strcmp(driver, "RT Image") == 0 )
        {
            this->SliceDriver2 = vtkBrpNavGUI::SLICE_DRIVER_RTIMAGE;
        }
    }
}




////////////////////////////////////////////////////////////////////
////////////////if for Checkbutton to XML+Coordinates send//////////////////////
///////////////////////////////////////////////////////////////////


#ifdef USE_NAVITRACK
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
        
      //int sp = atoi(this->UpdateRateEntry->GetWidget()->GetValue());
      int sp = 100;
      //float multi = atof(this->MultiFactorEntry->GetWidget()->GetValue());
        float multi = 1.0;
      this->OpenTrackerStream->SetSpeed(sp);
      this->OpenTrackerStream->SetMultiFactor(multi);
      this->OpenTrackerStream->SetStartTimer(1);
      this->OpenTrackerStream->ProcessTimerEvents();    
       
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
        
      
        if (checkedWorkPhaseStartUpButton)
        {                
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_START_UP;
        }
      
        if (checkedWorkPhasePlanningButton)
        {
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_PLANNING;
        }
        
        if (checkedWorkPhaseCalibarationButton)
        {
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_CALIBRATION; 
        }
        
        if (checkedWorkPhaseTargetingButton)
        {
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_TARGETING;
        }
      
        if (checkedWorkPhaseManualButton)
        {
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_MANUAL; 
        }
      
        if (checkedWorkPhaseEmergencyButton)
        {
            filtercommandkeys[0] = "workphase";
            filtercommandvalues[0] = BRPTPR_EMERGENCY;
        }
      
        this->OpenTrackerStream->SetOpenTrackerforBRPDataFlowValveFilter(filtercommandkeys, filtercommandvalues);
      
    }
  
    //08/02/2007 Philip Mewes TCL timer for resending Workphase command
    //defined in the Silcer<->robot Handshake Protokol. This is also used for
    //coordinates and orientation sending protokol
  
    if((received_robot_status==BRPTPR_Ready && checkedWorkPhaseStartUpButton) ||
       (received_robot_status==BRPTPR_Uncalibrated && checkedWorkPhaseCalibarationButton))
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
      
        if (checkedsendstartScanner)
        {
            scancommandkeys.resize(1);
            scancommandkeys[0] = "mrctrl_cmd";
            scancommandvalues.resize(1);
            scancommandvalues[0] = "START_SCAN";
        }

       if (checkedsendstopScanner)
       {
           scancommandkeys.resize(1);
           scancommandkeys[0] = "mrctrl_cmd";
           scancommandvalues.resize(1);
           scancommandvalues[0] = "STOP_SCAN"; 
       }

       if (checkedsendprepScanner)
       {
           scancommandkeys.resize(1);
           scancommandkeys[0] = "mrctrl_cmd";
           scancommandvalues.resize(1);
           scancommandvalues[0] = "PREP_SCAN"; 
       }
       
       if (checkedsendpauseScanner)
       {
           scancommandkeys.resize(1);
           scancommandkeys[0] = "mrctrl_cmd";
           scancommandvalues.resize(1);
           scancommandvalues[0] = "PAUSE_SCAN";
       }

       if (checkedsendresumeScanner)
       {
           scancommandkeys.resize(1);
           scancommandkeys[0] = "mrctrl_cmd";
           scancommandvalues.resize(1);
           scancommandvalues[0] = "RESUME_SCAN"; 
       }
       
       if (checkedsendnewexam)
       {
           scancommandkeys.resize(4);
           scancommandvalues.resize(4);
           
           scancommandkeys[0] = "mrctrl_cmd";
           scancommandkeys[1] = "patient_id";
           scancommandkeys[2] = "patient_name";
           scancommandkeys[3] = "patient_weight";
           
           scancommandvalues[0] = "NEW_EXAM";
           scancommandvalues[1] = this->positionbrppatientid->GetWidget()->GetValue ();
           scancommandvalues[2] = this->positionbrppatientname->GetWidget()->GetValue ();      
           scancommandvalues[3] = this->positionbrppatientweight->GetWidget()->GetValue ();
       }

       if (checkedsendsetprotocol)
       {
           scancommandkeys.resize(2);
           scancommandvalues.resize(2);
           
           scancommandkeys[0]= "mrctrl_cmd";
           scancommandkeys[1]= "protocol_name";
           
           scancommandvalues[0] = "LOAD_PROTOCOL"; 
           scancommandvalues[1]= this->positionbrpsetprotocol->GetWidget()->GetValue ();
       }

       std::cerr << "Sending Scanner controlling command:" <<std::endl;
       for (int i = 0; i < scancommandkeys.size(); i ++)
       {
           std::cerr << "    " << scancommandkeys[i] << std::endl;
           std::cerr << "    " << scancommandvalues[i] << std::endl;
       }
       
       this->OpenTrackerStream->SetOpenTrackerforScannerControll(scancommandkeys, scancommandvalues);
       
    }
}


#endif

vtkMRMLVolumeNode* vtkBrpNavGUI::AddVolumeNode(vtkSlicerVolumesLogic* volLogic,
                                               const char* volumeNodeName)
{

    std::cerr << "AddVolumeNode(): called." << std::endl;

    vtkMRMLVolumeNode *volumeNode = NULL;

    if (volumeNode == NULL)  // if real-time volume node has not been created
    {

        vtkMRMLVolumeDisplayNode *displayNode = NULL;
        vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();
        vtkImageData* image = vtkImageData::New();

        float fov = 300.0;
        image->SetDimensions(256, 256, 1);
        image->SetExtent(0, 255, 0, 255, 0, 0 );
        image->SetSpacing( fov/256, fov/256, 10 );
        //image->SetNumberOfScalarComponents( 1 );
        //image->SetOrigin( -127.5, -127.5, 0.5 );
        //image->SetOrigin( -fov/2, -fov/2, -5.0 );
        image->SetOrigin( -fov/2, -fov/2, -0.0 );
        //image->SetOrigin( 0.0, 0.0, 0.5 );
        image->SetScalarTypeToShort();
        image->AllocateScalars();
        
        short* dest = (short*) image->GetScalarPointer();
        if (dest)
        {
          memset(dest, 0x00, 256*256*sizeof(short));
          image->Update();
        }
        
        /*
        vtkSlicerSliceLayerLogic *reslice = vtkSlicerSliceLayerLogic::New();
        reslice->SetUseReslice(0);
        */
        scalarNode->SetAndObserveImageData(image);

        
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
            //colorLogic->Delete();
            
            volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());
            
            vtkDebugMacro("Name vol node "<<volumeNode->GetClassName());
            vtkDebugMacro("Display node "<<displayNode->GetClassName());
            
            volLogic->GetMRMLScene()->AddNode(volumeNode);
            vtkDebugMacro("Node added to scene");
            
            volLogic->SetActiveVolumeNode(volumeNode);
            volLogic->Modified();
        }

        //scalarNode->Delete();
        
        /*
        if (displayNode)
        {
            displayNode->Delete();
        }
        */

    }
    return volumeNode;
}



Image* vtkBrpNavGUI::DicomRead(const char* filename, int* width, int* height,
                 std::vector<float>& position, std::vector<float>& orientation)
{
  position.resize(3, 0.0);
  orientation.resize(4, 0.0);

  const   unsigned int   Dimension = 2;
  typedef unsigned short InputPixelType;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);

  typedef itk::GDCMImageIO           ImageIOType;
  ImageIOType::Pointer gdcmImageIO = ImageIOType::New();
  reader->SetImageIO( gdcmImageIO );

  try {
    reader->Update();
  } catch (itk::ExceptionObject & e) {
    std::cerr << "exception in file reader " << std::endl;
    std::cerr << e.GetDescription() << std::endl;
    std::cerr << e.GetLocation() << std::endl;
    return NULL;
  }

  char name[100];
  gdcmImageIO->GetPatientName(name);
  std::cerr << name << std::endl;

  double origin[3];
  double center[3];
  int    size[3];
  double spacing[3];

  for (int i = 0; i < 3;i ++) {
    origin[i]  = gdcmImageIO->GetOrigin(i);
    size[i]    = gdcmImageIO->GetDimensions(i);
    spacing[i] = gdcmImageIO->GetSpacing(i);
  }

  float imageDir[3][3];
  for (int i = 0; i < 3; i ++) {
    std::vector<double> v;
    v = gdcmImageIO->GetDirection(i);
    imageDir[i][0] = v[0];
    imageDir[i][1] = v[1];
    imageDir[i][2] = v[2];
  }

  // LPS to RAS
  origin[0] *= -1.0;
  origin[1] *= -1.0;
  imageDir[0][0] *= -1.0;
  imageDir[0][1] *= -1.0;
  imageDir[0][2] *= -1.0;
  imageDir[1][0] *= -1.0;
  imageDir[1][1] *= -1.0;
  imageDir[1][2] *= -1.0;

  std::cerr << "DICOM IMAGE:" << std::endl;
  std::cerr << " Dimension = ( "
            << size[0] << ", " << size[1] << ", " << size[2] << " )" << std::endl;
  std::cerr << " Origin    = ( "
            << origin[0] << ", " << origin[1] << ", " << origin[2] << " )" << std::endl;
  std::cerr << " Spacing   = ( "
            << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << " )" << std::endl;

  std::cerr << " Orientation: " << std::endl;
  std::cerr << "   " << imageDir[0][0] << ", " << imageDir[0][1] << ", " 
            << imageDir[0][2] << std::endl;
  std::cerr << "   " << imageDir[1][0] << ", " << imageDir[1][1] << ", "
            << imageDir[1][2] << std::endl;
  std::cerr << "   " << imageDir[2][0] << ", " << imageDir[2][1] << ", "
            << imageDir[2][2] << std::endl;

  InputImageType::Pointer    inputImage = reader->GetOutput();
  InputImageType::RegionType region   = inputImage->GetLargestPossibleRegion();


  // position is the center of the image
  double coffset[3];
  for (int i = 0; i < 3; i ++) {
    coffset[i] = ((size[i]-1)*spacing[i])/2.0;
  }

  for (int i = 0; i < 3; i ++) {
    position[i] = origin[i] + (coffset[0]*imageDir[i][0] + coffset[1]*imageDir[i][1]
                               + coffset[2]*imageDir[i][2]);
  }
  std::cerr << " Center   =  ( "
            << position[0] << ", " << position[1] << ", " << position[2] << " )" << std::endl;


  float matrix[3][3];
  float quat[4];
  MathUtils::matrixToQuaternion(imageDir, quat);
  for (int i = 0; i < 4; i ++) {
    orientation[i] = quat[i];
  }


  int w = size[0];
  int h = size[1];

  short* data = new short[w*h];
  InputImageType::IndexType index;

  for (int j = 0; j < h; j ++) {
    index[1] = j;
    for (int i = 0; i < w; i ++) {
      index[0] = w-i;
      data[j*w+i] = (short) inputImage->GetPixel(index);
    }
  }

  *width = w;
  *height = h;
  Image* img = new Image(size[0], size[1], sizeof(short), (void*)data);



  return img;

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
