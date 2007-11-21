/*=auto=========================================================================

  Portions (c) Copyright 2007 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkProstateNavLogic.h"

#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLLinearTransformNode.h"
#include "vtkSlicerColorLogic.h"

// for DICOM read
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkGDCMImageIO.h"
#include "itkSpatialOrientationAdapter.h"

// for communication with robot and scanner
#include "BRPTPRInterface.h"

vtkCxxRevisionMacro(vtkProstateNavLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkProstateNavLogic);

//---------------------------------------------------------------------------
const int vtkProstateNavLogic::PhaseTransitionMatrix[vtkProstateNavLogic::NumPhases][vtkProstateNavLogic::NumPhases] =
  {
               /*     next workphase     */
      /*    */ /* St  Pl  Cl  Tg  Mn  Em */
      /* St */ {  1,  1,  0,  0,  0,  1  },
      /* Pl */ {  1,  1,  1,  0,  0,  1  },
      /* Cl */ {  1,  1,  1,  1,  1,  1  },
      /* Tg */ {  1,  1,  1,  1,  1,  1  },
      /* Mn */ {  1,  1,  1,  1,  1,  1  },
      /* Em */ {  1,  1,  1,  1,  1,  1  },
  };

//---------------------------------------------------------------------------
vtkProstateNavLogic::vtkProstateNavLogic()
{
#ifndef IGSTK_OFF
    igstk::RealTimeClock::Initialize();
#endif

    this->CurrentPhase         = StartUp;
    this->PrevPhase            = StartUp;
    this->PhaseComplete        = false;
    this->PhaseTransitionCheck = true;
    this->OrientationUpdate    = false;

    this->RealtimeImageSerial = 0;
    this->RealtimeImageOrient = vtkProstateNavLogic::SLICE_RTIMAGE_PERP;

//    this->NeedRealtimeImageUpdate0 = 0;
//    this->NeedRealtimeImageUpdate1 = 0;
//    this->NeedRealtimeImageUpdate2 = 0;

#ifdef USE_NAVITRACK
    this->OpenTrackerStream = vtkIGTOpenTrackerStream::New();

    this->RealtimeVolumeNode = NULL;
#endif

    // Timer Handling

    this->DataCallbackCommand = vtkCallbackCommand::New();
    this->DataCallbackCommand->SetClientData( reinterpret_cast<void *> (this) );
    this->DataCallbackCommand->SetCallback(vtkProstateNavLogic::DataCallback);
#ifdef USE_NAVITRACK
    //this->OpenTrackerStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
    this->OpenTrackerStream->AddObserver(vtkCommand::ModifiedEvent, this->DataCallbackCommand);
#endif 

}


//---------------------------------------------------------------------------
vtkProstateNavLogic::~vtkProstateNavLogic()
{

    if (this->DataCallbackCommand)
    {
        this->DataCallbackCommand->Delete();
    }

#ifdef USE_NAVITRACK
    if (this->OpenTrackerStream)
    {
        this->OpenTrackerStream->RemoveObservers( vtkCommand::ModifiedEvent, this->DataCallbackCommand );
        this->OpenTrackerStream->Delete();
    }
#endif

}

//---------------------------------------------------------------------------
void vtkProstateNavLogic::PrintSelf(ostream& os, vtkIndent indent)
{
    this->vtkObject::PrintSelf(os, indent);

    os << indent << "vtkProstateNavLogic:             " << this->GetClassName() << "\n";

}

//---------------------------------------------------------------------------
void vtkProstateNavLogic::DataCallback(vtkObject *caller, 
                                       unsigned long eid, void *clientData, void *callData)
{
    vtkProstateNavLogic *self = reinterpret_cast<vtkProstateNavLogic *>(clientData);
    vtkDebugWithObjectMacro(self, "In vtkProstateNavLogic DataCallback");
    self->UpdateAll();
}

//---------------------------------------------------------------------------
void vtkProstateNavLogic::UpdateAll()
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
        px = 0.0;  py = 0.0;  pz = 0.0;
        nx = 0.0;  ny = 0.0;  nz = 1.0;
        tx = 1.0;  ty = 0.0;  tz = 0.0;
        sx = 0.0;  sy = 1.0;  sz = 0.0;
    }

    //std::cerr << "==== Locator position ====" << std::endl;
    //std::cerr << "  (px, py, pz) =  ( " << px << ", " << py << ", " << pz << " )" << std::endl;
    //std::cerr << "  (nx, ny, nz) =  ( " << nx << ", " << ny << ", " << nz << " )" << std::endl;
    //std::cerr << "  (tx, ty, tz) =  ( " << tx << ", " << ty << ", " << tz << " )" << std::endl;

    //Philip Mewes 17.07.2007: defining and sending te workphase (WP) commands depending of requestet WP
    // received_robot_status = NULL;
    
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

        if (vid && this->OrientationUpdate)
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
                
                if (rtimgslice == vtkProstateNavLogic::SLICE_RTIMAGE_PERP)  /* Perpendicular */
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
                else if (rtimgslice == vtkProstateNavLogic::SLICE_RTIMAGE_INPLANE90)  /* In-plane 90 */
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

        if (this->ImagingControl)
        {
            std::vector<float> pos;
            std::vector<float> quat;
            pos.resize(3);
            quat.resize(4);

            float scanTrans[3][3];  // Rotation matrix from axial plane to scan plane
            
            /* Parpendicular */
            if (rtimgslice == vtkProstateNavLogic::SLICE_RTIMAGE_PERP)
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
            else if (rtimgslice == vtkProstateNavLogic::SLICE_RTIMAGE_INPLANE90)
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
            else // if (rtimgslice == vtkProstateNavLogic::SLICE_RTIMAGE_INPLANE)
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
        if (this->UpdateLocator)
          {
            vtkTransform *transform = NULL;
            vtkTransform *transform_cb2 = NULL;

            this->OpenTrackerStream->SetLocatorTransforms();
            transform = this->OpenTrackerStream->GetLocatorNormalTransform();
            transform_cb2 = this->OpenTrackerStream->GetLocatorNormalTransform();
            //this->GUI->UpdateLocator(transform, transform_cb2);  // MOVE TO GUI

          }

        //if (!this->GUI->FreezeOrientationUpdate)
        if (this->OrientationUpdate)
        {
          //this->GUI->UpdateSliceDisplay(nx, ny, nz, tx, ty, tz, px, py, pz);  //MOVE TO GUI
        }
    }

    this->NeedRealtimeImageUpdate0 = 0;
    this->NeedRealtimeImageUpdate1 = 0;
    this->NeedRealtimeImageUpdate2 = 0;

}

//---------------------------------------------------------------------------
void vtkProstateNavLogic::AddRealtimeVolumeNode(vtkSlicerVolumesLogic* volLogic, const char* name)
{
    if (this->RealtimeVolumeNode == NULL)
        this->RealtimeVolumeNode = AddVolumeNode(volLogic, name);
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::SwitchWorkPhase(int newwp)
{
    if (IsPhaseTransitable(newwp))
    {
        PrevPhase     = CurrentPhase;
        CurrentPhase  = newwp;
        PhaseComplete = false;

        return 1;
    }
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::IsPhaseTransitable(int nextwp)
{
    if (nextwp < 0 || nextwp > NumPhases)
    {
        return 0;
    }
    
    if (PhaseTransitionCheck == 0)
    {
        return 1;
    }

    if (PhaseComplete)
    {
        return PhaseTransitionMatrix[CurrentPhase][nextwp];
    }
    else
    {
        return PhaseTransitionMatrix[PrevPhase][nextwp];
    }
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::ConnectTracker(const char* filename)
{
#ifdef USE_NAVITRACK
    int   speed = 100;         // speed
    float multi = 1.0;         // mutlti factor

    this->OpenTrackerStream->Init(filename);
    this->OpenTrackerStream->SetSpeed(speed);
    this->OpenTrackerStream->SetMultiFactor(multi);
    this->OpenTrackerStream->SetStartTimer(1);
    this->OpenTrackerStream->ProcessTimerEvents();    
#endif //USE_NAVITRACK
}


//---------------------------------------------------------------------------
int vtkProstateNavLogic::DisconnectTracker()
{
#ifdef USE_NAVITRACK
    this->OpenTrackerStream->StopPolling();
#endif // USE_NAVITRACK
}


//---------------------------------------------------------------------------
int vtkProstateNavLogic::RobotStop()
{

  std::cerr << "vtkProstateNavLogic::RobotStop()" << std::endl;

#ifdef USE_NAVITRACK
#endif // USE_NAVITRACK
}


//---------------------------------------------------------------------------
int vtkProstateNavLogic::RobotMoveTo(float px, float py, float pz,
                                     float nx, float ny, float nz,
                                     float tx, float ty, float tz)
{

  std::cerr << "vtkProstateNavLogic::RobotMoveTo()" << std::endl;

#ifdef USE_NAVITRACK

  if (this->OpenTrackerStream)
    {
    // temporally, orientation set to [0, 0, 0, 1];
    std::vector<float> orientation;
    orientation.clear();
    
    orientation.push_back(0.0);
    orientation.push_back(0.0);
    orientation.push_back(0.0);
    orientation.push_back(1.0);
    
    this->OpenTrackerStream->SetOrientationforRobot(px, py, pz,
                                                    orientation,
                                                    BRPTPR_TARGET, "command");
    }

#endif // USE_NAVITRACK
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::ScanStart()
{

  std::cerr << "vtkProstateNavLogic::ScanStart()" << std::endl;

#ifdef USE_NAVITRACK
  if (this->OpenTrackerStream)
    {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.resize(1);
    keys[0] = "mrctrl_cmd";
    values.resize(1);
    values[0] = "START_SCAN";
    
    this->OpenTrackerStream->SetOpenTrackerforScannerControll(keys, values);
    }
#endif // USE_NAVITRACK
  
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::ScanPause()
{

  std::cerr << "vtkProstateNavLogic::ScanPause()" << std::endl;

#ifdef USE_NAVITRACK
  if (this->OpenTrackerStream)
    {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.resize(1);
    keys[0] = "mrctrl_cmd";
    values.resize(1);
    values[0] = "PAUSE_SCAN";
    
    this->OpenTrackerStream->SetOpenTrackerforScannerControll(keys, values);
    }
#endif // USE_NAVITRACK
  
}

//---------------------------------------------------------------------------
int vtkProstateNavLogic::ScanStop()
{

  std::cerr << "vtkProstateNavLogic::ScanStop()" << std::endl;

#ifdef USE_NAVITRACK
  if (this->OpenTrackerStream)
    {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    keys.resize(1);
    keys[0] = "mrctrl_cmd";
    values.resize(1);
    values[0] = "STOP_SCAN";
    
    this->OpenTrackerStream->SetOpenTrackerforScannerControll(keys, values);
    }
#endif // USE_NAVITRACK
  
}

//---------------------------------------------------------------------------
vtkMRMLVolumeNode* vtkProstateNavLogic::AddVolumeNode(vtkSlicerVolumesLogic* volLogic,
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
        image->SetOrigin( -fov/2, -fov/2, -0.0 );
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

//---------------------------------------------------------------------------
Image* vtkProstateNavLogic::ReadCalibrationImage(const char* filename, int* width, int* height,
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
