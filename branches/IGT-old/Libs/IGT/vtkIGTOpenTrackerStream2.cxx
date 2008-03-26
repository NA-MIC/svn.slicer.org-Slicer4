
//#define USE_ZFRAME

#include "vtkIGTOpenTrackerStream2.h"
#include "vtkObjectFactory.h"

#include "vtkKWTkUtilities.h"
#include "vtkKWApplication.h"
#include "vtkCommand.h"
#include "vtkMath.h"
#include <OpenTracker/OpenTracker.h>
#include <OpenTracker/dllinclude.h>
#include <OpenTracker/input/SlicerNTModule.h>
#include <OpenTracker/core/Configurator.h>

#ifdef USE_ZFRAME
  #include <OpenTracker/input/BRPImageIOModule.h>
#endif

#include <vtksys/SystemTools.hxx>
#include "vtkCallbackCommand.h"
#include "sys/time.h"
#include <ctime>

vtkStandardNewMacro(vtkIGTOpenTrackerStream2);
vtkCxxRevisionMacro(vtkIGTOpenTrackerStream2, "$Revision: 1.0 $");


#define IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_XRES     256    // pixel
#define IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_YRES     256    // pixel
#define IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_FOV      300    // mm
#define IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_SLTHICK  5      // mm


vtkIGTOpenTrackerStream2::vtkIGTOpenTrackerStream2()
{
    this->Speed = 0;
    this->StartTimer = 0;
    this->LocatorNormalTransform = vtkTransform::New();
    this->LocatorNormalTransform_cb2 = vtkTransform::New();
    this->LocatorMatrix = vtkMatrix4x4::New();
    this->LocatorMatrix_cb2 = vtkMatrix4x4::New();// Identity
    /*
    this->LocatorNormalTransform = vtkTransform::New();
    this->LocatorMatrix = vtkMatrix4x4::New(); // Identity
    */
    this->RealtimeImageX       = IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_XRES;
    this->RealtimeImageY       = IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_YRES;
    this->RealtimeImageFov     = IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_FOV;
    this->RealtimeImageSlthick = IGT_OPENTRACKERSTREAM_DEFAULT_RTIMAGE_SLTHICK;
    this->RealtimeImageSerial  = 0;

    this->position_cb2_FS0= 0;
    this->position_cb2_FS1= 0;
    this->position_cb2_FS2= 0;

    //this->RealtimeImageData = Image::Image();
    this->RegMatrix = NULL;
    this->RegMatrix_cb2 = NULL;
    this->context = NULL;

}


vtkIGTOpenTrackerStream2::~vtkIGTOpenTrackerStream2()
{

    this->LocatorNormalTransform->Delete();
    this->LocatorMatrix->Delete();

    this->LocatorNormalTransform_cb2->Delete();
    this->LocatorMatrix_cb2->Delete();

    if (this->context)
    {
        delete this->context;
    }

}

void vtkIGTOpenTrackerStream2::Init(const char *configFile)
{
    fprintf(stderr,"config file: %s\n",configFile);

    //OT_REGISTER_MODULE(SlicerNTModule,NULL);

    addSPLModules();

    this->context = new Context(1); 
    // get callback module from the context
    CallbackModule * callbackMod = (CallbackModule *)context->getModule("CallbackConfig");
    

    // parse the configuration file
    context->parseConfiguration(configFile);  
    context->start();

    // sets the callback function
    callbackMod->setCallback( "cb1", (OTCallbackFunction*)&callbackF ,this);
    callbackMod->setCallback( "cb2", (OTCallbackFunction*)&callbackF_cb2 ,this);

}
//this second callbackF is for receiving Orientation and Position data from a Robot with a Needle device, philip


void vtkIGTOpenTrackerStream2::callbackF_cb2(Node&, Event &event, void *data_cb2)
{

   
    float position[3];
    float needleposition_cb2[3];
    float orientation_cb2[4];
    
    float norm_cb2[3];
    float transnorm_cb2[3];
    int   j;


    float position_cb2_FS;
    float orientation_cb2_FS[4];
    
    vtkIGTOpenTrackerStream2 *VOT_cb2 =(vtkIGTOpenTrackerStream2 *)data_cb2;



    // the original values are in the unit of meters

    position[0]=(float)(event.getPosition())[0] * VOT_cb2->MultiFactor; 
    position[1]=(float)(event.getPosition())[1] * VOT_cb2->MultiFactor;
    position[2]=(float)(event.getPosition())[2] * VOT_cb2->MultiFactor;

   
    orientation_cb2[0]=(float)(event.getOrientation())[0];
    orientation_cb2[1]=(float)(event.getOrientation())[1];
    orientation_cb2[2]=(float)(event.getOrientation())[2];
    orientation_cb2[3]=(float)(event.getOrientation())[3];
    
    VOT_cb2->position_cb2_FS0=(float)(event.getPosition())[0];
    VOT_cb2->position_cb2_FS1=(float)(event.getPosition())[1];
    VOT_cb2->position_cb2_FS2=(float)(event.getPosition())[2];

    VOT_cb2->orientation_cb2_FS0=(float)(event.getOrientation())[0];
    VOT_cb2->orientation_cb2_FS1=(float)(event.getOrientation())[1];
    VOT_cb2->orientation_cb2_FS2=(float)(event.getOrientation())[2];
    VOT_cb2->orientation_cb2_FS3=(float)(event.getOrientation())[3];
    

    //robot status    
    //   cout <<"event in NT"<<endl;
    if (event.hasAttribute("status"))
    {
        VOT_cb2->robot_Status = (std::string)event.getAttribute<std::string>("status","");
    }

    //Philip Mewes: 17.07.2007 This gets in addition to the Status Msg a Msg
    //which is precising the type of error
    if (event.hasAttribute("message"))
    {
        VOT_cb2->robot_message = (std::string)event.getAttribute<std::string>("message","");
    }

    VOT_cb2->needle_depth.resize(3, 0.0);
    if (event.hasAttribute("depth")) {
      VOT_cb2->needle_depth = (std::vector<float>)event.getAttribute <std::vector<float> >("depth", VOT_cb2->needle_depth);
    }

    
    if (VOT_cb2->quaternion2xyz(orientation_cb2, norm_cb2, transnorm_cb2) == NULL)
      {
        std::cerr << "vtkIGTOpenTrackerStream2 -- WARNING: Invalid quaternion received." << endl;
        std::cerr << " ORIENTATION = ( " 
                  << orientation_cb2[0] << ", "
                  << orientation_cb2[1] << ", "
                  << orientation_cb2[2] << ", "
                  << orientation_cb2[3] << ") " << std::endl;
        return;
      }

    // Apply the transform matrix 
    // to the postion, norm and transnorm
    if (VOT_cb2->RegMatrix)
        VOT_cb2->ApplyTransform(position, norm_cb2, transnorm_cb2);

    for (j=0; j<3; j++) {
        VOT_cb2->LocatorMatrix->SetElement(j,0,position[j]);
    }


    for (j=0; j<3; j++) {
        VOT_cb2->LocatorMatrix->SetElement(j,1,norm_cb2[j]);
    }

    for (j=0; j<3; j++) {
        VOT_cb2->LocatorMatrix->SetElement(j,2,transnorm_cb2[j]);
    }

    for (j=0; j<3; j++) {
      VOT_cb2->LocatorMatrix->SetElement(j,3,0);
    }

    for (j=0; j<3; j++) {
        VOT_cb2->LocatorMatrix->SetElement(3,j,0);
    }

    VOT_cb2->LocatorMatrix->SetElement(3,3,1);

    // get a 3x3 matrix from the quaterion
    float transform_matrix[3][3];
    vtkMath::QuaternionToMatrix3x3(orientation_cb2, transform_matrix);
    
    // get the "needle depth" vector(3,1) and multiply it by the robot orientation,
    // this will give the offsets in Slicer coordinates
    float needle_offset[3];
    for (j=0; j<3; j++) {
      needle_offset[j] = VOT_cb2->needle_depth[j];
    }
    // multiply the vector in-place
    vtkMath::Multiply3x3(transform_matrix, needle_offset, needle_offset);

    // add the needle offset to the robot position to get the needle top position
    VOT_cb2->needle_tip_cb2_FS0 = VOT_cb2->position_cb2_FS0 + needle_offset[0];
    VOT_cb2->needle_tip_cb2_FS1 = VOT_cb2->position_cb2_FS1 + needle_offset[1];
    VOT_cb2->needle_tip_cb2_FS2 = VOT_cb2->position_cb2_FS2 + needle_offset[2];

    // to make things simple, replace the robot position by the needle tip
    //  in the LocatorMatrix
    VOT_cb2->LocatorMatrix->SetElement(0,0,VOT_cb2->needle_tip_cb2_FS0);
    VOT_cb2->LocatorMatrix->SetElement(1,0,VOT_cb2->needle_tip_cb2_FS1);
    VOT_cb2->LocatorMatrix->SetElement(2,0,VOT_cb2->needle_tip_cb2_FS2);

}


void vtkIGTOpenTrackerStream2::callbackF(Node&, Event &event, void *data)
{

    float position[3];
    float orientation[4];
    float norm[3];
    float transnorm[3];
    int j;    
    
    vtkIGTOpenTrackerStream2 *VOT=(vtkIGTOpenTrackerStream2 *)data;

    // the original values are in the unit of meters

    position[0]=(float)(event.getPosition())[0] * VOT->MultiFactor; 
    position[1]=(float)(event.getPosition())[1] * VOT->MultiFactor;
    position[2]=(float)(event.getPosition())[2] * VOT->MultiFactor;

    orientation[0]=(float)(event.getOrientation())[0];
    orientation[1]=(float)(event.getOrientation())[1];
    orientation[2]=(float)(event.getOrientation())[2];
    orientation[3]=(float)(event.getOrientation())[3];
    
    VOT->quaternion2xyz(orientation, norm, transnorm);
    
    if (event.hasAttribute("image"))
    {
        VOT->RealtimeImageSerial = (VOT->RealtimeImageSerial + 1) % 32768;
        VOT->RealtimeImageData=(Image)event.getAttribute((Image*)NULL,"image");
        cout << "image size is " << VOT->RealtimeImageData.size() << endl;
        if (event.hasAttribute("xdim") && event.hasAttribute("ydim"))
        {
            VOT->RealtimeImageX = (int)event.getAttribute(std::string("xdim"),0);
            VOT->RealtimeImageY = (int)event.getAttribute(std::string("ydim"),0);
        } else {
            std::cerr << "No image size information." << std::endl;
        }

        
        if (event.hasAttribute("fov"))
        {
            VOT->RealtimeImageFov = (float)event.getAttribute(std::string("fov"),0.0);
        } else {
            std::cerr << "No image FOV information." << std::endl;
        }
        
        if (event.hasAttribute("slthick"))
        {
            VOT->RealtimeImageSlthick = (float)event.getAttribute(std::string("slthick"),0.0);
        } else {
            std::cerr << "No slice thickness information." << std::endl;
        }

    }
    
    // For debug 
    std::cerr << "Image size            : " << VOT->RealtimeImageX << " x " << VOT->RealtimeImageY << std::endl;
    std::cerr << "Image position        : (" << position[0] << ", " << position[1] << ", "
                                             << position[2] << ")"  << std::endl;
    std::cerr << "Image orientation     : (" << orientation[0] << ", " << orientation[1] << ", "
                                             << orientation[2] << ", " << orientation[3] << ")" << std::endl;
    std::cerr << "Image FOV             : " << VOT->RealtimeImageFov << std::endl;
    std::cerr << "Image slice thickness : " << VOT->RealtimeImageSlthick << std::endl;
    // end debug


    // Apply the transform matrix 
    // to the postion, norm and transnorm

    /*
    if (VOT->RegMatrix)
        VOT->ApplyTransform(position, norm, transnorm);

    for (j=0; j<3; j++) {
        VOT->LocatorMatrix->SetElement(j,0,position[j]);
    }


    for (j=0; j<3; j++) {
        VOT->LocatorMatrix->SetElement(j,1,norm[j]);
    }

    for (j=0; j<3; j++) {
        VOT->LocatorMatrix->SetElement(j,2,transnorm[j]);
    }

    for (j=0; j<3; j++) {
        VOT->LocatorMatrix->SetElement(j,3,0);
    }

    for (j=0; j<3; j++) {
        VOT->LocatorMatrix->SetElement(3,j,0);
    }

    VOT->LocatorMatrix->SetElement(3,3,1);
    */
}



void vtkIGTOpenTrackerStream2::StopPolling()
{
    context->close();
}



void vtkIGTOpenTrackerStream2::PollRealtime()
{
  if (context) {
    // cout <<"PollRealtime()"<< endlq;
    context->loopOnce();
  }
}



void vtkIGTOpenTrackerStream2::PrintSelf(ostream& os, vtkIndent indent)
{


}


int vtkIGTOpenTrackerStream2::quaternion2xyz(float* orientation, float *normal, float *transnormal) 
{
    float q0, qx, qy, qz;
    q0 = orientation[3];
    qx = orientation[0];
    qy = orientation[1];
    qz = orientation[2]; 

    // normalize quaternion
    double mod = sqrt(q0*q0+qx*qx+qy*qy+qz*qz);
    if (mod < 0.0001) // if the vector is zero (invalid quaternion)
    {
        return NULL;
    }
      
    q0 = (float)( q0 / mod );
    qx = (float)( qx / mod );
    qy = (float)( qy / mod );
    qz = (float)( qz / mod );

    transnormal[0] = 1-2*qy*qy-2*qz*qz;
    transnormal[1] = 2*qx*qy+2*qz*q0;
    transnormal[2] = 2*qx*qz-2*qy*q0;

    normal[0] = 2*qx*qz+2*qy*q0;
    normal[1] = 2*qy*qz-2*qx*q0;
    normal[2] = 1-2*qx*qx-2*qy*qy;

    return 1;
}


void vtkIGTOpenTrackerStream2::SetLocatorTransforms()
{
    // Get locator matrix
    float p[3], n[3], t[3], c[3];
    p[0] = this->LocatorMatrix->GetElement(0, 0);
    p[1] = this->LocatorMatrix->GetElement(1, 0);
    p[2] = this->LocatorMatrix->GetElement(2, 0);
    n[0] = this->LocatorMatrix->GetElement(0, 1);
    n[1] = this->LocatorMatrix->GetElement(1, 1);
    n[2] = this->LocatorMatrix->GetElement(2, 1);
    t[0] = this->LocatorMatrix->GetElement(0, 2);
    t[1] = this->LocatorMatrix->GetElement(1, 2);
    t[2] = this->LocatorMatrix->GetElement(2, 2);


    // Ensure N, T orthogonal:
    //    C = N x T
    //    T = C x N
    this->Cross(c, n, t);
    this->Cross(t, c, n);

    // Ensure vectors are normalized
    this->Normalize(n);
    this->Normalize(t);
    this->Normalize(c); 


    /*
    # Find transform, N, that brings the locator coordinate frame 
    # into the scanner frame.  Then invert N to M and set it to the locator's
    # userMatrix to position the locator within the world space.
    #
    # 1.) Concatenate a translation, T, TO the origin which is (-x,-y,-z)
    #     where the locator's position is (x,y,z).
    # 2.) Concatenate the R matrix.  If the locator's reference frame has
    #     axis Ux, Uy, Uz, then Ux is the TOP ROW of R, Uy is the second, etc.
    # 3.) Translate the cylinder so its tip is at the origin instead
    #     of the center of its tube.  Call this matrix C.
    # Then: N = C*R*T, M = Inv(N)
    #
    # (See page 419 and 429 of "Computer Graphics", Hearn & Baker, 1997,
    #  ISBN 0-13-530924-7)
    # 
    # The alternative approach used here is to find the transform, M, that
    # moves the scanner coordinate frame to the locator's.  
    # 
    # 1.) Translate the cylinder so its tip is at the origin instead
    #     of the center of its tube.  Call this matrix C.
    # 2.) Concatenate the R matrix.  If the locator's reference frame has
    #     axis Ux, Uy, Uz, then Ux is the LEFT COL of R, Uy is the second,etc.
    # 3.) Concatenate a translation, T, FROM the origin which is (x,y,z)
    #     where the locator's position is (x,y,z).
    # Then: M = T*R*C
    */
    vtkMatrix4x4 *locator_matrix = vtkMatrix4x4::New();
    vtkTransform *locator_transform = vtkTransform::New();

    // Locator's offset: p[0], p[1], p[2]
    float x0 = p[0];
    float y0 = p[1];
    float z0 = p[2];


    // Locator's coordinate axis:
    // Ux = T
    float Uxx = t[0];
    float Uxy = t[1];
    float Uxz = t[2];

    // Uy = -N
    float Uyx = -n[0];
    float Uyy = -n[1];
    float Uyz = -n[2];

    // Uz = Ux x Uy
    float Uzx = Uxy*Uyz - Uyy*Uxz;
    float Uzy = Uyx*Uxz - Uxx*Uyz;
    float Uzz = Uxx*Uyy - Uyx*Uxy;

    // Ux
    locator_matrix->SetElement(0, 0, Uxx);
    locator_matrix->SetElement(1, 0, Uxy);
    locator_matrix->SetElement(2, 0, Uxz);
    locator_matrix->SetElement(3, 0, 0);
    // Uy
    locator_matrix->SetElement(0, 1, Uyx);
    locator_matrix->SetElement(1, 1, Uyy);
    locator_matrix->SetElement(2, 1, Uyz);
    locator_matrix->SetElement(3, 1, 0);
    // Uz
    locator_matrix->SetElement(0, 2, Uzx);
    locator_matrix->SetElement(1, 2, Uzy);
    locator_matrix->SetElement(2, 2, Uzz);
    locator_matrix->SetElement(3, 2, 0);
    // Bottom row
    locator_matrix->SetElement(0, 3, 0);
    locator_matrix->SetElement(1, 3, 0);
    locator_matrix->SetElement(2, 3, 0);
    locator_matrix->SetElement(3, 3, 1);

    // Set the vtkTransform to PostMultiply so a concatenated matrix, C,
    // is multiplied by the existing matrix, M: C*M (not M*C)
    locator_transform->PostMultiply();
    // M = T*R*C

    
    // NORMAL PART

    locator_transform->Identity();
    // C:
    locator_transform->Translate(0, (100 / 2.0), 0);
    // R:
    locator_transform->Concatenate(locator_matrix);
    // T:
    locator_transform->Translate(x0, y0, z0);

    this->LocatorNormalTransform->DeepCopy(locator_transform);
   

    locator_matrix->Delete();
    locator_transform->Delete();


}



void vtkIGTOpenTrackerStream2::Normalize(float *a)
{
    float d;
    d = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);

    if (d == 0.0) return;

    a[0] = a[0] / d;
    a[1] = a[1] / d;
    a[2] = a[2] / d;
}



// a = b x c
void vtkIGTOpenTrackerStream2::Cross(float *a, float *b, float *c)
{
    a[0] = b[1]*c[2] - c[1]*b[2];
    a[1] = c[0]*b[2] - b[0]*c[2];
    a[2] = b[0]*c[1] - c[0]*b[1];
}


void vtkIGTOpenTrackerStream2::ApplyTransform(float *position, float *norm, float *transnorm)
{
    // Transform position, norm and transnorm
    // ---------------------------------------------------------
    float p[4];
    float n[4];
    float tn[4];

    for (int i = 0; i < 3; i++)
    {
        p[i] = position[i];
        n[i] = norm[i];
        tn[i] = transnorm[i];
    }
    p[3] = 1;     // translation affects a poistion
    n[3] = 0;     // translation doesn't affect an orientation
    tn[3] = 0;    // translation doesn't affect an orientation

    if (this->RegMatrix)
    {
      this->RegMatrix->MultiplyPoint(p, p);    // transform a position
      this->RegMatrix->MultiplyPoint(n, n);    // transform an orientation
      this->RegMatrix->MultiplyPoint(tn, tn);  // transform an orientation
    }
      
    for (int i = 0; i < 3; i++)
    {
        position[i] = p[i];
        norm[i] = n[i];
        transnorm[i] = tn[i];
    }
}


void vtkIGTOpenTrackerStream2::ProcessTimerEvents()
{
    if (this->StartTimer)
    {   
      this->PollRealtime();
      this->InvokeEvent (vtkCommand::ModifiedEvent);
      vtkKWTkUtilities::CreateTimerHandler(vtkKWApplication::GetMainInterp(), 
                                           100, this, "ProcessTimerEvents"); 
      // RSierra 3/8/07 The integer defines the update rate.
      // On my laptop there is no differenct in performance
      // (i.e. the CPU load is minimal and approx. 10% for update of 2). Is the value equivalent to ms?        
   } 
   else
   {
     this->StopPolling();
   }
}

void vtkIGTOpenTrackerStream2::SetTracker(std::vector<float> pos,std::vector<float> quat)
{
#ifdef USE_NAVITRACK
  SlicerNTModule * module = (SlicerNTModule *)context->getModule("SlicerConfig");
  module->SetTracker(pos,quat);
#endif
  
}

void vtkIGTOpenTrackerStream2::SetOpenTrackerforScannerControll(std::vector<std::string> scancommandkeys, std::vector<std::string> scancommandvalue)
{
#ifdef USE_NAVITRACK
  
  SlicerNTModule * module = (SlicerNTModule *)context->getModule("SlicerConfig");
  
  module->SetOpenTrackerforScannerControll(scancommandkeys, scancommandvalue);
#endif
}


void vtkIGTOpenTrackerStream2::SetOpenTrackerforBRPDataFlowValveFilter(std::vector<std::string> filtercommandkeys, std::vector<std::string> filtercommandvalue)
{
#ifdef USE_NAVITRACK

  SlicerNTModule * module = (SlicerNTModule *)context->getModule("SlicerConfig");
     
  module->SetOpenTrackerforBRPDataFlowValveFilter(filtercommandkeys, filtercommandvalue);
#endif
}



void vtkIGTOpenTrackerStream2::SetOrientationforRobot(float xsendrobotcoords, float ysendrobotcoords, float zsendrobotcoords, std::vector<float> sendrobotcoordsvector, std::string robotcommandvalue,std::string robotcommandkey)
{
#ifdef USE_NAVITRACK
  cout<<"opentrackerstream";
  SlicerNTModule * module = (SlicerNTModule *)context->getModule("SlicerConfig");
  
  module->SetOrientationforRobot(xsendrobotcoords, ysendrobotcoords, zsendrobotcoords, sendrobotcoordsvector,robotcommandvalue, robotcommandkey);
#endif
}


void vtkIGTOpenTrackerStream2::GetRealtimeImage(int* serial, vtkImageData* image)
{
  //std::cerr << "Serial : " << this->RealtimeImageSerial << ", " <<  *serial << std::endl;
  //std::cerr << "(xsize, ysize) = (" << RealtimeImageX << ", " << RealtimeImageY << ")" << std::endl;
    if (*serial != this->RealtimeImageSerial)
    {
        std::cerr << "Serial : " << this->RealtimeImageSerial << ", " <<  *serial << std::endl;
        *serial = this->RealtimeImageSerial;
        if (image && RealtimeImageData.size() > 0)
        {
          float spacing[3];
          spacing[0] = RealtimeImageFov / RealtimeImageX;
          spacing[1] = RealtimeImageFov / RealtimeImageY;
          spacing[2] = RealtimeImageSlthick;
          
          image->SetDimensions(RealtimeImageX, RealtimeImageY, 1);
          //image->SetExtent( xmin, xmax, ymin, ymax, zmin, zmax );
          image->SetExtent(0, RealtimeImageX-1, 0, RealtimeImageY-1, 0, 0 );
          image->SetNumberOfScalarComponents( 1 );
          image->SetOrigin( 0, 0, 0 );
          image->SetSpacing( spacing[0], spacing[1], spacing[2] );
          image->SetScalarTypeToShort();
          image->AllocateScalars();

          short* dest = (short*) image->GetScalarPointer();
          if (dest)
          {
            memcpy(dest, RealtimeImageData.image_ptr, RealtimeImageData.size());
            image->Update();
          }
        }
    }
    else
    {
    }
}

void vtkIGTOpenTrackerStream2::GetCoordsOrientforScanner(float* Orientation0, float* Orientation1, float* Orientation2, float* Orientation3, float* Position0, float* Position1, float* Position2)
{
   
  *Orientation0 = orientation_cb2_FS0;
  *Orientation1 = orientation_cb2_FS1;
  *Orientation2 = orientation_cb2_FS2;
  *Orientation3 = orientation_cb2_FS3;
  
  *Position0 = position_cb2_FS0;
  *Position1 = position_cb2_FS1;
  *Position2 = position_cb2_FS2;
  
}


void vtkIGTOpenTrackerStream2::GetDevicesStatus(std::string& received_robot_status,std::string& received_scanner_status,std::string& received_error_status)
{
  //   cout<< "robot Status (NT):  " << robot_Status <<endl;
  // cout<< "robot Message (NT):  " << robot_message <<endl;

    received_robot_status = robot_Status;
    received_error_status = robot_message;
  //  cout<<received_robot_status<<endl;


}


void vtkIGTOpenTrackerStream2::SetZFrameTrackingData(Image* img, int w, int h,
                                                    std::vector<float> pos, std::vector<float> ori)
{
#if defined(USE_ZFRAME) && defined(USE_NAVITRACK)
    BRPImageIOModule* module = (BRPImageIOModule*) context->getModule("BRPImageIOConfig");
    if (!module) {
        std::cerr << "Failed to get BRPImageIOModule." << std::endl;
    }
    else
    {
        module->setImage(*img, w, h, pos, ori);
    }
#endif
}
