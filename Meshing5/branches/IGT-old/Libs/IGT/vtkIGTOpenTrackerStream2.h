 // .NAME vtkIGTOpenTrackerStream2 - Central registry to provide control and I/O for
//  trackers and imagers
// .SECTION Description
// vtkIGTOpenTrackerStream2 registers arbitary number of trackers and imagers, created MRML nodes in the MRML secene. Designed and Coded by Nobuhiko Hata and Haiying Liu, Jan 12, 2007 @ NA-MIC All Hands Meeting, Salt Lake City, UT

#ifndef IGTOPENTRACKERSTREAM2_H
#define IGTOPENTRACKERSTREAM2_H


#include <string>
#include <vector>

#include "vtkIGTWin32Header.h" 
#include "vtkObject.h"
#include "vtkIGTDataStream.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"

#include "vtkImageData.h"


#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
#include <OpenTracker/types/Image.h>

using namespace ot;


class VTK_IGT_EXPORT vtkIGTOpenTrackerStream2 : public vtkIGTDataStream
{
public:


    static vtkIGTOpenTrackerStream2 *New();
    vtkTypeRevisionMacro(vtkIGTOpenTrackerStream2,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);

    vtkSetMacro(Speed,int);
    vtkSetMacro(MultiFactor,float);

    vtkSetMacro(StartTimer,int);

    vtkSetObjectMacro(RegMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(RegMatrix,vtkMatrix4x4);

   

     vtkGetObjectMacro(LocatorMatrix,vtkMatrix4x4);
   
     vtkSetMacro(position_cb2_FS0,float);
     vtkSetMacro(position_cb2_FS1,float);
     vtkSetMacro(position_cb2_FS2,float);

     vtkSetMacro(orientation_cb2_FS0,float);
     vtkSetMacro(orientation_cb2_FS1,float);
     vtkSetMacro(orientation_cb2_FS2,float);
     vtkSetMacro(orientation_cb2_FS3,float);
     
     
     // vtkSetMacro(robot_Status, char);
     

    vtkSetMacro(RealtimeImageX,int);
    vtkSetMacro(RealtimeImageY,int);
    // vtkSetMacro(RealtimeImageData,Image);
   
    vtkGetObjectMacro(LocatorNormalTransform,vtkTransform);
       
    /**
     * Constructor
     **/
    vtkIGTOpenTrackerStream2();


    //Description:
    //Destructor

    virtual ~vtkIGTOpenTrackerStream2 ( );
  

    void Init(const char *configFile);
    void StopPolling();
    void PollRealtime();
    void SetLocatorTransforms();
   

    void ProcessTimerEvents();

    
     
    static void callbackF(Node&, Event&, void *data);
    static void callbackF_cb2(Node&, Event&, void *data_cb2);
    //BTX
    void SetTracker(std::vector<float> pos,std::vector<float> quat);
    void SetOpenTrackerforScannerControll(std::vector<std::string> scancommandkeys, std::vector<std::string> scancommandvalue);
    void SetOpenTrackerforBRPDataFlowValveFilter(std::vector<std::string> filtercommandkeys, std::vector<std::string> filtercommandvalue);
    void SetOrientationforRobot(float xsendrobotcoords, float ysendrobotcoords, float zsendrobotcoords, std::vector<float> sendrobotcoordsvector, std::string robotcommandvalue,std::string robotcommandkey);
    void GetRealtimeImage(int*, vtkImageData* image);
    void GetDevicesStatus(std::string& received_robot_status,std::string& received_scanner_status, std::string& received_error_status);
    void SetZFrameTrackingData(Image*, int, int, std::vector<float>, std::vector<float>);

    //ETX
    
    //BTX
    
    void GetCoordsOrientforScanner(float* OrientationForScanner0,float* OrientationForScanner1, float* OrientationForScanner2, float* OrientationForScanner3, float* PositionForScanner0, float* PositionForScanner1, float* PositionForScanner2);
    //ETX

private:

    int Speed;
    int StartTimer;
    float MultiFactor;

    vtkMatrix4x4 *LocatorMatrix;
    vtkMatrix4x4 *LocatorMatrix_cb2;
    vtkMatrix4x4 *RegMatrix;
    vtkMatrix4x4 *RegMatrix_cb2;
    vtkTransform *LocatorNormalTransform;
    vtkTransform *LocatorNormalTransform_cb2;
                                          
    Context *context;
    
    float position_cb2_FS0;
    float position_cb2_FS1;
    float position_cb2_FS2;

    float orientation_cb2_FS0;
    float orientation_cb2_FS1;
    float orientation_cb2_FS2;
    float orientation_cb2_FS3;

    float needle_tip_cb2_FS0;
    float needle_tip_cb2_FS1;
    float needle_tip_cb2_FS2;
    

    //BTX
    std::string robot_Status;
    std::string robot_message;
    std::vector<float> needle_depth;
    //ETX

   
    
    float position_cb2[3];
    float orientation_cb2[4];
    

    Image RealtimeImageData;
    int   RealtimeImageSerial;
    int   RealtimeImageX;                // (pixel)
    int   RealtimeImageY;                // (pixel)
    float RealtimeImageFov;              // (mm)
    float RealtimeImageSlthick;          // (mm)


    void Normalize(float *a);
    void Cross(float *a, float *b, float *c);
    void ApplyTransform(float *position, float *norm, float *transnorm);
    void CloseConnection();

    int  quaternion2xyz(float* orientation, float *normal, float *transnormal); 

    
    
};

#endif // IGTOPENTRACKERSTREAM2_H

