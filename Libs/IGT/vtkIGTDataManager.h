// .NAME vtkIGTDataManager - Central registry to provide control and I/O for
//  trackers and imagers
// .SECTION Description
// vtkIGTDataManager registers arbitary number of trackers and imagers, created MRML nodes in the MRML secene. Designed and Coded by Nobuhiko Hata and Haiying Liu, Jan 12, 2007 @ NA-MIC All Hands Meeting, Salt Lake City, UT

#ifndef IGTDATAMANAGER_H
#define IGTDATAMANAGER_H


#include <string>
#include <vector>

#include "vtkIGTWin32Header.h" 
#include "vtkObject.h"
#include "vtkMRMLScene.h"
#include "vtkIGTMatrixState.h"
#include "vtkIGTDataStream.h"

#include "vtkMatrix4x4.h"
#include "vtkTransform.h"

#ifdef USE_OPENTRACKER
#ifdef OT_VERSION_20
#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
#endif
#ifdef OT_VERSION_13
#include "OpenTracker.h"
#include "common/CallbackModule.h"
#endif

using namespace ot;
#endif


class VTK_IGT_EXPORT vtkIGTDataManager : public vtkObject
{
public:


    static vtkIGTDataManager *New();
    vtkTypeRevisionMacro(vtkIGTDataManager,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);

    vtkSetMacro(Speed,int);
    vtkSetMacro(StartTimer,int);

    vtkSetStringMacro(Device);


    //Description:
    // Set MRML scene from the Slicer main routine
    vtkSetObjectMacro(MRMLScene,vtkMRMLScene);

    //Description:
    // Get MRML scene stored
    vtkGetObjectMacro(MRMLScene,vtkMRMLScene);

    vtkSetObjectMacro(RegMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(RegMatrix,vtkMatrix4x4);

    vtkGetObjectMacro(LocatorMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(LocatorNormalTransform,vtkTransform);

    /**
     * Constructor
     @ param buffersize: size of buufer (
     */
    vtkIGTDataManager();


    //Description:
    //Destructor
    virtual ~vtkIGTDataManager ( );

    //Description:
    //
    char *GetMRMLModelId(int index);

    //Descripton:
    //Reigster stream device (trackers and imagers). Parameters: int stram type (IGT_IMAGE_STREAM, IGT_MATRIX_DEVICE) defined in vtkIGTDataStream, vtkIGTDataStream
    void RegisterStreamDevice (int streamType, vtkIGTDataStream* datastream);

    void Init(char *configFile);
    void StopPolling();
    void PollRealtime();
    void SetLocatorTransforms();
    void ProcessTimerEvents();

#ifdef USE_OPENTRACKER
    static void callbackF(const Node&, const Event &event, void *data);
#endif

#ifdef USE_IGSTK
    void callbackF(double*, double*);
#endif



private:

    int Speed;
    int StartTimer;
    float Ratio;

    //Description.
    //MRML scene passed from the Module instantiating this class
    vtkMRMLScene *MRMLScene;
    vtkMatrix4x4 *LocatorMatrix;
    vtkMatrix4x4 *RegMatrix;
    vtkTransform *LocatorNormalTransform;
    char *Device;


//BTX
    std::vector<vtkIGTDataStream *> RegisteredDataStreams;
    std::vector<int> StreamTypes;  // matrix or image
    std::vector<char *> MRMLModelIds;  
//ETX

#ifdef USE_OPENTRACKER
    Context *context;
#endif

    void CreateMRMLNode(int streamType);
    void Normalize(float *a);
    void Cross(float *a, float *b, float *c);
    void ApplyTransform(float *position, float *norm, float *transnorm);
    void CloseConnection();

    void quaternion2xyz(float* orientation, float *normal, float *transnormal); 


};


#endif // IGTDATAMANAGER_H
