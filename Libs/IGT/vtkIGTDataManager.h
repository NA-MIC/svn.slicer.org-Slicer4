// .NAME vtkIGTDataManager - Central registry to provide control and I/O for
//  trackers and imagers
// .SECTION Description
// vtkIGTDataManager registers arbitary number of trackers and imagers, created MRML nodes in the MRML secene. Designed and Coded by Nobuhiko Hata and Haying Liu, Jan 12, 2006 @ NA-MIC All Hands Meeting, Salt Lake City, UT

#ifndef IGTDATAMANAGER_H
#define IGTDATAMANAGER_H


#include <string>
#include <vector>

#include "vtkIGTWin32Header.h" 
#include "vtkObject.h"
#include "vtkMRMLScene.h"
#include "vtkIGTMatrixState.h"
#include "vtkIGTDataStream.h"

#include "vtkCallbackCommand.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"

#include "vtkSlicerApplicationGUI.h"

#ifdef USE_OPENTRACKER
#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
using namespace ot;
#endif



class VTK_IGT_EXPORT vtkIGTDataManager : public vtkObject
{
public:


    static vtkIGTDataManager *New();
    vtkTypeRevisionMacro(vtkIGTDataManager,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);

    //Description:
    // Set MRML scene from the Slicer main routine
    vtkSetObjectMacro(MRMLScene,vtkMRMLScene);

    //Description:
    // Get MRML scene stored
    vtkGetObjectMacro(MRMLScene,vtkMRMLScene);

    vtkSetObjectMacro(RegMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(RegMatrix,vtkMatrix4x4);


    vtkSetObjectMacro(SlicerAppGUI,vtkSlicerApplicationGUI);


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
    char *GetMRMLId(int index);

    //Descripton:
    //Reigster stream device (trackers and imagers). Parameters: int stram type (IGT_IMAGE_STREAM, IGT_MATRIX_DEVICE) defined in vtkIGTDataStream, vtkIGTDataStream
    void RegisterStreamDevice (int streamType, vtkIGTDataStream* datastream);

    void Init(char *configFile);
    void StartReceivingData(int speed);
    void StopReceivingData();
    void ProcessTimerEvents();
    void PollRealtime();
    void SetKWEntry(int key, vtkKWEntry *entry);

#ifdef USE_OPENTRACKER
    static void callbackF(const Node&, const Event &event, void *data);
#endif


private:

    //Description.
    //MRML scene passed from the Module instantiating this class

    vtkMRMLScene* MRMLScene;
    int StopTimer;
    int Speed;
    vtkMatrix4x4 *LocatorMatrix;
    vtkMatrix4x4 *RegMatrix;
    vtkTransform *LocatorNormalTransform;


    vtkKWEntry *NREntry;
    vtkKWEntry *NAEntry;
    vtkKWEntry *NSEntry;
    vtkKWEntry *TREntry;
    vtkKWEntry *TAEntry;
    vtkKWEntry *TSEntry;
    vtkKWEntry *PREntry;
    vtkKWEntry *PAEntry;
    vtkKWEntry *PSEntry;

    vtkSlicerApplicationGUI *SlicerAppGUI;


//BTX
    std::vector<vtkIGTDataStream *> RegisteredDataStreams;
    std::vector<int> StreamTypes;  // matrix or image
    std::vector<char *> MRMLIds;  
//ETX

#ifdef USE_OPENTRACKER
    Context *context;
#endif

    void CreateMRMLNode(int streamType);
    void Normalize(float *a);
    void Cross(float *a, float *b, float *c);
    void SetLocatorTransforms();
    void UpdateLocator();
    void UpdateSliceDisplay(float px, float py, float pz);
    void ApplyTransform(float *position, float *norm, float *transnorm);
    void CloseConnection();

    void quaternion2xyz(float* orientation, float *normal, float *transnormal); 


};


#endif // IGTDATAMANAGER_H
