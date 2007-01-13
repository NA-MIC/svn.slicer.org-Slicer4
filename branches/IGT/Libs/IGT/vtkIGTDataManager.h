// .NAME vtkIGTDataManager - Central registry to provide control and I/O for
//  trackers and imagers
// .SECTION Description
// vtkIGTDataManager registers arbitary number of trackers and imagers, created MRML nodes in the MRML secene. Designed and Coded by Nobuhiko Hata and Haying Liu, Jan 12, 2006 @ NA-MIC All Hands Meeting, Salt Lake City, UT

#ifndef IGTDATAMANAGER_H
#define IGTDATAMANAGER_H


#include <string>
#include <vector>

#include "vtkObject.h"
#include "vtkMRMLScene.h"
#include "vtkIGTMatrixState.h"
#include "vtkIGTDataStream.h"


class vtkIGTDataManager : public vtkObject
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



    //Description:
    //Contructor
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

protected:

    //Descripton:
    //Create MRML Node for registered device. It is usually called after RegisterStreamDevice
    void CreateMRMLNode(int streamType);

private:

    //Description.
    //MRML scene passed from the Module instantiating this class

    vtkMRMLScene* MRMLScene;

//BTX
    std::vector<vtkIGTDataStream *> RegisteredDataStreams;
    std::vector<int> StreamTypes;  // matrix or image
    std::vector<char *> MRMLIds;  
//ETX


};

#endif // IGTDATAMANAGER_H
