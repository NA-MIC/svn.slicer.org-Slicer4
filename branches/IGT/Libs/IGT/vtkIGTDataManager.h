
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

    // Constructors/Destructors
    //  Magic lines for vtk and Slicer
    static vtkIGTDataManager *New();
    vtkTypeRevisionMacro(vtkIGTDataManager,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);

    vtkSetObjectMacro(MRMLScene,vtkMRMLScene);
    vtkGetObjectMacro(MRMLScene,vtkMRMLScene);

    /**
     * Constructor
     @ param buffersize: size of buufer (
     */
    vtkIGTDataManager();


    /**
     * Empty Destructor
     */
    virtual ~vtkIGTDataManager ( );

    char *GetMRMLId(int index);
//    void UpdateMatrixData(int index, vtkIGTMatrixState state);
    void RegisterStreamDevice (int streamType, vtkIGTDataStream* datastream);

protected:

    void CreateMRMLNode(int streamType);

private:

    vtkMRMLScene* MRMLScene;

//BTX
    std::vector<vtkIGTDataStream *> RegisteredDataStreams;
    std::vector<int> StreamTypes;  // matrix or image
    std::vector<char *> MRMLIds;  
//ETX


};

#endif // IGTDATAMANAGER_H
