
#ifndef IGTDATAMANAGER_H
#define IGTDATAMANAGER_H


#include <string>
#include <vector>

#include "vtkObject.h"
#include "vtkMRMLScene.h"
#include "vtkIGTMatrixState.h"


#define DEVICE_MAX_NUM 1024
#define IGT_MATRIX_STREAM 0
#define IGT_IMAGE_STREAM 1

class vtkIGTDataManager : public vtkObject
{
public:

  // Constructors/Destructors
  //  Magic lines for vtk and Slicer
    static vtkIGTDataManager *New();
      vtkTypeMacro(vtkIGTDataManager,vtkObject);
      void PrintSelf(ostream& os, vtkIndent indent);


  /**
   * Constructor
   @ param buffersize: size of buufer (
   */
  vtkIGTDataManager ();

  

  
/**
   * Empty Destructor
   */
  virtual ~vtkIGTDataManager ( );

  
  vtkIGTMatrixState* GetMatrixState(int devicenumber);

protected:

  /**
   * registering stream_device and allocating buffers of the nodes
   * users should not use any vtk classes in implementation (subclass) of this class
   * we shoud diffrentiate the datatype by flag
   @return:stream_id

  */   
  void Init ();
  int register_stream_device (int stream_type, vtkIGTDataManager* datastream);
  
  void create_mrml_node(int index_num);

private:

  // Static Private attribu tes
  //  


  // Private attributes
  //  
  


  int LastInputNum;
  int LastInputTime;
  vtkMRMLScene* scene;

  //BTX
  
  std::vector<vtkIGTDataStream*> RegisteredDataStream;
  std::vector<int> DeviceType;

  //ETX




};

#endif // IGTDATAMANAGER_H
