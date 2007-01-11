
#ifndef IGTDATASTREAM_H
#define IGTDATASTREAM_H


#include <string>
#include <vector>

#include "vtkObject.h"
#include "vtkIGTMatrixState.h"


#define DEVICE_MAX_NUM 1024
#define IGT_MATRIX_STREAM 0
#define IGT_IMAGE_STREAM 1

class vtkIGTDataStream : public vtkObject
{
public:

  // Constructors/Destructors
  //  Magic lines for vtk and Slicer
    static vtkIGTDataStream *New();
      vtkTypeMacro(vtkIGTDataStream,vtkObject);
      void PrintSelf(ostream& os, vtkIndent indent);


  /**
   * Constructor
   @ param buffersize: size of buufer (
   */
  vtkIGTDataStream ();

  

  
/**
   * Empty Destructor
   */
  virtual ~vtkIGTDataStream ( );

  

protected:

  /**
   * registering stream_device and allocating buffers of the nodes
   * users should not use any vtk classes in implementation (subclass) of this class
   * we shoud diffrentiate the datatype by flag
   @return:stream_id

  */   
  void Init ();
  int register_stream_device (int stream_type );
  
 
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

#endif // IGTDATASTREAM_H
