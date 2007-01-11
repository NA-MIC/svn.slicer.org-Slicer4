
#include "vtkIGTDataStream.h"
#include "vtkIGTMatrixState.h"
#include "vtkObjectFactory.h"

// Constructors/Destructors
//  

vtkIGTDataStream* vtkIGTDataStream::New()
{
  vtkObject* ret=vtkObjectFactory::CreateInstance("vtkIGTDataStream");
  if(ret)
    {
      return(vtkIGTDataStream*) ret;
    }
  return new vtkIGTDataStream;
}

vtkIGTDataStream::vtkIGTDataStream(){};



vtkIGTDataStream::~vtkIGTDataStream ( ) { }

//  
// Methods
//  

void vtkIGTDataStream::Init (int numbuffer ) {
  buffer_size = numbuffer;
  stream_device_hash = new void* [DEVICE_MAX_NUM];
}


/**
 */
int vtkIGTDataStream::register_stream_device ( int stream_type) {
  
  int this_device_id = num_registered_device +1;
  vtkIGTMatrixState *p_matrix;
  
  switch (stream_type) {
    case IGT_MATRIX_STREAM:
      p_matrix = new vtkIGTMatrixState[buffer_size];
      stream_device_hash[this_device_id] = (vtkIGTMatrixState*)p_matrix;
      
      break;
      
    case IGT_IMAGE_STREAM:
      cout << "" ;
      break;
      
    default:
      cout << "";
      return(-1);
      
    }
  
  return this_device_id;

}

