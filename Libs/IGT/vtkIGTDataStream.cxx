
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

}


/**
 */
int vtkIGTDataStream::register_stream_device ( int stream_type, vtkIGTDataStream* datastream) {
  
 
  vtkIGTMatrixState *p_matrix;
  //vtkIGTImageState *p_image;
  switch (stream_type) {
    case IGT_MATRIX_STREAM:
    
      p_matrix = new vtkIGTMatrixState;

      DeviceType->push_back(stream_type);
      RegisteredDataStream->push_back(datastream);
      create_mrml_node(DeviceType->size());

      return (DeviceType->size());
      
    case IGT_IMAGE_STREAM:
      //p_image = new vtkIGTImageState;
      //RegisteredMatrixState->push_back(p_image);
      DeviceIDs->push_back();
      RegisteredDataStream->push_back(datastream);
      return (DeviceType->size());     
    default:
      cout << "";
      return(-1);
      
    }
 
}


vtkIGTMatrixState* vtkIGTDataStream:GetMatrixState(int devicenumber){
  vtkIGTDataStream* p_data_stream;  
  p_data_stream = RegisteredDataStream->at(devicenumber);
  return(vtkIGTDataStream->vtkIGTMatrixState);
  
}

vtkIGTImageState* vtkIGTDataStream::GetImageState(int devicenumber){
  vtkIGTDataStream* p_data_stream;
  p_data_stream = RegisteredDataStream->at(devicenumber);
  return(vtkIGTDataStream->vtkIGTImageState);
}

void vtkIGTDataStream::StartMRMLUpdater() {
  //go through vtkIGTDataStream
  //access the matrix or image using GetMatrixState or GetImageState

}


create_mrml_node(int index_num) {

  // create mrml node 
  //use contents of DeviceType to check
}
