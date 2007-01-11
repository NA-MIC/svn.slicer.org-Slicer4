
#include "vtkIGTDataManager.h"
#include "vtkIGTMatrixState.h"
#include "vtkObjectFactory.h"

// Constructors/Destructors
//  

vtkIGTDataManager* vtkIGTDataManager::New()
{
  vtkObject* ret=vtkObjectFactory::CreateInstance("vtkIGTDataManager");
  if(ret)
    {
      return(vtkIGTDataManager*) ret;
    }
  return new vtkIGTDataManager;
}

vtkIGTDataManager::vtkIGTDataManager(){};



vtkIGTDataManager::~vtkIGTDataManager ( ) { }

//  
// Methods
//  

void vtkIGTDataManager::Init () {
}


/**
 */
int vtkIGTDataManager::register_stream_device ( int stream_type, vtkIGTDataManager* datastream) {
  
 
  vtkIGTMatrixState *p_matrix;
  //vtkIGTImageState *p_image;
  switch (stream_type) {
    case IGT_MATRIX_STREAM:
    
      p_matrix = new vtkIGTMatrixState;

      DeviceType.push_back(stream_type);
      RegisteredDataStream.push_back(datastream);
      create_mrml_node(DeviceType.size());

      return (DeviceType.size());
      
    case IGT_IMAGE_STREAM:
      //p_image = new vtkIGTImageState;
      //RegisteredMatrixState->push_back(p_image);
      DeviceType.push_back(stream_type);
      RegisteredDataStream.push_back(datastream);
      return (DeviceType.size());     
    default:
      cout << "";
      return(-1);
      
    }
 
}


vtkIGTMatrixState* vtkIGTDataManager::GetMatrixState(int devicenumber){
  vtkIGTDataStream* p_data_stream = RegisteredDataStream.at(devicenumber);
  return(->vtkIGTMatrixState);
  
}

vtkIGTImageState* vtkIGTDataManager::GetImageState(int devicenumber){
  vtkIGTDataStream* p_data_stream;
  p_data_stream = RegisteredDataStream->at(devicenumber);
  return(vtkIGTDataStream->vtkIGTImageState);
}

void vtkIGTDataManager::StartMRMLUpdater() {
  //go through vtkIGTDataStream
  //access the matrix or image using GetMatrixState or GetImageState

}


void vtkIGTDataManager::create_mrml_node(int index_num) {

  // create mrml node 
  //use contents of DeviceType to check
}
