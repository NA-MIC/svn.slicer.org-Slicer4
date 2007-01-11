



#include "vtkIGTOpenTrackerStream.h"
#include "vtkObjectFactory.h"

// Constructors/Destructors
//  

  vtkIGTOpenTrackerStream* vtkIGTOpenTrackerStream::New()
{
  vtkObject* ret=vtkObjectFactory::CreateInstance("vtkIGTOpenTrackerStream");
  if(ret)
    {
      return(vtkIGTOpenTrackerStream*) ret;
    }
  return new vtkIGTOpenTrackerStream;
}

vtkIGTOpenTrackerStream::vtkIGTOpenTrackerStream ( ) {
}

vtkIGTOpenTrackerStream::~vtkIGTOpenTrackerStream ( ) { }

//  
// Methods
//  


// Accessor methods
//  


// Other methods
//  


/**
 */
void vtkIGTOpenTrackerStream::initi_open_tracker ( ) {

}


/**
 */
void vtkIGTOpenTrackerStream::add_data_to_data_stream ( ) {

}


/**
 */
void vtkIGTOpenTrackerStream::callback_function ( ) {

}


/**
 */
void vtkIGTOpenTrackerStream::read_configuration_file ( ) {

}


