
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
