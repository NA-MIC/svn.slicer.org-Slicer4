



#include "vtkIGTOpenTrackerStream.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro (vtkIGTOpenTrackerStream);
vtkCxxRevisionMacro (vtkIGTOpenTrackerStream, "$Revision: 1.0 $");


vtkIGTOpenTrackerStream::vtkIGTOpenTrackerStream ( ) {
}

vtkIGTOpenTrackerStream::~vtkIGTOpenTrackerStream ( ) { }


void vtkIGTOpenTrackerStream::Init(char* configfile) 
{
   fprintf(stderr,"config file: %s\n",configfile);
  this->context = new Context(1); 
  // get callback module from the context
  CallbackModule * callbackMod = (CallbackModule *)context->getModule("CallbackConfig");
  
  context->parseConfiguration(configfile);  // parse the configuration file
  
  callbackMod->setCallback( "cb1", (CallbackFunction*)&callbackF ,this);    // sets the callback function
  
  
  context->start();

}
  


void vtkIGTOpenTrackerStream::callbackF(const Node&,const Event &event, void *data) 
{
    float position[3];
    float orientation[4];
    float norm[3];
    float transnorm[3];
    int j;
    
    vtkIGTOpenTrackerStream *VOT=(vtkIGTOpenTrackerStream *)data;
    
    // the original values are in the unit of meters
    position[0]=(float)(event.getPosition())[0] * 1000.0; 
    position[1]=(float)(event.getPosition())[1] * 1000.0;
    position[2]=(float)(event.getPosition())[2] * 1000.0;
    
    orientation[0]=(float)(event.getOrientation())[0];
    orientation[1]=(float)(event.getOrientation())[1];
    orientation[2]=(float)(event.getOrientation())[2];
    orientation[3]=(float)(event.getOrientation())[3];

    VOT->quaternion2xyz(orientation, norm, transnorm);

    vtkIGTMatrixState *state = VOT->GetMatrixState();
    vtkMatrix4x4 *matrix = state->GetMatrix();


    for (j=0; j<3; j++) {
        matrix->SetElement(j,0,position[j]);
    }


    for (j=0; j<3; j++) {
        matrix->SetElement(j,1,norm[j]);
    }

    for (j=0; j<3; j++) {
        matrix->SetElement(j,2,transnorm[j]);
    }

    for (j=0; j<3; j++) {
        matrix->SetElement(j,3,0);
    }

    for (j=0; j<3; j++) {
        matrix->SetElement(3,j,0);
    }

    matrix->SetElement(3,3,1);


}




void vtkIGTOpenTrackerStream::quaternion2xyz(float* orientation, float *normal, float *transnormal) 
{
    float q0, qx, qy, qz;

    q0 = orientation[3];
    qx = orientation[0];
    qy = orientation[1];
    qz = orientation[2]; 

    transnormal[0] = 1-2*qy*qy-2*qz*qz;
    transnormal[1] = 2*qx*qy+2*qz*q0;
    transnormal[2] = 2*qx*qz-2*qy*q0;

    normal[0] = 2*qx*qz+2*qy*q0;
    normal[1] = 2*qy*qz-2*qx*q0;
    normal[2] = 1-2*qx*qx-2*qy*qy;
}



void vtkIGTOpenTrackerStream::PrintSelf(ostream& os, vtkIndent indent)
{

}


