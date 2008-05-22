#include "vtkMiniBirdInstrumentTracker.h"
#include "vtkObjectFactory.h"

#include <cmath>
#include "vtkMatrix4x4.h"

#include <windows.h>
#include "bird.h"  


vtkCxxRevisionMacro(vtkMiniBirdInstrumentTracker, "$Revision 1.0$");
vtkStandardNewMacro(vtkMiniBirdInstrumentTracker);


vtkMiniBirdInstrumentTracker::vtkMiniBirdInstrumentTracker()
: m_NumberOfDevices(2)//, m_DataLogger("mini_bird.txt")
{
    this->Transform = vtkMatrix4x4::New();
    this->PosX = 0.0;
    this->PosY = 0.0;
    this->PosZ = 0.0;
    this->Phi = 0.0;
    this->Theta = 0.0;
    this->Roll = 0.0;

    //WORD comport[LOCAL_MAX_BIRDS + 1];
    WORD comport[3];
    BIRDSYSTEMCONFIG sysconfig;
    BIRDDEVICECONFIG* devconfig = new BIRDDEVICECONFIG[m_NumberOfDevices];
    
    // tell driver to look on comport 4 for master bird
    //comport[0] = 4;

    comport[1] = 5; //4;
    comport[2] = 0;

    // wake up a flock of birds using the RS232 interface
    //birdRS232WakeUp(1,TRUE,1,comport,115200,2000,2000);    // wake-up call to stand alone bird
    birdRS232WakeUp(1,false,2,comport,115200,2000,2000);        // wake-up call to flock of birds


    // get the system configuration
    birdGetSystemConfig(1,&sysconfig);

    // get a device configuration for each device
    for (int i = 1; i <= m_NumberOfDevices; i++)
    {
        birdGetDeviceConfig(1,i,&devconfig[i]);  
        devconfig[i].byDataFormat = BDF_POSITIONMATRIX;
        birdSetDeviceConfig(1,i,&devconfig[i]);  
    }

    birdStartFrameStream(1);
    //m_DataLogger.AddDataDouble("Staring minibird Stream", 0.0);
    //m_DataLogger.WriteBufferToFile();
    //delete[] devconfig;
}

vtkMiniBirdInstrumentTracker::~vtkMiniBirdInstrumentTracker()
{
    birdStopFrameStream(1);
    birdShutDown(1);
    //m_DataLogger.AddDataDouble("Ending minibird Stream", 0.0);

    this->Transform->Delete();
}


void vtkMiniBirdInstrumentTracker::CalcInstrumentPos()
{
    static bool running = false;
    if (birdFrameReady(1))// && !running)
    {
        running = true;
        BIRDFRAME frame;
        birdGetMostRecentFrame(1,&frame);

        BIRDREADING *bird_US;
        BIRDREADING *bird_instrument;

        bird_US = &frame.reading[2]; 
        bird_instrument = &frame.reading[1]; 

        vtkMatrix4x4* orientation_US = vtkMatrix4x4::New();
        vtkMatrix4x4* orientation_instrument = vtkMatrix4x4::New();
        orientation_US->Identity();
        orientation_instrument->Identity();
        for (int iMat=0; iMat<3; iMat++)
        {
            for (int jMat=0; jMat<3; jMat++)
            {
                orientation_US->SetElement(iMat,jMat, (1.0/32767.0) * bird_US->matrix.n[iMat][jMat]);
                orientation_instrument->SetElement(iMat,jMat, (1.0/32767.0) * bird_instrument->matrix.n[iMat][jMat]);
            }
        }
        double pos_scale = 36.0;

        orientation_instrument->SetElement(0,3, bird_instrument->position.nX * pos_scale / 32767.0);
        orientation_instrument->SetElement(1,3, bird_instrument->position.nY * pos_scale / 32767.0);
        orientation_instrument->SetElement(2,3, bird_instrument->position.nZ * pos_scale / 32767.0);

        orientation_US->SetElement(0,3, bird_US->position.nX * pos_scale / 32767.0);
        orientation_US->SetElement(1,3, bird_US->position.nY * pos_scale / 32767.0);
        orientation_US->SetElement(2,3, bird_US->position.nZ * pos_scale / 32767.0);

        vtkMatrix4x4* orientation_US_inverse = vtkMatrix4x4::New();
        orientation_US_inverse->DeepCopy(orientation_US);
        orientation_US_inverse->Invert();

        //BEGIN: IMAGE SIZE DEPENDANT
        vtkMatrix4x4* US_to_image = vtkMatrix4x4::New(); 
        US_to_image->Identity();
        US_to_image->SetElement(0,3, 64);

        vtkMatrix4x4* bird_to_US = vtkMatrix4x4::New();
        bird_to_US->Identity();
        bird_to_US->SetElement(1, 3, 1.2);
        bird_to_US->SetElement(2, 3, -2.9);

        vtkMatrix4x4* scale_transform = vtkMatrix4x4::New();
        scale_transform->Identity();
        const double DEPTH = 8.0; //cm
        scale_transform->SetElement(0,0, (DEPTH/8.0) * 25.4 / 0.640); //0.456;// (mm/inch) / (mm/voxel) = (voxel/inch) in x
        scale_transform->SetElement(1,1, (DEPTH/8.0) * 25.4 / 0.640);// (mm/inch) / (mm/voxel) = (voxel/inch) in y
        scale_transform->SetElement(2,2, (DEPTH/8.0) * 25.4 / 0.403);// (mm/inch) / (mm/voxel) = (voxel/inch) in z
        //END: IMAGE SIZE DEPENDANT

        vtkMatrix4x4* axis_switch = vtkMatrix4x4::New();
        axis_switch->Zero();
        axis_switch->SetElement(0,1, -1.0);
        axis_switch->SetElement(1,2, -1.0);
        axis_switch->SetElement(2,0, 1.0);
        axis_switch->SetElement(3,3, 1.0);
        

        vtkMatrix4x4* total_transform = vtkMatrix4x4::New();
        vtkMatrix4x4* partial_transform = vtkMatrix4x4::New();
        
        vtkMatrix4x4* temp1 = vtkMatrix4x4::New();
        vtkMatrix4x4::Multiply4x4(bird_to_US, axis_switch, partial_transform);
        vtkMatrix4x4::Multiply4x4(partial_transform, orientation_US_inverse, temp1);
        vtkMatrix4x4::Multiply4x4(temp1, orientation_instrument, partial_transform);
//        partial_transform = bird_to_US * axis_switch *  orientation_US_inverse * orientation_instrument;

        vtkMatrix4x4::Multiply4x4(US_to_image , scale_transform, temp1);
        vtkMatrix4x4::Multiply4x4(temp1, partial_transform, total_transform);
        //total_transform = US_to_image * scale_transform * partial_transform;

        temp1->Delete();

        double instrument_tip_pos[4] = {5.0, 0.0, 0.25, 1.0};
        double* transformed_tip_pos = total_transform->MultiplyDoublePoint(instrument_tip_pos);
        //transformed_tip_position = total_transform * instrument_tip_position;


        this->Transform->DeepCopy(partial_transform);
        this->Transform->SetElement(0,3,0.0);
        this->Transform->SetElement(1,3,0.0);
        this->Transform->SetElement(2,3,0.0);
        this->Transform->SetElement(3,3,1.0);


        this->PosX = ( transformed_tip_pos[0] );
        this->PosY = ( transformed_tip_pos[1] );
        this->PosZ = ( transformed_tip_pos[2]);
        
        //double phi = atan2(-partial_transform->GetElement(1,0), partial_transform->GetElement(0,0)); //( total_transform(1,0) / cos(theta) );
        //double gamma = atan2( partial_transform->GetElement(2,1), partial_transform->GetElement(2,2) );
        //double theta = asin( -partial_transform->GetElement(2,0) );

        //m_DataLogger.AddMatrix("total transform", total_transform);
        //const double RAD_TO_DEG = 180.0/3.14159265; 
        //m_InstrumentPosition.SetAngleTheta( theta * RAD_TO_DEG ); 
        //m_InstrumentPosition.SetAnglePhi( -phi * RAD_TO_DEG );
        //m_InstrumentPosition.SetRollAngle( gamma * RAD_TO_DEG );

        //m_DataLogger.AddData3DPos("instrument position", orientation_instrument.get_column(3));
        //m_DataLogger.AddData3DPos("US Position", orientation_US.get_column(3));
        //m_DataLogger.WriteBufferToFile();

        running = false;
    }
}

