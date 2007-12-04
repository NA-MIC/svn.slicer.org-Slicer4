#include "vtkObjectFactory.h"

#include <vtksys/SystemTools.hxx>

#include "vtkProstateNavDataStream.h"
#include "vtkIGTMessageGenericAttribute.h"
#include "vtkIGTMessageImageDataAttribute.h"
#include "vtkImageData.h"
#include "vtkMath.h"


vtkStandardNewMacro(vtkProstateNavDataStream);
vtkCxxRevisionMacro(vtkProstateNavDataStream, "$Revision: 1.0 $");

vtkProstateNavDataStream::vtkProstateNavDataStream()
{
  this->AttrSetRobot   = NULL;
  this->AttrSetScanner = NULL;
  this->NeedleMatrix   = vtkMatrix4x4::New();
  
}

vtkProstateNavDataStream::~vtkProstateNavDataStream()
{
}


void vtkProstateNavDataStream::Init(const char *configFile)
{
  Superclass::Init(configFile);
}

void vtkProstateNavDataStream::PrintSelf(ostream& os, vtkIndent indent)
{
}

void vtkProstateNavDataStream::AddCallbacks()
{

  // Callback for the Robot
  this->AttrSetRobot = vtkIGTMessageAttributeSet::New();
  if (this->AttrSetRobot)
    {
    this->AttrSetRobot->AddAttribute("position",    (std::vector<float>*)NULL);
    this->AttrSetRobot->AddAttribute("orientation", (std::vector<float>*)NULL);
    this->AttrSetRobot->AddAttribute("status",      (std::string*)NULL);
    this->AttrSetRobot->AddAttribute("message",     (std::string*)NULL);
    this->AttrSetRobot->AddAttribute("depth",       (std::vector<float>*)NULL);
    }

  // Callback for the Scanner
  this->AttrSetScanner = vtkIGTMessageAttributeSet::New();
  if (this->AttrSetScanner)
    {
    this->AttrSetScanner->AddAttribute("image",   (vtkImageData*)NULL);
    this->AttrSetScanner->AddAttribute("fov",     (float*)NULL);
    this->AttrSetScanner->AddAttribute("slthick", (float*)NULL);
    }

  this->AddCallback("cb_robot",
                    (vtkIGTMessageAttributeSet::MessageHandlingFunction*)OnRecieveMessageFromRobot,
                    this->AttrSetRobot, NULL);
  this->AddCallback("cb_scanner",
                    (vtkIGTMessageAttributeSet::MessageHandlingFunction*)OnRecieveMessageFromScanner,
                    this->AttrSetScanner, NULL);

} 


void vtkProstateNavDataStream::OnRecieveMessageFromRobot(vtkIGTMessageAttributeSet* attrSet, void* arg)
{
  std::vector<float> position;
  std::vector<float> orientation;

  attrSet->GetAttribute("position", &position);
  attrSet->GetAttribute("orientation", &orientation);

  float ori[4];
  float norm[3];
  float transnorm[3];
  ori[0] = orientation[0];
  ori[1] = orientation[1];
  ori[2] = orientation[2];
  ori[3] = orientation[3];

  vtkProstateNavDataStream* ds = dynamic_cast<vtkProstateNavDataStream*>(attrSet->GetOpenTrackerStream());

  ds->QuaternionToXYZ(ori, norm, transnorm);
  std::cerr << " ORIENTATION = ( " 
            << ori[0] << ", "
            << ori[1] << ", "
            << ori[2] << ", "
            << ori[3] << ") " << std::endl;
  return;

  int j;
  for (j=0; j<3; j++)
    {
    ds->NeedleMatrix->SetElement(j,0,position[j]);
    }
  for (j=0; j<3; j++)
    {
    ds->NeedleMatrix->SetElement(j,1,norm[j]);
    }
  for (j=0; j<3; j++)
    {
    ds->NeedleMatrix->SetElement(j,2,transnorm[j]);
    }
  for (j=0; j<3; j++)
    {
    ds->NeedleMatrix->SetElement(j,3,0);
    }
  for (j=0; j<3; j++)
    {
    ds->NeedleMatrix->SetElement(3,j,0);
    }

  ds->NeedleMatrix->SetElement(3,3,1);

  // get a 3x3 matrix from the quaterion
  float transform_matrix[3][3];
  vtkMath::QuaternionToMatrix3x3(ori, transform_matrix);
  
  // get the "needle depth" vector(3,1) and multiply it by the robot orientation,
  // this will give the offsets in Slicer coordinates

  std::vector<float> depth;
  attrSet->GetAttribute("depth", &depth);

  float needle_offset[3];
  for (j=0; j<3; j++) {
    needle_offset[j] = depth[j];
  }

  // multiply the vector in-place
  vtkMath::Multiply3x3(transform_matrix, needle_offset, needle_offset);
  
  // to make things simple, replace the robot position by the needle tip
  //  in the LocatorMatrix
  ds->NeedleMatrix->SetElement(0, 0, position[0] + needle_offset[0]);
  ds->NeedleMatrix->SetElement(1, 0, position[1] + needle_offset[1]);
  ds->NeedleMatrix->SetElement(2, 0, position[2] + needle_offset[2]);
}


void vtkProstateNavDataStream::OnRecieveMessageFromScanner(vtkIGTMessageAttributeSet* attrSet, void* arg)
{

}



