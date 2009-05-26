/*=========================================================================

  Program:   Open IGT Link -- Example for Tracker Client Program
  Module:    $RCSfile: $
  Language:  C++
  Date:      $Date: $
  Version:   $Revision: $

  Copyright (c) Insight Software Consortium. All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include <iostream>
#include <math.h>
#include <cstdlib>

#include "igtlOSUtil.h"
#include "igtlImageMessage.h"
#include "igtlTransformMessage.h"
#include "igtlClientSocket.h"

#include "EndoNavTrackerCLP.h"

#include "vtkWin32VideoSource.h"
#include "vtkTrackingSystem.h"
#include "vtkTrackFileData.h"

#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkFieldData.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"

#include "vtkGlobFileNames.h"
#include "vtkImageReader.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkStructuredPointsReader.h"

int SendSensorMatrix(igtl::ClientSocket::Pointer &socket, vtkMatrix4x4* sensorMatrix);

int SendSensorImage(igtl::ClientSocket::Pointer &socket, vtkImageData *img, vtkMatrix4x4 *sensorMatrix);

int GetTestImage(igtl::ImageMessage::Pointer& msg, const char* dir, int i);
void GetRandomSensorMatrix(igtl::Matrix4x4& matrix);
void GetVTKMatrix(igtl::Matrix4x4& matrix, vtkMatrix4x4* vtkMatrix);


void printUsage()
{
  std::cerr << "Usage: " << " <Hostname> <Port> <Fps> <imgdir>"    << std::endl;
  std::cerr << "    <Hostname> : IP or host name"                    << std::endl;
  std::cerr << "    <Port>     : Port # (18944 in Slicer default)"   << std::endl;
  std::cerr << "    <Fps>      : Frequency (fps) to send coordinate" << std::endl;
  std::cerr << "    <ImageDirectory>   : file directory, where \"igtlTestImage[1-5].raw\" are placed." << std::endl;
  std::cerr << "                 (usually, in the Examples/Imager/img directory.)" << std::endl;
}

int main(int argc, char* argv[])
{
  
  //------------------------------------------------------------
  // Parse Arguments

  PARSE_ARGS;
  
  int    interval = (int) (1000.0 / Fps);
  
  //
  // Source
  //

  vtkWin32VideoSource *videoSource = vtkWin32VideoSource::New();

  vtkTrackingSystem *tracker = vtkTrackingSystem::New();

  vtkTrackFileData *trackFileData = vtkTrackFileData::New();

  int numSourceFiles = 0;
  bool device = true;
  int hasTracker = 0;

  if (TrackingSource == std::string("TrackingDevice") )
    {    
    hasTracker = tracker->OpenConnection();
    if (CaptureVideo)
      {
      videoSource->Initialize ();
      videoSource->SetFrameBufferSize(100);
      videoSource->SetNumberOfOutputFrames(100);
      videoSource->SetFrameSize(ImageDimensionX,ImageDimensionY,1);
      }
    } 
  else 
    {
    device = false;
    numSourceFiles = trackFileData->InitilizeRead(SourceDirectory.c_str(), SourcePattern.c_str());
    }

  //
  // Destination
  //

  igtl::ClientSocket::Pointer socket;
  
  if (SendToServer)
    {
    //------------------------------------------------------------
    // Establish Connection
    socket = igtl::ClientSocket::New();
    int r = socket->ConnectToServer(Hostname.c_str(), Port);
    
    if (r != 0)
      {
      std::cerr << "Cannot connect to the server." << std::endl;
      exit(0);
      }
    }
  else if (Record)
    {
    trackFileData->InitilizeWrite(RecordDirectory.c_str(), RecordPrefix.c_str());
    }
  else
    {
    std::cerr << "No destiantion selected" << std::endl;
    exit(0);
    }

  int index = 0;
  int inFileCount = 0;
  int outFileCount = 0;
  
  vtkImageData *img = NULL;
  vtkMatrix4x4 *sensorMatrix = NULL;
  vtkMatrix4x4 *calibMatrix = NULL;
  vtkMatrix4x4 *regMatrix = NULL;

  //------------------------------------------------------------
  // loop

  while (1) 
    {
    img = NULL;
    sensorMatrix = NULL;
    calibMatrix = NULL;
    regMatrix = NULL;

    double timeStamp = 0;

    if (device && CaptureVideo)
      {
      videoSource->Grab();
      img = videoSource->GetOutput();
      }
   if (device && hasTracker)
      {
      tracker->Poll();
      for (int n=0; n<tracker->GetNumSensors(); n++)
        {
        // Get TRANSFORM        
        //

        // TODO: what do we do with multiple sensors???
        // For now use just one
        if (n = VideoChannel)
          {
          sensorMatrix = vtkMatrix4x4::New();
          sensorMatrix->DeepCopy(tracker->GetSensorMatrix(n));
          timeStamp = tracker->GetSensorTimeStamp(n);
          }
        }
      } //if (device && hasTracker)

    if (!device)
      {
      // Source Files
      if (inFileCount >= numSourceFiles)
        {
        if (CycleFiles)
          {
          inFileCount = 0;
          }
        else
          {
          break;
          }
        }
        
      trackFileData->ReadStep(inFileCount, timeStamp, &calibMatrix, &regMatrix, &sensorMatrix, &img);
      inFileCount++;

      } //if (!device)

    if (Record)
      {
      if (!device)
        {
        outFileCount = inFileCount;
        }

      // Temporary for debugging
      if (sensorMatrix == NULL)
        {
        sensorMatrix = vtkMatrix4x4::New();
        igtl::Matrix4x4 matrix;
        GetRandomSensorMatrix(matrix);
        GetVTKMatrix(matrix, sensorMatrix);
        }
      if (calibMatrix == NULL)
        {
        calibMatrix = vtkMatrix4x4::New();
        calibMatrix->Identity();
        }
      if (regMatrix == NULL)
        {
        regMatrix = vtkMatrix4x4::New();
        regMatrix->Identity();
        }

      trackFileData->WriteStep(outFileCount, timeStamp, calibMatrix, regMatrix, sensorMatrix, img);

      outFileCount++;
      }
    
    if (SendToServer)
      {
      SendSensorImage(socket, img, sensorMatrix);
      SendSensorMatrix(socket, sensorMatrix);
      }

     if (calibMatrix) 
       {
       calibMatrix->Delete();
       }
     if (regMatrix) 
       {
       regMatrix->Delete();
       }
     if (sensorMatrix) 
       {
       sensorMatrix->Delete();
       }
    
    igtl::Sleep(interval); // wait
      
    index ++;

    // file to file do it once
    if (Record && !device && outFileCount>=numSourceFiles)
      {
      break;
      }
    } // While(1);

  //------------------------------------------------------------
  // Close connection
  socket->CloseSocket();

  if (device)
    {
    videoSource->Stop();
    videoSource->ReleaseSystemResources();
    tracker->CloseConnection();
    }
  
  videoSource->Delete();
  tracker->Delete();
    
  return EXIT_SUCCESS;

}

//------------------------------------------------------------
// Function to send sensor matrix.
int SendSensorMatrix(igtl::ClientSocket::Pointer &socket, vtkMatrix4x4 *sensorMatrix)
{

  igtl::Matrix4x4 matrix;

  if (sensorMatrix)
    {
    for (int n=0; n<4; n++) 
      {
      for (int m=0; m<4; m++) 
        {
        matrix[n][m] = sensorMatrix->GetElement(n,m);
        }
      }
    }
  else 
    {
    return 0;
    //GetRandomSensorMatrix(matrix);
    }
      
  // Allocate Transform Message Class
  igtl::TransformMessage::Pointer transMsg = igtl::TransformMessage::New();
  transMsg->SetDeviceName("Tracker");
  transMsg->SetMatrix(matrix);
  transMsg->Pack();
  return socket->Send(transMsg->GetPackPointer(), transMsg->GetPackSize());
}

//------------------------------------------------------------
// Function to send sensor matrix.
int SendSensorImage(igtl::ClientSocket::Pointer &socket, vtkImageData *img, vtkMatrix4x4 *sensorMatrix)
{
  int   size[]     = {1, 1, 1};       // image dimension
  float spacing[]  = {1.0, 1.0, 1.0};     // spacing (mm/pixel)
  int   svsize[]   = {1, 1, 1};       // sub-volume size
  int   svoffset[] = {0, 0, 0};           // sub-volume offset
  int   scalarType = igtl::ImageMessage::TYPE_UINT8;// scalar type

  if (img)
    {
    img->GetDimensions(size);
    for (int j=0; j<3; j++) 
      {
      svsize[j] = size[j];
      }
    }
  else 
    {
    return 0;
    }

  int dtype = VTK_UNSIGNED_CHAR;
  //dtype = img->GetPointData()->GetScalars()->GetDataType();
  //dtype = img->GetScalarType();
  
  switch (dtype) {
  case VTK_FLOAT:
    scalarType = igtl::ImageMessage::TYPE_FLOAT32;
    break;
  case VTK_DOUBLE:
    scalarType = igtl::ImageMessage::TYPE_FLOAT64;
    break;
  case VTK_UNSIGNED_CHAR:
    scalarType = igtl::ImageMessage::TYPE_UINT8;
    break;
  case VTK_UNSIGNED_INT:
    scalarType = igtl::ImageMessage::TYPE_UINT32;
    break;
  case VTK_UNSIGNED_SHORT:
    scalarType = igtl::ImageMessage::TYPE_UINT16;
    break;
  default:
    break;
  }
  //------------------------------------------------------------
  // Create a new IMAGE type message
  igtl::ImageMessage::Pointer imgMsg = igtl::ImageMessage::New();
  imgMsg->SetDimensions(size);
  imgMsg->SetSpacing(spacing);
  imgMsg->SetScalarType(scalarType);
  imgMsg->SetDeviceName("ImagerClient");
  imgMsg->SetSubVolume(size, svoffset);
  imgMsg->AllocateScalars();
  
  //------------------------------------------------------------
  // Set image data (See GetTestImage() bellow for the details)
  //GetTestImage(imgMsg, RecordDirectory.c_str(), i % 5);
  
  memcpy(imgMsg->GetScalarPointer(), img->GetScalarPointer(), imgMsg->GetImageSize());
  
  // TODO add matrix to image from video sensor channel
  //------------------------------------------------------------
  // Get  orientation matrix and set it.
  igtl::Matrix4x4 matrix;

  if (sensorMatrix)
    {
    for (int n=0; n<4; n++) 
      {
      for (int m=0; m<4; m++) 
        {
        matrix[n][m] = sensorMatrix->GetElement(n,m);
        }
      }
    imgMsg->SetMatrix(matrix);
    }
  else 
    {
    GetRandomSensorMatrix(matrix);
    imgMsg->SetMatrix(matrix);
    }
  
  //------------------------------------------------------------
  // Pack (serialize) and send
  imgMsg->Pack();
  
  return socket->Send(imgMsg->GetPackPointer(), imgMsg->GetPackSize());
}

//------------------------------------------------------------
// Function to read test image data
int GetTestImage(igtl::ImageMessage::Pointer& msg, const char* dir, int i)
{

  //------------------------------------------------------------
  // Check if image index is in the range
  if (i < 0 && i >= 5) 
    {
    std::cerr << "Image index is invalid." << std::endl;
    return 0;
    }

  //------------------------------------------------------------
  // Generate path to the raw image file
  char filename[128];
  sprintf(filename, "%s/igtlTestImage%d.raw", dir, i+1);
  std::cerr << "Reading " << filename << "...";

  //------------------------------------------------------------
  // Load raw data from the file
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL)
    {
    std::cerr << "File opeining error: " << filename << std::endl;
    return 0;
    }
  int fsize = msg->GetImageSize();
  size_t b = fread(msg->GetScalarPointer(), 1, fsize, fp);

  fclose(fp);

  std::cerr << "done." << std::endl;

  return 1;
}

//------------------------------------------------------------
// Function to generate random matrix.
void GetRandomSensorMatrix(igtl::Matrix4x4& matrix)
{
  float position[3];
  float orientation[4];


  // random position
  static float phi = 0.0;
  position[0] = 50.0 * cos(phi);
  position[1] = 50.0 * sin(phi);
  position[2] = 0;
  phi = phi + 0.2;

  // random orientation
  static float theta = 0.0;
  orientation[0]=0.0;
  orientation[1]=0.6666666666*cos(theta);
  orientation[2]=0.577350269189626;
  orientation[3]=0.6666666666*sin(theta);
  theta = theta + 0.1;

  //igtl::Matrix4x4 matrix;
  igtl::QuaternionToMatrix(orientation, matrix);

  matrix[0][0] = 1.0;  matrix[1][0] = 0.0;  matrix[2][0] = 0.0; matrix[3][0] = 0.0;
  matrix[0][1] = 0.0;  matrix[1][1] = -1.0;  matrix[2][1] = 0.0; matrix[3][1] = 0.0;
  matrix[0][2] = 0.0;  matrix[1][2] = 0.0;  matrix[2][2] = 1.0; matrix[3][2] = 0.0;
  matrix[0][3] = 0.0;  matrix[1][3] = 0.0;  matrix[2][3] = 0.0; matrix[3][3] = 1.0;

  matrix[0][3] = position[0];
  matrix[1][3] = position[1];
  matrix[2][3] = position[2];


  
  igtl::PrintMatrix(matrix);
}


void GetVTKMatrix(igtl::Matrix4x4& matrix, vtkMatrix4x4* vtkMatrix)
{
  if (vtkMatrix == NULL)
    {
    return;
    }
  else
    {
    for (int n=0; n<4; n++) 
      {
      for (int m=0; m<4; m++) 
        {
        vtkMatrix->SetElement(n,m, matrix[n][m]);
        }
      }

    }
}
