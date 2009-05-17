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

#include "EndoNavRecorderCLP.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"

int ReceiveTransform(igtl::Socket::Pointer& socket, igtl::MessageHeader::Pointer& header, vtkMatrix4x4 *matrix);
int ReceivePosition(igtl::Socket::Pointer& socket, igtl::MessageHeader::Pointer& header, float position[3], float quaternion[4]);

int ReceiveImage(igtl::Socket::Pointer& socket, igtl::MessageHeader::Pointer& header, vtkImageData);
int ReceiveStatus(igtl::Socket::Pointer& socket, igtl::MessageHeader::Pointer& header);

void printUsage()
{
  std::cerr << "Usage: " << " <Port> <imgdir>"    << std::endl;
  std::cerr << "    <Port>     : Port # (18944 in Slicer default)"   << std::endl;
  std::cerr << "    <ImageDirectory>   : file directory, where image are placed." << std::endl;
}

int main(int argc, char* argv[])
{
  
  PARSE_ARGS;


  //------------------------------------------------------------
  // Parse Arguments


  //------------------------------------------------------------
  // Establish Connection
  
  igtl::ServerSocket::Pointer serverSocket;
  serverSocket = igtl::ServerSocket::New();
  serverSocket->CreateServer(Port);

  igtl::Socket::Pointer socket;
  
  // Connection loop
  while (1)
    {
    //------------------------------------------------------------
    // Waiting for Connection
    socket = serverSocket->WaitForConnection(1000);
    
    if (socket.IsNotNull()) // if client connected
      {
      // Create a message buffer to receive header
      igtl::MessageHeader::Pointer headerMsg;
      headerMsg = igtl::MessageHeader::New();
      
      //------------------------------------------------------------
      // Recieving loop
      while (1)
        {
        
        // Initialize receive buffer
        headerMsg->InitPack();
        
        // Receive generic header from the socket
        int r = socket->Receive(headerMsg->GetPackPointer(), headerMsg->GetPackSize());
        if (r != headerMsg->GetPackSize())
          {
          break;
          }

        // Deserialize the header
        headerMsg->Unpack();

        // Check data type and receive data body
        if (strcmp(headerMsg->GetDeviceType(), "TRANSFORM") == 0)
          {
          ReceiveTransform(socket, headerMsg);
          }
        else if (strcmp(headerMsg->GetDeviceType(), "POSITION") == 0)
          {
          ReceivePosition(socket, headerMsg);
          }
        else if (strcmp(headerMsg->GetDeviceType(), "IMAGE") == 0)
          {
          ReceiveImage(socket, headerMsg);
          }
        else if (strcmp(headerMsg->GetDeviceType(), "STATUS") == 0)
          {
          ReceiveStatus(socket, headerMsg);
          }
        else
          {
          // if the data type is unknown, skip reading.
          socket->Skip(headerMsg->GetBodySizeToRead(), 0);
          }
        }
      }
    }
    
  //------------------------------------------------------------
  // Close connection (The example code never reachs to this section ...)
  
  socket->CloseSocket();

}
 
  
  
  
  
  
  
  
  
  
  int i = 0;
  while (1) 
    {
    //------------------------------------------------------------
    // size parameters
    int   size[]     = {256, 256, 1};       // image dimension
    float spacing[]  = {1.0, 1.0, 5.0};     // spacing (mm/pixel)
    int   svsize[]   = {256, 256, 1};       // sub-volume size
    int   svoffset[] = {0, 0, 0};           // sub-volume offset
    int   scalarType = igtl::ImageMessage::TYPE_UINT8;// scalar type

    videoSource->Grab();
    vtkImageData *img = videoSource->GetOutput();

    img->GetDimensions(size);
    int dtype = img->GetScalarType();

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
    //GetTestImage(imgMsg, ImageDirectory.c_str(), i % 5);

    memcpy(imgMsg->GetScalarPointer(), img->GetScalarPointer(), imgMsg->GetImageSize());


    //------------------------------------------------------------
    // Get random orientation matrix and set it.
    igtl::Matrix4x4 matrix;
    GetRandomTestMatrix(matrix);
    imgMsg->SetMatrix(matrix);

    //------------------------------------------------------------
    // Pack (serialize) and send
    imgMsg->Pack();
    socket->Send(imgMsg->GetPackPointer(), imgMsg->GetPackSize());


    ////////// TRANSFORM
    //------------------------------------------------------------
    // Allocate Transform Message Class

    igtl::TransformMessage::Pointer transMsg = igtl::TransformMessage::New();
    transMsg->SetDeviceName("Tracker");
    transMsg->SetMatrix(matrix);
    transMsg->Pack();
    socket->Send(transMsg->GetPackPointer(), transMsg->GetPackSize());

    igtl::Sleep(interval); // wait

    i ++;
    }

  //------------------------------------------------------------
  // Close connection
  socket->CloseSocket();

  videoSource->Stop();
  videoSource->ReleaseSystemResources();
  videoSource->Delete();
  return EXIT_SUCCESS;

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
void GetRandomTestMatrix(igtl::Matrix4x4& matrix)
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


