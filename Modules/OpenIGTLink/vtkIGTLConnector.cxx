/*==========================================================================

Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $HeadURL: $
Date:      $Date: $
Version:   $Revision: $

==========================================================================*/

#include "vtkObjectFactory.h"

#include <vtksys/SystemTools.hxx>

//#include "vtkSocketCommunicator.h"
#include "vtkMultiThreader.h"
#include "vtkServerSocket.h"
#include "vtkClientSocket.h"

#include "igtl_util.h"
#include "igtl_header.h"
#include "igtl_image.h"
#include "igtl_transform.h"
#include "crc32.h"

#include "vtkIGTLConnector.h"

vtkStandardNewMacro(vtkIGTLConnector);
vtkCxxRevisionMacro(vtkIGTLConnector, "$Revision: 1.0 $");

vtkIGTLConnector::vtkIGTLConnector()
{
  this->Type   = TYPE_NOT_DEFINED;
  this->State  = STATE_OFF;

  //this->Communicator = vtkSocketCommunicator::New();
  this->Thread = vtkMultiThreader::New();
  this->ServerStopFlag = false;
  this->ThreadID = -1;
  this->ServerSocket = vtkServerSocket::New(); 
  this->ServerHostname = "localhost";
  this->ServerPort = 18944;
}

vtkIGTLConnector::~vtkIGTLConnector()
{
}


void vtkIGTLConnector::PrintSelf(ostream& os, vtkIndent indent)
{
}

int vtkIGTLConnector::SetTypeServer(int port)
{
  this->Type = TYPE_SERVER;
  this->ServerPort = port;
  return 1;
}

int vtkIGTLConnector::SetTypeClient(char* hostname, int port)
{
  this->Type = TYPE_CLIENT;
  this->ServerPort = port;
  this->ServerHostname = hostname;
  return 1;
}

int vtkIGTLConnector::SetTypeClient(std::string hostname, int port)
{
  this->Type = TYPE_CLIENT;
  this->ServerPort = port;
  this->ServerHostname = hostname;
  return 1;
}


int vtkIGTLConnector::Start()
{
  // Check if type is defined.
  if (this->Type == vtkIGTLConnector::TYPE_NOT_DEFINED)
    {
      std::cerr << "Connector type is not defined." << std::endl;
      return 0;
    }

  // Check if thread is detached
  if (this->ThreadID >= 0)
    {
      std::cerr << "Thread exists." << std::endl;
      return 0;
    }

  this->ServerStopFlag = false;
  this->ThreadID = this->Thread->SpawnThread(vtkIGTLConnector::ThreadFunction, this);

  return 1;
}


int vtkIGTLConnector::Stop()
{
  // Check if thread exists
  if (this->ThreadID >= 0)
    {
      // NOTE: Thread should be killed by activating ServerStopFlag.
      //this->ServerStopFlag = true;
      this->Thread->TerminateThread(this->ThreadID);
      this->ThreadID = -1;
      this->State = STATE_OFF;
      return 1;
    }
  else
    {
      return 0;
    }
}


void* vtkIGTLConnector::ThreadFunction(void* ptr)
{

  //vtkIGTLConnector* igtlcon = static_cast<vtkIGTLConnector*>(ptr);
  vtkMultiThreader::ThreadInfo* vinfo = 
    static_cast<vtkMultiThreader::ThreadInfo*>(ptr);
  vtkIGTLConnector* igtlcon = static_cast<vtkIGTLConnector*>(vinfo->UserData);
  
  igtlcon->State = STATE_WAIT_CONNECTION;

  // Communication -- common to both Server and Client
  while (!igtlcon->ServerStopFlag)
    {
      std::cerr << "vtkOpenIGTLinkLogic::ThreadFunction(): alive." << std::endl;
      vtkClientSocket* socket = igtlcon->WaitForConnection();
      if (socket != NULL)
        {
          igtlcon->State = STATE_CONNECTED;
          std::cerr << "vtkOpenIGTLinkLogic::ThreadFunction(): Client Connected." << std::endl;
          igtlcon->ReceiveController(socket);
          igtlcon->State = STATE_WAIT_CONNECTION;
        }
    }
  igtlcon->ThreadID = -1;
  igtlcon->State = STATE_OFF;

}


vtkClientSocket* vtkIGTLConnector::WaitForConnection()
{
  vtkClientSocket* socket = NULL;

  if (this->Type == TYPE_CLIENT)
    {
      socket = vtkClientSocket::New();
    }
  else
    {
      this->ServerSocket->CreateServer(this->ServerPort);
    }

  while (!this->ServerStopFlag)
    {
      if (this->Type == TYPE_SERVER)
        {
          std::cerr << "vtkIGTLConnector: Waiting for client @ port #"
                    << this->ServerPort << std::endl;
          socket = this->ServerSocket->WaitForConnection(1000);
          if (socket != NULL) // if client connected
            {
              std::cerr << "vtkIGTLConnector: connected." << std::endl;
              return socket;
            }
        }
      else if (this->Type == TYPE_CLIENT) // if this->Type == TYPE_CLIENT
        {
          std::cerr << "vtkIGTLConnector: Connecting to server..." << std::endl;
          int r = socket->ConnectToServer(this->ServerHostname.c_str(), this->ServerPort);
          if (r == 0) // if connected to server
            {
              return socket;
            }
          else
            {
              break;
            }
        }
      else
        {
          this->ServerStopFlag = true;
        }
    }

  if (socket != NULL)
    {
      socket->Delete();
    }

  return NULL;
}


int vtkIGTLConnector::ReceiveController(vtkClientSocket* socket)
{
  igtl_header header;

  if (!socket)
    {
      return 0;
    }

  while (!this->ServerStopFlag)
    {

      // check if connection is alive
      if (!socket->GetConnected())
        {
          break;
        }

      int r = socket->Receive(&header, IGTL_HEADER_SIZE);

      if (r != IGTL_HEADER_SIZE)
        {
          std::cerr << "Irregluar size." << std::endl;
        }

      igtl_header_convert_byte_order(&header);  
      char deviceType[13];
      deviceType[12] = 0;
      memcpy((void*)deviceType, header.name, 8);
      
      char deviceName[21];
      deviceName[20] = 0;
      memcpy((void*)deviceName, header.device_name, 20);
      
      std::cerr << "deviceType  = " << deviceType << std::endl;;  
      std::cerr << "deviceName  = " << deviceName << std::endl;;  
      
      if (header.version != IGTL_HEADER_VERSION)
        {
          vtkErrorMacro("Unsupported OpenIGTLink version.");
          break;
        }

      /*
      if (this->AppLogic)
        {
          this->Scene = this->AppLogic->GetMRMLScene();
        }
      
      if (this->Scene)
        {
          vtkCollection* collection = this->Scene->GetNodesByName(deviceName);
          int num = collection->GetNumberOfItems();
          int newNode;
          newNode = (num == 0)? 1:0;
      */         
 
      if (strcmp("IMAGE", deviceType) == 0)
        {
          this->ReceiveImage(socket, deviceName, header.body_size, header.crc);
        }
      else if (strcmp("TRANSFORM", deviceType))
        {
          this->ReceiveTransform(socket, deviceName, header.body_size, header.crc);
        }

      /*
        }
        else
        {
        vtkErrorMacro("Cannot get MRML Scene");
        }
      */
    }

  socket->CloseSocket();
  return 0;
}

int vtkIGTLConnector::ReceiveImage(vtkClientSocket* socket, const char* deviceName,
                 long long bodySize, long long crc)
{

#if 0
  std::cerr << "ReceiveImage  is called  " << std::endl;

  vtkMRMLScalarVolumeNode* volumeNode;
  
  igtl_image_header imgheader;

  //int read = Tcl_Read(channel, (char *)&imgheader, IGTL_IMAGE_HEADER_SIZE);
  int read = socket->Receive(&header, IGTL_HEADER_SIZE);
  
  if (read != IGTL_IMAGE_HEADER_SIZE)
    {
      vtkErrorMacro ("Only read " << read << " but expected to read " << IGTL_IMAGE_HEADER_SIZE << "\n");
      return;
    }

  igtl_image_convert_byte_order(&imgheader);

  std::cerr << "image format version = " << imgheader.version << std::endl;

  unsigned char imgType = imgheader.data_type;
  unsigned char scalarType = imgheader.scalar_type;

  std::cerr << "scalar type = " << (int)scalarType << std::endl;
  std::cerr << "image type = " << (int)imgType << std::endl;

  std::cerr << "size[0] =  " << imgheader.size[0] << ", "
            << "size[1] =  " << imgheader.size[1] << ", "
            << "size[2] =  " << imgheader.size[2] << ", "
            << std::endl;

  std::cerr << "subvol_size[0] =  " << imgheader.subvol_size[0] << ", "
            << "subvol_size[1] =  " << imgheader.subvol_size[1] << ", "
            << "subvol_size[2] =  " << imgheader.subvol_size[2] << ", "
            << std::endl;

  float tx = imgheader.matrix[0];
  float ty = imgheader.matrix[1];
  float tz = imgheader.matrix[2];
  float sx = imgheader.matrix[3];
  float sy = imgheader.matrix[4];
  float sz = imgheader.matrix[5];
  float nx = imgheader.matrix[6];
  float ny = imgheader.matrix[7];
  float nz = imgheader.matrix[8];
  float px = imgheader.matrix[9];
  float py = imgheader.matrix[10];
  float pz = imgheader.matrix[11];

  std::cerr << "matrix = "<< std::endl;
  std::cerr << tx << ", " << ty << ", " << tz << std::endl;
  std::cerr << sx << ", " << sy << ", " << sz << std::endl;
  std::cerr << nx << ", " << ny << ", " << nz << std::endl;
  std::cerr << px << ", " << py << ", " << pz << std::endl;

  vtkImageData* imageData;
  if (newNode)
    {
      volumeNode = vtkMRMLScalarVolumeNode::New();
      volumeNode->SetName(deviceName);
      volumeNode->SetDescription("Received by OpenIGTLink");

      imageData = vtkImageData::New();

      imageData->SetDimensions(imgheader.size[0], imgheader.size[1], imgheader.size[2]);
      imageData->SetNumberOfScalarComponents(1);
      
      // Scalar type
      //  TBD: Long might not be 32-bit in some platform.
      switch (imgheader.scalar_type)
        {
        case IGTL_IMAGE_STYPE_TYPE_INT8:
          imageData->SetScalarTypeToChar();
          break;
        case IGTL_IMAGE_STYPE_TYPE_UINT8:
          imageData->SetScalarTypeToUnsignedChar();
          break;
        case IGTL_IMAGE_STYPE_TYPE_INT16:
          imageData->SetScalarTypeToShort();
          break;
        case IGTL_IMAGE_STYPE_TYPE_UINT16:
          imageData->SetScalarTypeToUnsignedShort();
          break;
        case IGTL_IMAGE_STYPE_TYPE_INT32:
          imageData->SetScalarTypeToUnsignedLong();
          break;
        case IGTL_IMAGE_STYPE_TYPE_UINT32:
          imageData->SetScalarTypeToUnsignedLong();
          break;
        default:
          vtkErrorMacro ("Invalid Scalar Type\n");
          break;
        }

      imageData->AllocateScalars();
      volumeNode->SetAndObserveImageData(imageData);
      imageData->Delete();

      this->Scene->AddNode(volumeNode);
      this->AppLogic->GetSelectionNode()->SetReferenceActiveVolumeID(volumeNode->GetID());
      this->AppLogic->PropagateVolumeSelection();
      
    }
  else
    {
      vtkCollection* collection = this->Scene->GetNodesByName(deviceName);
      volumeNode = vtkMRMLScalarVolumeNode::SafeDownCast(collection->GetItemAsObject(0));
    }

  // Get vtk image from MRML node
  imageData = volumeNode->GetImageData();

  // TODO:
  // It should be checked here if the dimension of vtkImageData
  // and arrived data is same.

  int bytes = igtl_image_get_data_size(&imgheader);

  if (imgheader.size[0] == imgheader.subvol_size[0] &&
      imgheader.size[1] == imgheader.subvol_size[1] &&
      imgheader.size[2] == imgheader.subvol_size[2] )
    {
      // In case that volume size == sub-volume size,
      // image is read directly to the memory area of vtkImageData
      // for better performance. 
      read = Tcl_Read(channel, (char *) imageData->GetScalarPointer(), bytes);
      if (read != bytes)
        {
          vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
          return;
        }
    }
  else
    {
      // In case of volume size != sub-volume size,
      // image is loaded into ImageReadBuffer, then copied to
      // the memory area of vtkImageData.

      if (bytes != this->ImageReadBufferSize)
        {
          if (this->ImageReadBuffer)
            {
              delete this->ImageReadBuffer;
            }
          this->ImageReadBufferSize = bytes;
          this->ImageReadBuffer = new char[bytes];
        }
      
      read = Tcl_Read(channel, (char *) this->ImageReadBuffer, bytes);
      if (read != bytes)
        {
          vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
          return;
        }

      // Check scalar size
      int scalarSize;
      switch (imgheader.scalar_type)
        {
        case IGTL_IMAGE_STYPE_TYPE_INT8:
        case IGTL_IMAGE_STYPE_TYPE_UINT8:
          scalarSize = 1;
          break;
        case IGTL_IMAGE_STYPE_TYPE_INT16:
        case IGTL_IMAGE_STYPE_TYPE_UINT16:
          scalarSize = 2;
          break;
        case IGTL_IMAGE_STYPE_TYPE_INT32:
        case IGTL_IMAGE_STYPE_TYPE_UINT32:
          scalarSize = 4;
          break;
        default:
          scalarSize = 0;
          vtkErrorMacro ("Invalid Scalar Type\n");
          break;
        }
        
      char* imgPtr = (char*) imageData->GetScalarPointer();
      char* bufPtr = this->ImageReadBuffer;
      int sizei = imgheader.size[0];
      int sizej = imgheader.size[1];
      int sizek = imgheader.size[2];
      int subsizei = imgheader.subvol_size[0];

      int bg_i = imgheader.subvol_offset[0];
      int ed_i = bg_i + imgheader.subvol_size[0];
      int bg_j = imgheader.subvol_offset[1];
      int ed_j = bg_j + imgheader.subvol_size[1];
      int bg_k = imgheader.subvol_offset[2];
      int ed_k = bg_k + imgheader.subvol_size[2];
      
      for (int k = bg_k; k < ed_k; k ++)
        {
          for (int j = bg_j; j < ed_j; j ++)
            {
              memcpy(&imgPtr[(sizei*sizej*k + sizei*j + bg_i)*scalarSize],
                     bufPtr, subsizei*scalarSize);
              bufPtr += subsizei*scalarSize;
            }
        }
    }


  // normalize
  float psi = sqrt(tx*tx + ty*ty + tz*tz);
  float psj = sqrt(sx*sx + sy*sy + sz*sz);
  float psk = sqrt(nx*nx + ny*ny + nz*nz);

  tx = tx / psi;
  ty = ty / psi;
  tz = tz / psi;
  sx = sx / psj;
  sy = sy / psj;
  sz = sz / psj;
  nx = nx / psk;
  ny = ny / psk;
  nz = nz / psk;


  float hfovi = psi * imgheader.size[0] / 2.0;
  float hfovj = psj * imgheader.size[1] / 2.0;
  float hfovk = psk * imgheader.size[2] / 2.0;

  float cx = tx * hfovi + sx * hfovj + nx * hfovk;
  float cy = ty * hfovi + sy * hfovj + ny * hfovk;
  float cz = tz * hfovi + sz * hfovj + nz * hfovk;

  px = px - cx;
  py = py - cy;
  pz = pz - cz;


  // set volume orientation
  vtkMatrix4x4* rtimgTransform = vtkMatrix4x4::New();
  rtimgTransform->Identity();
  rtimgTransform->SetElement(0, 0, tx);
  rtimgTransform->SetElement(1, 0, ty);
  rtimgTransform->SetElement(2, 0, tz);
  
  rtimgTransform->SetElement(0, 1, sx);
  rtimgTransform->SetElement(1, 1, sy);
  rtimgTransform->SetElement(2, 1, sz);
  
  rtimgTransform->SetElement(0, 2, nx);
  rtimgTransform->SetElement(1, 2, ny);
  rtimgTransform->SetElement(2, 2, nz);

  rtimgTransform->SetElement(0, 3, px);
  rtimgTransform->SetElement(1, 3, py);
  rtimgTransform->SetElement(2, 3, pz);


  rtimgTransform->Invert();
  volumeNode->SetRASToIJKMatrix(rtimgTransform);


//  if (lps) { // LPS coordinate
//    vtkMatrix4x4* lpsToRas = vtkMatrix4x4::New();
//    lpsToRas->Identity();
//    lpsToRas->SetElement(0, 0, -1);
//    lpsToRas->SetElement(1, 1, -1);
//    lpsToRas->Multiply4x4(lpsToRas, rtimgTransform, rtimgTransform);
//    lpsToRas->Delete();
//  }


  px = px + cx;
  py = py + cy;
  pz = pz + cz;


  //volumeNode->SetAndObserveImageData(imageData);
  volumeNode->Modified();
  vtkMRMLSliceNode* slnode = 
    vtkMRMLSliceNode::SafeDownCast(this->Scene->GetNodeByID("vtkMRMLSliceNode1"));
  slnode->SetSliceToRASByNTP(nx, ny, nz, tx, ty, tz, px, py, pz, 0);
#endif

}

int vtkIGTLConnector::ReceiveTransform(vtkClientSocket* socket, const char* deviceName, long long bodySize, long long crc)
{
  
}
