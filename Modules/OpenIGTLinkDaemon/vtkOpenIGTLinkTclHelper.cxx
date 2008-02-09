/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkOpenIGTLinkTclHelper.cxx,v $
  Date:      $Date: 2006/01/06 17:58:00 $
  Version:   $Revision: 1.4 $

  =========================================================================auto=*/

/* 
 * vtkOpenIGTLinkTclHelper allows access to Tcl and vtk routines in the same class
 * inspired by vtkTkRenderWidget and similar classes.
 */


#include <stdlib.h>

#include "vtkOpenIGTLinkTclHelper.h"
#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkTclUtil.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkFloatArray.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"

#ifdef WIN32
#include "Winsock2.h"
#else
#include "arpa/inet.h"
#endif /* LINUX */

#include "vtkMRMLScalarVolumeNode.h"

vtkCxxRevisionMacro(vtkOpenIGTLinkTclHelper, "$Revision: 1.4 $");
vtkStandardNewMacro(vtkOpenIGTLinkTclHelper);


vtkOpenIGTLinkTclHelper::vtkOpenIGTLinkTclHelper()
{
  this->Interp = NULL;
  this->ImageData = NULL;
  this->VolumeNode = NULL;
  this->Matrix = NULL;
  this->MeasurementFrame = vtkMatrix4x4::New();
  this->MeasurementFrame->Identity();
}


vtkOpenIGTLinkTclHelper::~vtkOpenIGTLinkTclHelper() 
{ 
}


void 
vtkOpenIGTLinkTclHelper::SetInterpFromCommand(unsigned long tag)
{
  vtkCommand *c = this->GetCommand(tag);
  vtkTclCommand *tc = (vtkTclCommand *) c;

  this->Interp = tc->Interp;
}

void 
vtkOpenIGTLinkTclHelper::SendImageDataScalars(char *sockname)
{
  int mode;
  Tcl_Channel channel = Tcl_GetChannel(this->Interp, sockname, &mode);

  if ( ! (mode & TCL_WRITABLE) )
    {   vtkErrorMacro ("Socket " << sockname << " is not writable\n");
      return;
    }

  if ( this->ImageData == NULL )
    {   vtkErrorMacro("Image Data is NULL");
      return;
    }

  int dims[3];
  this->ImageData->GetDimensions(dims);
  int bytes = this->ImageData->GetScalarSize() * 
    this->ImageData->GetNumberOfScalarComponents() * 
    dims[0] * dims[1] * dims[2];

  int written = Tcl_WriteRaw(channel, (char *) this->ImageData->GetScalarPointer(), bytes);
  Tcl_Flush(channel);

  if ( written != bytes )
    {   vtkErrorMacro ("Only wrote " << written << " but expected to write " << bytes << "\n");
      return;
    }
}


inline int is_little_endian() {
  short a = 1; return ((char*)&a)[0];
}

#define BYTE_SWAP_INT16(S) (((S) & 0xFF) << 8 \
                            | (((S) >> 8) & 0xFF))
#define BYTE_SWAP_INT32(L) ((BYTE_SWAP_INT16 ((L) & 0xFFFF) << 16) \
                            | BYTE_SWAP_INT16 (((L) >> 16) & 0xFFFF))
#define BYTE_SWAP_INT64(LL) ((BYTE_SWAP_INT32 ((LL) & 0xFFFFFFFF) << 32) \
                             | BYTE_SWAP_INT32 (((LL) >> 32) & 0xFFFFFFFF))

void 
vtkOpenIGTLinkTclHelper::OnReceiveOpenIGTLinkMessage(char *sockname)
{
  int mode;

  std::cerr << "vtkOpenIGTLinkTclHelper::OnRecieveOpenIGTLinkMessage(char *sockname) !!!" << std::endl;;

  Tcl_Channel channel = Tcl_GetChannel(this->Interp, sockname, &mode);
    
  if ( ! (mode & TCL_READABLE) )
    {   vtkErrorMacro ("Socket " << sockname << " is not readable" << "\n");
      return;
    }

  unsigned char header[54];
  int bytes = 54;
  int read = Tcl_Read(channel, (char *) header, bytes);

  if (read != bytes)
    {
      vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

  unsigned short version;
  memcpy((void*)&version, &header[0], sizeof(unsigned short));
  version = ntohs(version);

  char deviceType[9];
  deviceType[8] = 0;
  memcpy((void*)deviceType, &header[2], 8);
  
  char deviceName[21];
  deviceName[20] = 0;
  memcpy((void*)deviceName, &header[10], 20);

  std::cerr << "deviceType  = " << deviceType << std::endl;;  
  std::cerr << "deviceName  = " << deviceName << std::endl;;  

  if (version != 1)
    {
      vtkErrorMacro("Unsupported OpenIGTLink version.");
      return;
    }

  long long bodySize;
  memcpy((void*)&bodySize, &header[38], sizeof(long long));

  long long crc;
  memcpy((void*)&crc, &header[46], sizeof(long long));

  if (is_little_endian())
    {
      bodySize = BYTE_SWAP_INT64(bodySize);
      crc = BYTE_SWAP_INT64(crc);
    }

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
      
      if (strcmp("IMAGE", deviceType) == 0)
        {
          ReceiveImage(channel, deviceName, bodySize, crc, newNode);
        }
      else if (strcmp("tracking", deviceType))
        {
          ReceiveTracker(channel, deviceName, bodySize, crc, newNode);
        }
    }
  else
    {
      vtkErrorMacro("Cannot get MRML Scene");
    }
}
  
// Read a stream of numbers from vtkSocketCommunicator::SendTagged 
// and put it int the Matrix ivar
void 
vtkOpenIGTLinkTclHelper::PerformVTKSocketHandshake(char *sockname)
{

  std::cerr << "PerformVTKSocketHandshake(char *sockname) is called " << std::endl;
  int mode;

  Tcl_Channel channel = Tcl_GetChannel(this->Interp, sockname, &mode);

  if ( ! (mode & TCL_READABLE) )
    {   vtkErrorMacro ("Socket " << sockname << " is not readable" << "\n");
      return;
    }

  // read the tag, but ignore it
  int bytes = 9;
  char handshake[9];
  int read = Tcl_Read(channel, (char *) &handshake, bytes);

  if ( read != bytes )
    {   vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

  int written = Tcl_WriteRaw(channel, (char *) handshake, bytes);
  Tcl_Flush(channel);

  if ( written != bytes )
    {   vtkErrorMacro ("Only wrote " << written << " but expected to write " << bytes << "\n");
      return;
    }

}

void 
vtkOpenIGTLinkTclHelper::SendMessage(char *sockname)
{
  int mode;
  Tcl_Channel channel = Tcl_GetChannel(this->Interp, sockname, &mode);
  if ( ! (mode & TCL_WRITABLE) )
    {   vtkErrorMacro ("Socket " << sockname << " is not writable\n");
      return;
    }

  char m = 1;
  int tag = 17;
  int bytes = 1; 
  // all messages share the same tag: 17
  int written = Tcl_WriteRaw(channel, (char *)&tag, sizeof(int));
  // bytes to be sent for the message
  written = Tcl_WriteRaw(channel, (char *)&bytes, sizeof(bytes));
  // the one byte message
  written = Tcl_WriteRaw(channel, &m, bytes);

  Tcl_Flush(channel);

  if ( written != bytes )
    {   vtkErrorMacro ("Only wrote " << written << " but expected to write " << bytes << "\n");
      return;
    }
}

void
vtkOpenIGTLinkTclHelper::ReceiveImage(Tcl_Channel channel, char* deviceName, long long bodySize, long long crc, int newNode)
{

  std::cerr << "ReceiveImage  is called  " << std::endl;

  vtkMRMLScalarVolumeNode* volumeNode;
  
  unsigned char imgheader[72];
  int bytes = 72;
  int read = Tcl_Read(channel, (char *)imgheader, bytes);
  
  if (read != bytes)
    {
      vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

  unsigned short version;
  memcpy((void*)&version, &imgheader[0], 2);
  version = ntohs(version);

  std::cerr << "image format version = " << version << std::endl;

  unsigned char imgType = imgheader[2];
  unsigned char scalarType = imgheader[3];

  std::cerr << "scalar type = " << (int)scalarType << std::endl;
  std::cerr << "image type = " << (int)imgType << std::endl;

  unsigned short ssize[3];
  int size[3];
  memcpy((void*)ssize, &imgheader[6], 2*3);
  for (int i = 0; i < 3; i ++)
    {
      size[i] = (int)ntohs(ssize[i]);
    }

  std::cerr << "size[0] =  " << size[0] << ", "
            << "size[1] =  " << size[1] << ", "
            << "size[2] =  " << size[2] << ", "
            << std::endl;

  unsigned int imatrix[12];
  memcpy((void*)imatrix, &imgheader[12], sizeof(unsigned int)*12);
  for (int i = 0; i < 12; i ++)
    {
      imatrix[i] = ntohl(imatrix[i]);
    }

  float* matrix = (float*) imatrix;

  unsigned short subpos[3];
  unsigned short subsize[3];
  memcpy((void*)subpos, &imgheader[60], sizeof(unsigned short)*3);
  memcpy((void*)subsize, &imgheader[66], sizeof(unsigned short)*3);
  for (int i = 0; i < 3; i ++)
    {
      subpos[i] = ntohs(subpos[i]);
      subsize[i] = ntohs(subsize[i]);
    }

  float tx = matrix[0];
  float ty = matrix[1];
  float tz = matrix[2];
  float sx = matrix[3];
  float sy = matrix[4];
  float sz = matrix[5];
  float nx = matrix[6];
  float ny = matrix[7];
  float nz = matrix[8];
  float px = matrix[9];
  float py = matrix[10];
  float pz = matrix[11];

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
      imageData->SetDimensions(size[0], size[1], size[2]);
      imageData->SetNumberOfScalarComponents(1);
      imageData->SetScalarTypeToShort();  // should be set according to scalar type
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

  imageData = volumeNode->GetImageData();
  std::cerr << "size[0] =  " << size[0] << ", "
            << "size[1] =  " << size[1] << ", "
            << "size[2] =  " << size[2] << ", "
            << std::endl;
  bytes = size[0]*size[1]*size[2]*sizeof(short);
  std::cerr << "image size  = " << bytes << std::endl;  
  read = Tcl_Read(channel, (char *) imageData->GetScalarPointer(), bytes);

  if (read != bytes)
    {
      vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

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

//  if (lps) { // LPS coordinate
//    vtkMatrix4x4* lpsToRas = vtkMatrix4x4::New();
//    lpsToRas->Identity();
//    lpsToRas->SetElement(0, 0, -1);
//    lpsToRas->SetElement(1, 1, -1);
//    lpsToRas->Multiply4x4(lpsToRas, rtimgTransform, rtimgTransform);
//    lpsToRas->Delete();
//  }


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

  float hfovi = psi * size[0] / 2.0;
  float hfovj = psj * size[1] / 2.0;

  rtimgTransform->Invert();
  volumeNode->SetRASToIJKMatrix(rtimgTransform);

  float cx = tx * hfovi + sx * hfovj;
  float cy = ty * hfovi + sy * hfovj;
  float cz = tz * hfovi + sz * hfovj;

  px = px + cx;
  py = py + cy;
  pz = pz + cz;

  volumeNode->SetAndObserveImageData(imageData);
  vtkMRMLSliceNode* slnode = vtkMRMLSliceNode::SafeDownCast(this->Scene->GetNodeByID("vtkMRMLSliceNode1"));
  slnode->SetSliceToRASByNTP(nx, ny, nz, tx, ty, tz, px, py, pz, 0);
}


void
vtkOpenIGTLinkTclHelper::ReceiveTracker(Tcl_Channel channel, char* deviceName, long long size, long long crc, int newNode)
{
  
}


void 
vtkOpenIGTLinkTclHelper::ReceiveMatrix(char *sockname)
{
  int mode;
  Tcl_Channel channel = Tcl_GetChannel(this->Interp, sockname, &mode);
    
  if ( ! (mode & TCL_READABLE) )
    {   vtkErrorMacro ("Socket " << sockname << " is not readable" << "\n");
      return;
    }

  if ( this->Matrix == NULL )
    {   vtkErrorMacro ("Matrix is NULL");
      return;
    }

  // read the tag, but ignore it
  int tag;
  int bytes = sizeof(int);
  int read = Tcl_Read(channel, (char *) &tag, bytes);

  if ( read != bytes )
    {   vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

  // read the number of elements
  int length;
  bytes = sizeof(int);
  read = Tcl_Read(channel, (char *) &length, bytes);

  if ( read != bytes )
    {   vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }

  if ( length != 12*8 )
    {   vtkErrorMacro ("Packet of " << length << " sent, but expected " << 12*8 << "\n");
      return;
    }

  // read the actual elements
  double elements[12];
  bytes = 12*8;
  read = Tcl_Read(channel, (char *) &elements, 12*8);

  if ( read != bytes )
    {   vtkErrorMacro ("Only read " << read << " but expected to read " << bytes << "\n");
      return;
    }
  
  vtkMatrix4x4 *localMatrix = vtkMatrix4x4::New();
  localMatrix->Identity();
  unsigned int counter = 0;
  for (unsigned int i = 0; i < 4; i++)
    {
    for (unsigned int j = 0; j < 3; j++)
      {
      localMatrix->SetElement( j, i, elements[counter++] );
      }
    }
  this->Matrix->DeepCopy(localMatrix);
  localMatrix->Delete();
}

const char *
vtkOpenIGTLinkTclHelper::Execute (char *Command)
{
  int res;
#if TCL_MAJOR_VERSION == 8 && TCL_MINOR_VERSION <= 2
  res = Tcl_GlobalEval(this->Interp, this->Command);
#else
  res = Tcl_EvalEx(this->Interp, Command, -1, TCL_EVAL_GLOBAL);
#endif  

  return Tcl_GetStringResult (this->Interp);
}


void vtkOpenIGTLinkTclHelper::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Interp: " << this->Interp << "\n";
  os << indent << "ImageData: " << this->ImageData << "\n";

}

