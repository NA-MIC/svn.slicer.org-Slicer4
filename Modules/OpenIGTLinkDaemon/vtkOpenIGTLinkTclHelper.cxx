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
#include "vtkMRMLLinearTransformNode.h"

#include "igtl_header.h"
#include "igtl_image.h"
#include "igtl_transform.h"

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
  this->ImageReadBuffer = NULL;
  this->ImageReadBufferSize = 0;
}


vtkOpenIGTLinkTclHelper::~vtkOpenIGTLinkTclHelper() 
{ 
  if (this->Matrix)
    {
      this->Matrix->Delete();
    }
  if (this->MeasurementFrame)
    {
      this->MeasurementFrame->Delete();
    }
  if (this->ImageReadBuffer)
    {
      this->ImageReadBufferSize = 0;
      delete this->ImageReadBuffer;
    }
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

  igtl_header header;
  int read = Tcl_Read(channel, (char *) &header, IGTL_HEADER_SIZE);

  if (read != IGTL_HEADER_SIZE)
    {
      vtkErrorMacro ("Only read " << read << " but expected to read " << IGTL_HEADER_SIZE << "\n");
      return;
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
      return;
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
          ReceiveImage(channel, deviceName, header.body_size, header.crc, newNode);
        }
      else if (strcmp("TRANSFORM", deviceType))
        {
          ReceiveTransform(channel, deviceName, header.body_size, header.crc, newNode);
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
  
  igtl_image_header imgheader;

  int read = Tcl_Read(channel, (char *)&imgheader, IGTL_IMAGE_HEADER_SIZE);
  
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


  float hfovi = psi * imgheader.size[0] / 2.0;
  float hfovj = psj * imgheader.size[1] / 2.0;

  rtimgTransform->Invert();
  volumeNode->SetRASToIJKMatrix(rtimgTransform);

  float cx = tx * hfovi + sx * hfovj;
  float cy = ty * hfovi + sy * hfovj;
  float cz = tz * hfovi + sz * hfovj;

  px = px + cx;
  py = py + cy;
  pz = pz + cz;

  //volumeNode->SetAndObserveImageData(imageData);
  volumeNode->Modified();
  vtkMRMLSliceNode* slnode = 
    vtkMRMLSliceNode::SafeDownCast(this->Scene->GetNodeByID("vtkMRMLSliceNode1"));
  slnode->SetSliceToRASByNTP(nx, ny, nz, tx, ty, tz, px, py, pz, 0);
}


void
vtkOpenIGTLinkTclHelper::ReceiveTransform(Tcl_Channel channel, char* deviceName, long long size, long long crc, int newNode)
{
  float matrix[12];

  vtkMRMLLinearTransformNode* transformNode;

  int read = Tcl_Read(channel, (char *)matrix, IGTL_TRANSFORM_SIZE);
  igtl_transform_convert_byte_order(matrix);

  if (newNode)
    {
      transformNode = vtkMRMLLinearTransformNode::New();
      transformNode->SetName(deviceName);
      transformNode->SetDescription("Received by OpenIGTLink");

      vtkMatrix4x4* transform = vtkMatrix4x4::New();
      transform->Identity();

      //transformNode->SetAndObserveImageData(transform);
      transformNode->ApplyTransform(transform);
      transform->Delete();

      this->Scene->AddNode(transformNode);
    }
  else
    {
      vtkCollection* collection = this->Scene->GetNodesByName(deviceName);
      transformNode = vtkMRMLLinearTransformNode::SafeDownCast(collection->GetItemAsObject(0));
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
  
  // set volume orientation
  vtkMatrix4x4* transform = vtkMatrix4x4::New();
  vtkMatrix4x4* transformToParent = transformNode->GetMatrixTransformToParent();
  transform->Identity();
  transform->SetElement(0, 0, tx);
  transform->SetElement(1, 0, ty);
  transform->SetElement(2, 0, tz);

  transform->SetElement(0, 1, sx);
  transform->SetElement(1, 1, sy);
  transform->SetElement(2, 1, sz);

  transform->SetElement(0, 2, nx);
  transform->SetElement(1, 2, ny);
  transform->SetElement(2, 2, nz);

  transform->SetElement(0, 3, px);
  transform->SetElement(1, 3, py);
  transform->SetElement(2, 3, pz);

  transformToParent->DeepCopy(transform);
  transform->Delete();
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

