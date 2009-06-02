/*==========================================================================

  Portions (c) Copyright 2008 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $HeadURL: http://svn.slicer.org/Slicer3/branches/EndoTracking/Modules/EndoNavIF/vtkIGTLToViewerImage.cxx $
  Date:      $Date: 2009-04-30 23:14:25 -0400 (Thu, 30 Apr 2009) $
  Version:   $Revision: 9377 $

==========================================================================*/

#include <vtksys/SystemTools.hxx>

#include "vtkObjectFactory.h"
#include "vtkIGTLToViewerImage.h"

#include "vtkSlicerColorLogic.h"

#include "vtkImageData.h"
#include "vtkRenderer.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "igtlImageMessage.h"

vtkStandardNewMacro(vtkIGTLToViewerImage);
vtkCxxRevisionMacro(vtkIGTLToViewerImage, "$Revision: 9377 $");


//---------------------------------------------------------------------------
vtkIGTLToViewerImage::vtkIGTLToViewerImage()
{
  this->SliceViewer = NULL;
  this->SliceGUI = NULL;
  this->ImageData = NULL;
  this->NodeCreated = 0;
  this->ImageViewerUS = NULL;

}


//---------------------------------------------------------------------------
vtkIGTLToViewerImage::~vtkIGTLToViewerImage()
{
  if (this->ImageData)
    {
    this->ImageData->Delete();
    }
}


//---------------------------------------------------------------------------
void vtkIGTLToViewerImage::PrintSelf(ostream& os, vtkIndent indent)
{
}


//---------------------------------------------------------------------------
vtkMRMLNode* vtkIGTLToViewerImage::CreateNewNode(vtkMRMLScene* scene, const char* name)
{
  vtkMRMLVolumeNode *volumeNode = NULL;

  //vtkMRMLVolumeDisplayNode *displayNode = NULL;
  vtkMRMLScalarVolumeDisplayNode *displayNode = NULL;
  vtkMRMLScalarVolumeNode *scalarNode = vtkMRMLScalarVolumeNode::New();
  this->ImageData = vtkImageData::New();
  
  //float fov = 256.0;
  this->ImageData->SetDimensions(3, 3, 1);
  this->ImageData->SetExtent(0, 2, 0, 2, 0, 0 );
  //this->ImageData->SetSpacing( fov/256, fov/256, 10 );
  this->ImageData->SetSpacing(1.0, 1.0, 1.0);
  //this->ImageData->SetOrigin( fov/2, -fov/2, -0.0 );
  this->ImageData->SetOrigin(0.0, 0.0, 0.0);
  this->ImageData->SetNumberOfScalarComponents(1);
  this->ImageData->SetScalarTypeToShort();
  this->ImageData->AllocateScalars();
  
  short* dest = (short*) this->ImageData->GetScalarPointer();
  if (dest)
    {
    memset(dest, 0x00, 3*3*sizeof(short));
    this->ImageData->Update();
    }
  
  /*
    vtkSlicerSliceLayerLogic *reslice = vtkSlicerSliceLayerLogic::New();
    reslice->SetUseReslice(0);
  */
  scalarNode->SetAndObserveImageData(this->ImageData);
  
  
  /* Based on the code in vtkSlicerVolumeLogic::AddHeaderVolume() */
  //displayNode = vtkMRMLVolumeDisplayNode::New();
  displayNode = vtkMRMLScalarVolumeDisplayNode::New();
  scalarNode->SetLabelMap(0);
  volumeNode = scalarNode;
  
  if (volumeNode != NULL)
    {
    volumeNode->SetName(name);
    scene->SaveStateForUndo();
    
    vtkDebugMacro("Setting scene info");
    volumeNode->SetScene(scene);
    volumeNode->SetDescription("Received by EndoNav");
    
    displayNode->SetScene(scene);
    
    
    double range[2];
    vtkDebugMacro("Set basic display info");
    volumeNode->GetImageData()->GetScalarRange(range);
    range[0] = 0.0;
    range[1] = 256.0;
    displayNode->SetLowerThreshold(range[0]);
    displayNode->SetUpperThreshold(range[1]);
    displayNode->SetWindow(range[1] - range[0]);
    displayNode->SetLevel(0.5 * (range[1] + range[0]) );
    
    vtkDebugMacro("Adding node..");
    scene->AddNode(displayNode);
    
    //displayNode->SetDefaultColorMap();
    vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
    displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultVolumeColorNodeID());
    //colorLogic->Delete();
    
    volumeNode->SetAndObserveDisplayNodeID(displayNode->GetID());
    
    vtkDebugMacro("Name vol node "<<volumeNode->GetClassName());
    vtkDebugMacro("Display node "<<displayNode->GetClassName());
    
    scene->AddNode(volumeNode);
    vtkDebugMacro("Node added to scene");
    }
  
  scalarNode->Delete();
  displayNode->Delete();

  this->NodeCreated = 1;

  return volumeNode;
}


//---------------------------------------------------------------------------
vtkIntArray* vtkIGTLToViewerImage::GetNodeEvents()
{
  return NULL;
}


//---------------------------------------------------------------------------
int vtkIGTLToViewerImage::IGTLToMRML(igtl::MessageBase::Pointer buffer, vtkMRMLNode* node)
{
  if (strcmp(node->GetNodeTagName(), "Volume") != 0)
    {
    //std::cerr << "Invalid node!!!!" << std::endl;
    return 0;
    }

  // Create a message buffer to receive image data
  igtl::ImageMessage::Pointer imgMsg;
  imgMsg = igtl::ImageMessage::New();
  imgMsg->Copy(buffer); // !! TODO: copy makes performance issue.

  // Deserialize the transform data
  // If you want to skip CRC check, call Unpack() without argument.
  int c = imgMsg->Unpack(1);

  if ((c & igtl::MessageHeader::UNPACK_BODY) == 0) // if CRC check fails
    {
    // TODO: error handling
    return 0;
    }

  if (node == NULL)
    {
    return 0;
    }

  // Retrive the image data
  int   size[3];          // image dimension
  float spacing[3];       // spacing (mm/pixel)
  int   svsize[3];        // sub-volume size
  int   svoffset[3];      // sub-volume offset
  int   scalarType;       // scalar type
  igtl::Matrix4x4 matrix; // Image origin and orientation matrix

  scalarType = imgMsg->GetScalarType();
  imgMsg->GetDimensions(size);
  imgMsg->GetSpacing(spacing);
  imgMsg->GetSubVolume(svsize, svoffset);
  imgMsg->GetMatrix(matrix);

  // check if the IGTL data fits to the current MRML node
  vtkMRMLScalarVolumeNode* volumeNode = vtkMRMLScalarVolumeNode::SafeDownCast(node);

  if (volumeNode == NULL || !volumeNode->GetImageData() )
    {
    return 0;
    }

  // Get vtk image from MRML node
  this->ImageData = volumeNode->GetImageData();
  int dsize[3];
  int dscalarType;

  this->ImageData->GetDimensions(dsize);

  dscalarType = this->ImageData->GetScalarType();
  if (dsize[0] != size[0] || dsize[1] != size[1] || dsize[2] != size[2] ||
      scalarType != dscalarType)
    {
    this->ImageData->Delete();
    this->ImageData = vtkImageData::New();
    this->ImageData->SetDimensions(size[0], size[1], size[2]);
    this->ImageData->SetExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1);
    this->ImageData->SetOrigin(0.0, 0.0, 0.0);
    this->ImageData->SetSpacing(1.0, 1.0, 1.0);
    this->ImageData->SetNumberOfScalarComponents(1);
    //this->ImageData->SetScalarType(scalarType);
    switch (scalarType)
      {
      case igtl::ImageMessage::TYPE_INT8:
        this->ImageData->SetScalarTypeToChar();
        break;
      case igtl::ImageMessage::TYPE_UINT8:
        this->ImageData->SetScalarTypeToUnsignedChar();
        break;
      case igtl::ImageMessage::TYPE_INT16:
        this->ImageData->SetScalarTypeToShort();
        break;
      case igtl::ImageMessage::TYPE_UINT16:
        this->ImageData->SetScalarTypeToUnsignedShort();
        break;
      case igtl::ImageMessage::TYPE_INT32:
        this->ImageData->SetScalarTypeToUnsignedLong();
        break;
      case igtl::ImageMessage::TYPE_UINT32:
        this->ImageData->SetScalarTypeToUnsignedLong();
        break;
      case igtl::ImageMessage::TYPE_FLOAT32:
        this->ImageData->SetScalarTypeToFloat();
        break;
      case igtl::ImageMessage::TYPE_FLOAT64:
        this->ImageData->SetScalarTypeToDouble();
        break;
      default:
        vtkErrorMacro ("Invalid Scalar Type\n");
        break;
      }
    this->ImageData->AllocateScalars();
    volumeNode->SetAndObserveImageData(this->ImageData);
    }

  float tx = matrix[0][0];
  float ty = matrix[1][0];
  float tz = matrix[2][0];
  float sx = matrix[0][1];
  float sy = matrix[1][1];
  float sz = matrix[2][1];
  float nx = matrix[0][2];
  float ny = matrix[1][2];
  float nz = matrix[2][2];
  float px = matrix[0][3];
  float py = matrix[1][3];
  float pz = matrix[2][3];

  // TODO:
  // It should be checked here if the dimension of vtkImageData
  // and arrived data is same.

  //vtkErrorMacro("IGTL image size = " << bytes);
  if (imgMsg->GetImageSize() == imgMsg->GetSubVolumeImageSize())
    {
    // In case that volume size == sub-volume size,
    // image is read directly to the memory area of vtkImageData
    // for better performance. 
    memcpy(this->ImageData->GetScalarPointer(),
           imgMsg->GetScalarPointer(), imgMsg->GetSubVolumeImageSize());
    }
  else
    {
    // In case of volume size != sub-volume size,
    // image is loaded into ImageReadBuffer, then copied to
    // the memory area of vtkImageData.
    
    // Check scalar size
    int scalarSize = imgMsg->GetScalarSize();
    
    char* imgPtr = (char*) this->ImageData->GetScalarPointer();
    char* bufPtr = (char*) imgMsg->GetScalarPointer();
    int sizei = size[0];
    int sizej = size[1];
    //int sizek = size[2];
    int subsizei = svsize[0];
    
    int bg_i = svoffset[0];
    //int ed_i = bg_i + svsize[0];
    int bg_j = svoffset[1];
    int ed_j = bg_j + svsize[1];
    int bg_k = svoffset[2];
    int ed_k = bg_k + svsize[2];
    
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
  float ntx = tx / psi;
  float nty = ty / psi;
  float ntz = tz / psi;
  float nsx = sx / psj;
  float nsy = sy / psj;
  float nsz = sz / psj;
  float nnx = nx / psk;
  float nny = ny / psk;
  float nnz = nz / psk;

  // Shift the center
  // NOTE: The center of the image should be shifted due to different
  // definitions of image origin between VTK (Slicer) and EndoNav;
  // EndoNav image has its origin at the center, while VTK image
  // has one at the corner.

  //float hfovi = psi * (size[0]-1) / 2.0;
  //float hfovj = psj * (size[1]-1) / 2.0;
  //float hfovk = psk * (size[2]-1) / 2.0;
  float hfovi = spacing[0] * psi * (size[0]-1) / 2.0;
  float hfovj = spacing[1] * psj * (size[1]-1) / 2.0;
  float hfovk = spacing[2] * psk * (size[2]-1) / 2.0;

  //float hfovk = 0;

  float cx = ntx * hfovi + nsx * hfovj + nnx * hfovk;
  float cy = nty * hfovi + nsy * hfovj + nny * hfovk;
  float cz = ntz * hfovi + nsz * hfovj + nnz * hfovk;
  px = px - cx;
  py = py - cy;
  pz = pz - cz;

  // set volume orientation
  vtkMatrix4x4* rtimgTransform = vtkMatrix4x4::New();
  rtimgTransform->Identity();
  rtimgTransform->SetElement(0, 0, ntx*spacing[0]);
  rtimgTransform->SetElement(1, 0, nty*spacing[0]);
  rtimgTransform->SetElement(2, 0, ntz*spacing[0]);
  rtimgTransform->SetElement(0, 1, nsx*spacing[1]);
  rtimgTransform->SetElement(1, 1, nsy*spacing[1]);
  rtimgTransform->SetElement(2, 1, nsz*spacing[1]);
  rtimgTransform->SetElement(0, 2, nnx*spacing[2]);
  rtimgTransform->SetElement(1, 2, nny*spacing[2]);
  rtimgTransform->SetElement(2, 2, nnz*spacing[2]);
  rtimgTransform->SetElement(0, 3, px);
  rtimgTransform->SetElement(1, 3, py);
  rtimgTransform->SetElement(2, 3, pz);

  //rtimgTransform->Invert();
  //volumeNode->SetRASToIJKMatrix(rtimgTransform);
  if (this->NodeCreated)
    {
    double range[2];
    range[0] = 0.0;
    range[1] = 256.0;
    volumeNode->GetImageData()->GetScalarRange(range);
    vtkMRMLScalarVolumeDisplayNode *displayNode = vtkMRMLScalarVolumeDisplayNode::SafeDownCast(volumeNode->GetDisplayNode());
    displayNode->SetLowerThreshold(range[0]);
    displayNode->SetUpperThreshold(range[1]);
    displayNode->SetWindow(range[1] - range[0]);
    displayNode->SetLevel(0.5 * (range[1] + range[0]) );

    volumeNode->SetIJKToRASMatrix(rtimgTransform);
    this->CenterImage(volumeNode);
    volumeNode->Modified();

    if ( this->SliceViewer)
      {
      this->SliceViewer->SetImageData(this->ImageData);
      this->FitImage();
      //this->SliceViewer->GetRenderWidget()->Reset();
      this->SliceViewer->Render();
      }
    else
      {
      this->ImageViewerUS->SetInput(this->ImageData);
      this->ImageViewerUS->Render();
      }
    }
  else
    {
    volumeNode->DisableModifiedEventOn();

    this->ImageData->RemoveAllObservers();

    double x,y,z;
    volumeNode->GetOrigin(x,y,z);
    this->ImageData->SetOrigin(x,y,z);
    volumeNode->GetSpacing(x,y,z);
    this->ImageData->SetSpacing(x,y,z);

    if ( this->SliceViewer)
      {
      this->SliceViewer->SetImageData(this->ImageData);
      //this->SliceViewer->GetRenderWidget()->Reset();
      //this->FitImage();
      this->SliceViewer->Render();
      }
    else
      {
      this->ImageViewerUS->SetInput(this->ImageData);
      this->ImageViewerUS->Render();
      }

    volumeNode->SetAndObserveImageData(this->ImageData);

    volumeNode->DisableModifiedEventOff();
    }

  rtimgTransform->Delete();

  // The following line is Necessary to update volume rendering
  // in VolumeRenderingCuda (Suggested by Nicholas Herlambang)
  // volumeNode->GetImageData()->Modified();
   
  this->NodeCreated = 0;

  return 1;
}


//---------------------------------------------------------------------------
int vtkIGTLToViewerImage::MRMLToIGTL(unsigned long event, vtkMRMLNode* mrmlNode, int* size, void** igtlMsg)
{
  
  return 0;
}


void vtkIGTLToViewerImage::CenterImage(vtkMRMLVolumeNode *volumeNode)
{
    if ( volumeNode )
      {
      vtkImageData *image = volumeNode->GetImageData();
      if (image) 
        {
        vtkMatrix4x4 *ijkToRAS = vtkMatrix4x4::New();
        volumeNode->GetIJKToRASMatrix(ijkToRAS);

        double dimsH[4];
        double rasCorner[4];
        int *dims = image->GetDimensions();
        dimsH[0] = dims[0] - 1;
        dimsH[1] = dims[1] - 1;
        dimsH[2] = dims[2] - 1;
        dimsH[3] = 0.;
        ijkToRAS->MultiplyPoint(dimsH, rasCorner);

        double origin[3];
        int i;
        for (i = 0; i < 3; i++)
          {
          origin[i] = -0.5 * rasCorner[i];
          }
        volumeNode->SetDisableModifiedEvent(1);
        volumeNode->SetOrigin(origin);
        volumeNode->SetDisableModifiedEvent(0);
        volumeNode->InvokePendingModifiedEvent();

        ijkToRAS->Delete();
        }
      }
}


void vtkIGTLToViewerImage::FitImage()
{

  int *rSize = this->SliceGUI->GetSliceViewer()->GetRenderWidget()->GetRenderer()->GetSize();
  this->SliceGUI->GetLogic()->FitSliceToAll ( rSize[0], rSize[1] );
  this->SliceGUI->GetSliceNode()->UpdateMatrices( );
  this->SliceGUI->GetLogic()->Modified();
  this->SliceGUI->GetSliceNode()->Modified( );
  // this->GetApplicationGUI()->GetSlicesControlGUI()->RequestFOVEntriesUpdate();
}
