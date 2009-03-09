/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkCellWallSegmentLogic.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

// needed for bcopy
#include <strings.h>

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "vtkCellWallSegmentLogic.h"
#include "vtkCellWallSegment.h"
#include "vtkCellWallVisSeg.h"

#include "vtkMRMLScene.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkImageReader2.h"

vtkCellWallSegmentLogic* vtkCellWallSegmentLogic::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCellWallSegmentLogic");
  if(ret)
    {
      return (vtkCellWallSegmentLogic*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkCellWallSegmentLogic;
}


//----------------------------------------------------------------------------
vtkCellWallSegmentLogic::vtkCellWallSegmentLogic()
{
  this->CellWallSegmentNode = NULL;
  this->VisSegInstance = new vtkCellWallVisSeg();  
  this->Reader = NULL;
}

//----------------------------------------------------------------------------
vtkCellWallSegmentLogic::~vtkCellWallSegmentLogic()
{
  vtkSetMRMLNodeMacro(this->CellWallSegmentNode, NULL);
}

//----------------------------------------------------------------------------
void vtkCellWallSegmentLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  
}



void vtkCellWallSegmentLogic::InitializeMRMLVolume(char* filename)
{

  // check if MRML node is present 
  if (this->CellWallSegmentNode == NULL)
    {
    vtkErrorMacro("No input CellWallSegmentNode found");
    return;
    }
  
   // trim the extension off the filename and add .ids to point to the binary
  int filenamelength;
  char fileids[256];
   filenamelength = strlen(filename);
   strncpy(fileids,filename,filenamelength-4);
   fileids[filenamelength-4]='\0';
   // add the .ids extension to the binary component of the datafile
   strcat(fileids,".ids");

  // read the volume and assign it to an output node.  This way, the volume
    // will appear in the Slicer viewers after it is read in, and it can be 
    // used 
  
   RIMAGEDEF& rimage = this->VisSegInstance->getRimage();
    if(Reader==NULL) Reader = vtkImageReader2::New();
      Reader->SetDataScalarTypeToUnsignedChar();
      Reader->SetDataExtent(0, rimage.nx-1, 0, rimage.ny-1, 0, rimage.nz-1);
      Reader->SetDataSpacing(1.0, 1.0, rimage.aspratio);  
      Reader->SetFileDimensionality(3);
      Reader->SetDataOrigin(0,0,0);
      Reader->SetFileName(fileids);
      Reader->Modified();
      Reader->Update();

    
    // find output volume
    vtkMRMLScalarVolumeNode *outVolume =  vtkMRMLScalarVolumeNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->CellWallSegmentNode->GetOutputVolumeRef()));
    if (outVolume == NULL)
      {
      vtkErrorMacro("No output volume found with id= " << this->CellWallSegmentNode->GetOutputVolumeRef());
      return;
      }
    

  // copy RASToIJK matrix, and other attributes from input to output
  std::string name (outVolume->GetName());
  std::string id (outVolume->GetID());

//  outVolume->CopyOrientation(inVolume);
//  outVolume->SetAndObserveTransformNodeID(inVolume->GetTransformNodeID());

  outVolume->SetName(name.c_str());
  //outVolume->SetID(id.c_str());
 
  //this->VisSegInstance->compute2DBoundary(1);
  
  // create filter
  
  // set ouput of the filter to VolumeNode's ImageData
  // TODO FIX the bug of the image is deallocated unless we do DeepCopy
 vtkImageData* image = vtkImageData::New(); 
  image->DeepCopy( this->Reader->GetOutput() );
  outVolume->SetAndObserveImageData(image);
  image->Delete();
  outVolume->SetModifiedSinceRead(1);

  //outVolume->SetImageData(this->GradientAnisotropicDiffusionImageFilter->GetOutput());

  // delete the filter
  this->Reader->Delete();
}



void vtkCellWallSegmentLogic::InitializeMRMLSegmentationVolume()
{

  // check if MRML node is present 
  if (this->CellWallSegmentNode == NULL)
    {
    vtkErrorMacro("No input CellWallSegmentNode found");
    return;
    }
  
  // read the segmentation volume and assign it to an output node.  This way, the volume
    // will appear in the Slicer viewers after it is read in, and it can be 
    // used 
  
  RIMAGEDEF& rimage = this->VisSegInstance->getRimage();
   unsigned char *pixbuf = this->VisSegInstance->getPixbuf();
   
   // make new VTK image and point it to the pixbuf voxels so it can be displayed
   vtkImageData* image = vtkImageData::New();
      bcopy(pixbuf,image->GetScalarPointer(),(rimage.nx*rimage.ny*rimage.nz));
      image->SetScalarTypeToUnsignedChar();
      //image->SetExtent(0, rimage.nx-1, 0, rimage.ny-1, 0, rimage.nz-1);
      image->SetSpacing(1.0, 1.0, rimage.aspratio);  
      //*** don't know the depth
      image->SetDimensions(rimage.nx,rimage.ny,1);
      image->SetOrigin(0,0,0);
      image->Modified();
      image->Update();
    
    // find output volume
    vtkMRMLScalarVolumeNode *segmentVolume =  vtkMRMLScalarVolumeNode::SafeDownCast(this->GetMRMLScene()->GetNodeByID(this->CellWallSegmentNode->GetSegmentationVolumeRef()));
    if (segmentVolume == NULL)
      {
      vtkErrorMacro("No segment volume found with id= " << this->CellWallSegmentNode->GetSegmentationVolumeRef());
      return;
      }
    

  // copy RASToIJK matrix, and other attributes from input to output
  std::string name (segmentVolume->GetName());
  std::string id (segmentVolume->GetID());

//  outVolume->CopyOrientation(inVolume);
//  outVolume->SetAndObserveTransformNodeID(inVolume->GetTransformNodeID());

  segmentVolume->SetName(name.c_str());
  segmentVolume->SetAndObserveImageData(image);
  image->Delete();
  segmentVolume->SetModifiedSinceRead(1);

}





void vtkCellWallSegmentLogic::Perform2DSegmentation() 
{
    vtkDebugMacro("vtkCellWallSegmentLogic: 2D segmentation " << "\n");

    // find the currently selected fiducial list and set the cell center and cell boundary points
    // to the values in the fiducial points.
    
    char* fiducialListID = this->CellWallSegmentNode->GetFiducialListRef();
    vtkMRMLFiducialListNode *fidList = vtkMRMLFiducialListNode::SafeDownCast(
                this->GetMRMLScene()->GetNodeByID(fiducialListID));

    int numPoints = fidList->GetNumberOfFiducials();
    if (numPoints == 2) 
    { 
        vtkDebugMacro("hurray! found two points");
        float *center, *boundary;
        double centerAsDoubles[4], boundaryAsDoubles[4]; 
        center = fidList->GetNthFiducialXYZ(0);
        boundary = fidList->GetNthFiducialXYZ(1);
        vtkDebugMacro("found points: center ("<< center[0] << center[1] << center[2] << "\n");
        cout  << "found points: center ("<< center[0] << " " << center[1] << " " << center[2] << ")" << endl;
        cout  << "found points: boundary ("<< boundary[0] << " " << boundary[1] << " " << boundary[2] << ")" << endl;
        for (int j=0;j<3;j++) {
            centerAsDoubles[j] = (double)center[j];
            boundaryAsDoubles[j] = (double)boundary[j];
        }
        // set the value as a homogeneous coordinate 
        centerAsDoubles[3] = 1.0;  boundaryAsDoubles[3] = 1.0;
       
        this->VisSegInstance->setCellCenter(centerAsDoubles);
        this->VisSegInstance->setCellEdge(boundaryAsDoubles);
        cout  << "found points: center ("<< centerAsDoubles[0] << " " << centerAsDoubles[1] << " " << centerAsDoubles[2] << ")" << endl;
        cout  << "found points: boundary ("<< boundaryAsDoubles[0] << " " << boundaryAsDoubles[1] << " " << boundaryAsDoubles[2] << ")" << endl;
        this->VisSegInstance->compute2DBoundary(0);
        // copy to output volume (segmentation)
        cout << "2D boundary complete. copy to segmentation volume" << endl;
        this->InitializeMRMLSegmentationVolume();
    } else 
        vtkErrorMacro("Fiducial list did not contain two points")
}


 void vtkCellWallSegmentLogic::Perform3DSegmentation() {cout << "3D Segmentation" << endl;}
