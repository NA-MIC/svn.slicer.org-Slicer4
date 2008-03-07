/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkOpenIGTLinkTclHelper.h,v $
  Date:      $Date: 2006/01/06 17:58:01 $
  Version:   $Revision: 1.4 $

=========================================================================auto=*/
// .NAME vtkOpenIGTLinkTclHelper - a bridge between tcl and vtk
// -- this class allows binary data to be read from a tcl channel 
//    and put into a vtk data structure (vtkImageData for now)

// .SECTION Description

// .SECTION See Also
// vtkTkRenderWidget 


#ifndef __vtkOpenIGTLinkTclHelper_h
#define __vtkOpenIGTLinkTclHelper_h

#include "vtkTcl.h"
#include "vtkObject.h"
#include "vtkMRMLScene.h"
#include "vtkImageData.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMatrix4x4.h"
#include "vtkOpenIGTLinkDaemonWin32Header.h"
#include "vtkSlicerApplicationLogic.h"

//typedef stract {
//  char deviceName[20];
//  long long crc;       // 64 bit CRC
//  long long bodySize;  // 64 bit size of the body
//} OpenIGTLinkHeader;

class VTK_OPENIGTLINKDAEMON_EXPORT vtkOpenIGTLinkTclHelper : public vtkObject
{

public:
    void PrintSelf(ostream& os, vtkIndent indent);

    vtkTypeRevisionMacro(vtkOpenIGTLinkTclHelper, vtkObject);

    vtkSetObjectMacro(ImageData, vtkImageData);
    vtkGetObjectMacro(ImageData, vtkImageData);
    vtkSetObjectMacro(Matrix, vtkMatrix4x4);
    vtkGetObjectMacro(Matrix, vtkMatrix4x4);
    //vtkGetObjectMacro(Scene, vtkMRMLScene);
    //vtkSetObjectMacro(Scene, vtkMRMLScene);
    void SetScene(vtkMRMLScene* s)
    {
      this->Scene = s;
    }
    
    void SetAppLogic(vtkSlicerApplicationLogic* al)
    {
      this->AppLogic = al;
    }
      
    static vtkOpenIGTLinkTclHelper *New();

    void SetInterpFromCommand(unsigned long tag);

    void SendImageDataScalars(char *sockname);

    void OnReceiveOpenIGTLinkMessage(char *sockname);
    void ReceiveMatrix(char *sockname);
    void SendMessage(char *sockname);

    void ReceiveImage(Tcl_Channel channel, char* deviceName, long long size, long long crc, int newNode);
    void ReceiveTransform(Tcl_Channel channel, char* deviceName, long long size, long long crc, int newNode);

    const char *Execute (char *Command);

    void PerformVTKSocketHandshake(char *sockname);

protected:
    vtkOpenIGTLinkTclHelper();
    ~vtkOpenIGTLinkTclHelper(); 

    vtkSlicerApplicationLogic* AppLogic;
    vtkMRMLScene *Scene;
    vtkMRMLVolumeNode *VolumeNode;
    vtkImageData *ImageData;    
    vtkMatrix4x4 *MeasurementFrame;    
    vtkMatrix4x4 *Matrix;    
    Tcl_Interp *Interp;           /* Tcl interpreter */
    char* ImageReadBuffer;
    int   ImageReadBufferSize;
};

#endif
