/*=auto=========================================================================

  Portions (c) Copyright 2006 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: $
  Date:      $Date: $
  Version:   $Revision: $

=========================================================================auto=*/

// .NAME vtkRealTimeImagingLogic - slicer logic class for Locator module 
// .SECTION Description
// This class manages the logic associated with tracking device for
// IGT. 


#ifndef __vtkRealTimeImagingLogic_h
#define __vtkRealTimeImagingLogic_h

#include "vtkRealTimeImagingWin32Header.h"
#include "vtkSlicerBaseLogic.h"
#include "vtkSlicerLogic.h"

#include "vtkMRMLModelNode.h"
#include "vtkMRML.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLVectorVolumeNode.h"

#include "vtkPoints.h"
#include "vtkMatrix4x4.h"
#include "vtkUnsignedShortArray.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkSlicerColorLogic.h"


#ifdef USE_OPENTRACKER
#ifdef OT_VERSION_20
#include "OpenTracker/OpenTracker.h"
#include "OpenTracker/common/CallbackModule.h"
#endif
#ifdef OT_VERSION_13
#include "OpenTracker.h"
#include "common/CallbackModule.h"
#endif

#include "Image.h"
#if defined(_WIN32) || defined(__CYGWIN__)
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif
#include "stdio.h"
using namespace ot;
#endif



class VTK_REALTIMEIMAGING_EXPORT vtkRealTimeImagingLogic : public vtkSlicerLogic 
{
public:

    // The Usual vtk class functions
    static vtkRealTimeImagingLogic *New();
    vtkTypeRevisionMacro(vtkRealTimeImagingLogic,vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent);


    vtkGetObjectMacro(LocatorMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(LandmarkTransformMatrix,vtkMatrix4x4);
    vtkGetObjectMacro(LocatorNormalTransform,vtkTransform);
    vtkGetObjectMacro(PixelArray,vtkUnsignedShortArray);
    vtkGetObjectMacro(scalarNode,vtkMRMLScalarVolumeNode);
    //vtkGetObjectMacro(OTInputImage,short);

    vtkSetMacro(UseRegistration,int);
    vtkGetMacro(UseRegistration,int);

    vtkGetMacro(NumberOfPoints,int);

    void Init(const char *configfile);
    void CloseConnection();
    void PollRealtime();

#ifdef USE_OPENTRACKER
    static void callbackF(const Node&, const Event &event, void *data);
#endif

    // t1, t2, t3: target landmarks 
    // s1, s2, s3: source landmarks 
    void AddPoint(int id, float t1, float t2, float t3, float s1, float s2, float s3);
    int DoRegistration();

    void SetNumberOfPoints(int no);

    // Description:
    // Description:
    // Update logic state when MRML scene chenges
    void ProcessMRMLEvents(vtkObject * caller, unsigned long event, void * callData);

    void SetLocatorTransforms();
    void UpdateSliceImages();

    // RSierra, S DiMaio; adopted this funtion from vtkSlicerVolumesLogic.cxx, Jan 10 2007
    vtkMRMLVolumeNode* AddRealTimeVolumeNode(const char* volname);

    //simond
    short                   OTInputImage[256*256];

protected:

    vtkRealTimeImagingLogic();
    ~vtkRealTimeImagingLogic();
    vtkRealTimeImagingLogic(const vtkRealTimeImagingLogic&);
    void operator=(const vtkRealTimeImagingLogic&);

#ifdef USE_OPENTRACKER
    Context *context;
#endif

    int NumberOfPoints;
    int UseRegistration;
    vtkPoints *SourceLandmarks;
    vtkPoints *TargetLandmarks;
    vtkMatrix4x4 *LandmarkTransformMatrix;
    vtkTransform *LocatorNormalTransform;

//BTX
    float p[3], n[3];
    float pOld[3], nOld[3];
//ETX


    void quaternion2xyz(float* orientation, float *normal, float *transnormal); 
    vtkMatrix4x4 *LocatorMatrix;

    void ApplyTransform(float *position, float *norm, float *transnorm);

    void Normalize(float *a);
    void Cross(float *a, float *b, float *c);

  //simond
  vtkMRMLScalarVolumeNode *scalarNode;
  vtkUnsignedShortArray   *PixelArray;
};

#endif


  
