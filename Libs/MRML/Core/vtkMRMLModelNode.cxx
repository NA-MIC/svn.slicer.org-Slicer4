/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See COPYRIGHT.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLModelNode.cxx,v $
Date:      $Date: 2006/03/03 22:26:39 $
Version:   $Revision: 1.3 $

=========================================================================auto=*/

// MRML includes
#include "vtkEventBroker.h"
#include "vtkMRMLFreeSurferProceduralColorNode.h"
#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkMRMLModelStorageNode.h"
#include "vtkMRMLTransformNode.h"
#include "vtkMRMLScene.h"

// VTK includes
#include <vtkAlgorithmOutput.h>
#include <vtkAssignAttribute.h>
#include <vtkCallbackCommand.h>
#include <vtkCellData.h>
#include <vtkColorTransferFunction.h>
#include <vtkFloatArray.h>
#include <vtkGeneralTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTrivialProducer.h>
#include <vtkVersion.h>

// STD includes
#include <cassert>
#include <sstream>

//----------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLModelNode);

//----------------------------------------------------------------------------
vtkMRMLModelNode::vtkMRMLModelNode()
{
#if (VTK_MAJOR_VERSION <= 5)
  this->PolyData = NULL;
#else
  this->PolyDataConnection = NULL;
#endif
}

//----------------------------------------------------------------------------
vtkMRMLModelNode::~vtkMRMLModelNode()
{
  this->SetAndObservePolyData(NULL);
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();
  this->Superclass::Copy(anode);
  vtkMRMLModelNode* modelNode = vtkMRMLModelNode::SafeDownCast(anode);
  if (modelNode && modelNode->GetPolyData())
    {
    // Only copy bulk data if it exists - this handles the case
    // of restoring from SceneViews, where the nodes will not
    // have bulk data.
#if (VTK_MAJOR_VERSION <= 5)
    this->SetAndObservePolyData(modelNode->GetPolyData());
#else
    this->SetPolyDataConnection(modelNode->GetPolyDataConnection());
#endif
    }
  this->EndModify(disabledModify);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData )
{
  this->Superclass::ProcessMRMLEvents(caller, event, callData);

#if (VTK_MAJOR_VERSION <= 5)
  if (this->GetPolyData() == vtkPolyData::SafeDownCast(caller) &&
      this->GetPolyData() != 0 &&
#else
  if (this->PolyDataConnection &&
      this->PolyDataConnection->GetProducer() == vtkAlgorithm::SafeDownCast(caller) &&
#endif
      event ==  vtkCommand::ModifiedEvent)
    {
    this->StorableModifiedTime.Modified();
    this->InvokeEvent(vtkMRMLModelNode::PolyDataModifiedEvent, NULL);
    }
}

//----------------------------------------------------------------------------
vtkMRMLModelDisplayNode* vtkMRMLModelNode::GetModelDisplayNode()
{
  return vtkMRMLModelDisplayNode::SafeDownCast(this->GetDisplayNode());
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "\nPoly Data:";
  if (this->GetPolyData())
    {
    os << "\n";
    this->GetPolyData()->PrintSelf(os, indent.GetNextIndent());
    }
  else
    {
    os << " none\n";
    }
}

//----------------------------------------------------------------------------
void vtkMRMLModelNode::SetAndObservePolyData(vtkPolyData *polyData)
{
#if (VTK_MAJOR_VERSION <= 5)
  if (polyData == this->PolyData)
    {
    return;
    }

  vtkPolyData* oldPolyData = this->PolyData;

  this->PolyData = polyData;

  if (this->PolyData != NULL)
    {
    vtkEventBroker::GetInstance()->AddObservation(
      this->PolyData, vtkCommand::ModifiedEvent, this, this->MRMLCallbackCommand );
    this->PolyData->Register(this);
    }

  this->SetPolyDataToDisplayNodes();

  if (oldPolyData != NULL)
    {
    vtkEventBroker::GetInstance()->RemoveObservations (
      oldPolyData, vtkCommand::ModifiedEvent, this, this->MRMLCallbackCommand );
    oldPolyData->UnRegister(this);
    }

  this->StorableModifiedTime.Modified();
  this->Modified();
  this->InvokeEvent( vtkMRMLModelNode::PolyDataModifiedEvent , this);
}
#else
  if (polyData == 0)
    {
    this->SetPolyDataConnection(0);
    }
  else
    {
    vtkNew<vtkTrivialProducer> tp;
    tp->SetOutput(polyData);
    this->SetPolyDataConnection(tp->GetOutputPort());
    }
}

//---------------------------------------------------------------------------
vtkPolyData* vtkMRMLModelNode::GetPolyData()
{
  vtkAlgorithm* producer = this->PolyDataConnection ?
    this->PolyDataConnection->GetProducer() : 0;
  return vtkPolyData::SafeDownCast(
    producer ? producer->GetOutputDataObject(
      this->PolyDataConnection->GetIndex()) : 0);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode
::SetPolyDataConnection(vtkAlgorithmOutput *newPolyDataConnection)
{
  if (newPolyDataConnection == this->PolyDataConnection)
    {
    return;
    }

  vtkAlgorithm* oldPolyDataAlgorithm = this->PolyDataConnection ?
    this->PolyDataConnection->GetProducer() : 0;

  this->PolyDataConnection = newPolyDataConnection;

  vtkAlgorithm* polyDataAlgorithm = this->PolyDataConnection ?
    this->PolyDataConnection->GetProducer() : 0;
  if (polyDataAlgorithm != NULL)
    {
    vtkEventBroker::GetInstance()->AddObservation(
      polyDataAlgorithm, vtkCommand::ModifiedEvent, this, this->MRMLCallbackCommand );
    polyDataAlgorithm->Register(this);
    }

  this->SetPolyDataToDisplayNodes();

  if (oldPolyDataAlgorithm != NULL)
    {
    vtkEventBroker::GetInstance()->RemoveObservations (
      oldPolyDataAlgorithm, vtkCommand::ModifiedEvent, this, this->MRMLCallbackCommand );
    oldPolyDataAlgorithm->UnRegister(this);
    }

  this->StorableModifiedTime.Modified();
  this->Modified();
  this->InvokeEvent( vtkMRMLModelNode::PolyDataModifiedEvent , this);
}
#endif

//---------------------------------------------------------------------------
void vtkMRMLModelNode::AddPointScalars(vtkDataArray *array)
{
  this->AddScalars(array, vtkAssignAttribute::POINT_DATA);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::AddCellScalars(vtkDataArray *array)
{
  this->AddScalars(array, vtkAssignAttribute::CELL_DATA);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::AddScalars(vtkDataArray *array, int location)
{
  if (array == NULL)
    {
    return;
    }
  if (this->GetPolyData() == NULL)
    {
    vtkErrorMacro("AddScalars: No polydata on model "
                  << (this->GetName() ? this->GetName() : "no_name"));
    return;
    }
  vtkDataSetAttributes* data =
    (location == vtkAssignAttribute::POINT_DATA ?
     vtkDataSetAttributes::SafeDownCast(this->GetPolyData()->GetPointData()) :
     vtkDataSetAttributes::SafeDownCast(this->GetPolyData()->GetCellData()));

  int numScalars = data->GetNumberOfArrays();
  vtkDebugMacro("Model node has " << numScalars << " scalars now, "
                << "adding " << array->GetName());

  data->AddArray(array);
  if (numScalars > 0)
    {
    // set the scalars
    data->SetActiveScalars(array->GetName());
    }
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::RemoveScalars(const char *scalarName)
{
  if (scalarName == NULL)
    {
    vtkErrorMacro("Scalar name is null");
    return;
    }
  if (this->GetPolyData() == NULL)
    {
    vtkErrorMacro("RemoveScalars: No poly data on model "
                  << (this->GetName() ? this->GetName() : "no_name"));
    return;
    }
  // try removing the array from the points first
  if (this->GetPolyData()->GetPointData())
    {
    this->GetPolyData()->GetPointData()->RemoveArray(scalarName);
    // it's a void method, how to check if it succeeded?
    }
  // try the cells
  if (this->GetPolyData()->GetCellData())
    {
    this->GetPolyData()->GetCellData()->RemoveArray(scalarName);
    }
}


//---------------------------------------------------------------------------
const char * vtkMRMLModelNode::GetActivePointScalarName(int type)
{
  if (this->GetPolyData() == NULL ||
      this->GetPolyData()->GetPointData() == NULL)
    {
    return NULL;
    }
  vtkAbstractArray* attributeArray =
    this->GetPolyData()->GetPointData()->GetAbstractAttribute(type);
  return attributeArray ? attributeArray->GetName() : NULL;
}

//---------------------------------------------------------------------------
const char * vtkMRMLModelNode::GetActiveCellScalarName(int type)
{
  if (this->GetPolyData() == NULL ||
      this->GetPolyData()->GetCellData() == NULL)
    {
    return NULL;
    }
  vtkAbstractArray* attributeArray =
    this->GetPolyData()->GetCellData()->GetAbstractAttribute(type);
  return attributeArray ? attributeArray->GetName() : NULL;
}

//---------------------------------------------------------------------------
bool vtkMRMLModelNode::HasPointScalarName(const char* scalarName)
{
  if (this->GetPolyData() == NULL ||
      this->GetPolyData()->GetPointData() == NULL)
    {
    return false;
    }
  return static_cast<bool>(
    this->GetPolyData()->GetPointData()->HasArray(scalarName));
}

//---------------------------------------------------------------------------
bool vtkMRMLModelNode::HasCellScalarName(const char* scalarName)
{
  if (this->GetPolyData() == NULL ||
      this->GetPolyData()->GetCellData() == NULL)
    {
    return false;
    }
  return static_cast<bool>(
    this->GetPolyData()->GetCellData()->HasArray(scalarName));
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::GetAttributeTypeFromString(const char* typeName)
{
  if (typeName != NULL && (strcmp(typeName, "") != 0))
    {
    for (int a = 0; a < vtkDataSetAttributes::NUM_ATTRIBUTES; a++)
      {
      if (strcmp(typeName, vtkDataSetAttributes::GetAttributeTypeAsString(a)) == 0)
        {
        return a;
        }
      }
    }
  return -1;
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::SetActivePointScalars(const char *scalarName, int attributeType)
{
  if (this->GetPolyData() == NULL)
    {
    return -1;
    }
#if (VTK_MAJOR_VERSION <= 5)
  this->GetPolyData()->Update();
#else
  this->PolyDataConnection->GetProducer()->Update();
#endif
  vtkPointData* pointData = this->GetPolyData()->GetPointData();
  return pointData? pointData->SetActiveAttribute(
    scalarName, attributeType) : 0;
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::SetActiveCellScalars(const char *scalarName, int attributeType)
{
  if (this->GetPolyData() == NULL)
    {
    return -1;
    }
#if (VTK_MAJOR_VERSION <= 5)
  this->GetPolyData()->Update();
#else
  this->PolyDataConnection->GetProducer()->Update();
#endif
  vtkCellData* cellData = this->GetPolyData()->GetCellData();
  return cellData ? cellData->SetActiveAttribute(
    scalarName, attributeType) : 0;
}

//---------------------------------------------------------------------------
int vtkMRMLModelNode::CompositeScalars(const char* backgroundName, const char* overlayName,
                                       float overlayMin, float overlayMax,
                                       int showOverlayPositive, int showOverlayNegative,
                                       int reverseOverlay)
{

    if (backgroundName == NULL || overlayName == NULL)
      {
      vtkErrorMacro("CompositeScalars: one of the input array names is null");
      return 0;
      }

    bool haveCurvScalars = false;
    // is there a curv scalar in the composite?
    if (strstr(backgroundName, "curv") != NULL ||
        strstr(overlayName, "curv") != NULL)
      {
      haveCurvScalars = true;
      }

    // get the scalars to composite, putting any curv file in scalars 1
    vtkDataArray *scalars1, *scalars2;
    if (!haveCurvScalars ||
        strstr(backgroundName, "curv") != NULL)
      {
      scalars1 = this->GetPolyData()->GetPointData()->GetScalars(backgroundName);
      scalars2 = this->GetPolyData()->GetPointData()->GetScalars(overlayName);
      }
    else
      {
      scalars1 = this->GetPolyData()->GetPointData()->GetScalars(overlayName);
      scalars2 = this->GetPolyData()->GetPointData()->GetScalars(backgroundName);
      }
    if (scalars1 == NULL || scalars2 == NULL)
      {
      vtkErrorMacro("CompositeScalars: unable to find the named scalar arrays " << backgroundName << " and/or " << overlayName);
      return 0;
      }
    if (scalars1->GetNumberOfTuples() != scalars1->GetNumberOfTuples())
      {
      vtkErrorMacro("CompositeScalars: sizes of scalar arrays don't match");
      return 0;
      }
    // Get the number of elements and initialize the composed scalar
    // array.
    int cValues = 0;
    cValues = scalars1->GetNumberOfTuples();

    vtkFloatArray* composedScalars = vtkFloatArray::New();

    std::stringstream ss;
    ss << backgroundName;
    ss << "+";
    ss << overlayName;
    std::string composedName = std::string(ss.str());
    composedScalars->SetName(composedName.c_str());
    composedScalars->Allocate( cValues );
    composedScalars->SetNumberOfComponents( 1 );

    // For each value, check the overlay value. If it's < min, use
    // the background value. If we're reversing, reverse the overlay
    // value. If we're not showing one side, use the background
    // value. If we are showing curvature (and have it), the
    // background value is our curvature value.
    float overlayMid = 0.5 * (overlayMax - overlayMin) + overlayMin; // 2.0;
    vtkDebugMacro("CompositeScalars: using overlay mid = " << overlayMid);
    float overlay = 0.0;
    float background = 0.0;
    for( int nValue = 0; nValue < cValues; nValue++ )
      {
      background = scalars1->GetTuple1(nValue);
      overlay = scalars2->GetTuple1(nValue);

      if( reverseOverlay )
        {
        overlay = -overlay;
        }
      if( overlay > 0 && !showOverlayPositive )
        {
        overlay = 0;
        }

      if( overlay < 0 && !showOverlayNegative )
        {
        overlay = 0;
        }

      // Insert the appropriate color into the composed array.
      if( overlay < overlayMin &&
          overlay > -overlayMin )
        {
        composedScalars->InsertNextValue( background );
        }
      else
        {
        composedScalars->InsertNextValue( overlay );
        }
      }

    // set up a colour node
    vtkMRMLProceduralColorNode *colorNode = vtkMRMLProceduralColorNode::New();
    colorNode->SetName(composedName.c_str());
    // set the type to avoid error messages when copy it, as the default is -1
    colorNode->SetType(vtkMRMLFreeSurferProceduralColorNode::Custom);
    vtkColorTransferFunction *func = colorNode->GetColorTransferFunction();

    // adapted from FS code that assumed that one scalar was curvature, the
    // other heat overlay
    const double EPS = 0.00001; // epsilon
    double curvatureMin = 0;

    if (haveCurvScalars)
      {
      curvatureMin = 0.5;
      }
    bool bUseGray = true;
    if( overlayMin <= curvatureMin )
      {
      curvatureMin = overlayMin - EPS;
      bUseGray = false;
      }
    func->AddRGBPoint( -overlayMax, 0, 1, 1 );
    func->AddRGBPoint( -overlayMid, 0, 0, 1 );
    func->AddRGBPoint( -overlayMin, 0, 0, 1 );

    if( bUseGray && overlayMin != 0 )
      {
      func->AddRGBPoint( -overlayMin + EPS, 0.5, 0.5, 0.5 );
      if( haveCurvScalars)
        {
        func->AddRGBPoint( -curvatureMin - EPS, 0.5, 0.5, 0.5 );
        }
      }
    if( haveCurvScalars && overlayMin != 0 )
      {
      func->AddRGBPoint( -curvatureMin, 0.6, 0.6, 0.6 );
      func->AddRGBPoint(  0,            0.6, 0.6, 0.6 );
      func->AddRGBPoint(  EPS,          0.4, 0.4, 0.4 );
      func->AddRGBPoint(  curvatureMin, 0.4, 0.4, 0.4 );
      }

    if ( bUseGray && overlayMin != 0 )
      {
      if( haveCurvScalars )
        {
        func->AddRGBPoint( curvatureMin + EPS, 0.5, 0.5, 0.5 );
        }
      func->AddRGBPoint( overlayMin - EPS, 0.5, 0.5, 0.5 );
      }

    func->AddRGBPoint( overlayMin, 1, 0, 0 );
    func->AddRGBPoint( overlayMid, 1, 0, 0 );
    func->AddRGBPoint( overlayMax, 1, 1, 0 );

    func->Build();

    // use the new colornode
    this->Scene->AddNode(colorNode);
    vtkDebugMacro("CompositeScalars: created color transfer function, and added proc color node to scene, id = " << colorNode->GetID());
    if (colorNode->GetID() != NULL)
      {
      this->GetModelDisplayNode()->SetAndObserveColorNodeID(colorNode->GetID());
      this->GetModelDisplayNode()->SetScalarRange(-overlayMax, overlayMax);
      }

    // add the new scalars
    this->AddPointScalars(composedScalars);

    // make them active
    this->GetModelDisplayNode()->SetActiveScalarName(composedName.c_str());

    // clean up
    colorNode->Delete();
    colorNode = NULL;
    composedScalars->Delete();
    composedScalars = NULL;

    return 1;
}

//---------------------------------------------------------------------------
bool vtkMRMLModelNode::CanApplyNonLinearTransforms()const
{
  return true;
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::ApplyTransform(vtkAbstractTransform* transform)
{
  if (this->GetPolyData() == 0)
    {
    return;
    }
  vtkTransformPolyDataFilter* transformFilter = vtkTransformPolyDataFilter::New();
#if (VTK_MAJOR_VERSION <= 5)
  transformFilter->SetInput(this->GetPolyData());
#else
  transformFilter->SetInputConnection(this->PolyDataConnection);
#endif

  transformFilter->SetTransform(transform);
  transformFilter->Update();

#if (VTK_MAJOR_VERSION <= 5)
  bool isInPipeline = !vtkTrivialProducer::SafeDownCast(
    this->GetPolyData() ? this->GetPolyData()->GetProducerPort()->GetProducer() : 0);
#else
  bool isInPipeline = vtkTrivialProducer::SafeDownCast(
    this->PolyDataConnection ? this->PolyDataConnection->GetProducer() : 0);
#endif
  vtkSmartPointer<vtkPolyData> polyData;
  if (isInPipeline)
    {
    polyData = vtkSmartPointer<vtkPolyData>::New();
    }
  else
    {
    polyData = this->GetPolyData();
    }
  polyData->DeepCopy(transformFilter->GetOutput());
  if (isInPipeline)
    {
#if (VTK_MAJOR_VERSION <= 5)
    this->SetAndObservePolyData(polyData);
#else
    this->SetPolyDataConnection(transformFilter->GetOutputPort());
#endif
    }
  transformFilter->Delete();
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::GetRASBounds(double bounds[6])
{
  this->Superclass::GetRASBounds( bounds);

  if (this->GetPolyData() == NULL)
    {
    return;
    }

  this->GetPolyData()->ComputeBounds();

  double boundsLocal[6];
  this->GetPolyData()->GetBounds(boundsLocal);

  this->TransformBoundsToRAS(boundsLocal, bounds);

}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::TransformBoundsToRAS(double inputBounds_Local[6], double outputBounds_RAS[6])
{
  vtkMRMLTransformNode *transformNode = this->GetParentTransformNode();
  if ( !transformNode )
    {
    // node is not transformed, therefore RAS=local
    for (int i=0; i<6; i++)
      {
      outputBounds_RAS[i]=inputBounds_Local[i];
      }
    return;
    }

  vtkNew<vtkGeneralTransform> transformLocalToRAS;
  transformNode->GetTransformToWorld(transformLocalToRAS.GetPointer());

  double cornerPoints_Local[8][4] =
    {
    {inputBounds_Local[0], inputBounds_Local[2], inputBounds_Local[4], 1},
    {inputBounds_Local[0], inputBounds_Local[3], inputBounds_Local[4], 1},
    {inputBounds_Local[0], inputBounds_Local[2], inputBounds_Local[5], 1},
    {inputBounds_Local[0], inputBounds_Local[3], inputBounds_Local[5], 1},
    {inputBounds_Local[1], inputBounds_Local[2], inputBounds_Local[4], 1},
    {inputBounds_Local[1], inputBounds_Local[3], inputBounds_Local[4], 1},
    {inputBounds_Local[1], inputBounds_Local[2], inputBounds_Local[5], 1},
    {inputBounds_Local[1], inputBounds_Local[3], inputBounds_Local[5], 1}
    };

  // initialize bounds with point 0
  double* cornerPoint_RAS = transformLocalToRAS->TransformDoublePoint(cornerPoints_Local[0]);
  for ( int i=0; i<3; i++)
    {
    outputBounds_RAS[2*i]   = cornerPoint_RAS[i];
    outputBounds_RAS[2*i+1] = cornerPoint_RAS[i];
    }

  // update bounds with the rest of the points
  for ( int i=1; i<8; i++)
    {
    cornerPoint_RAS = transformLocalToRAS->TransformPoint( cornerPoints_Local[i] );
    for (int n=0; n<3; n++)
      {
      if (cornerPoint_RAS[n] < outputBounds_RAS[2*n]) // min bound
        {
        outputBounds_RAS[2*n] = cornerPoint_RAS[n];
        }
      if (cornerPoint_RAS[n] > outputBounds_RAS[2*n+1]) // max bound
        {
        outputBounds_RAS[2*n+1] = cornerPoint_RAS[n];
        }
      }
   }
}

//---------------------------------------------------------------------------
vtkMRMLStorageNode* vtkMRMLModelNode::CreateDefaultStorageNode()
{
  return vtkMRMLStorageNode::SafeDownCast(vtkMRMLModelStorageNode::New());
}
//---------------------------------------------------------------------------
void vtkMRMLModelNode::OnNodeReferenceAdded(vtkMRMLNodeReference *reference)
{
  if (std::string(reference->GetReferenceRole()) == this->DisplayNodeReferenceRole)
    {
    this->UpdateDisplayNodePolyData(vtkMRMLDisplayNode::SafeDownCast(reference->ReferencedNode));
    }
  Superclass::OnNodeReferenceAdded(reference);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::OnNodeReferenceModified(vtkMRMLNodeReference *reference)
{
   this->UpdateDisplayNodePolyData(vtkMRMLDisplayNode::SafeDownCast(reference->ReferencedNode));
   Superclass::OnNodeReferenceModified(reference);
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode::UpdateDisplayNodePolyData(vtkMRMLDisplayNode *dnode)
{
  vtkMRMLModelDisplayNode* modelDisplayNode =
    vtkMRMLModelDisplayNode::SafeDownCast(dnode);
  if (modelDisplayNode)
    {
    this->SetPolyDataToDisplayNode(modelDisplayNode);
    }
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode
::SetPolyDataToDisplayNodes()
{
  int ndisp = this->GetNumberOfDisplayNodes();
  for (int n=0; n<ndisp; n++)
    {
    vtkMRMLModelDisplayNode *dnode = vtkMRMLModelDisplayNode::SafeDownCast(
      this->GetNthDisplayNode(n));
    if (dnode)
      {
      this->SetPolyDataToDisplayNode(dnode);
      }
    }
}

//---------------------------------------------------------------------------
void vtkMRMLModelNode
::SetPolyDataToDisplayNode(vtkMRMLModelDisplayNode* modelDisplayNode)
{
  assert(modelDisplayNode);
#if (VTK_MAJOR_VERSION <= 5)
  modelDisplayNode->SetInputPolyData(this->PolyData);
#else
  modelDisplayNode->SetInputPolyDataConnection(this->GetPolyDataConnection());
#endif
}

//---------------------------------------------------------------------------
bool vtkMRMLModelNode::GetModifiedSinceRead()
{
  return this->Superclass::GetModifiedSinceRead() ||
    (this->GetPolyData() && this->GetPolyData()->GetMTime() > this->GetStoredTime());
}
