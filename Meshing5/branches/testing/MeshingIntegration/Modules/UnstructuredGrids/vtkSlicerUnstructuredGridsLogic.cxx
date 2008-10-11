/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerUnstructuredGridsLogic.cxx,v $
  Date:      $Date: 2006/01/06 17:56:48 $
  Version:   $Revision: 1.58 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"
#include <itksys/SystemTools.hxx> 
#include <itksys/Directory.hxx> 

#include "vtkSlicerUnstructuredGridsLogic.h"

#include "vtkMRMLUnstructuredGridNode.h"
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMRMLUnstructuredGridDisplayNode.h"
#include "vtkSlicerColorLogic.h"

vtkCxxRevisionMacro(vtkSlicerUnstructuredGridsLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerUnstructuredGridsLogic);

//----------------------------------------------------------------------------
vtkSlicerUnstructuredGridsLogic::vtkSlicerUnstructuredGridsLogic()
{
  this->ActiveUnstructuredGridNode = NULL;
}

//----------------------------------------------------------------------------
vtkSlicerUnstructuredGridsLogic::~vtkSlicerUnstructuredGridsLogic()
{
  if (this->ActiveUnstructuredGridNode != NULL)
    {
        this->ActiveUnstructuredGridNode->Delete();
        this->ActiveUnstructuredGridNode = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsLogic::ProcessMRMLEvents(vtkObject * /*caller*/, 
                                            unsigned long /*event*/, 
                                            void * /*callData*/)
{
  // TODO: implement if needed
}

//----------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsLogic::SetActiveUnstructuredGridNode(vtkMRMLUnstructuredGridNode *activeNode)
{
  vtkSetMRMLNodeMacro(this->ActiveUnstructuredGridNode, activeNode );
  this->Modified();
}

//----------------------------------------------------------------------------
int vtkSlicerUnstructuredGridsLogic::AddUnstructuredGrids (const char* dirname, const char* suffix )
{
  std::string ssuf = suffix;
  itksys::Directory dir;
  dir.Load(dirname);
 
  int nfiles = dir.GetNumberOfFiles();
  int res = 1;
  for (int i=0; i<nfiles; i++) {
    const char* filename = dir.GetFile(i);
    std::string sname = filename;
    if (!itksys::SystemTools::FileIsDirectory(filename))
      {
      if ( sname.find(ssuf) != std::string::npos )
        {
        std::string fullPath = std::string(dir.GetPath())
            + "/" + filename;
        if (this->AddUnstructuredGrid((char *)fullPath.c_str()) == NULL) 
          {
          res = 0;
          }
        }
      }
  }
  return res;
}

//----------------------------------------------------------------------------
vtkMRMLUnstructuredGridNode* vtkSlicerUnstructuredGridsLogic::AddUnstructuredGrid (const char* filename)
{
  vtkMRMLUnstructuredGridNode *UnstructuredGridNode = vtkMRMLUnstructuredGridNode::New();
  vtkMRMLUnstructuredGridDisplayNode *displayNode = vtkMRMLUnstructuredGridDisplayNode::New();
  vtkMRMLUnstructuredGridStorageNode *mStorageNode = vtkMRMLUnstructuredGridStorageNode::New();
  vtkMRMLStorageNode *storageNode = NULL;
  
  mStorageNode->SetFileName(filename);
   
  if (mStorageNode->ReadData(UnstructuredGridNode) != 0)
    {
    storageNode = mStorageNode;
    }
  
  if (storageNode != NULL)
    {
    const itksys_stl::string fname(filename);
    itksys_stl::string name = itksys::SystemTools::GetFilenameName(fname);
    UnstructuredGridNode->SetName(name.c_str());

    this->GetMRMLScene()->SaveStateForUndo();

    UnstructuredGridNode->SetScene(this->GetMRMLScene());
    storageNode->SetScene(this->GetMRMLScene());
    displayNode->SetScene(this->GetMRMLScene()); 

    this->GetMRMLScene()->AddNodeNoNotify(storageNode);  
    this->GetMRMLScene()->AddNodeNoNotify(displayNode);
    UnstructuredGridNode->SetStorageNodeID(storageNode->GetID());
    UnstructuredGridNode->SetAndObserveDisplayNodeID(displayNode->GetID());  
    displayNode->SetUnstructuredGrid(UnstructuredGridNode->GetUnstructuredGrid());
    
    this->GetMRMLScene()->AddNode(UnstructuredGridNode);  

    //this->Modified();  

    UnstructuredGridNode->Delete();
    }
  else
    {
    vtkDebugMacro("Couldn't read file, returning null UnstructuredGrid node: " << filename);
    UnstructuredGridNode->Delete();
    UnstructuredGridNode = NULL;
    }
  mStorageNode->Delete();
  displayNode->Delete();

  return UnstructuredGridNode;  
}
//----------------------------------------------------------------------------
int vtkSlicerUnstructuredGridsLogic::SaveUnstructuredGrid (const char* filename, vtkMRMLUnstructuredGridNode *UnstructuredGridNode)
{
   if (UnstructuredGridNode == NULL || filename == NULL)
    {
    return 0;
    }
  
  vtkMRMLUnstructuredGridStorageNode *storageNode = NULL;
  vtkMRMLStorageNode *snode = UnstructuredGridNode->GetStorageNode();
  if (snode != NULL)
    {
    storageNode = vtkMRMLUnstructuredGridStorageNode::SafeDownCast(snode);
    }
  if (storageNode == NULL)
    {
    storageNode = vtkMRMLUnstructuredGridStorageNode::New();
    storageNode->SetScene(this->GetMRMLScene());
    this->GetMRMLScene()->AddNode(storageNode);  
    UnstructuredGridNode->SetStorageNodeID(storageNode->GetID());
    storageNode->Delete();
    }

  //storageNode->SetAbsoluteFileName(true);
  storageNode->SetFileName(filename);

  int res = storageNode->WriteData(UnstructuredGridNode);

  
  return res;

}


//----------------------------------------------------------------------------
void vtkSlicerUnstructuredGridsLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);

  os << indent << "vtkSlicerUnstructuredGridsLogic:             " << this->GetClassName() << "\n";

  os << indent << "ActiveUnstructuredGridNode: " <<
    (this->ActiveUnstructuredGridNode ? this->ActiveUnstructuredGridNode->GetName() : "(none)") << "\n";
}

//----------------------------------------------------------------------------
int vtkSlicerUnstructuredGridsLogic::AddScalar(const char* filename, vtkMRMLUnstructuredGridNode *UnstructuredGridNode)
{
  if (UnstructuredGridNode == NULL ||
      filename == NULL)
    {
    vtkErrorMacro("UnstructuredGrid node or file name are null.");
    return 0;
    }  

   // get the storage node and use it to read the scalar file
  vtkMRMLUnstructuredGridStorageNode *storageNode = NULL;
  vtkMRMLStorageNode *snode = UnstructuredGridNode->GetStorageNode();
  if (snode != NULL)
    {
    storageNode = vtkMRMLUnstructuredGridStorageNode::SafeDownCast(snode);
    }
  if (storageNode == NULL)
    {
    vtkErrorMacro("UnstructuredGrid "  << UnstructuredGridNode->GetName() << " does not have a freesurfer storage node associated with it, cannot load scalar overlay.");
    return 0;
    }
  storageNode->SetFileName(filename);
  storageNode->ReadData(UnstructuredGridNode);

  // check to see if the UnstructuredGrid display node has a colour node already
  vtkMRMLUnstructuredGridDisplayNode *displayNode = UnstructuredGridNode->GetUnstructuredGridDisplayNode();
  if (displayNode == NULL)
    {
    vtkWarningMacro("UnstructuredGrid " << UnstructuredGridNode->GetName() << "'s display node is null\n");
    }
  else
    {
    vtkMRMLColorNode *colorNode = vtkMRMLColorNode::SafeDownCast(displayNode->GetColorNode());
    if (colorNode == NULL)
      {
      vtkSlicerColorLogic *colorLogic = vtkSlicerColorLogic::New();
      displayNode->SetAndObserveColorNodeID(colorLogic->GetDefaultModelColorNodeID());
      colorLogic->Delete();
      }
    
    }
  return 1;
}
