/*=auto=========================================================================

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Program:   3D Slicer
  Module:    $RCSfile: vtkSlicerModelsLogic.cxx,v $
  Date:      $Date: 2006/01/06 17:56:48 $
  Version:   $Revision: 1.58 $

=========================================================================auto=*/

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"
#include <itksys/SystemTools.hxx> 
#include <itksys/Directory.hxx> 

#include "vtkSlicerModelsLogic.h"

#include "vtkMRMLModelNode.h"
#include "vtkMRMLModelStorageNode.h"
#include "vtkMRMLModelDisplayNode.h"
#include "vtkSlicerColorLogic.h"
#include "vtkMRMLFreeSurferModelStorageNode.h"
#include "vtkMRMLFreeSurferModelOverlayStorageNode.h"
#include "vtkMRMLUnstructuredGridStorageNode.h"
#include "vtkMRMLUnstructuredGridDisplayNode.h"

#include "vtkUnstructuredGridReader.h"
#include "vtkDataReader.h"

vtkCxxRevisionMacro(vtkSlicerModelsLogic, "$Revision: 1.9.12.1 $");
vtkStandardNewMacro(vtkSlicerModelsLogic);

//----------------------------------------------------------------------------
vtkSlicerModelsLogic::vtkSlicerModelsLogic()
{
  this->ActiveModelNode = NULL;
}

//----------------------------------------------------------------------------
vtkSlicerModelsLogic::~vtkSlicerModelsLogic()
{
  if (this->ActiveModelNode != NULL)
    {
        this->ActiveModelNode->Delete();
        this->ActiveModelNode = NULL;
    }
}

//----------------------------------------------------------------------------
void vtkSlicerModelsLogic::ProcessMRMLEvents(vtkObject * /*caller*/, 
                                            unsigned long /*event*/, 
                                            void * /*callData*/)
{
  // TODO: implement if needed
}

//----------------------------------------------------------------------------
void vtkSlicerModelsLogic::SetActiveModelNode(vtkMRMLModelNode *activeNode)
{
  vtkSetMRMLNodeMacro(this->ActiveModelNode, activeNode );
  this->Modified();
}

//----------------------------------------------------------------------------
int vtkSlicerModelsLogic::AddModels (const char* dirname, const char* suffix )
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
        if (this->AddModel((char *)fullPath.c_str()) == NULL) 
          {
          res = 0;
          }
        }
      }
  }
  return res;
}



//----------------------------------------------------------------------------
vtkMRMLModelNode* vtkSlicerModelsLogic::AddModel (const char* filename)
{
  vtkMRMLModelNode *modelNode = vtkMRMLModelNode::New();
  vtkMRMLUnstructuredGridNode *ugNode = vtkMRMLUnstructuredGridNode::New();
  vtkMRMLModelDisplayNode *displayNode = vtkMRMLModelDisplayNode::New();
  vtkMRMLUnstructuredGridDisplayNode *ugDisplayNode = vtkMRMLUnstructuredGridDisplayNode::New();
  vtkMRMLModelStorageNode *mStorageNode = vtkMRMLModelStorageNode::New();
  vtkMRMLUnstructuredGridStorageNode *ugStorageNode = vtkMRMLUnstructuredGridStorageNode::New();
  vtkMRMLFreeSurferModelStorageNode *fsmStorageNode = vtkMRMLFreeSurferModelStorageNode::New();
  fsmStorageNode->SetUseStripper(0);  // turn off stripping by default (breaks some pickers)
  vtkMRMLStorageNode *storageNode = NULL;

  // check for local or remote files
  int useURI = 0; // false;
  if (this->GetMRMLScene()->GetCacheManager() != NULL)
    {
    useURI = this->GetMRMLScene()->GetCacheManager()->IsRemoteReference(filename);
    vtkDebugMacro("AddModel: file name is remote: " << filename);
    }
  
  itksys_stl::string name;
  const char *localFile;
  if (useURI)
    {
    mStorageNode->SetURI(filename);
    fsmStorageNode->SetURI(filename);
    ugStorageNode->SetURI(filename);
    // reset filename to the local file name
    localFile = ((this->GetMRMLScene())->GetCacheManager())->GetFilenameFromURI(filename);
    }
  else
    {
    mStorageNode->SetFileName(filename);
    fsmStorageNode->SetFileName(filename);
    ugStorageNode->SetFileName(filename);
  
    localFile = filename;
    }
  const itksys_stl::string fname(localFile);
  // the model name is based on the file name (itksys call should work even if
  // file is not on disk yet)
  name = itksys::SystemTools::GetFilenameName(fname);
  vtkDebugMacro("AddModel: got model name = " << name.c_str());
  
  vtkDataReader *datatypeTestReader = vtkDataReader::New();
  datatypeTestReader->SetFileName(filename);
   
  // check to see which node can read this type of file
  if (mStorageNode->SupportedFileType(name.c_str()))
    {
      // at this point, we know that the file is either a PolyData or UnstructuredGrid
      // file, because it passed the filename test.  Instantiate a data reader to check if
      // which storage node should be used. 
      
        if (datatypeTestReader->IsFilePolyData())
        {
            vtkDebugMacro("input dataset was poly data");
            storageNode = mStorageNode;
        }
        else 
        {    
            // treat the volume as an unstructured grid, so use the correct
            // storage, display, and main reference nodes
            vtkDebugMacro("input dataset was ugrid data");
            storageNode = ugStorageNode;
            modelNode->Delete();
            modelNode = ugNode;   
            displayNode->Delete();
            displayNode = ugDisplayNode;
        }
    }
  else if (fsmStorageNode->SupportedFileType(name.c_str()))
    {
    vtkDebugMacro("AddModel: have a freesurfer type model file.");
    storageNode = fsmStorageNode;
    }
 
  if (storageNode != NULL)
    {
    modelNode->SetName(name.c_str());

    this->GetMRMLScene()->SaveStateForUndo();

    modelNode->SetScene(this->GetMRMLScene());
    storageNode->SetScene(this->GetMRMLScene());
    displayNode->SetScene(this->GetMRMLScene()); 

    this->GetMRMLScene()->AddNodeNoNotify(storageNode);  
    this->GetMRMLScene()->AddNodeNoNotify(displayNode);
    modelNode->SetAndObserveStorageNodeID(storageNode->GetID());
    modelNode->SetAndObserveDisplayNodeID(displayNode->GetID());  
    
    // if this displayable node is really a UGrid, then we need to assign its unstructured grid data, otherwise
    // we should assign polydata
    if (datatypeTestReader->IsFileUnstructuredGrid())
    {
        vtkMRMLUnstructuredGridDisplayNode::SafeDownCast(displayNode)->SetUnstructuredGrid(vtkMRMLUnstructuredGridNode::SafeDownCast(modelNode)->GetUnstructuredGrid());
        cout << "set Ugrid data from node as display node dataset" << endl;
    }
    else
    {
        cout << "set poly data from node as display node dataset" << endl;
        displayNode->SetPolyData(modelNode->GetPolyData());
    }
   
    this->GetMRMLScene()->AddNode(modelNode);  

    //this->Modified();  

    // the scene points to it still
    modelNode->Delete();

    // now set up the reading
    vtkDebugMacro("AddModel: calling read on the storage node");
    storageNode->ReadData(modelNode);
    }
  else
    {
    vtkDebugMacro("Couldn't read file, returning null model node: " << filename);
    modelNode->Delete();
    modelNode = NULL;
    }
  mStorageNode->Delete();
  fsmStorageNode->Delete();
  displayNode->Delete();

  return modelNode;  
}
//----------------------------------------------------------------------------
int vtkSlicerModelsLogic::SaveModel (const char* filename, vtkMRMLModelNode *modelNode)
{
   if (modelNode == NULL || filename == NULL)
    {
    vtkErrorMacro("SaveModel: unable to proceed, filename is " <<
                  (filename == NULL ? "null" : filename) <<
                  ", model node is " <<
                  (modelNode == NULL ? "null" : modelNode->GetID()));
    return 0;
    }
  
  vtkMRMLModelStorageNode *storageNode = NULL;
  vtkMRMLStorageNode *snode = modelNode->GetStorageNode();
  if (snode != NULL)
    {
    storageNode = vtkMRMLModelStorageNode::SafeDownCast(snode);
    }
  if (storageNode == NULL)
    {
    storageNode = vtkMRMLModelStorageNode::New();
    storageNode->SetScene(this->GetMRMLScene());
    this->GetMRMLScene()->AddNode(storageNode);  
    modelNode->SetAndObserveStorageNodeID(storageNode->GetID());
    storageNode->Delete();
    }

  // check for a remote file
  if ((this->GetMRMLScene()->GetCacheManager() != NULL) &&
      this->GetMRMLScene()->GetCacheManager()->IsRemoteReference(filename))
    {
    storageNode->SetURI(filename);
    }
  else
    {
    storageNode->SetFileName(filename);
    }
  
  int res = storageNode->WriteData(modelNode);

  
  return res;

}


//----------------------------------------------------------------------------
void vtkSlicerModelsLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);

  os << indent << "vtkSlicerModelsLogic:             " << this->GetClassName() << "\n";

  os << indent << "ActiveModelNode: " <<
    (this->ActiveModelNode ? this->ActiveModelNode->GetName() : "(none)") << "\n";
}

//----------------------------------------------------------------------------
int vtkSlicerModelsLogic::AddScalar(const char* filename, vtkMRMLModelNode *modelNode)
{
  if (modelNode == NULL ||
      filename == NULL)
    {
    vtkErrorMacro("Model node or file name are null.");
    return 0;
    }  

  vtkMRMLFreeSurferModelOverlayStorageNode *fsmoStorageNode = vtkMRMLFreeSurferModelOverlayStorageNode::New();
  vtkMRMLStorageNode *storageNode = NULL;
  
  // check for local or remote files
  int useURI = 0; //false ;
  if (this->GetMRMLScene()->GetCacheManager() != NULL)
    {
    useURI = this->GetMRMLScene()->GetCacheManager()->IsRemoteReference(filename);
    vtkDebugMacro("AddModel: file name is remote: " << filename);
    }

  const char *localFile;
  if (useURI)
    {
    fsmoStorageNode->SetURI(filename);
    // add other overlay storage nodes here
    localFile = ((this->GetMRMLScene())->GetCacheManager())->GetFilenameFromURI(filename);
    }
  else
    {
    fsmoStorageNode->SetFileName(filename);
    // add other overlay storage nodes here
    localFile = filename;
    }

  // check to see if it can read it
  if (fsmoStorageNode->SupportedFileType(localFile))
    {
    storageNode = fsmoStorageNode;
    }

  // check to see if the model display node has a colour node already
  vtkMRMLModelDisplayNode *displayNode = modelNode->GetModelDisplayNode();
  if (displayNode == NULL)
    {
    vtkWarningMacro("Model " << modelNode->GetName() << "'s display node is null\n");
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

  if (storageNode != NULL)
    {
    this->GetMRMLScene()->SaveStateForUndo();
    storageNode->SetScene(this->GetMRMLScene());
    this->GetMRMLScene()->AddNodeNoNotify(storageNode);
    // now add this as another storage node on the model
    modelNode->AddAndObserveStorageNodeID(storageNode->GetID());

    // now read, since all the id's are set up
    vtkDebugMacro("AddScalar: calling read data now.");
    if (this->GetDebug()) { storageNode->DebugOn(); }
    storageNode->ReadData(modelNode);
    }
  fsmoStorageNode->Delete();
  
  return 1;
}


