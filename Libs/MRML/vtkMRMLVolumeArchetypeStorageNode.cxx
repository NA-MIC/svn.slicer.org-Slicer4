/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkMRMLVolumeArchetypeStorageNode.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.6 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>
#include "vtksys/Directory.hxx"

#include "vtkMRMLConfigure.h" // MRML_USE*

#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"
#include "vtkImageChangeInformation.h"
#include "vtkMRMLVolumeArchetypeStorageNode.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLScalarVolumeNode.h"

#ifdef MRML_USE_vtkTeem
#include "vtkMRMLVectorVolumeNode.h"
#endif

#include "vtkMatrix4x4.h"
#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkPointData.h"
#include "vtkStringArray.h"
#include "vtkITKArchetypeImageSeriesReader.h"
#include "vtkITKArchetypeImageSeriesScalarReader.h"
#include "vtkITKArchetypeImageSeriesVectorReaderFile.h"
#include "vtkITKArchetypeImageSeriesVectorReaderSeries.h"
#include "vtkITKImageWriter.h"

//------------------------------------------------------------------------------
vtkMRMLVolumeArchetypeStorageNode* vtkMRMLVolumeArchetypeStorageNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeArchetypeStorageNode");
  if(ret)
    {
    return (vtkMRMLVolumeArchetypeStorageNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLVolumeArchetypeStorageNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLVolumeArchetypeStorageNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeArchetypeStorageNode");
  if(ret)
    {
    return (vtkMRMLVolumeArchetypeStorageNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLVolumeArchetypeStorageNode;
}

//----------------------------------------------------------------------------
vtkMRMLVolumeArchetypeStorageNode::vtkMRMLVolumeArchetypeStorageNode()
{
  this->CenterImage = 0;
  this->SingleFile  = 0;
  this->UseOrientationFromFile = 1;
}

//----------------------------------------------------------------------------
vtkMRMLVolumeArchetypeStorageNode::~vtkMRMLVolumeArchetypeStorageNode()
{
}

void vtkMRMLVolumeArchetypeStorageNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
  vtkIndent indent(nIndent);
  {
  std::stringstream ss;
  ss << this->CenterImage;
  of << indent << " centerImage=\"" << ss.str() << "\"";
  }
  {
  std::stringstream ss;
  ss << this->SingleFile;
  of << indent << " singleFile=\"" << ss.str() << "\"";
  }
  {
  std::stringstream ss;
  ss << this->UseOrientationFromFile;
  of << indent << " UseOrientationFromFile=\"" << ss.str() << "\"";
  }
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeArchetypeStorageNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  Superclass::ReadXMLAttributes(atts);

  const char* attName;
  const char* attValue;
  while (*atts != NULL) 
    {
    attName = *(atts++);
    attValue = *(atts++);
    if (!strcmp(attName, "centerImage")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->CenterImage;
      }
    if (!strcmp(attName, "singleFile")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->SingleFile;
      }
    if (!strcmp(attName, "UseOrientationFromFile")) 
      {
      std::stringstream ss;
      ss << attValue;
      ss >> this->UseOrientationFromFile;
      }
    }

  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, StorageID
void vtkMRMLVolumeArchetypeStorageNode::Copy(vtkMRMLNode *anode)
{
  int disabledModify = this->StartModify();

  Superclass::Copy(anode);
  vtkMRMLVolumeArchetypeStorageNode *node = (vtkMRMLVolumeArchetypeStorageNode *) anode;

  this->SetCenterImage(node->CenterImage);
  this->SetSingleFile(node->SingleFile);
  this->SetUseOrientationFromFile(node->UseOrientationFromFile);

  this->EndModify(disabledModify);
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeArchetypeStorageNode::PrintSelf(ostream& os, vtkIndent indent)
{  
  vtkMRMLStorageNode::PrintSelf(os,indent);
  os << indent << "CenterImage:   " << this->CenterImage << "\n";
  os << indent << "SingleFile:   " << this->SingleFile << "\n";
  os << indent << "UseOrientationFromFile:   " << this->UseOrientationFromFile << "\n";
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeArchetypeStorageNode::ProcessParentNode(vtkMRMLNode *parentNode)
{
  this->ReadData(parentNode);
}

//----------------------------------------------------------------------------
int vtkMRMLVolumeArchetypeStorageNode::ReadData(vtkMRMLNode *refNode)
{

  // do not read if if we are not in the scene (for example inside snapshot)
  if ( !refNode->GetAddToScene() )
    {
    return 1;
    }
   
  // test whether refNode is a valid node to hold a volume
  if ( !( refNode->IsA("vtkMRMLScalarVolumeNode") || refNode->IsA("vtkMRMLVectorVolumeNode" ) ) )
    {
    //vtkErrorMacro("Reference node is not a vtkMRMLVolumeNode");
    return 0;         
    }
  if (this->GetFileName() == NULL && this->GetURI() == NULL) 
    {
    vtkErrorMacro("ReadData: both filename and uri are null.");
    return 0;
    }
  
  Superclass::StageReadData(refNode);
  if ( this->GetReadState() != this->TransferDone )
    {
    // remote file download hasn't finished
    vtkWarningMacro("ReadData: read state is pending, remote download hasn't finished yet");
    return 0;
    }
  else
    {
    vtkDebugMacro("ReadData: read state is ready, URI = " << (this->GetURI() == NULL ? "null" : this->GetURI()) << ", filename = " << (this->GetFileName() == NULL ? "null" : this->GetFileName()));
    }
  
  std::string fullName = this->GetFullNameFromFileName();
  vtkDebugMacro("ReadData: got full archetype name " << fullName);

  if (fullName == std::string("")) 
    {
    vtkErrorMacro("ReadData: File name not specified");
    return 0;
    }
  
  vtkMRMLVolumeNode *volNode = NULL;
  vtkITKArchetypeImageSeriesReader* reader = NULL;
  
#ifdef MRML_USE_vtkTeem
  if ( refNode->IsA("vtkMRMLVectorVolumeNode") ) 
    {
    volNode = dynamic_cast <vtkMRMLVectorVolumeNode *> (refNode);
    // 
    // decide if we want to use a vector file reader (e.g. for multi-component nrrd)
    // or a vector series reader (e.g. for a sequence of rgb image files)
    // - note these are different classes because they are too big to compile
    //   as a single class on some systems
    //
    vtkITKArchetypeImageSeriesVectorReaderFile *readerFile = vtkITKArchetypeImageSeriesVectorReaderFile::New();
    vtkITKArchetypeImageSeriesVectorReaderSeries *readerSeries = vtkITKArchetypeImageSeriesVectorReaderSeries::New();

    readerFile->SetArchetype(fullName.c_str());
    readerFile->SetSingleFile( this->GetSingleFile() );
    readerFile->SetUseOrientationFromFile( this->GetUseOrientationFromFile() );

    readerSeries->SetArchetype(fullName.c_str());
    readerSeries->SetSingleFile( this->GetSingleFile() );
    readerSeries->SetUseOrientationFromFile( this->GetUseOrientationFromFile() );

    try 
      {
        readerSeries->UpdateInformation();
      }
    catch ( ... )
      {
      readerFile->Delete();
      readerSeries->Delete();
      return 0;
      }
    if ( readerSeries->GetNumberOfFileNames() == 1 )
      {
      reader = readerFile;
      reader->UpdateInformation();
      readerSeries->Delete();
      }
    else
      {
      vtkDebugMacro("ReadData: readerSeries number of file names = " << readerSeries->GetNumberOfFileNames());
      reader = readerSeries;
      readerFile->Delete();
      }
    if (reader->GetNumberOfComponents() < 3)
      {
      reader->Delete();
      return 0;
      }
    }
  else 
    #endif
    if ( refNode->IsA("vtkMRMLScalarVolumeNode") ) 
    {
    volNode = dynamic_cast <vtkMRMLScalarVolumeNode *> (refNode);
    reader = vtkITKArchetypeImageSeriesScalarReader::New();  
    reader->SetSingleFile( this->GetSingleFile() );
    reader->SetUseOrientationFromFile( this->GetUseOrientationFromFile() );
    }

  reader->AddObserver( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);

  if (volNode->GetImageData()) 
    {
    volNode->SetAndObserveImageData (NULL);
    }

  // set the list of file names on the reader
  reader->ResetFileNames();
  reader->SetArchetype(fullName.c_str());

  // TODO: this is a workaround for an issue in itk::ImageSeriesReader
  // where is assumes that all the filenames that have been passed
  // to it are a dimension smaller than the image it is asked to create
  // (i.e. a list of .jpg files that form a volume).
  // In our case though, we can have file lists that include both the
  // header and bulk data, like .hdr/.img pairs.  So we need to 
  // be careful not to send extra filenames to the reader if the 
  // format is multi-file for the same volume
  //
  // check for Analyze and similar format- if the archetype is 
  // one of those, then don't send the rest of the list
  //
  std::string fileExt(itksys::SystemTools::GetFilenameLastExtension(fullName));
  if ( fileExt != std::string(".hdr") 
      && fileExt != std::string(".img") 
      && fileExt != std::string(".mhd") 
      && fileExt != std::string(".nhdr") )
    {
    for (int n = 0; n < this->GetNumberOfFileNames(); n++)
      {
      std::string nthFileName = this->GetFullNameFromNthFileName(n);
      vtkDebugMacro("ReadData: got full name for " << n << "th file: " << nthFileName << ", adding it to reader, current num files on it = " << reader->GetNumberOfFileNames());
      reader->AddFileName(nthFileName.c_str());
      }
    }

  reader->SetOutputScalarTypeToNative();
  reader->SetDesiredCoordinateOrientationToNative();
  if (this->CenterImage) 
    {
    reader->SetUseNativeOriginOff();
    }
  else
    {
    reader->SetUseNativeOriginOn();
    }

  int result = 1;
  try
    {
    vtkDebugMacro("ReadData: right before reader update, reader num files = " << reader->GetNumberOfFileNames());
    reader->Update();
    }
    catch (...)
    {
    vtkErrorMacro("ReadData: Cannot read file, fullName = " << fullName.c_str() << ", node num file names = " << this->GetNumberOfFileNames() << ", reader num files = " << reader->GetNumberOfFileNames() << ", reader archetype = " << reader->GetArchetype());
    if (reader->GetFileName(0) != NULL)
      {
      vtkErrorMacro("reader 0th file name = " << reader->GetFileName(0) );
      }
    reader->RemoveObservers( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);
    reader->Delete();
    return 0;
    }

  if (reader->GetOutput() == NULL 
      || reader->GetOutput()->GetPointData()->GetScalars()->GetNumberOfTuples() == 0) 
    {
    vtkErrorMacro("ReadData: Unable to read data from file: " << fullName.c_str() );
    reader->RemoveObservers( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);
    reader->Delete();
    return 0;
    }
  
  if ( !volNode->IsA("vtkMRMLVectorVolumeNode") &&
        volNode->IsA("vtkMRMLScalarVolumeNode") && reader->GetNumberOfComponents() != 1 ) 
    {
    volNode->SetAndObserveImageData(NULL);
    vtkErrorMacro("ReadData: Not a scalar volume file: " << fullName.c_str() );
    reader->RemoveObservers( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);
    reader->Delete();
    return 0;
    }

  // set volume attributes
  volNode->SetAndObserveStorageNodeID(this->GetID());
  volNode->SetMetaDataDictionary( reader->GetMetaDataDictionary() );

  // get all the file names from the reader
  if (reader->GetNumberOfFileNames() > 1)
    {
    vtkDebugMacro("Number of file names = " << reader->GetNumberOfFileNames() << ", number of slice location = " << reader->GetNumberOfSliceLocation());
    // include the archtype, file 0, in the storage node's file list
    for (unsigned int n = 0; n < reader->GetNumberOfFileNames(); n++)
      {
      const char *thisFileName = reader->GetFileName(n);
      int currentSize = this->AddFileName(thisFileName);
      vtkDebugMacro("After adding file " << n << ", filename = " << thisFileName << " to this storage node's list, current size of the list = " << currentSize);
      }
    }
  //TODO update scene to send Modified event
 
  vtkImageChangeInformation *ici = vtkImageChangeInformation::New();
  ici->SetInput (reader->GetOutput());
  ici->SetOutputSpacing( 1, 1, 1 );
  ici->SetOutputOrigin( 0, 0, 0 );
  ici->Update();

  if (ici->GetOutput() == NULL)
    {
    vtkErrorMacro("vtkMRMLVolumeArchetypeStorageNode: Cannot read file: " << fullName.c_str() );
    reader->RemoveObservers( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);
    reader->Delete();
    ici->Delete();
    return 0;
    }
  else
    {
    volNode->SetAndObserveImageData (ici->GetOutput());
    volNode->ModifiedSinceReadOff();
    }

  vtkMatrix4x4* mat = reader->GetRasToIjkMatrix();
  if ( mat == NULL )
    {
    vtkErrorMacro ("Reader returned NULL RasToIjkMatrix");
    }
  volNode->SetRASToIJKMatrix(mat);

  reader->RemoveObservers( vtkCommand::ProgressEvent,  this->MRMLCallbackCommand);
  reader->Delete();
  ici->Delete();

  this->SetReadStateIdle();
  
  return result;
}

//----------------------------------------------------------------------------
int vtkMRMLVolumeArchetypeStorageNode::WriteData(vtkMRMLNode *refNode)
{
  int result = 1;

  // test whether refNode is a valid node to hold a volume
  if (!refNode->IsA("vtkMRMLScalarVolumeNode") ) 
    {
    vtkErrorMacro("Reference node is not a vtkMRMLVolumeNode");
    return 0;
    }
  
  vtkMRMLVolumeNode *volNode = NULL;
  
  if ( refNode->IsA("vtkMRMLScalarVolumeNode") ) 
    {
    volNode = vtkMRMLScalarVolumeNode::SafeDownCast(refNode);
    }
  
  if (volNode->GetImageData() == NULL) 
    {
    vtkErrorMacro("cannot write ImageData, it's NULL");
    return 0;
    }
  
  // update the file list
  std::string moveFromDir = this->UpdateFileList(refNode, 1);

 
  std::string fullName = this->GetFullNameFromFileName();  
  if (fullName == std::string("")) 
    {
    vtkErrorMacro("WriteData: File name not specified");
    return 0;
    }

  bool moveSucceeded = true;
  if (moveFromDir != std::string(""))
    {
    // the temp writing went okay, just move the files from there to where
    // they're supposed to go. It will fail if the temp dir is on a different
    // device, so fall back to a second write in that case.
    std::string targetDir = vtksys::SystemTools::GetParentDirectory(fullName.c_str());
    vtkDebugMacro("WriteData: moving files from temp dir " << moveFromDir << " to target dir " << targetDir);

    vtksys::Directory dir;
    dir.Load(moveFromDir.c_str());
    vtkDebugMacro("WriteData: tempdir " << moveFromDir.c_str() << " has " << dir.GetNumberOfFiles() << " in it");
    size_t fileNum;
    vtksys_stl::vector<vtksys_stl::string> targetPathComponents;
    vtksys::SystemTools::SplitPath(targetDir.c_str(), targetPathComponents);
    vtksys_stl::vector<vtksys_stl::string> sourcePathComponents;
    vtksys::SystemTools::SplitPath(moveFromDir.c_str(), sourcePathComponents);
    for (fileNum = 0; fileNum <  dir.GetNumberOfFiles(); ++fileNum)
      {
      const char *thisFile = dir.GetFile(static_cast<unsigned long>(fileNum));
      // skip the dirs
      if (strcmp(thisFile,".") &&
          strcmp(thisFile,".."))
        {
        targetPathComponents.push_back(thisFile);
        sourcePathComponents.push_back(thisFile);
        std::string targetFile = vtksys::SystemTools::JoinPath(targetPathComponents);
        // does the target file already exist?
        if (vtksys::SystemTools::FileExists(targetFile.c_str(), true))
          {
          // remove it
          vtkWarningMacro("WriteData: removing old version of file " << targetFile);
          if (!vtksys::SystemTools::RemoveFile(targetFile.c_str()))
            {
            vtkErrorMacro("WriteData: unable to remove old version of file " << targetFile);
            }
          }
        std::string sourceFile = vtksys::SystemTools::JoinPath(sourcePathComponents);
        vtkDebugMacro("WriteData: moving file number " << fileNum << ", " << sourceFile << " to " << targetFile);
        // thisFile needs a full path it's bare
        int renameReturn = std::rename(sourceFile.c_str(), targetFile.c_str());
        if (renameReturn != 0 )
          {
          perror( "Error renaming file" );
          vtkErrorMacro( "WriteData: Error renaming file to " << targetFile << ", renameReturn = " << renameReturn );
          // fall back to doing a second write
          moveSucceeded = false;
          break;
          }
        targetPathComponents.pop_back();
        sourcePathComponents.pop_back();
        }
      }
    
    }
  else
    {
    // didn't move it
    moveSucceeded = false;
    }

  if (!moveSucceeded)
    {
    vtkDebugMacro("WriteData: writing out file with archetype " << fullName);
    
    vtkITKImageWriter *writer = vtkITKImageWriter::New();
    writer->SetFileName(fullName.c_str());
  
    writer->SetInput( volNode->GetImageData() );
    writer->SetUseCompression(this->GetUseCompression());
    if(this->WriteFileFormat)
      {
      writer->SetImageIOClassName(
                                  this->GetScene()->GetDataIOManager()->GetFileFormatHelper()->
                                  GetClassNameFromFormatString(this->WriteFileFormat));
      }
    
    // set volume attributes
    vtkMatrix4x4* mat = vtkMatrix4x4::New();
    volNode->GetRASToIJKMatrix(mat);
    writer->SetRasToIJKMatrix(mat);
    
    try
      {
      writer->Write();
      }
    catch (...)
      {
      result = 0;
      }
    mat->Delete();
    writer->Delete();    
    }
  
  Superclass::StageWriteData(refNode);
  
  return result;

}

//----------------------------------------------------------------------------
int vtkMRMLVolumeArchetypeStorageNode::SupportedFileType(const char *fileName)
{
  // check to see which file name we need to check
  std::string name;
  if (fileName)
    {
    name = std::string(fileName);
    }
  else if (this->FileName != NULL)
    {
    name = std::string(this->FileName);
    }
  else if (this->URI != NULL)
    {
    name = std::string(this->URI);
    }
  else
    {
    vtkWarningMacro("SupportedFileType: no file name to check");
    return 0;
    }

  // for now, return 1
  return 1;
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeArchetypeStorageNode::InitializeSupportedWriteFileTypes()
{
  Superclass::InitializeSupportedWriteFileTypes();
  vtkStringArray* supportedFormats = this->GetScene()->GetDataIOManager()->
    GetFileFormatHelper()->GetITKSupportedWriteFileFormats();
  for(int i=0; i<supportedFormats->GetNumberOfTuples(); i++)
    {
    this->SupportedWriteFileTypes->InsertNextValue(
      supportedFormats->GetValue(i));
    }
}

//----------------------------------------------------------------------------
std::string vtkMRMLVolumeArchetypeStorageNode::UpdateFileList(vtkMRMLNode *refNode, int move)
{
  std::string returnString = "";
  // test whether refNode is a valid node to hold a volume
  if (!refNode->IsA("vtkMRMLScalarVolumeNode") ) 
    {
    vtkErrorMacro("Reference node is not a vtkMRMLVolumeNode");
    return returnString;
    }
  
  vtkMRMLVolumeNode *volNode = NULL;
  volNode = vtkMRMLScalarVolumeNode::SafeDownCast(refNode);
  
  if (volNode == NULL || volNode->GetImageData() == NULL) 
    {
    vtkErrorMacro("UpdateFileList: cannot write ImageData, it's NULL");
    return returnString;
    }

  std::string oldName = this->GetFileName();  
  if (oldName == std::string("")) 
    {
    vtkErrorMacro("UpdateFileList: File name not specified");
    return returnString;
    }

  // clear out the old file list
  this->ResetFileNameList();
  
  // make a new dir to write temporary stuff out to
//  vtksys_stl::vector<vtksys_stl::string> pathComponents;
  // get the base dir of the destination
  /*
  // get the cache dir and make a subdir in it.
  if (this->GetScene() &&
      this->GetScene()->GetCacheManager() &&
      this->GetScene()->GetCacheManager()->GetRemoteCacheDirectory())
    {
    vtksys::SystemTools::SplitPath(this->GetScene()->GetCacheManager()->GetRemoteCacheDirectory(), pathComponents);
    }
  else
    {
    vtkWarningMacro("UpdateFileList: Unable to get remote cache dir, using current dir for temp dir.");
    }
  */
  // get the original directory
  vtksys_stl::string originalDir = vtksys::SystemTools::GetParentDirectory(oldName.c_str());
  vtksys_stl::vector<vtksys_stl::string> pathComponents;
  vtksys::SystemTools::SplitPath(originalDir.c_str(), pathComponents);
  // add a temp dir to it
  pathComponents.push_back(std::string("TempWrite"));
  std::string tempDir = vtksys::SystemTools::JoinPath(pathComponents);
  vtkDebugMacro("UpdateFileList: deleting and then re-creating temp dir "<< tempDir.c_str());
  vtksys::SystemTools::RemoveADirectory(tempDir.c_str());
  vtksys::SystemTools::MakeDirectory(tempDir.c_str());
  
  // make a new name,
  pathComponents.push_back(vtksys::SystemTools::GetFilenameName(oldName));
  std::string tempName = vtksys::SystemTools::JoinPath(pathComponents);
  vtkDebugMacro("UpdateFileList: new archetype file name = " << tempName.c_str());

  // set up the writer and write
  vtkITKImageWriter *writer = vtkITKImageWriter::New();
  writer->SetFileName(tempName.c_str());
  
  writer->SetInput( volNode->GetImageData() );
  writer->SetUseCompression(this->GetUseCompression());
  if(this->WriteFileFormat)
    {
    if (this->GetScene() &&
        this->GetScene()->GetDataIOManager() &&
        this->GetScene()->GetDataIOManager()->GetFileFormatHelper())
      {
      writer->SetImageIOClassName(this->GetScene()->GetDataIOManager()->GetFileFormatHelper()->
                                  GetClassNameFromFormatString(this->WriteFileFormat));
      }
    }

  // set volume attributes
  vtkMatrix4x4* mat = vtkMatrix4x4::New();
  volNode->GetRASToIJKMatrix(mat);
  writer->SetRasToIJKMatrix(mat);

  int result = 1;
  try
    {
    writer->Write();
    }
    catch (...)
    {
    result = 0;
    }
  mat->Delete();
  writer->Delete();

  // look through the new dir and populate the file list
  vtksys::Directory dir;
  dir.Load(tempDir.c_str());
  vtkDebugMacro("UpdateFileList: tempdir " << tempDir.c_str() << " has " << dir.GetNumberOfFiles() << " in it");
  size_t fileNum;

  // take the archetype and temp dir off of the path
  pathComponents.pop_back();
  pathComponents.pop_back();
  vtksys_stl::string localDirectory = vtksys::SystemTools::JoinPath(pathComponents);
  vtksys_stl::string relativePath;
  if (volNode->GetScene() != NULL &&
      strcmp(volNode->GetScene()->GetRootDirectory(), "") != 0)
    {
    // use the scene's root dir, all the files in the list will be
    // relative to it (the relative path is how you go from the root dir to
    // the dir in which the volume is saved)
    relativePath = vtksys::SystemTools::RelativePath(volNode->GetScene()->GetRootDirectory(), localDirectory.c_str());
    vtkDebugMacro("UpdateFileList: got the scene root dir " << volNode->GetScene()->GetRootDirectory());
    }
  else
    {
    // use the archetype's directory, so that all the files in the list will
    // be relative to it
    relativePath = vtksys::SystemTools::RelativePath(originalDir.c_str(), localDirectory.c_str());
    }
  vtkDebugMacro("UpdateFileList: using prefix of relative path " << relativePath.c_str());
  // now get ready to join the relative path to thisFile
  vtksys_stl::vector<vtksys_stl::string> relativePathComponents;
  vtksys::SystemTools::SplitPath(relativePath.c_str(), relativePathComponents);
  for (fileNum = 0; fileNum < dir.GetNumberOfFiles(); ++fileNum)
    {
    // skip the dirs
    const char *thisFile = dir.GetFile(static_cast<unsigned long>(fileNum));
    if (strcmp(thisFile,".") &&
        strcmp(thisFile,".."))
      {
      vtkDebugMacro("UpdateFileList: adding file number " << fileNum << ", " << thisFile);
      // at this point, the file name is bare of a directory, turn it into a
      // relative path from the original archetype
      relativePathComponents.push_back(thisFile);
      vtksys_stl::string relativeFile =  vtksys::SystemTools::JoinPath(relativePathComponents);
      relativePathComponents.pop_back();
      vtkDebugMacro("UpdateFileList: " << fileNum << ", using relative file name " << relativeFile.c_str());
      this->AddFileName(relativeFile.c_str());
      }
    }
  // restore the old file name
  vtkDebugMacro("UpdateFileList: resetting file name to " << oldName.c_str());
  this->SetFileName(oldName.c_str());

  if (move != 1)
    {
    // clean up temp directory
    vtkDebugMacro("UpdateFileList: removing temp dir " << tempDir);
    vtksys::SystemTools::RemoveADirectory(tempDir.c_str());
    return std::string("");
    }
  else
    {
    vtkDebugMacro("UpdateFileList: returning temp dir " << tempDir);
    return tempDir;
    }
}
