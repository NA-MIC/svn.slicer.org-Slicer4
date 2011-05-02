#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"
#include "vtkImageChangeInformation.h"
#include "vtkMRMLAnnotationLinesStorageNode.h"
#include "vtkMRMLAnnotationLineDisplayNode.h"
#include "vtkMRMLAnnotationLinesNode.h"
#include "vtkStringArray.h"

//------------------------------------------------------------------------------
vtkMRMLAnnotationLinesStorageNode* vtkMRMLAnnotationLinesStorageNode::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLAnnotationLinesStorageNode");
  if(ret)
    {
    return (vtkMRMLAnnotationLinesStorageNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLAnnotationLinesStorageNode;
}

//----------------------------------------------------------------------------

vtkMRMLNode* vtkMRMLAnnotationLinesStorageNode::CreateNodeInstance()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLAnnotationLinesStorageNode");
  if(ret)
    {
    return (vtkMRMLAnnotationLinesStorageNode*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkMRMLAnnotationLinesStorageNode;
}

//----------------------------------------------------------------------------
vtkMRMLAnnotationLinesStorageNode::vtkMRMLAnnotationLinesStorageNode()
{
}

//----------------------------------------------------------------------------
vtkMRMLAnnotationLinesStorageNode::~vtkMRMLAnnotationLinesStorageNode()
{
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::WriteXML(ostream& of, int nIndent)
{
  Superclass::WriteXML(of, nIndent);
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::ReadXMLAttributes(const char** atts)
{

  Superclass::ReadXMLAttributes(atts);

}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, StorageID
void vtkMRMLAnnotationLinesStorageNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::PrintSelf(ostream& os, vtkIndent indent)
{  
  vtkMRMLStorageNode::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::ProcessParentNode(vtkMRMLNode *parentNode)
{
  this->ReadData(parentNode);
}
//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::ReadAnnotationLineDisplayProperties(vtkMRMLAnnotationLineDisplayNode *refNode, std::string lineString, std::string preposition)
{
  if (refNode == NULL)
    {
      vtkErrorMacro("ReadAnnotationLineDisplayProperties: unable to get associated AnnotationPointDisplayNode"); 
      return -1;
    }

  int flag = Superclass::ReadAnnotationDisplayProperties(refNode, lineString, preposition);
  if (flag)
    {
      return flag; 
    }

  int pointOffset = preposition.size();
  preposition.insert(0,"# ");

  if (lineString.find(preposition + "LineThickness = ") != std::string::npos)
    {
     std::string str = lineString.substr(18 + pointOffset,std::string::npos);
     vtkDebugMacro("Getting LineThickness, substr = " << str);
     float size = atof(str.c_str());
     refNode->SetLineThickness(size);
     return 1;
    }

  return 0;
}

 
//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::ReadAnnotationLinesData(vtkMRMLAnnotationLinesNode *refNode, char line[1024],
                                   int typeColumn, int startIDColumn, int endIDColumn, int selColumn,  int visColumn, int numColumns)
{
  if (!refNode)
    {
      return -1;
    }

  if (typeColumn)
    {
      vtkErrorMacro("Type column has to be zero !");
      return -1;
    }

  // is it empty?

  if (line[0] == '\0')
    {
      vtkDebugMacro("Empty line, skipping:\n\"" << line << "\"");
      return 1;
    }

  vtkDebugMacro("got a line: \n\"" << line << "\""); 
  std::string attValue(line);
  int size = std::string(this->GetAnnotationStorageType()).size();
 
  if (attValue.compare(0,size,this->GetAnnotationStorageType()))
    {
      return 0;
    }

  int sel = 1, vis = 1;
  std::string annotation;
  
  // Jump over type 
  size_t  startPos =attValue.find("|",0) +1;
  size_t  endPos =attValue.find("|",startPos);
  int columnNumber = 1;
  vtkIdType coordID[2] = {-1, -1};
  while (startPos != std::string::npos && (columnNumber < numColumns)) 
    {
      if (startPos != endPos) 
    {
      const char* ptr;
      if (endPos == std::string::npos) 
        {
          ptr = attValue.substr(startPos,endPos).c_str();
        }
      else
        {
          ptr = attValue.substr(startPos,endPos-startPos).c_str(); 
        }
      
      if (columnNumber == startIDColumn)
        {
          coordID[0] = atof(ptr);
        }
      else if (columnNumber == endIDColumn)
        {
          coordID[1] = atof(ptr);
        }
     else if (columnNumber == selColumn)
        {
          sel = atoi(ptr);
        }
      else if (columnNumber == visColumn)
        {
          vis = atoi(ptr);
        }
    }
      startPos = endPos +1;
      endPos =attValue.find("|",startPos);
      columnNumber ++;
    }

  if (refNode->AddLine(coordID[0],coordID[1], sel, vis) < 0 ) 
    {
      vtkErrorMacro("Error adding list to list, coordID = " << coordID[0] << " " << coordID[1]);
      return -1;
    }
  return 1;
}

//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::ReadAnnotationLinesProperties(vtkMRMLAnnotationLinesNode *vtkNotUsed(refNode), char line[1024], int &typeColumn,
                                    int& startIDColumn,    int& endIDColumn, int& selColumn, int& visColumn, int& numColumns)
{
 if (line[0] != '#' || line[1] != ' ') 
    {
      return 0;
    }


  vtkDebugMacro("Comment line, checking:\n\"" << line << "\"");
  // TODO: parse out the display node settings
  // if there's a space after the hash, try to find options
  std::string preposition = std::string("# ") + this->GetAnnotationStorageType();
  vtkIdType  pointOffset = std::string(this->GetAnnotationStorageType()).size();;
  
  vtkDebugMacro("Have a possible option in line " << line);
  std::string lineString = std::string(line);


  if (lineString.find(preposition + "Columns = ") != std::string::npos)
    {
      std::string str = lineString.substr(12 + pointOffset, std::string::npos);
      
      vtkDebugMacro("Getting column order for the fids, substr = " << str.c_str());
      // reset all of them
      typeColumn= startIDColumn = endIDColumn = selColumn = visColumn = -1;
      numColumns = 0;
      char *columns = (char *)str.c_str();
      char *ptr = strtok(columns, "|");
      while (ptr != NULL)
    {
      if (strcmp(ptr, "type") == 0)
        {
          typeColumn = numColumns ;
        }
      else if (strcmp(ptr, "startPointID") == 0)
        {
          startIDColumn =  numColumns;
        }
      else if (strcmp(ptr, "endPointID") == 0)
        {
          endIDColumn =  numColumns;
        }
      else if (strcmp(ptr, "sel") == 0)
        {
          selColumn =  numColumns;
        }
      else if (strcmp(ptr, "vis" ) == 0)
        {
          visColumn =  numColumns;
        }
      ptr = strtok(NULL, "|");
      numColumns++;
    }
      // set the total number of columns
      vtkDebugMacro("Got " << numColumns << " columns, type = " << typeColumn << ", startPointID = " << startIDColumn << ",  endPointID = " << endIDColumn << ", sel = " <<  selColumn << ", vis = " << visColumn);
      return 1;
    }
  return 0;
}


//----------------------------------------------------------------------------
// assumes that the node is already reset
int vtkMRMLAnnotationLinesStorageNode::ReadAnnotation(vtkMRMLAnnotationLinesNode *refNode)
{

  if (refNode == NULL)
    {
      vtkErrorMacro("ReadAnnotation: unable to cast input node " << refNode->GetID() << " to a annotation node");
      return 0;
    }

  if (!Superclass::ReadAnnotation(refNode)) 
    {
      return 0;
    }

  // open the file for reading input
  fstream fstr;
  if (!this->OpenFileToRead(fstr, refNode))
    {
      return 0;
    }

  
 vtkMRMLAnnotationLineDisplayNode *aLineDisplayNode = refNode->GetAnnotationLineDisplayNode();

  
  // turn off modified events
  int modFlag = refNode->GetDisableModifiedEvent();
  refNode->DisableModifiedEventOn();
  char line[1024];
  // default column ordering for annotation info - this is exactly the same as for fiducial
  // first pass: line will have label,x,y,z,selected,visible
  int typePointColumn = 0;
  int startIDColumn = 1;
  int endIDColumn = 2;
  int selPointColumn  = 4;
  int visPointColumn  = 5;
  int numPointColumns = 6;

  while (fstr.good())
    {
    fstr.getline(line, 1024);
    
    // does it start with a #?
        // Property
    if (line[0] == '#')
      {
        if (line[1] == ' ') 
          {
            if (!this->ReadAnnotationLinesProperties(refNode, line, typePointColumn, startIDColumn, endIDColumn, selPointColumn, visPointColumn, numPointColumns))
              {
            if (this->ReadAnnotationLineDisplayProperties(aLineDisplayNode, line,this->GetAnnotationStorageType()) < 0 )
              {
                return 0;
              }
              }
          }
      }
        else
          {
        if (this->ReadAnnotationLinesData(refNode, line, typePointColumn, startIDColumn, endIDColumn, selPointColumn,  
                              visPointColumn, numPointColumns) < 0 ) 
          {
        return 0;
          }
      }
    }   
    refNode->SetDisableModifiedEvent(modFlag);

    fstr.close();

    return 1;

}

//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::ReadData(vtkMRMLNode *refNode)
{
  // do not read if if we are not in the scene (for example inside snapshot)
  if ( !this->GetAddToScene() || !refNode->GetAddToScene() )
    {
      return 1;
    }

  // cast the input node
  vtkMRMLAnnotationLinesNode *aNode = dynamic_cast <vtkMRMLAnnotationLinesNode *> (refNode);

  if (aNode == NULL)
    {
    vtkErrorMacro("ReadData: unable to cast input node " << refNode->GetID() << " to a annotation control point node");
    return 0;
    }

  // clear out the list
  aNode->ResetAnnotations();

  if (!this->ReadAnnotation(aNode))
    {
      return 0;
    }

  this->SetReadStateIdle();
  
  // make sure that the list node points to this storage node
  aNode->SetAndObserveStorageNodeID(this->GetID());
  
  // mark it unmodified since read
  aNode->ModifiedSinceReadOff();

  aNode->InvokeEvent(vtkMRMLScene::NodeAddedEvent, aNode);//vtkMRMLAnnotationNode::DisplayModifiedEvent);

  return 1;
}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::WriteAnnotationLineDisplayProperties(fstream& of, vtkMRMLAnnotationLineDisplayNode *refNode, std::string preposition)
{
  if (!refNode)
   {
     return;
   }
  this->WriteAnnotationDisplayProperties(of,refNode, preposition);

  preposition.insert(0,"# ");
  of << preposition + "LineThickness = " << refNode->GetLineThickness() << endl;
}

//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::WriteAnnotationLinesProperties(fstream& of, vtkMRMLAnnotationLinesNode *refNode)
{
   // put down a header
  if (refNode == NULL)
    {
    vtkErrorMacro("WriteAnnotationLinesProperties: ref node is null");
    return 0;
    }

  vtkMRMLAnnotationLineDisplayNode *annDisNode = refNode->GetAnnotationLineDisplayNode();
  if (annDisNode == NULL)
    {
    vtkErrorMacro("WriteAnnotationLinesProperties: no annotation line display node");
    return 0;
    }

  this->WriteAnnotationLineDisplayProperties(of, annDisNode, this->GetAnnotationStorageType());
  of << "# " << this->GetAnnotationStorageType() << "Columns = type|startPointID|endPointID|sel|vis" << endl;

  return 1;

}

//----------------------------------------------------------------------------
void vtkMRMLAnnotationLinesStorageNode::WriteAnnotationLinesData(fstream& of, vtkMRMLAnnotationLinesNode *refNode)
{
  if (!refNode)
    {
      return;
    }
  for (int i = 0; i < refNode->GetNumberOfLines(); i++)
    {
      vtkIdType pointIDs[2];
      refNode->GetEndPointsId(i,pointIDs);
      int sel = refNode->GetAnnotationAttribute(i, vtkMRMLAnnotationLinesNode::LINE_SELECTED);
      int vis = refNode->GetAnnotationAttribute(i, vtkMRMLAnnotationLinesNode::LINE_VISIBLE);
      of << this->GetAnnotationStorageType() << "|" << pointIDs[0] << "|" <<  pointIDs[1]  << "|" << sel << "|" << vis << endl;   
    }

}

//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::WriteData(vtkMRMLNode *refNode)
{
  // open the file for writing
  fstream of;
  if (!this->OpenFileToWrite(of)) 
    {
    vtkWarningMacro(" vtkMRMLAnnotationLinesStorageNode::WriteData: can't open file to write");
    return 0;
    } 

  int flag = this->WriteData(refNode,of);

  of.close();

  Superclass::StageWriteData(refNode);

  vtkDebugMacro(" vtkMRMLAnnotationLinesStorageNode::WriteData: returning flag " << flag);
  return flag;
}


//----------------------------------------------------------------------------
int vtkMRMLAnnotationLinesStorageNode::WriteData(vtkMRMLNode *refNode, fstream& of)
{
  int retval = Superclass::WriteData(refNode,of);
  if (!retval)
    {
    vtkWarningMacro("vtkMRMLAnnotationLinesStorageNode::WriteData with stream: can't call WriteData on superclass, retval = " << retval);
    return 0;
    }

  // test whether refNode is a valid node to hold a volume
  if ( !( refNode->IsA("vtkMRMLAnnotationLinesNode") ) )
    {
    vtkErrorMacro("Reference node is not a proper vtkMRMLAnnotationLinesNode");
    return 0;         
    }


  // cast the input nod
  vtkMRMLAnnotationLinesNode *aNode = dynamic_cast <vtkMRMLAnnotationLinesNode *> (refNode);

  if (aNode == NULL)
    {
    vtkErrorMacro("WriteData: unable to cast input node " << refNode->GetID() << " to a known annotation line node");
    return 0;
    }

  // Control Points Properties
  if (!WriteAnnotationLinesProperties(of, aNode))
    {
    vtkWarningMacro("vtkMRMLAnnotationLinesStorageNode::WriteData with stream: error writing annotation lines properties");
    return 0;
    }

  WriteAnnotationLinesData(of, aNode);

  vtkDebugMacro("vtkMRMLAnnotationLinesStorageNode::WriteData with stream: returning 1");
  return 1;
}

