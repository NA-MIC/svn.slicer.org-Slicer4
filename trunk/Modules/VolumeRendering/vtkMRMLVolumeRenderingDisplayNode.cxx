#include "vtkMRMLVolumeRenderingDisplayNode.h"
#include "vtkMRMLNode.h"
#include <iostream>
#include <sstream>
#include <string>


vtkMRMLVolumeRenderingDisplayNode* vtkMRMLVolumeRenderingDisplayNode::New()
{
        // First try to create the object from the vtkObjectFactory
        vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeRenderingDisplayNode");
        if(ret)
        {
                return (vtkMRMLVolumeRenderingDisplayNode*)ret;
        }
        // If the factory was unable to create the object, then create it here.
        return new vtkMRMLVolumeRenderingDisplayNode;
}

vtkMRMLNode* vtkMRMLVolumeRenderingDisplayNode::CreateNodeInstance(void)
{
        // First try to create the object from the vtkObjectFactory
        vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLVolumeRenderingDisplayNode");
        if(ret)
        {
                return (vtkMRMLVolumeRenderingDisplayNode*)ret;
        }
        // If the factory was unable to create the object, then create it here.
        return new vtkMRMLVolumeRenderingDisplayNode;
}

vtkMRMLVolumeRenderingDisplayNode::vtkMRMLVolumeRenderingDisplayNode(void)
{        
}

vtkMRMLVolumeRenderingDisplayNode::~vtkMRMLVolumeRenderingDisplayNode(void)
{
}
void vtkMRMLVolumeRenderingDisplayNode::WriteXML(ostream& of, int nIndent)
{
  //// Write all attributes not equal to their defaults
  //
  //Superclass::WriteXML(of, nIndent);
  //
  //vtkIndent indent(nIndent);
  //
  //of << " type=\"" << this->GetType() << "\"";

  //if (this->FileName != NULL)
  //  {
  //  of << " filename=\"" << this->FileName << "\"";
  //  }
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeRenderingDisplayNode::ReadXMLAttributes(const char** atts)
{

  //Superclass::ReadXMLAttributes(atts);

  //const char* attName;
  //const char* attValue;
  //while (*atts != NULL) 
  //  {
  //  attName = *(atts++);
  //  attValue = *(atts++);
  //  if (!strcmp(attName, "name"))
  //    {
  //    this->SetName(attValue);
  //    }
  //  else if (!strcmp(attName, "type")) 
  //    {
  //    int type;
  //    std::stringstream ss;
  //    ss << attValue;
  //    ss >> type;
  //    this->SetType(type);
  //    }
  //  else if (!strcmp(attName, "filename"))
  //    {
  //    this->SetFileName(attValue);
  //    // read in the file with the colours
  //    std::cout << "Reading file " << this->FileName << endl;
  //    this->ReadFile();
  //    }
  //  }
  //vtkDebugMacro("Finished reading in xml attributes, list id = " << this->GetID() << " and name = " << this->GetName() << endl);
}

//----------------------------------------------------------------------------
int vtkMRMLVolumeRenderingDisplayNode::ReadFile ()
{
  vtkErrorMacro("Subclass has not implemented ReadFile.");
  return 0;
}

//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLVolumeRenderingDisplayNode::Copy(vtkMRMLNode *anode)
{
  Superclass::Copy(anode);
  //vtkMRMLColorNode *node = (vtkMRMLColorNode *) anode;

  //if (node->Type != -1)
  //  {
  //  // not using SetType, as that will basically recreate a new color node,
  //  // very slow
  //  this->Type = node->Type;
  //  }
  //this->SetFileName(node->FileName);
  //this->SetNoName(node->NoName);

  //// copy names
  //this->Names = node->Names;
  //
  //this->NamesInitialised = node->NamesInitialised;
}

//----------------------------------------------------------------------------
void vtkMRMLVolumeRenderingDisplayNode::PrintSelf(ostream& os, vtkIndent indent)
{
  
 /* Superclass::PrintSelf(os,indent);

  os << indent << "Name: " <<
      (this->Name ? this->Name : "(none)") << "\n";
  

  os << indent << "Type: (" << this->GetTypeAsString() << ")\n";

  os << indent << "NoName = " <<
    (this->NoName ? this->NoName : "(not set)") <<  "\n";

  os << indent << "Names array initialised: " << (this->GetNamesInitialised() ? "true" : "false") << "\n";
  
  if (this->Names.size() > 0)
    {
    os << indent << "Color Names:\n";
    for (unsigned int i = 0; (int)i < this->Names.size(); i++)
      {
      os << indent << indent << i << " " << this->GetColorName(i) << endl;
      }
    }*/
}

//-----------------------------------------------------------

void vtkMRMLVolumeRenderingDisplayNode::UpdateScene(vtkMRMLScene *scene)
{
    Superclass::UpdateScene(scene);
    /*
    if (this->GetStorageNodeID() == NULL) 
    {
        //vtkErrorMacro("No reference StorageNodeID found");
        return;
    }

    vtkMRMLNode* mnode = scene->GetNodeByID(this->StorageNodeID);
    if (mnode) 
    {
        vtkMRMLStorageNode *node  = dynamic_cast < vtkMRMLStorageNode *>(mnode);
        node->ReadData(this);
        //this->SetAndObservePolyData(this->GetPolyData());
    }
    */
}


//---------------------------------------------------------------------------
void vtkMRMLVolumeRenderingDisplayNode::ProcessMRMLEvents ( vtkObject *caller,
                                           unsigned long event, 
                                           void *callData )
{
  Superclass::ProcessMRMLEvents(caller, event, callData);
/*
  vtkMRMLColorDisplayNode *dnode = this->GetDisplayNode();
  if (dnode != NULL && dnode == vtkMRMLColorDisplayNode::SafeDownCast(caller) &&
      event ==  vtkCommand::ModifiedEvent)
    {
        this->InvokeEvent(vtkMRMLColorNode::DisplayModifiedEvent, NULL);
    }
*/
  return;
}

//---------------------------------------------------------------------------
int vtkMRMLVolumeRenderingDisplayNode::GetFirstType()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return -1;
}

//---------------------------------------------------------------------------
int vtkMRMLVolumeRenderingDisplayNode::GetLastType()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return -1;
}

//---------------------------------------------------------------------------
const char * vtkMRMLVolumeRenderingDisplayNode::GetTypeAsString()
{
  vtkErrorMacro("Subclass has not over ridden this method");
  return "(unknown)";
}

int vtkMRMLVolumeRenderingDisplayNode::getPiecewiseFunctionString(vtkPiecewiseFunction* function, char* result)
{
    std::ostringstream resultStream;
    double anfang=0;
    double ende=500;
    int size=(ende-anfang)*10;
    double* resultArray=new double[size];
    double* it;
    function->GetTable(anfang,ende,size,resultArray);
    it=resultArray;
    for(int i=0;i<size;i++)
    {
        resultStream<<&it<<"\t";
        it++;
    }
    char* p = new char[resultStream.str().length()+1];
    resultStream.str().copy(p,std::string::npos);
    p[resultStream.str().length()]=0;
    return (resultStream.str().length()+1);
    result=p;
    
}
void  vtkMRMLVolumeRenderingDisplayNode::getColorTransferFunctionString(vtkColorTransferFunction* function, char* result)
{
}
void vtkMRMLVolumeRenderingDisplayNode::GetPiecewiseFunctionFromString(char* string,vtkPiecewiseFunction* result)
{
}
void vtkMRMLVolumeRenderingDisplayNode::GetColorTransferFunction(char* string, vtkColorTransferFunction* result)
{
}

