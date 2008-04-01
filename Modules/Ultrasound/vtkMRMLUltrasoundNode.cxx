#include "vtkMRMLUltrasoundNode.h"
#include "vtkMRMLNode.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <string>

vtkMRMLUltrasoundNode* vtkMRMLUltrasoundNode::New()
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLUltrasoundNode");
    if(ret)
    {
        return (vtkMRMLUltrasoundNode*)ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkMRMLUltrasoundNode;
}

vtkMRMLNode* vtkMRMLUltrasoundNode::CreateNodeInstance(void)
{
    // First try to create the object from the vtkObjectFactory
    vtkObject* ret = vtkObjectFactory::CreateInstance("vtkMRMLUltrasoundNode");
    if(ret)
    {
        return (vtkMRMLUltrasoundNode*)ret;
    }
    // If the factory was unable to create the object, then create it here.
    return new vtkMRMLUltrasoundNode;
}

vtkMRMLUltrasoundNode::vtkMRMLUltrasoundNode(void)
{   
}

vtkMRMLUltrasoundNode::~vtkMRMLUltrasoundNode(void)
{
}
void vtkMRMLUltrasoundNode::WriteXML(ostream& of, int nIndent)
{
    // Write all attributes not equal to their defaults

    Superclass::WriteXML(of, nIndent);

    //vtkIndent indent(nIndent);
    of << " references=\""<<this->References.size()<<" ";
    for(unsigned int i=0;i<this->References.size();i++)
    {
        of<<this->References.at(i);
        if(i!=(this->References.size()-1))
        {
            of<<" ";
        }
    }
    of<<"\"";
}

//----------------------------------------------------------------------------
void vtkMRMLUltrasoundNode::ReadXMLAttributes(const char** atts)
{
    Superclass::ReadXMLAttributes(atts);

    const char* attName;
    const char* attValue;
    while (*atts!=NULL){
        attName= *(atts++);
        attValue= *(atts++);
        if(!strcmp(attName,"references"))
        {
            int size=0;
            std::stringstream ss;
            ss << attValue;
            ss>>size;
            for(int i=0;i<size;i++)
            {
                std::string str;
                ss>>str;
                this->AddReference(str.c_str());
            }
        }
    }
        vtkDebugMacro("Finished reading in xml attributes, list id = " << this->GetID() << " and name = " << this->GetName() << endl);
}


//----------------------------------------------------------------------------
// Copy the node's attributes to this object.
// Does NOT copy: ID, FilePrefix, Name, ID
void vtkMRMLUltrasoundNode::Copy(vtkMRMLNode *anode)
{
    Superclass::Copy(anode);
    this->CopyParameterset(anode);

}
void vtkMRMLUltrasoundNode::CopyParameterset(vtkMRMLNode *anode)
{
    //cast
    vtkMRMLUltrasoundNode *node = (vtkMRMLUltrasoundNode *) anode;

}

//----------------------------------------------------------------------------
void vtkMRMLUltrasoundNode::PrintSelf(ostream& os, vtkIndent indent)
{
    Superclass::PrintSelf(os,indent);
}

//-----------------------------------------------------------

void vtkMRMLUltrasoundNode::UpdateScene(vtkMRMLScene *scene)
{
    Superclass::UpdateScene(scene);
}


//---------------------------------------------------------------------------
void vtkMRMLUltrasoundNode::ProcessMRMLEvents ( vtkObject *caller,
                                                    unsigned long event, 
                                                    void *callData )
{
    Superclass::ProcessMRMLEvents(caller, event, callData);
    return;
}



void vtkMRMLUltrasoundNode::AddReference(std::string id)
{
    //test if we already have a reference
    if(this->HasReference(id))
    {
        return;
    }
    else 
    {
        this->References.push_back(id);
    }

}
bool vtkMRMLUltrasoundNode::HasReference(std::string id)
{
    //loop over vector and comparing
    for(unsigned int i=0;i<this->References.size();i++)
    {
        if(strcmp(this->References.at(i).c_str(),id.c_str())==0)
        {
            return true;
        }
    }
    return false;
}
void vtkMRMLUltrasoundNode::RemoveReference(std::string id)
{
    vtkErrorMacro("Not yet implemented");

}
