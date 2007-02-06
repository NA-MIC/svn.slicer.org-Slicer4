#include "WFStepObject.h"
#include <iostream>
using namespace WFEngine::nmWFStepObject;

WFStepObject::WFStepObject()
{
    this->m_ID = "";
    this->m_name = "";
    this->m_desc = "";
    this->m_guiDesc = "";
}

WFStepObject::~WFStepObject()
{
    
}

WFStepObject *WFStepObject::New()
{
    return new WFStepObject();
}

void WFStepObject::SetID(std::string &ID)
{
    this->m_ID = ID;
}

void WFStepObject::SetName(std::string &name)
{
    this->m_name = name;
}

void WFStepObject::AddNextStepID(std::string &nextStepID)
{
    this->m_nextSteps.push_back(nextStepID);
}

void WFStepObject::SetVariableMapping(std::string &from, std::string &to)
{
    this->m_varMapping.insert(std::make_pair(from,to));
}

void WFStepObject::SetDescription(std::string &stepDesc)
{
    this->m_desc = stepDesc;
}

void WFStepObject::SetGUIDescription(std::string &stepGUIDesc)
{
    if(strcmp(stepGUIDesc.c_str(),"") != 0)
    {
        if(strcmp(stepGUIDesc.substr(0,5).c_str(), "<?xml") != 0)
        {
            std::string xmlHeader = "<?xml version=\"1.0\" encoding=\"utf-8\"?>";
            xmlHeader.append(stepGUIDesc);
            stepGUIDesc = xmlHeader;
        }
        this->m_guiDesc = stepGUIDesc;        
    }    
}

std::string WFStepObject::GetID()
{
    return this->m_ID;
}

std::string WFStepObject::GetName()
{
    return this->m_name;
}

std::string WFStepObject::GetNextStepID()
{
    if(this->m_nextSteps.size() > 0)
        return this->m_nextSteps[0];
    else
        return "";           
}

std::string WFStepObject::GetDescription()
{
    return this->m_desc;
}

std::string WFStepObject::GetGUIDescription()
{
    return this->m_guiDesc;
}
