#include "vtkWFStepHandler.h"

//#include <WFDirectInterface.h>

#include "ModuleDescription.h"
#include "ModuleDescriptionParser.h"

vtkWFStepHandler::vtkWFStepHandler()
{
    this->m_errorMSG = "";
    this->m_nextStepFuncLoaded = false;
    this->m_nextStepFuncTCL = NULL;
    this->m_validated = false;
    this->m_validationFuncLoaded = false;
    this->m_validationFuncTCL = NULL;
    
    this->m_paramNames = "";
    this->m_paramValues = "";
}

vtkWFStepHandler::~vtkWFStepHandler()
{
}

int vtkWFStepHandler::InitializeWFEngine()
{
    this->m_wfDI = WFDirectInterface::New();
    
    if(this->m_wfDI->InitializeWFEngine())
    {
        this->m_initialized = true;
        return SUCC;
    }
    else FAIL;
}

std::vector<vtkWFStepHandler::workflowDesc*>* vtkWFStepHandler::GetKnownWorkflowDescriptions()
{
    if(!this->m_initialized)
    {
        return NULL;
    }        
//    std::vector<WFDirectInterface::workflowDesc*> knownWFs;
    return this->m_wfDI->GetAllKnownWorkflows();

//    std::cout<<"knownWFs.size() "<<knownWFs.size()<<std::endl;
}

vtkWFStepHandler *vtkWFStepHandler::New()
{
    return new vtkWFStepHandler;
}

void vtkWFStepHandler::LoadNextStepFunction(const char* tclFunc)
{
    this->m_nextStepFuncLoaded = false;
    this->m_nextStepFuncTCL = tclFunc;
    if(this->m_nextStepFuncTCL && strcmp(this->m_nextStepFuncTCL, "") != 0)
        this->m_nextStepFuncLoaded = true;
}

void vtkWFStepHandler::LoadStepValidationFunction(const char* tclFunc)
{
    this->m_validationFuncLoaded = false;
    this->m_validationFuncTCL = tclFunc;
    if(this->m_validationFuncTCL && strcmp(this->m_validationFuncTCL, "") != 0)
        this->m_validationFuncLoaded = true;
}

int vtkWFStepHandler::GetNextStepID()
{
    this->m_errorMSG = "";
    
    if(!this->GetApplication())
    {
        this->m_errorMSG = "Failed to load vtkKWApplication! Please use vtkWFStepConditionChecker::SetApplication to initialize the application object!";
        return ERR;
    }
    
    if(!this->m_nextStepFuncLoaded)
    {
        this->m_errorMSG = "vtkWFStepConditionChecker: No nextStepFunction loaded!";
        return ERR;
    }
    
    std::string tclFunction = this->m_nextStepFuncTCL;
    tclFunction.append(" \n\r getNextID");
    std::cout<<tclFunction<<std::endl;
    const char* returnValue = this->Script(tclFunction.c_str());
    std::cout<<returnValue<<std::endl;
    
    if(strcmp(returnValue,"true") == 0)
    {
        return SUCC;
    }
    else
    {
        return FAIL;
    }
}

int vtkWFStepHandler::ValidateStep()
{
    this->m_errorMSG = "";
    
    if(!this->GetApplication())
    {
        this->m_errorMSG = "Failed to load vtkKWApplication! Please use vtkWFStepConditionChecker::SetApplication to initialize the application object!";
        return ERR;
    }
    
    if(!this->m_validationFuncLoaded)
    {
        this->m_errorMSG = "vtkWFStepConditionChecker: No validationFunction loaded!";
        return ERR;
    }
    
    if(this->m_parameterToValueMap.size() == 0)
    {
        this->m_errorMSG = "vtkWFStepConditionChecker: No parameters added!";
        return ERR;
    }
    
    std::string names = "";
    std::string values = "";
    
    std::map<std::string, std::string>::iterator iter;
    for(iter = this->m_parameterToValueMap.begin(); iter != this->m_parameterToValueMap.end(); iter++)
    {
        if(names == "")
        {
            names = "{" + iter->first + "}";
            values = "{" + iter->second + "}";
        }
        else
        {
            names.append(" {" + iter->first + "}");
            values.append(" {" + iter->second + "}");
        }
    }
    
    std::string tclFunction = this->m_validationFuncTCL;
    tclFunction.append(" \n\r set names { %s }; set values { %s }; validate $names $values");
    std::cout<<tclFunction<<"\n\r"<<names<<"\n\r"<<values<<std::endl;
    const char* returnValue = this->Script(tclFunction.c_str(), names.c_str(), values.c_str());
    std::cout<<returnValue<<std::endl;
    
    if(strcmp(returnValue,"true") == 0)
    {
        return SUCC;
    }
    else
    {
        //process errormesages
        int startPos = 0;
        std::map<std::string, std::string>::iterator iter = this->m_parameterToValueMap.begin();
        std::string errorString = returnValue;
        
        this->m_parameterToErrorMap.clear();
        
        bool exit = false;
        while(!exit)           
        {
            int endPos = errorString.find("}", startPos);
            if(endPos != std::string::npos)
            {
                std::cout<<"Error: "<<errorString.substr(startPos + 1, endPos - (startPos + 1))<<" "<<startPos<<":"<<endPos<<std::endl;
                this->m_parameterToErrorMap.insert(std::make_pair(iter->first, errorString.substr(startPos + 1, endPos - (startPos + 1))));
                
                if(endPos < errorString.size() - 1)
                {
                    startPos = endPos + 2;
                    iter++;
                }
                else
                    exit = true;                          
            }
            else
                exit = true;
        }
        return FAIL;
    }
}

std::string vtkWFStepHandler::GetLastError()
{
    return this->m_errorMSG;
}

void vtkWFStepHandler::AddParameter(const char* name, const char* value)
{
    this->m_parameterToValueMap.insert(std::make_pair(name,value));
}

int vtkWFStepHandler::LoadNextWorkStep()
{
    int validated = SUCC;
    if(this->m_validationFuncLoaded)
    {
        validated = this->ValidateStep();        
    }
    else
        validated = SUCC;
    
    if(validated == SUCC)
    {
        this->m_curWFStepObject = this->m_wfDI->getNextWorkStep();
        
        if(this->m_curWFStepObject)
        {
            this->LoadNextStepFunction(this->m_curWFStepObject->GetTCLNextWorkstepFunction().c_str());
            this->LoadStepValidationFunction(this->m_curWFStepObject->GetTCLValidationFunction().c_str());
            
        }
        
        this->m_parameterToErrorMap.clear();
    }
    
    this->m_parameterToValueMap.clear();    
    
    return validated;
}

int vtkWFStepHandler::LoadBackWorkStep()
{
    this->m_curWFStepObject = this->m_wfDI->getBackWorkStep();
    
    if(this->m_curWFStepObject)
        return SUCC;
    else
        return FAIL;
}

WFEngine::nmWFStepObject::WFStepObject *vtkWFStepHandler::GetLoadedWFStep()
{
    return this->m_curWFStepObject;
}
int vtkWFStepHandler::CloseWorkflowSession()
{
    this->m_wfDI->CloseWorkflowManager();
    
    this->m_curModuleDescription = NULL;
    this->m_curWFStepObject = NULL;
    this->m_errorMSG = "";
    this->m_nextStepFuncLoaded = false;
    this->m_nextStepFuncTCL = NULL;
    
    this->m_validated = false;
    this->m_validationFuncLoaded = false;
    this->m_validationFuncTCL = NULL;
    
    
    this->m_paramNames.clear();
    this->m_paramValues.clear();
    this->m_parameterToErrorMap.clear();
    this->m_parameterToValueMap.clear();
    
    return SUCC;
}

void vtkWFStepHandler::LoadNewWorkflowSession(std::string workflowFilename)
{
    this->m_wfDI->loadWorkflowFromFile(workflowFilename);
}

ModuleDescription *vtkWFStepHandler::GetCurrentModuleDescription()
{
    ModuleDescriptionParser curMDParser;

    this->m_curModuleDescription = new ModuleDescription;
    std::string guiDesc = this->m_curWFStepObject->GetGUIDescription();
    if(guiDesc != "")
    {
        std::cout<<"guiParser: ";
        std::cout<<curMDParser.Parse(guiDesc, *(this->m_curModuleDescription))<<std::endl;
    }
    else
        this->m_curModuleDescription = NULL;
        
    return this->m_curModuleDescription;
}

std::map<std::string, std::string> *vtkWFStepHandler::GetValidationErrorMap()
{
    std::cout<<"error-size: "<<this->m_parameterToErrorMap.size()<<std::endl;
    if(this->m_parameterToErrorMap.size() == 0)
    {
        return NULL;
    }
    else
        return &(this->m_parameterToErrorMap);
}
