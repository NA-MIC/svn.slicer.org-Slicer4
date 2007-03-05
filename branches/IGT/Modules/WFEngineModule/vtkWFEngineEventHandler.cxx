#include "vtkWFEngineEventHandler.h"

#include <vtkCallbackCommand.h>
#include <vtkKWObject.h>
#include <vtkMRMLScene.h>
#include <vtkMRMLFiducialListNode.h>

#include "vtkWFEngineModuleGUI.h"

#include <string>
#include <sstream>

vtkWFEngineEventHandler::vtkWFEngineEventHandler()
{
    this->m_workflowCB = NULL;
    this->m_mrmlScene = NULL;
    this->m_id = NULL;
    this->m_eventName = NULL;
}

vtkWFEngineEventHandler::~vtkWFEngineEventHandler()
{
    if(this->m_workflowCB)
    {
        this->m_workflowCB->Delete();
        this->m_workflowCB = NULL;
    }
    
    this->m_mrmlScene = NULL;
}

vtkWFEngineEventHandler *vtkWFEngineEventHandler::New()
{
    return new vtkWFEngineEventHandler;
}

void vtkWFEngineEventHandler::ProcessWorkflowEvents(vtkObject *caller, unsigned long event, void *clientData, void *callData)
{
    std::cout<<"vtkWFEngineEventHandler::ProcessWorkflowEvents"<<std::endl;
    vtkWFEngineModuleGUI *curWFGUI = (vtkWFEngineModuleGUI*)caller;
    vtkWFEngineEventHandler *curWFEventHandler = (vtkWFEngineEventHandler*)callData;
    std::string test = "";
    
    if(curWFEventHandler)
    {
        std::cout<<"EVENT catched: "<<curWFEventHandler->m_eventName<<std::endl;
        if(curWFEventHandler->m_eventName && strcmp(curWFEventHandler->m_eventName, "addFiducial") == 0)
        {
            if(curWFEventHandler->m_mrmlScene)
            {
                // get Parameter from the m_mrmlScene
                
                vtkMRMLNode *tmpWFEngineNode = curWFEventHandler->m_mrmlScene->GetNodeByID("vtkMRMLWFEngineModuleNode1");
                vtkMRMLWFEngineModuleNode *wfEngineModuleNode = vtkMRMLWFEngineModuleNode::SafeDownCast(tmpWFEngineNode);
                if(wfEngineModuleNode)
                {
                    std::string fidNameParameter = curWFEventHandler->m_id;
                    fidNameParameter.append(".fidName");
                    std::string fidXParameter = curWFEventHandler->m_id;
                    fidXParameter.append(".fidX");
                    std::string fidYParameter = curWFEventHandler->m_id;
                    fidYParameter.append(".fidY");
                    std::string fidZParameter = curWFEventHandler->m_id;
                    fidZParameter.append(".fidZ");
                    
                    const char* fidNameValue = wfEngineModuleNode->GetAttribute(fidNameParameter.c_str());
                    const char* fidXValue = wfEngineModuleNode->GetAttribute(fidXParameter.c_str());
                    const char* fidYValue = wfEngineModuleNode->GetAttribute(fidYParameter.c_str());
                    const char* fidZValue = wfEngineModuleNode->GetAttribute(fidZParameter.c_str());
                    
                    // is there one list?
                    vtkMRMLFiducialListNode *activeFiducialListNode = NULL;
                    vtkMRMLNode *node = curWFEventHandler->m_mrmlScene->GetNextNodeByClass("vtkMRMLFiducialListNode");
                    if (node == NULL)
                    {
                        std::cerr << "ERROR: No Fiducial List, adding one first!\n";                    
                        vtkMRMLNode *newList = curWFEventHandler->m_mrmlScene->CreateNodeByClass("vtkMRMLFiducialListNode");      
                        if (newList != NULL)
                        {
                            curWFEventHandler->m_mrmlScene->AddNode(newList);
                            activeFiducialListNode = vtkMRMLFiducialListNode::SafeDownCast(newList);
    //                        newList->Delete();
                        }              
                    }
                    else
                    {
                        activeFiducialListNode = vtkMRMLFiducialListNode::SafeDownCast(node);
                    }
                    
                    if(activeFiducialListNode)
                    {
                        int fidID;
                        fidID = activeFiducialListNode->AddFiducial();
                        if(fidNameValue)
                        {
                            activeFiducialListNode->SetNthFiducialLabelText(fidID, fidNameValue);   
                        }                        
                        if(fidXValue && fidYValue && fidZValue)
                        {
                            std::ostringstream strvalue;
                            std::string a_double;
                            double x;
                            a_double = fidXValue;
                            std::istringstream sx(a_double);
                            sx >> x;
                            
                            double y;
                            a_double = fidYValue;
                            std::istringstream sy(a_double);
                            sy >> y;
                            
                            double z;
                            a_double = fidZValue;
                            std::istringstream sz(a_double);
                            sz >> z;
                                    
                            
                            activeFiducialListNode->SetNthFiducialXYZ(fidID, x, y, z);
                        }
                    }                                
                }
                
            }        
        }
    }
}

void vtkWFEngineEventHandler::AddWorkflowObservers(vtkWFEngineModuleGUI *curModuleGUI)
{
    this->m_workflowCB = vtkCallbackCommand::New();
    this->m_workflowCB->SetClientData(this);
    this->m_workflowCB->SetCallback(&vtkWFEngineEventHandler::ProcessWorkflowEvents);
    
    if(curModuleGUI)
    {
        curModuleGUI->AddObserver(vtkWFEngineModuleGUI::WorkflowHandleEvent, this->m_workflowCB);
    }          
}

void vtkWFEngineEventHandler::SetMRMLScene(vtkMRMLScene *scene)
{
    this->m_mrmlScene = scene;
}

vtkMRMLScene *vtkWFEngineEventHandler::GetMRMLScene()
{
    return this->m_mrmlScene;
}

void vtkWFEngineEventHandler::SetCurrentStepID(const char *id)
{
    this->m_id = id;
}

void vtkWFEngineEventHandler::SetEventName(const char *name)
{
    this->m_eventName = name;
}
