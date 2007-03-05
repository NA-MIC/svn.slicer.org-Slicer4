#ifndef VTKWFENGINEEVENTHANDLER_H_
#define VTKWFENGINEEVENTHANDLER_H_

#include <vtkKWObject.h>

class vtkObject;
class vtkCallbackCommand;
class vtkWFEngineModuleGUI;
class vtkMRMLScene;

class vtkWFEngineEventHandler : public vtkKWObject
{
public:
    static vtkWFEngineEventHandler *New();
    
    void AddWorkflowObservers(vtkWFEngineModuleGUI *curModuleGUI);
        
    void SetMRMLScene(vtkMRMLScene *scene);
    
    vtkMRMLScene *GetMRMLScene();
    
    void SetEventName(const char* name);
    void SetCurrentStepID(const char* id);
protected:
    vtkWFEngineEventHandler();
    virtual ~vtkWFEngineEventHandler();
    
    static void ProcessWorkflowEvents(vtkObject *caller, unsigned long event, void *clientData, void *callData);
    
private:
    
    vtkCallbackCommand *m_workflowCB;
    
    vtkMRMLScene *m_mrmlScene;
    
    const char* m_id;
    const char* m_eventName;
};

#endif /*VTKWFENGINEEVENTHANDLER_H_*/
