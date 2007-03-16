#ifndef VTKSLICERPARAMETERWIDGET_H_
#define VTKSLICERPARAMETERWIDGET_H_

#include <vector>
#include <vtkKWObject.h>
#include <string>
#include <map>
#include <ModuleParameter.h>
// Description:
// This Widget builds out of the "module Xml description" a vtkKWWidget.


class vtkKWWidget;
class vtkSlicerModuleLogic;
class vtkCallbackCommand;
class vtkKWCoreWidget;
class vtkMRMLNode;
class vtkKWLabel;
//BTX
class ModuleDescription;
//ETX

class vtkSlicerParameterWidget : public vtkKWObject
{
public:
    static vtkSlicerParameterWidget *New();
    vtkTypeMacro(vtkSlicerParameterWidget,vtkKWObject);
    
    void SetParent(vtkKWWidget *parent);
    void SetMRMLNode(vtkMRMLNode *mrmlNode);
    vtkMRMLNode *GetMRMLNode();
    void SetSlicerModuleLogic(vtkSlicerModuleLogic *logic);
    void SetModuleDescription(ModuleDescription *modDescription);
    //BTX
    std::vector<ModuleParameter> *GetCurrentParameters();
    
    struct ParameterWidgetChangedStruct{
        vtkKWWidget *inputWidget;        
        ModuleParameter *widgetParameter;
    };
    
    void SetWidgetID(std::string ID);
    
    std::string GetValueByName(std::string name);
    void SetErrorMap(std::map<std::string, std::string> *errorMap);
    //ETX
    bool IsCreated();
    int CreateWidgets();
    
    vtkKWWidget *GetNextWidget();
    
    bool end();
    void reset();
    int size();
    int currentIndex();
    
    //BTX
    enum{
      ParameterWidgetChangedEvent = 10000
    };
    
    enum{
        ERR = -1,
        FAIL = 0,
        SUCC = 1        
    };
    //ETX
protected:
    vtkSlicerParameterWidget();
    virtual ~vtkSlicerParameterWidget();
    
    void copyModuleParameters(const ModuleParameter &from, const ModuleParameter &to);
    
    void AddGUIObservers();
    
    void AddParameterAndEventToWidget(vtkKWCoreWidget* parentWidget, ModuleParameter widgetParameter);
    
    // Description:
    // create new attribute for the widget in the mrml node or get the value from the node
    void UpdateMRMLForWidget(vtkKWCoreWidget* parentWidget, ModuleParameter widgetParameter);
    
    // Callback routine that is used for creating a new node.  This
    // method is needed to avoid recursive calls to GUICallback()
    static void GUIChangedCallback( vtkObject *__caller,
                                unsigned long eid, void *__clientData, void *callData );    

//    vtkCallbackCommand *GUIChangedCallbackCommand;
    const char* GetValueFromWidget(vtkKWWidget *widg);
    
    void SetValueForWidget(vtkKWCoreWidget *widg, const char* value);
    //BTX
    struct callBackDataStruct{
        ModuleParameter widgetParameter;
        vtkSlicerParameterWidget *parentClass;
    };
    
    struct moduleParameterWidgetStruct{
        std::vector<ModuleParameter> *modParams;
        vtkKWCoreWidget *paramWidget;
    };
    
    const char *GetAttributeName(std::string name);
        
    std::string GetErrorByParamName(std::string name);
    //ETX
    void DeleteInternalLists();
    
    void Initialize();
        
private:
    //BTX
    std::vector<moduleParameterWidgetStruct*> *m_InternalWidgetParamList;
    std::vector<moduleParameterWidgetStruct*>::iterator m_InternalIterator;
    
//    std::map<vtkKWCoreWidget*, ModuleParameter> *m_internalWidgetToParamMap; 
    std::string m_curWidgetLabel;
    std::string m_widgID;

    std::map<std::string, std::string> *m_paramToErrorMap;
    //ETX
    vtkKWWidget *m_ParentWidget;
    vtkSlicerModuleLogic *m_ModuleLogic;
    ModuleDescription *m_ModuleDescription;
    vtkMRMLNode *m_MRMLNode;
    
    
    
    bool m_End;
    bool m_Created;
    int m_CurrentIndex;    
};

#endif /*VTKSLICERPARAMETERWIDGET_H_*/
