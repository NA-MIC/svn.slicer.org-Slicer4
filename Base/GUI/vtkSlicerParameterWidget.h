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
    
    typedef struct{
        vtkKWWidget *inputWidget;
        ModuleParameter *widgetParameter;
    } ParameterWidgetChangedStruct;
    
    void SetWidgetID(std::string ID);
    //ETX
    bool IsCreated();
    void CreateWidgets();
    
    vtkKWWidget *GetNextWidget();    
    
    bool end();
    void reset();
    int size();
    int currentIndex();
    
    //BTX
    enum{
      ParameterWidgetChangedEvent = 10000
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
    typedef struct{
        ModuleParameter widgetParameter;
        vtkSlicerParameterWidget *parentClass;
    } callBackDataStruct;
    
    typedef struct{
        std::vector<ModuleParameter> *modParams;
        vtkKWCoreWidget *paramWidget;
    } moduleParameterWidgetStruct;
    
    const char *GetAttributeName(std::string name);
    //ETX
        
private:
    //BTX
    std::vector<moduleParameterWidgetStruct*> *m_InternalWidgetParamList;
    std::vector<moduleParameterWidgetStruct*>::iterator m_InternalIterator;
    
    std::map<vtkKWCoreWidget*, ModuleParameter> *m_internalWidgetToParamMap; 
    std::string m_curWidgetLabel;
    std::string m_widgID;
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
