#ifndef VTKSLICERPARAMETERWIDGET_H_
#define VTKSLICERPARAMETERWIDGET_H_

#include <vector>
#include <vtkKWObject.h>
// Description:
// This Widget builds out of the "module Xml description" a vtkKWWidget.


class vtkKWWidget;
class vtkSlicerModuleLogic;
//BTX
class ModuleDescription;
class ModuleParameter;
//ETX

class vtkSlicerParameterWidget : public vtkKWObject
{
public:
    static vtkSlicerParameterWidget *New();
    vtkTypeMacro(vtkSlicerParameterWidget,vtkKWObject);
    
    void SetParent(vtkKWWidget *parent);
    void SetSlicerModuleLogic(vtkSlicerModuleLogic *logic);
    void SetModuleDescription(ModuleDescription *modDescription);
    //BTX
    typedef struct{
        const ModuleParameter *modParam;
        vtkKWWidget *paramWidget;
    } moduleParameterWidgetStruct;       
    //ETX
    bool IsCreated();
    void CreateWidgets();
    
    vtkKWWidget *GetNextWidget();
    const ModuleParameter *GetCurrentParameter();
    
    bool end();
    void reset();
    int size();
    int currentIndex();
protected:
    vtkSlicerParameterWidget();
    virtual ~vtkSlicerParameterWidget();
        
private:
    //BTX
    std::vector<moduleParameterWidgetStruct*> *m_InternalWidgetParamList;
    std::vector<moduleParameterWidgetStruct*>::iterator m_InternalIterator;
    //ETX
    vtkKWWidget *m_ParentWidget;
    vtkSlicerModuleLogic *m_ModuleLogic;
    ModuleDescription *m_ModuleDescription;
    
    bool m_End;
    bool m_Created;
    int m_CurrentIndex;    
};

#endif /*VTKSLICERPARAMETERWIDGET_H_*/
