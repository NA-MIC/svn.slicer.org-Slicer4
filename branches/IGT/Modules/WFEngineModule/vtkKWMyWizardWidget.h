#ifndef VTKKWMYWIZARDWIDGET_H_
#define VTKKWMYWIZARDWIDGET_H_

#include <vtkKWWizardWidget.h>

class vtkKWWizardWorkflow;
class vtkKWMyWizardWorkflow;
class vtkKWProgressGauge;
class vtkKWComboBoxWithLabel;

class vtkKWMyWizardWidget : public vtkKWWizardWidget
{
public:
    static vtkKWMyWizardWidget *New();
    
    vtkTypeRevisionMacro(vtkKWMyWizardWidget,vtkKWWizardWidget);
    
    vtkKWWizardWorkflow *GetWizardWorkflow();
    vtkKWMyWizardWorkflow *GetMyWizardWorkflow();
    
    void SetNumberOfUnprocessedSteps(int steps);
    int GetNumberOfUnprocessedSteps();
    void SetNumberOfProcessedSteps(int steps);
    //BTX
    enum{
        nextButtonClicked = 10000,
        backButtonClicked
    };
    //ETX
    
    void UpdateProcessGauge();
    
    virtual void Delete();
protected:
    vtkKWMyWizardWidget();
    virtual ~vtkKWMyWizardWidget();
    // Description:
    // Create the widget.
    virtual void CreateWidget();
    
    static void NextButtonClicked(vtkObject* obj, unsigned long,void*, void*);
    static void BackButtonClicked(vtkObject* obj, unsigned long,void*, void*);
    
    static void NavigationStackChanged(vtkObject* obj, unsigned long,void*, void*);
private:
    vtkKWMyWizardWorkflow *WizardWorkflow;
    
    vtkKWProgressGauge *m_wfAdvancementPG;
    vtkKWComboBoxWithLabel *m_historyCBWL;
    
    int m_numberOfUnprocessedSteps;
    int m_numberOfProcessedSteps;
    
    vtkKWMyWizardWidget(const vtkKWMyWizardWidget&);   // Not implemented.
    void operator=(const vtkKWMyWizardWidget&);  // Not implemented.
};

#endif /*VTKKWMYWIZARDWIDGET_H_*/
