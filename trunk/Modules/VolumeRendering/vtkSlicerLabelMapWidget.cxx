#include "vtkSlicerLabelMapWidget.h"

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkKWTreeWithScrollbars.h"
#include "vtkKWTree.h"
#include "vtkKWPushButton.h"
#include "vtkKWLabel.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerLabelmapTree.h"
#include "vtkSlicerLabelmapElement.h"


vtkStandardNewMacro (vtkSlicerLabelMapWidget);
//vtkCxxRevisionMacro (vtkSlicerLabelMapWidget, "$Revision: 1.0 $");
vtkSlicerLabelMapWidget::vtkSlicerLabelMapWidget(void)
{
    this->Tree=NULL;
    this->ChangeAll=NULL;
}

vtkSlicerLabelMapWidget::~vtkSlicerLabelMapWidget(void)
{
    if(this->ChangeAll!=NULL)
    {
        this->Script("pack forget %s",this->ChangeAll->GetWidgetName());
        this->ChangeAll->RemoveObservers(vtkCommand::AnyEvent);
        this->ChangeAll->SetParent(NULL);
        this->ChangeAll->Delete();
        this->ChangeAll=NULL;
    }
    if(this->Tree!=NULL)
    {
        this->Script("pack forget %s",this->Tree->GetWidgetName());
        this->Tree->RemoveObservers(vtkSlicerLabelmapTree::SingleLabelEdited);
        this->Tree->SetParent(NULL);
        this->Tree->Delete();
        this->Tree=NULL;
    }

}

void vtkSlicerLabelMapWidget::CreateWidget(void)
{
    this->Superclass::CreateWidget();
    this->ChangeAll=vtkSlicerLabelmapElement::New();
    this->ChangeAll->SetParent(this);
    this->ChangeAll->Create();
    double color[3];
    color[0]=0.5;
    color[1]=0.5;
    color[2]=0.5;
    this->ChangeAll->Init(INT_MIN,"ALL COLORS",color,.2,20);
    this->ChangeAll->ChangeOpacity(-1);
    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",this->ChangeAll->GetWidgetName());
    this->ChangeAll->AddObserver(vtkCommand::AnyEvent,(vtkCommand *)this->GUICallbackCommand);
    this->Tree=vtkSlicerLabelmapTree::New();
    this->Tree->SetParent(this);
    this->Tree->Create();
    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",this->Tree->GetWidgetName());
    this->Tree->AddObserver(vtkSlicerLabelmapTree::SingleLabelEdited,(vtkCommand *) this->GUICallbackCommand);


}
void vtkSlicerLabelMapWidget::ProcessWidgetEvents(vtkObject *caller, unsigned long event, void *callData)
{
    vtkSlicerLabelmapElement *callerLabelmap=vtkSlicerLabelmapElement::SafeDownCast(caller);

    if(callerLabelmap==this->ChangeAll)
    {
        int *opacities=(int*) callData;
        this->Tree->ChangeAllOpacities(opacities[1]);
        this->InvokeEvent(vtkSlicerLabelMapWidget::NeedForRenderEvent);
        return;
    }
    vtkSlicerLabelmapTree *callerLabelmapTree=vtkSlicerLabelmapTree::SafeDownCast(caller);
    if(callerLabelmapTree==this->Tree&&event==vtkSlicerLabelmapTree::SingleLabelEdited)
    {

        this->ChangeAll->ChangeOpacity(-1);
        this->InvokeEvent(vtkSlicerLabelMapWidget::NeedForRenderEvent);
    }

}
void vtkSlicerLabelMapWidget::UpdateGuiElements(void)
{
    //Update the tree
    if(this->Tree==NULL)
    {
        vtkErrorMacro("Call Init before updating GUI Elements");
    }
    else
    {
        this->Tree->UpdateGuiElements();
    }
    //Update the all Buttons
    //Not needed yet

}
