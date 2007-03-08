#include "vtkKWMyWizardWidget.h"

#include <vtkObjectFactory.h>

#include <vtkKWWizardWorkflow.h>
#include "vtkKWMyWizardWorkflow.h"
#include <vtkKWPushButton.h>
#include <vtkCallbackCommand.h>

#include <vtkKWComboBoxWithLabel.h>
#include <vtkKWProgressGauge.h>

#include <vtkKWApplication.h>
#include <vtkKWSeparator.h>
#include <vtkKWComboBox.h>
#include <vtkKWLabel.h>

//----------------------------------------------------------------------------
vtkStandardNewMacro( vtkKWMyWizardWidget );
vtkCxxRevisionMacro(vtkKWMyWizardWidget, "$Revision: 1.3 $");

vtkKWMyWizardWidget::vtkKWMyWizardWidget()
{
    this->TitleFrame              = NULL;
    this->TitleLabel              = NULL;
    this->SubTitleLabel           = NULL;
    this->TitleIconLabel          = NULL;

    this->SeparatorAfterTitleArea = NULL;

    this->LayoutFrame             = NULL;
    this->PreTextLabel            = NULL;
    this->ClientArea              = NULL;
    this->PostTextLabel           = NULL;
    this->ErrorTextLabel          = NULL;

    this->SeparatorBeforeButtons  = NULL;

    this->ButtonFrame             = NULL;
    this->BackButton              = NULL;
    this->NextButton              = NULL;
    this->FinishButton            = NULL;
    this->CancelButton            = NULL;
    this->HelpButton              = NULL;
    this->OKButton                = NULL;

    this->WizardWorkflow          = vtkKWMyWizardWorkflow::New();

    this->BackButtonVisibility    = 1;
    this->NextButtonVisibility    = 1;
    this->FinishButtonVisibility  = 1;
    this->CancelButtonVisibility  = 1;
    this->HelpButtonVisibility    = 0;
    this->OKButtonVisibility      = 1;

    this->ButtonsPosition   = vtkKWWizardWidget::ButtonsPositionBottom;
    
    this->m_wfAdvancementPG = NULL;
    this->m_historyCBWL = NULL;
}

vtkKWMyWizardWidget::~vtkKWMyWizardWidget()
{
}

vtkKWWizardWorkflow *vtkKWMyWizardWidget::GetWizardWorkflow()
{
    return this->WizardWorkflow;
}

vtkKWMyWizardWorkflow *vtkKWMyWizardWidget::GetMyWizardWorkflow()
{
    return this->WizardWorkflow;
}

//----------------------------------------------------------------------------
void vtkKWMyWizardWidget::CreateWidget()
{
  // Check if already created

  if (this->IsCreated())
    {
    vtkErrorMacro("class already created");
    return;
    }
  
  if(!this->WizardWorkflow)
  {
      this->WizardWorkflow = vtkKWMyWizardWorkflow::New();
  }
  if(!this->WizardWorkflow->GetApplication())
  {
      this->WizardWorkflow->SetApplication(this->GetApplication());
  }

  // Call the superclass to create the whole widget
  this->Superclass::WizardWorkflow = this->WizardWorkflow;
  this->Superclass::CreateWidget();
  
  // Add some workflow related items into the wizard gui
  if(!this->m_historyCBWL)
  {
      this->m_historyCBWL = vtkKWComboBoxWithLabel::New();
  }

  this->m_historyCBWL->SetParent(this->TitleFrame);
  this->m_historyCBWL->SetBackgroundColor(this->GetTitleAreaBackgroundColor());
  this->m_historyCBWL->GetWidget()->SetBackgroundColor(this->GetTitleAreaBackgroundColor());
  this->m_historyCBWL->Create();
  
//  this->m_historyCBWL->SetBackgroundColor(this->GetTitleAreaBackgroundColor());
  this->m_historyCBWL->SetBackgroundColor(1.0,0,0);
  this->m_historyCBWL->GetWidget()->SetBackgroundColor(1.0,0,0);
  this->m_historyCBWL->GetLabel()->SetBackgroundColor(1.0,0,0);
//  this->m_historyCBWL->GetWidget()->SetBackgroundColor(this->GetTitleAreaBackgroundColor());
  
  this->m_historyCBWL->SetLabelText("History:");
  
  this->Script("grid %s -row 0 -column 2 -sticky nswe -padx 8",
          this->m_historyCBWL->GetWidgetName());

  this->Script("grid columnconfigure %s 2 -weight 0",
          this->TitleFrame->GetWidgetName());
  
//  this->GetApplication()->Script("pack %s -side right -anchor ne -expand y -fill both -padx 2 -pady 2", 
//          cbStepHistory->GetWidgetName());
  
  // Add some workflow related items into the wizard gui
  if(!this->m_wfAdvancementPG)
  {
      this->m_wfAdvancementPG = vtkKWProgressGauge::New();
  }
  
  this->m_wfAdvancementPG->SetParent(this->TitleFrame);
  this->m_wfAdvancementPG->Create();  
  
  this->Script("grid %s -row 1 -column 2 -sticky nwe -padx 8",
          this->m_wfAdvancementPG->GetWidgetName());

  this->Script("grid columnconfigure %s 2 -weight 0",
          this->TitleFrame->GetWidgetName());
//  
//  if (this->ButtonsPosition == vtkKWWizardWidget::ButtonsPositionBottom) {
//        this->Script("pack %s -side left -fill x -padx 0 -pady 0",
//                pgWorkflow->GetWidgetName());
//    } else {
//        
//        this->Script("pack %s -side left -fill x -padx 0 -pady 0 -before %s",
//                pgWorkflow->GetWidgetName(),
//                this->SeparatorBeforeButtons->GetWidgetName());
//    }

////  cbStepHistory->SetLabelText("History:");
//  
//  this->GetApplication()->Script("pack %s -side left -anchor ne -expand y -fill both -padx 2 -pady 2", 
//          pgWorkflow->GetWidgetName());
  
  
  vtkCallbackCommand *nextBtnClicked = vtkCallbackCommand::New();
  nextBtnClicked->SetClientData(this);
  nextBtnClicked->SetCallback(&vtkKWMyWizardWidget::NextButtonClicked);
  this->NextButton->AddObserver(vtkKWPushButton::InvokedEvent, nextBtnClicked);
  
  vtkCallbackCommand *backBtnClicked = vtkCallbackCommand::New();
  backBtnClicked->SetClientData(this);
  backBtnClicked->SetCallback(&vtkKWMyWizardWidget::BackButtonClicked);
  this->BackButton->AddObserver(vtkKWPushButton::InvokedEvent, backBtnClicked);
  
  vtkCallbackCommand *navStackChanged = vtkCallbackCommand::New();
  navStackChanged->SetClientData(this);
  navStackChanged->SetCallback(&vtkKWMyWizardWidget::NavigationStackChanged);
  this->AddObserver(vtkKWWizardWorkflow::NavigationStackedChangedEvent, navStackChanged);
}

void vtkKWMyWizardWidget::NextButtonClicked(vtkObject* obj, unsigned long,void* callbackData, void*)
{
    std::cout<<"nextButtonClicked"<<std::endl;
    vtkKWMyWizardWidget *myWizWidg = (vtkKWMyWizardWidget*)callbackData;
    if(myWizWidg)
    {
        myWizWidg->InvokeEvent(vtkKWMyWizardWidget::nextButtonClicked);       
    }
}

void vtkKWMyWizardWidget::BackButtonClicked(vtkObject* obj, unsigned long,void* callbackData, void*)
{
    std::cout<<"backButtonClicked"<<std::endl;
    vtkKWMyWizardWidget *myWizWidg = (vtkKWMyWizardWidget*)callbackData;
    if(myWizWidg)
    {
        myWizWidg->InvokeEvent(vtkKWMyWizardWidget::backButtonClicked);
    }    
}

void vtkKWMyWizardWidget::NavigationStackChanged(vtkObject* obj, unsigned long,void* callbackData, void* clientData)
{
    vtkKWMyWizardWidget *myWizWidg = (vtkKWMyWizardWidget*)callbackData;            
    myWizWidg->UpdateProcessGauge();
}

int vtkKWMyWizardWidget::GetNumberOfUnprocessedSteps()
{
    return this->m_numberOfUnprocessedSteps;
}

void vtkKWMyWizardWidget::SetNumberOfUnprocessedSteps(int steps)
{
    this->m_numberOfUnprocessedSteps = steps;
}

void vtkKWMyWizardWidget::SetNumberOfProcessedSteps(int steps)
{
    this->m_numberOfProcessedSteps = steps;
}

void vtkKWMyWizardWidget::UpdateProcessGauge()
{
    double percent = 0;
    if(this->WizardWorkflow->GetCurrentStep() == this->WizardWorkflow->GetFinishStep())
    {
        percent = 100;
    }
    else if(this->WizardWorkflow->GetCurrentStep() == this->WizardWorkflow->GetInitialStep())
    {
        percent = 0;
    }
    else
    {
        // subtract 2 from the actual navigation stack because of the intial and last step
        int stepAmount = this->m_numberOfUnprocessedSteps + this->m_numberOfProcessedSteps + 1;
        int processedSteps = this->m_numberOfProcessedSteps + 1;
        percent = (processedSteps * 100 / stepAmount);        
    }
            
    this->m_wfAdvancementPG->SetValue(percent);
}

void vtkKWMyWizardWidget::Delete()
{
    if(this->m_wfAdvancementPG)
    {
        this->m_wfAdvancementPG->Unpack();
        this->m_wfAdvancementPG->Delete();
        this->m_wfAdvancementPG = NULL;
    }
    
    if(this->m_historyCBWL)
    {
        this->m_historyCBWL->Unpack();
        this->m_historyCBWL->Delete();
        this->m_historyCBWL = NULL;
    }
    
    this->Superclass::Delete();
}
