#include "vtkObject.h"
#include "vtkObjectFactory.h"

#include "vtkMRMLVolumeNode.h"
#include "vtkMRMLVolumeDisplayNode.h"
#include "vtkMRMLLabelMapVolumeDisplayNode.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWEntrySet.h"
#include "vtkKWPushButton.h"
#include "vtkKWEntry.h"
#include "vtkKWLabel.h"
#include "vtkKWComboBox.h"
#include "vtkKWTextWithScrollbars.h"
#include "vtkKWText.h"
#include "vtkKWCheckButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWLoadSaveDialog.h"

#include "vtkSlicerGradientEditorWidget.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include <sstream>

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerGradientEditorWidget);
vtkCxxRevisionMacro (vtkSlicerGradientEditorWidget, "$Revision: 1.0 $");

//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::AddWidgetObservers ( )
{    
    this->EnableMatrixButton->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->RotateButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NegativeButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->SwapButton->AddObserver(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
    for(int i=0; i<ARRAY_LENGTH;i ++){
        this->Checkbuttons[i]->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    }
}

void vtkSlicerGradientEditorWidget::RemoveWidgetObservers( )
{
    this->EnableMatrixButton->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->RotateButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    this->NegativeButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
    this->SwapButton->RemoveObservers(vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand);
    for(int i=0; i<ARRAY_LENGTH;i ++){
        this->Checkbuttons[i]->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand);
    }
}


void vtkSlicerGradientEditorWidget::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "vtkSlicerGradientEditorWidget: " << this->GetClassName ( ) << "\n";
}
void vtkSlicerGradientEditorWidget::ProcessWidgetEvents ( vtkObject *caller, unsigned long event, void *callData )
{
    if(event == vtkKWCheckButton::SelectedStateChangedEvent 
        && (this->Checkbuttons[0] == vtkKWCheckButton::SafeDownCast(caller)
        ||  this->Checkbuttons[1] == vtkKWCheckButton::SafeDownCast(caller)
        ||  this->Checkbuttons[2] == vtkKWCheckButton::SafeDownCast(caller))){
            //check how many Checkbuttons are selected
            int numberSelected = 0;
            for(int i=0; i<ARRAY_LENGTH;i++){
                if(this->Checkbuttons[i]->GetSelectedState()){
                    numberSelected++;
                }
            }
            //enable/disable buttons
            if (numberSelected >= 1){
                this->NegativeButton->SetEnabled(1);
                if(numberSelected == 2){
                    this->SwapButton->SetEnabled(1);}
                else {this->SwapButton->SetEnabled(0);}
                if(numberSelected == 1){
                    this->RotateButton->SetEnabled(1);
                    this->LabelAngle->SetEnabled(1);
                    this->Angle->SetEnabled(1);                }
                else {
                    this->RotateButton->SetEnabled(0);
                    this->Angle->SetEnabled(0);
                    this->LabelAngle->SetEnabled(0);
                }
            }
            else {
                this->NegativeButton->SetEnabled(0);
                this->RotateButton->SetEnabled(0);
                this->LabelAngle->SetEnabled(0);
                this->Angle->SetEnabled(0);
            }
    }
    else if(this->EnableMatrixButton == vtkKWCheckButton::SafeDownCast(caller) && event == vtkKWCheckButton::SelectedStateChangedEvent){
        for (int i=0; i<NUMBER_MATRIX_VALUES; i++){
            this->Matrix->GetWidget(i)->SetEnabled(this->EnableMatrixButton->GetSelectedState());
        }
    }
    else if (this->RotateButton == vtkKWPushButton::SafeDownCast(caller)  && event == vtkKWPushButton::InvokedEvent){
        vtkMatrix4x4 *vtkMatrixIn = vtkMatrix4x4::New();
        int n=0;
        vtkMatrixIn->Zero();
        //fill the vtkMatrix with the values from the gui
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                vtkMatrixIn->SetElement(i,j, this->Matrix->GetWidget(n)->GetValueAsDouble());
                n++;
            }            
        }
        vtkTransform* transform = vtkTransform::New();
        transform->SetMatrix(vtkMatrixIn);
        //rotate
        if(this->Checkbuttons[0]->GetSelectedState()){
            transform->RotateX(this->Angle->GetValueAsDouble());
        }
        if(this->Checkbuttons[1]->GetSelectedState()){
            transform->RotateY(this->Angle->GetValueAsDouble());
        }
        if(this->Checkbuttons[2]->GetSelectedState()){
            transform->RotateZ(this->Angle->GetValueAsDouble());
        }  
        vtkMatrix4x4 *vtkMatrixOut = transform->GetMatrix();
        int m=0;
        //fill the guiMatrix with the new values
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                this->Matrix->GetWidget(m)->SetValueAsDouble(vtkMatrixOut->GetElement(i,j));
                m++;
            }
        }
        int k=1;
    }
    else if (this->SwapButton == vtkKWPushButton::SafeDownCast(caller)  && event == vtkKWPushButton::InvokedEvent){
        int firstSelectedCheckbox  = -1;        
        int secondSelectedCheckbox = -1;
        //seek for selected checkboxes
        for(int i=0; i<ARRAY_LENGTH;i++){
            if(this->Checkbuttons[i]->GetSelectedState()){
                firstSelectedCheckbox = i;
                break;
            }
        }
        for(int i=ARRAY_LENGTH-1; i>=0;i--){
            if(this->Checkbuttons[i]->GetSelectedState()){
                secondSelectedCheckbox = i;
                break;
            }
        }
        int firstPosition = firstSelectedCheckbox*3;
        int secondPosition = secondSelectedCheckbox*3;
        //swap values
        for(int j=secondPosition; j<secondPosition+3; j++){
            const char* value = this->Matrix->GetWidget(j)->GetValue();
            this->Matrix->GetWidget(j)->SetValue(this->Matrix->GetWidget(firstPosition)->GetValue());
            this->Matrix->GetWidget(firstPosition)->SetValue(value);
            firstPosition++;
        }
    }
    else if (this->NegativeButton == vtkKWPushButton::SafeDownCast(caller)  && event == vtkKWPushButton::InvokedEvent){
        for(unsigned int j=0;j<ARRAY_LENGTH;j++){
            //seek selected checkbuttons
            if(this->Checkbuttons[j]->GetSelectedState()){            
                for(unsigned int i=0+j*3;i<3+j*3;i++){
                    int currentValue = this->Matrix->GetWidget(i)->GetValueAsInt();
                    //change sign of values != 0
                    if(currentValue != 0){
                        currentValue = currentValue-(2*currentValue);
                        this->Matrix->GetWidget(i)->SetValueAsInt(currentValue);
                    }
                }//end for2
            }// end if selected
        }//end for1
    }
}

vtkSlicerGradientEditorWidget::vtkSlicerGradientEditorWidget(void)
{
    this->EnableMatrixButton = NULL;
    this->Matrix = NULL;
    this->RotateButton = NULL;
    this->NegativeButton = NULL;
    this->SwapButton = NULL;
    this->Angle = NULL;
    this->LabelAngle = NULL;
    this->RunButton = NULL;
    this->Mask = NULL;
    this->Gradients = NULL;
    this->TestFrame = NULL;
    this->MeasurementFrame = NULL;
    this->GradientsFrame = NULL;
    this->LoadsaveFrame = NULL;
    this->LoadButton = NULL;
    this->SaveButton = NULL;
    this->LoadGradientsButton = NULL;
}

vtkSlicerGradientEditorWidget::~vtkSlicerGradientEditorWidget(void)
{
    if (this->GradientsFrame)
    {
        this->GradientsFrame->SetParent (NULL);
        this->GradientsFrame->Delete();
        this->GradientsFrame = NULL;
    }
    if (this->Gradients)
    {
        this->Gradients->SetParent (NULL);
        this->Gradients->Delete();
        this->Gradients = NULL;
    }
    if (this->Mask)
    {
        this->Mask->SetParent (NULL);
        this->Mask->Delete();
        this->Mask = NULL;
    }
    if (this->RunButton)
    {
        this->RunButton->SetParent (NULL);
        this->RunButton->Delete();
        this->RunButton = NULL;
    }
    if (this->SaveButton)
    {
        this->SaveButton->SetParent (NULL);
        this->SaveButton->Delete();
        this->SaveButton = NULL;
    }
    if (this->LoadButton)
    {
        this->LoadButton->SetParent (NULL);
        this->LoadButton->Delete();
        this->LoadButton = NULL;
    }
    if (this->LoadsaveFrame)
    {
        this->LoadsaveFrame->SetParent (NULL);
        this->LoadsaveFrame->Delete();
        this->LoadsaveFrame = NULL;
    }
    if (this->GradientsFrame)
    {
        this->GradientsFrame->SetParent (NULL);
        this->GradientsFrame->Delete();
        this->GradientsFrame = NULL;
    }
    if (this->MeasurementFrame)
    {
        this->MeasurementFrame->SetParent (NULL);
        this->MeasurementFrame->Delete();
        this->MeasurementFrame = NULL;
    }
    if (this->TestFrame)
    {
        this->TestFrame->SetParent (NULL);
        this->TestFrame->Delete();
        this->TestFrame = NULL;
    }
    if (this->LabelAngle)
    {
        this->LabelAngle->SetParent (NULL);
        this->LabelAngle->Delete();
        this->LabelAngle = NULL;
    }
    if (this->Angle)
    {
        this->Angle->SetParent (NULL);
        this->Angle->Delete();
        this->Angle = NULL;
    }
    if (this->SwapButton)
    {
        this->SwapButton->SetParent (NULL);
        this->SwapButton->Delete();
        this->SwapButton = NULL;
    }
    if (this->NegativeButton)
    {
        this->NegativeButton->SetParent (NULL);
        this->NegativeButton->Delete();
        this->NegativeButton = NULL;
    }
    if (this->RotateButton)
    {
        this->RotateButton->SetParent (NULL);
        this->RotateButton->Delete();
        this->RotateButton = NULL;
    }
    if (this->Matrix)
    {
        this->Matrix->SetParent (NULL);
        this->Matrix->Delete();
        this->Matrix = NULL;
    }
    if (this->EnableMatrixButton)
    {
        this->EnableMatrixButton->SetParent (NULL);
        this->EnableMatrixButton->Delete();
        this->EnableMatrixButton = NULL;
    }
    if (this->LoadGradientsButton)
    {
        this->LoadGradientsButton->SetParent (NULL);
        this->LoadGradientsButton->Delete();
        this->LoadGradientsButton = NULL;
    }
    for (int i=0; i<3; i++){
        this->Checkbuttons[i]->SetParent (NULL);
        this->Checkbuttons[i]->Delete();
        this->Checkbuttons[i] = NULL;
    }
    this->RemoveWidgetObservers();
}
//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::CreateWidget ( )
{
    // Check if already created
    if (this->IsCreated()){
        vtkErrorMacro(<< this->GetClassName() << " already created");
        return;
    }

    // Call the superclass to create the whole widget
    this->Superclass::CreateWidget();

    //Create the measurement frame
    this->MeasurementFrame = vtkKWFrameWithLabel::New ( );
    this->MeasurementFrame->SetParent ( this->GetParent() );
    this->MeasurementFrame->Create ( );
    this->MeasurementFrame->SetLabelText ("Measurement Frame");

    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
        this->MeasurementFrame->GetWidgetName());

    // Create Checkbutton1
    this->EnableMatrixButton = vtkKWCheckButton::New();
    this->EnableMatrixButton->SetParent(this->MeasurementFrame->GetFrame());
    this->EnableMatrixButton->SetText("Enable Matrix");
    this->EnableMatrixButton->SelectedStateOn();
    this->EnableMatrixButton->Create();

    // Create matrix for measurement frame
    this->Matrix = vtkKWEntrySet::New();
    this->Matrix->SetParent(this->MeasurementFrame->GetFrame());
    this->Matrix->Create();
    this->Matrix->SetMaximumNumberOfWidgetsInPackingDirection(3);
    this->Matrix->SetPadX(10);
    this->Matrix->SetWidgetsPadX(4);

    const char *matrixValues [NUMBER_MATRIX_VALUES] = {"1", "0", "0", "0", "1", "0", "0", "0", "1"};

    for (int i=0; i<NUMBER_MATRIX_VALUES; i++){
        this->Matrix->AddWidget(i);
        this->Matrix->GetWidget(i)->SetWidth(3);
        this->Matrix->GetWidget(i)->SetEnabled(0);
        this->Matrix->GetWidget(i)->SetValue(matrixValues[i]);
    }

    for(int i=0; i< ARRAY_LENGTH; i++){
        vtkKWCheckButton* checkButton = vtkKWCheckButton::New();
        this->Checkbuttons[i] = checkButton;
        this->Checkbuttons[i]->SetParent(this->MeasurementFrame->GetFrame());
        this->Checkbuttons[i]->Create();
        this->Script("grid %s -row 4 -column %d -sticky n", 
            checkButton->GetWidgetName(), i);
    }

    // buttons for options
    this->NegativeButton = vtkKWPushButton::New();
    this->NegativeButton->SetParent(this->MeasurementFrame->GetFrame());
    this->NegativeButton->Create();
    this->NegativeButton->SetText("Negative Selected");
    this->NegativeButton->SetWidth(15);
    this->NegativeButton->SetEnabled(0);

    this->SwapButton = vtkKWPushButton::New();
    this->SwapButton->SetParent(this->MeasurementFrame->GetFrame());
    this->SwapButton->Create();
    this->SwapButton->SetText("Swap Selected");
    this->SwapButton->SetWidth(15);
    this->SwapButton->SetEnabled(0);

    this->RotateButton = vtkKWPushButton::New();
    this->RotateButton->SetParent(this->MeasurementFrame->GetFrame());
    this->RotateButton->Create();
    this->RotateButton->SetText("Rotate Selected");
    this->RotateButton->SetWidth(15);
    this->RotateButton->SetEnabled(0);

    //Create Angle label
    this->LabelAngle = vtkKWLabel::New();
    this->LabelAngle->SetParent(this->MeasurementFrame->GetFrame());
    this->LabelAngle->Create();
    this->LabelAngle->SetText("Angle: ");

    //Create Angle combobox
    this->Angle = vtkKWComboBox::New();
    this->Angle->SetParent(this->MeasurementFrame->GetFrame());
    this->Angle->Create();
    this->Angle->SetWidth(4);
    this->Angle->SetValue("+90");

    const char *angleValues [] = {"+90", "-90", "+180", "-180", "+30", "-30"};
    for (int i=0; i<sizeof(angleValues)/sizeof(angleValues[0]); i++){
        this->Angle->AddValue(angleValues[i]);
    }

    //pack all elements
    this->Script("grid %s -row 0 -column 0 -columnspan 3 -sticky n", 
        this->EnableMatrixButton->GetWidgetName());
    this->Script("grid %s -row 1 -column 0 -columnspan 3 -rowspan 3 -sticky nes", 
        this->Matrix->GetWidgetName());
    this->Script("grid %s -row 3 -column 3 -sticky ne", 
        this->NegativeButton->GetWidgetName());
    this->Script("grid %s -row 2 -column 3 -sticky ne", 
        this->SwapButton->GetWidgetName());
    this->Script("grid %s -row 1 -column 3 -sticky ne", 
        this->RotateButton->GetWidgetName());
    this->Script("grid %s -row 1 -column 4 -sticky ne", 
        this->LabelAngle->GetWidgetName());
    this->Script("grid %s -row 1 -column 5 -sticky ne", 
        this->Angle->GetWidgetName());

    //Create gradient frame 
    this->GradientsFrame = vtkKWFrameWithLabel::New ( );
    this->GradientsFrame->SetParent ( this->GetParent() );
    this->GradientsFrame->Create ( );
    this->GradientsFrame->SetLabelText ("Gradients");

    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
        this->GradientsFrame->GetWidgetName());

    //#$GradientsFrame CollapseFrame

    //Create load button
    this->LoadGradientsButton = vtkKWLoadSaveButtonWithLabel::New();
    this->LoadGradientsButton->SetParent(this->GradientsFrame->GetFrame());
    this->LoadGradientsButton->Create();
    this->LoadGradientsButton->SetLabelText("Load Gradients from .txt/.nrrd File");
    this->LoadGradientsButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Open .txt/.nrrd File");
    this->LoadGradientsButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {Textfile} {.txt} } { {NHRDfile} {.nrrd} }");
    this->LoadGradientsButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->LoadGradientsButton->GetWidgetName());

    //Create textfield for Gradients
    this->Gradients = vtkKWTextWithScrollbars::New();
    this->Gradients->SetParent(this->GradientsFrame->GetFrame());
    this->Gradients->Create();
    this->Gradients->GetWidget()->SetText("DWMRI_b-value:=800\nDWMRI_gradient_0000:= 0 0 0 0\nDWMRI_gradient_0001:= 0 0 0 0\nDWMRI_gradient_0002:= -0.8238094 -0.4178235 -0.3830949\nDWMRI_gradient_0003:= -0.5681645 0.5019867 -0.6520725\n...");
    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
        this->Gradients->GetWidgetName());

    //Create test frame 
    this->TestFrame = vtkKWFrameWithLabel::New ( );
    this->TestFrame->SetParent ( this->GetParent() );
    this->TestFrame->Create ( );
    this->TestFrame->SetLabelText ("Test");
    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
        this->TestFrame->GetWidgetName());

    //Create ROI menu
    this->Mask = vtkSlicerNodeSelectorWidget::New();
    this->Mask->SetParent(this->TestFrame->GetFrame());
    this->Mask->Create();
    this->Mask->SetLabelText("Input ROI: ");
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->Mask->GetWidgetName());

    //Create Run button
    this->RunButton = vtkKWPushButton::New();
    this->RunButton->SetParent(this->TestFrame->GetFrame());
    this->RunButton->Create();
    this->RunButton->SetText("Run");
    this->RunButton->SetWidth(7);
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->RunButton->GetWidgetName());

    //Create load/save frame 
    this->LoadsaveFrame = vtkKWFrameWithLabel::New ( );
    this->LoadsaveFrame->SetParent ( this->GetParent() );
    this->LoadsaveFrame->Create ( );
    this->LoadsaveFrame->SetLabelText ("Load/Save");
    this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2", 
        this->LoadsaveFrame->GetWidgetName());

    //Create save button
    this->SaveButton = vtkKWLoadSaveButtonWithLabel::New();
    this->SaveButton->SetParent(this->LoadsaveFrame->GetFrame());
    this->SaveButton->Create();
    this->SaveButton->SetLabelText("Save parameters to NHDR");
    this->SaveButton->SetWidth(150);
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->SaveButton->GetWidgetName());

    //Create load button
    this->LoadButton = vtkKWLoadSaveButtonWithLabel::New();
    this->LoadButton->SetParent(this->LoadsaveFrame->GetFrame());
    this->LoadButton->Create();
    this->LoadButton->SetLabelText("Load existing NHDR");
    this->LoadButton->SetWidth(150);    
    this->LoadButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Open .nrrd File");
    this->LoadButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {NHRDfile} {.nrrd} }");
    this->LoadButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->LoadButton->GetWidgetName());

    this->AddWidgetObservers();
} 
