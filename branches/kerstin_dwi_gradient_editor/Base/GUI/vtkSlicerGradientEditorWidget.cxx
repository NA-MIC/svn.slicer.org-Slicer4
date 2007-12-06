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


//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerGradientEditorWidget );
vtkCxxRevisionMacro (vtkSlicerGradientEditorWidget, "$Revision: 1.0 $");

//---------------------------------------------------------------------------

vtkSlicerGradientEditorWidget::vtkSlicerGradientEditorWidget(void)
{
    this->Checkbutton1 = NULL;
    this->Checkbutton2 = NULL;
    this->Checkbutton3 = NULL;
    this->Matrix = NULL;
    this->Rotate = NULL;
    this->Negative = NULL;
    this->Swap = NULL;
    this->Angle = NULL;
    this->LabelAngle = NULL;
    this->TestFrame = NULL;
    this->MeasurementFrame = NULL;
    this->GradientsFrame = NULL;
    this->LoadsaveFrame = NULL;
    this->LoadButton = NULL;
    this->SaveButton = NULL;
    this->Run = NULL;
    this->Mask = NULL;
    this->Gradients = NULL;
    this->GradientsFrame = NULL;
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
    if (this->Run)
    {
        this->Run->SetParent (NULL);
        this->Run->Delete();
        this->Run = NULL;
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
    if (this->Swap)
    {
        this->Swap->SetParent (NULL);
        this->Swap->Delete();
        this->Swap = NULL;
    }
    if (this->Negative)
    {
        this->Negative->SetParent (NULL);
        this->Negative->Delete();
        this->Negative = NULL;
    }
    if (this->Rotate)
    {
        this->Rotate->SetParent (NULL);
        this->Rotate->Delete();
        this->Rotate = NULL;
    }
    if (this->Matrix)
    {
        this->Matrix->SetParent (NULL);
        this->Matrix->Delete();
        this->Matrix = NULL;
    }
    if (this->Checkbutton3)
    {
        this->Checkbutton3->SetParent (NULL);
        this->Checkbutton3->Delete();
        this->Checkbutton3 = NULL;
    }
    if (this->Checkbutton2)
    {
        this->Checkbutton2->SetParent (NULL);
        this->Checkbutton2->Delete();
        this->Checkbutton2 = NULL;
    }
    if (this->Checkbutton1)
    {
        this->Checkbutton1->SetParent (NULL);
        this->Checkbutton1->Delete();
        this->Checkbutton1 = NULL;
    }
}
//---------------------------------------------------------------------------
void vtkSlicerGradientEditorWidget::CreateWidget ( )
{
    // Check if already created
    if (this->IsCreated())
    {
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
    this->EnableMatrix = vtkKWCheckButton::New();
    this->EnableMatrix->SetParent(this->MeasurementFrame->GetFrame());
    this->EnableMatrix->SetText("Enable Matrix");
    this->EnableMatrix->Create();

    // Create matrix for measurement frame
    this->Matrix = vtkKWEntrySet::New();
    this->Matrix->SetParent(this->MeasurementFrame->GetFrame());
    this->Matrix->Create();
    this->Matrix->SetMaximumNumberOfWidgetsInPackingDirection(3);
    this->Matrix->SetPadX(10);
    this->Matrix->SetWidgetsPadX(4);

    for (int i=0; i<9; i++){
        this->Matrix->AddWidget(i);
        this->Matrix->GetWidget(i)->SetWidth(3);
        this->Matrix->GetWidget(i)->SetEnabled(0);
    }

    this->Matrix->GetWidget(0)->SetValue("1");
    this->Matrix->GetWidget(1)->SetValue("0");
    this->Matrix->GetWidget(2)->SetValue("0");

    this->Matrix->GetWidget(3)->SetValue("0");
    this->Matrix->GetWidget(4)->SetValue("1");
    this->Matrix->GetWidget(5)->SetValue("0");

    this->Matrix->GetWidget(6)->SetValue("0");
    this->Matrix->GetWidget(7)->SetValue("0");
    this->Matrix->GetWidget(8)->SetValue("1");

    // Create Checkbutton1
    this->Checkbutton1 = vtkKWCheckButton::New();
    this->Checkbutton1->SetParent(this->MeasurementFrame->GetFrame());
    this->Checkbutton1->Create();

    // Create Checkbutton2
    this->Checkbutton2 = vtkKWCheckButton::New();
    this->Checkbutton2->SetParent(this->MeasurementFrame->GetFrame());
    this->Checkbutton2->Create();

    // Create Checkbutton3
    this->Checkbutton3 = vtkKWCheckButton::New();
    this->Checkbutton3->SetParent(this->MeasurementFrame->GetFrame());
    this->Checkbutton3->Create();

    // buttons for options
    this->Negative = vtkKWPushButton::New();
    this->Negative->SetParent(this->MeasurementFrame->GetFrame());
    this->Negative->Create();
    this->Negative->SetText("Negative Selected");
    this->Negative->SetWidth(15);

    this->Swap = vtkKWPushButton::New();
    this->Swap->SetParent(this->MeasurementFrame->GetFrame());
    this->Swap->Create();
    this->Swap->SetText("Swap Selected");
    this->Swap->SetWidth(15);

    this->Rotate = vtkKWPushButton::New();
    this->Rotate->SetParent(this->MeasurementFrame->GetFrame());
    this->Rotate->Create();
    this->Rotate->SetText("Rotate Selected");
    this->Rotate->SetWidth(15);

    //Create Angle label
    this->LabelAngle = vtkKWLabel::New();
    this->LabelAngle->SetParent(this->MeasurementFrame->GetFrame());
    this->LabelAngle->Create();
    this->LabelAngle->SetText("Angle: ");

    //$labelAngle SetEnabled 0

    //Create Angle combobox
    this->Angle = vtkKWComboBox::New();
    this->Angle->SetParent(this->MeasurementFrame->GetFrame());
    this->Angle->Create();
    this->Angle->SetWidth(4);
    this->Angle->SetValue("+90");

    this->Angle->AddValue("+90");
    this->Angle->AddValue("-90");
    this->Angle->AddValue("+180");
    this->Angle->AddValue("-180");

    
    this->Script("grid %s -row 0 -column 0 -columnspan 3 -sticky n", 
        this->EnableMatrix->GetWidgetName());
    this->Script("grid %s -row 1 -column 0 -columnspan 3 -rowspan 3 -sticky nes", 
        this->Matrix->GetWidgetName());
    this->Script("grid %s -row 4 -column 0 -sticky n", 
        this->Checkbutton1->GetWidgetName());
    this->Script("grid %s -row 4 -column 1 -sticky n", 
        this->Checkbutton2->GetWidgetName());
    this->Script("grid %s -row 4 -column 2 -sticky n", 
        this->Checkbutton3->GetWidgetName());
    this->Script("grid %s -row 3 -column 3 -sticky ne", 
        this->Negative->GetWidgetName());
    this->Script("grid %s -row 2 -column 3 -sticky ne", 
        this->Swap->GetWidgetName());
    this->Script("grid %s -row 1 -column 3 -sticky ne", 
        this->Rotate->GetWidgetName());
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

    //Create ROI
    this->Mask = vtkSlicerNodeSelectorWidget::New();
    this->Mask->SetParent(this->TestFrame->GetFrame());
    this->Mask->Create();
    this->Mask->SetLabelText("Input ROI: ");
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->Mask->GetWidgetName());

    //Create Run button
    this->Run = vtkKWPushButton::New();
    this->Run->SetParent(this->TestFrame->GetFrame());
    this->Run->Create();
    this->Run->SetText("Run");
    this->Run->SetWidth(7);
    this->Script("pack %s -side top -anchor ne -padx 2 -pady 2", 
        this->Run->GetWidgetName());

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
} 
