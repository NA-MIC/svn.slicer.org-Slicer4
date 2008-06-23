#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkProperty.h"

#include "vtkSlicerUnstructuredGridDisplayWidget.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"
#include "vtkKWScale.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"

#include "vtkMRMLUnstructuredGridNode.h"
#include "vtkMRMLUnstructuredGridDisplayNode.h"

// to get at the colour logic to set a default color node
#include "vtkKWApplication.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkSlicerColorGUI.h"
#include "vtkSlicerColorLogic.h"

// for scalars
#include "vtkPointData.h"
#include "vtkCellData.h"

//#include "vtkMRMLColorProceduralFreeSurferNode.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerUnstructuredGridDisplayWidget );
vtkCxxRevisionMacro ( vtkSlicerUnstructuredGridDisplayWidget, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerUnstructuredGridDisplayWidget::vtkSlicerUnstructuredGridDisplayWidget ( )
{

    this->UnstructuredGridDisplayNode = NULL;
    this->UnstructuredGridNode = NULL;

    this->VisibilityButton = NULL;
    this->ScalarVisibilityButton = NULL;
    this->ScalarMenu = NULL;
    this->ColorSelectorWidget = NULL;
    this->ClippingButton = NULL;
    this->OpacityScale = NULL;
    this->SurfaceMaterialPropertyWidget = NULL;
    this->ProcessingMRMLEvent = 0;
    this->ProcessingWidgetEvent = 0;

    this->UpdatingMRML = 0;
    this->UpdatingWidget = 0;

}


//---------------------------------------------------------------------------
vtkSlicerUnstructuredGridDisplayWidget::~vtkSlicerUnstructuredGridDisplayWidget ( )
{
  this->RemoveMRMLObservers();
  this->RemoveWidgetObservers();

  if (this->VisibilityButton)
    {
    this->VisibilityButton->SetParent(NULL);
    this->VisibilityButton->Delete();
    this->VisibilityButton = NULL;
    }
  if (this->ScalarVisibilityButton)
    {
    this->ScalarVisibilityButton->SetParent(NULL);
    this->ScalarVisibilityButton->Delete();
    this->ScalarVisibilityButton = NULL;
    }
  if (this->ScalarMenu)
    {
    this->ScalarMenu->SetParent(NULL);
    this->ScalarMenu->Delete();
    this->ScalarMenu = NULL;
    }
   if (this->ColorSelectorWidget)
    {
    this->ColorSelectorWidget->SetParent(NULL);
    this->ColorSelectorWidget->Delete();
    this->ColorSelectorWidget = NULL;
    }
  if (this->ClippingButton)
    {
    this->ClippingButton->SetParent(NULL);
    this->ClippingButton->Delete();
    this->ClippingButton = NULL;
    }
  if (this->OpacityScale)
    {
    this->OpacityScale->SetParent(NULL);
    this->OpacityScale->Delete();
    this->OpacityScale = NULL;
    }
  if (this->SurfaceMaterialPropertyWidget)
    {
    this->SurfaceMaterialPropertyWidget->SetParent(NULL);
    this->SurfaceMaterialPropertyWidget->Delete();
    this->SurfaceMaterialPropertyWidget = NULL;
    }
  if (this->ChangeColorButton)
    {
    this->ChangeColorButton->SetParent(NULL);
    this->ChangeColorButton->Delete();
    this->ChangeColorButton= NULL;
    }
  vtkSetAndObserveMRMLNodeMacro(this->UnstructuredGridNode, NULL);
  vtkSetAndObserveMRMLNodeMacro(this->UnstructuredGridDisplayNode, NULL);
  this->SetMRMLScene ( NULL );
  
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "vtkSlicerUnstructuredGridDisplayWidget: " << this->GetClassName ( ) << "\n";
    // print widgets?
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::SetUnstructuredGridDisplayNode ( vtkMRMLUnstructuredGridDisplayNode *node )
{ 
  // 
  // Set the member variables and do a first process
  //
  vtkSetAndObserveMRMLNodeMacro(this->UnstructuredGridDisplayNode, node);

  if ( node )
    {
    this->UpdateWidget();
    }
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::SetUnstructuredGridNode ( vtkMRMLUnstructuredGridNode *node )
{ 
  // 
  // Set the member variables and do a first process
  //
  vtkSetAndObserveMRMLNodeMacro(this->UnstructuredGridNode, node);

  if ( node )
    {
    this->UpdateWidget();
    }
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::ProcessWidgetEvents ( vtkObject *caller,
                                                         unsigned long event, void *callData )
{
  if (this->ProcessingMRMLEvent != 0 || this->ProcessingWidgetEvent != 0)
    {
    vtkDebugMacro("ProcessMRMLEvents already processing " << this->ProcessingMRMLEvent);
    return;
    }
  
  this->ProcessingWidgetEvent = event;
 
  if (this->UnstructuredGridDisplayNode != NULL && 
    !(vtkKWSurfaceMaterialPropertyWidget::SafeDownCast(caller) == this->SurfaceMaterialPropertyWidget && event == this->SurfaceMaterialPropertyWidget->GetPropertyChangedEvent()) &&
    !(vtkKWScale::SafeDownCast(caller) == this->OpacityScale->GetWidget() && event == vtkKWScale::ScaleValueChangingEvent) &&
    !(vtkKWScale::SafeDownCast(caller) == this->OpacityScale->GetWidget() && event == vtkKWScale::ScaleValueChangedEvent))
    {
    if (this->MRMLScene->GetNodeByID(this->UnstructuredGridDisplayNode->GetID()))
      {
      this->MRMLScene->SaveStateForUndo(this->UnstructuredGridDisplayNode);
      }
    }
  
  this->UpdateMRML();

  this->ProcessingWidgetEvent = 0;

}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::UpdateMRML()
{
  if (this->UpdatingMRML || this->UpdatingWidget)
    {
    return;
    }

  this->UpdatingMRML = 1;

  if ( this->UnstructuredGridDisplayNode )
    {
    this->UnstructuredGridDisplayNode->SetVisibility(this->VisibilityButton->GetWidget()->GetSelectedState());
    this->UnstructuredGridDisplayNode->SetScalarVisibility(this->ScalarVisibilityButton->GetWidget()->GetSelectedState());
    // get the value of the button, it's the selected item in the menu
    this->UnstructuredGridDisplayNode->SetActiveScalarName(this->ScalarMenu->GetWidget()->GetValue());
    vtkDebugMacro("Set display node active scalar name to " << this->UnstructuredGridDisplayNode->GetActiveScalarName());
    if (this->ColorSelectorWidget->GetSelected() != NULL)
      {
      vtkMRMLColorNode *color =
        vtkMRMLColorNode::SafeDownCast(this->ColorSelectorWidget->GetSelected());
      if (color != NULL &&
          this->UnstructuredGridDisplayNode->GetColorNodeID() == NULL ||
          strcmp(this->UnstructuredGridDisplayNode->GetColorNodeID(), color->GetID()) != 0)
        {
        // there's a change, set it
        vtkDebugMacro("UpdateMRML: setting the display node's color node to " << color->GetID());
        this->UnstructuredGridDisplayNode->SetAndObserveColorNodeID(color->GetID());
        }
      }
    this->UnstructuredGridDisplayNode->SetClipping(this->ClippingButton->GetWidget()->GetSelectedState());
    this->UnstructuredGridDisplayNode->SetOpacity(this->OpacityScale->GetWidget()->GetValue());
    if (this->SurfaceMaterialPropertyWidget->GetProperty() == NULL)
      {
      vtkProperty *prop = vtkProperty::New();
      this->SurfaceMaterialPropertyWidget->SetProperty(prop);
      prop->Delete();
      }

    this->UnstructuredGridDisplayNode->SetAmbient(this->SurfaceMaterialPropertyWidget->GetProperty()->GetAmbient());
    this->UnstructuredGridDisplayNode->SetDiffuse(this->SurfaceMaterialPropertyWidget->GetProperty()->GetDiffuse());
    this->UnstructuredGridDisplayNode->SetSpecular(this->SurfaceMaterialPropertyWidget->GetProperty()->GetSpecular());
    this->UnstructuredGridDisplayNode->SetPower(this->SurfaceMaterialPropertyWidget->GetProperty()->GetSpecularPower());
    double *rgb = this->ChangeColorButton->GetColor();
    double *rgb1 = UnstructuredGridDisplayNode->GetColor();
    if (fabs(rgb[0]-rgb1[0]) > 0.001 ||
        fabs(rgb[1]-rgb1[1]) > 0.001 ||
        fabs(rgb[2]-rgb1[2]) > 0.001)
      {
      this->UnstructuredGridDisplayNode->SetColor(this->ChangeColorButton->GetColor());
      }
    }
  this->UpdatingMRML = 0;

}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::ProcessMRMLEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
  if ( !this->UnstructuredGridDisplayNode )
    {
    return;
    }
  if (this->ProcessingMRMLEvent != 0 || this->ProcessingWidgetEvent != 0)
    {
    vtkDebugMacro("ProcessMRMLEvents already processing " << this->ProcessingMRMLEvent);
    return;
    }
  
  this->ProcessingMRMLEvent = event;
  

  if (this->UnstructuredGridDisplayNode == vtkMRMLUnstructuredGridDisplayNode::SafeDownCast(caller) &&
      event == vtkCommand::ModifiedEvent)
    {
    this->UpdateWidget();
    }
  this->ProcessingMRMLEvent = 0;
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::UpdateWidget()
{
  if (this->UpdatingMRML || this->UpdatingWidget)
    {
    return;
    }
  this->UpdatingWidget = 1;
  
  if ( this->UnstructuredGridDisplayNode == NULL )
    {
    this->UpdatingWidget = 0;
    return;
    }
  
  // get the UnstructuredGrid node so can get at it's scalars
  if (this->UnstructuredGridNode != NULL &&
      this->UnstructuredGridNode->GetPolyData() != NULL)
    {
    this->ScalarVisibilityButton->SetEnabled(1); 
    this->ScalarMenu->SetEnabled(1);
    this->ColorSelectorWidget->SetEnabled(1);

    // populate the scalars menu from the UnstructuredGrid node
    int numPointScalars;
    int numCellScalars;
    if (this->UnstructuredGridNode->GetPolyData()->GetPointData() != NULL)
      {
      numPointScalars = this->UnstructuredGridNode->GetPolyData()->GetPointData()->GetNumberOfArrays();
      }
    else
      {
      numPointScalars = 0;
      }
    if (this->UnstructuredGridNode->GetPolyData()->GetCellData() != NULL)
      {
      numCellScalars = this->UnstructuredGridNode->GetPolyData()->GetCellData()->GetNumberOfArrays();
      }
    else
      {
      numCellScalars = 0;
      }
    vtkDebugMacro("numPointScalars = " << numPointScalars << ", numCellScalars = " << numCellScalars);
    this->ScalarMenu->GetWidget()->GetMenu()->DeleteAllItems();
    // clear the button text
    this->ScalarMenu->GetWidget()->SetValue("");
    for (int p = 0; p < numPointScalars; p++)
      {
      vtkDebugMacro("Adding point scalar " << p << " " << this->UnstructuredGridNode->GetPolyData()->GetPointData()->GetArray(p)->GetName());
      this->ScalarMenu->GetWidget()->GetMenu()->AddRadioButton(this->UnstructuredGridNode->GetPolyData()->GetPointData()->GetArray(p)->GetName());
      }
    for (int c = 0; c < numCellScalars; c++)
      {
      vtkDebugMacro("Adding cell scalar " << c << " " << this->UnstructuredGridNode->GetPolyData()->GetCellData()->GetArray(c)->GetName());
      this->ScalarMenu->GetWidget()->GetMenu()->AddRadioButton(this->UnstructuredGridNode->GetPolyData()->GetCellData()->GetArray(c)->GetName());
      }
    } 
  else 
    { 
    this->ScalarVisibilityButton->SetEnabled(0); 
    this->ScalarMenu->SetEnabled(0);
    this->ColorSelectorWidget->SetEnabled(0);
    vtkDebugMacro("UnstructuredGridNode is null, can't set up the scalars menu\n"); 
    }
  
  this->VisibilityButton->GetWidget()->SetSelectedState(this->UnstructuredGridDisplayNode->GetVisibility());
  this->ScalarVisibilityButton->GetWidget()->SetSelectedState(this->UnstructuredGridDisplayNode->GetScalarVisibility());
  
  // set the active one if it's not already set
  this->ScalarMenu->GetWidget()->GetMenu()->SelectItem(this->UnstructuredGridDisplayNode->GetActiveScalarName());
  // get the color node
  if (this->UnstructuredGridDisplayNode->GetColorNode() != NULL)
    {
    vtkMRMLColorNode *color =
      vtkMRMLColorNode::SafeDownCast(this->ColorSelectorWidget->GetSelected());
    if (color == NULL ||
        strcmp(this->UnstructuredGridDisplayNode->GetColorNodeID(), color->GetID()) != 0)
      {
      this->ColorSelectorWidget->SetSelected(this->UnstructuredGridDisplayNode->GetColorNode());
      }
    }
  else
    {
    // clear the selection
    this->ColorSelectorWidget->SetSelected(NULL);
    }
  this->ClippingButton->GetWidget()->SetSelectedState(this->UnstructuredGridDisplayNode->GetClipping());
  this->OpacityScale->GetWidget()->SetValue(this->UnstructuredGridDisplayNode->GetOpacity());
  if (this->SurfaceMaterialPropertyWidget->GetProperty() == NULL)
    {
    vtkProperty *prop = vtkProperty::New();
    this->SurfaceMaterialPropertyWidget->SetProperty(prop);
    prop->Delete();
    }
  
  this->SurfaceMaterialPropertyWidget->GetProperty()->SetAmbient(this->UnstructuredGridDisplayNode->GetAmbient());
  this->SurfaceMaterialPropertyWidget->GetProperty()->SetDiffuse(this->UnstructuredGridDisplayNode->GetDiffuse());
  this->SurfaceMaterialPropertyWidget->GetProperty()->SetSpecular(this->UnstructuredGridDisplayNode->GetSpecular());
  this->SurfaceMaterialPropertyWidget->GetProperty()->SetSpecularPower(this->UnstructuredGridDisplayNode->GetPower());
  double *rgb = this->ChangeColorButton->GetColor();
  double *rgb1 = UnstructuredGridDisplayNode->GetColor();
  if (fabs(rgb[0]-rgb1[0]) > 0.001 ||
      fabs(rgb[1]-rgb1[1]) > 0.001 ||
      fabs(rgb[2]-rgb1[2]) > 0.001)
    {
    this->ChangeColorButton->SetColor(this->UnstructuredGridDisplayNode->GetColor());
    }

  this->SurfaceMaterialPropertyWidget->Update();
  this->UpdatingWidget = 0;

}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::AddMRMLObservers ( )
{
}

//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::RemoveMRMLObservers ( )
{
  if (this->UnstructuredGridDisplayNode)
    {
    this->UnstructuredGridDisplayNode->RemoveObservers(vtkCommand::ModifiedEvent,
                                            (vtkCommand *)this->MRMLCallbackCommand );
    }
  
  if (this->UnstructuredGridNode)
    {
    this->UnstructuredGridNode->RemoveObservers(vtkCommand::ModifiedEvent,
                                     (vtkCommand *)this->MRMLCallbackCommand );
    }
  
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::RemoveWidgetObservers ( ) {
  this->VisibilityButton->GetWidget()->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ScalarVisibilityButton->GetWidget()->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ScalarMenu->GetWidget()->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ClippingButton->GetWidget()->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->ChangeColorButton->RemoveObservers(vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->SurfaceMaterialPropertyWidget->RemoveObservers(this->SurfaceMaterialPropertyWidget->GetPropertyChangedEvent(), (vtkCommand *)this->GUICallbackCommand );
  this->SurfaceMaterialPropertyWidget->RemoveObservers(this->SurfaceMaterialPropertyWidget->GetPropertyChangingEvent(), (vtkCommand *)this->GUICallbackCommand );
  
  this->ColorSelectorWidget->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
}


//---------------------------------------------------------------------------
void vtkSlicerUnstructuredGridDisplayWidget::CreateWidget ( )
{
  // Check if already created

  if (this->IsCreated())
    {
    vtkErrorMacro(<< this->GetClassName() << " already created");
    return;
    }
  
  // Call the superclass to create the whole widget
  
  this->Superclass::CreateWidget();
  
  // ---
  // DISPLAY FRAME            
  vtkKWFrame *UnstructuredGridDisplayFrame = vtkKWFrame::New ( );
  UnstructuredGridDisplayFrame->SetParent ( this->GetParent() );
  UnstructuredGridDisplayFrame->Create ( );
/*
  UnstructuredGridDisplayFrame->SetLabelText ("Display");
  UnstructuredGridDisplayFrame->CollapseFrame ( );
*/
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 UnstructuredGridDisplayFrame->GetWidgetName() );

  // Don't select child classes (like FiberBundles)
  //this->UnstructuredGridSelectorWidget->ChildClassesEnabledOff();

  this->VisibilityButton = vtkKWCheckButtonWithLabel::New();
  this->VisibilityButton->SetParent ( UnstructuredGridDisplayFrame );
  this->VisibilityButton->Create ( );
  this->VisibilityButton->SetLabelText("Visibility");
  this->VisibilityButton->SetBalloonHelpString("set UnstructuredGrid visibility.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->VisibilityButton->GetWidgetName() );

  // a frame to hold the scalar related widgets
  vtkKWFrame *scalarFrame = vtkKWFrame::New();
  scalarFrame->SetParent( UnstructuredGridDisplayFrame );
  scalarFrame->Create();
  this->Script("pack %s -side top -anchor nw -fill x -pady 0 -in %s",
                 scalarFrame->GetWidgetName(),
                 UnstructuredGridDisplayFrame->GetWidgetName());

  // scalar visibility
  this->ScalarVisibilityButton = vtkKWCheckButtonWithLabel::New();
  this->ScalarVisibilityButton->SetParent ( scalarFrame );
  this->ScalarVisibilityButton->Create ( );
  this->ScalarVisibilityButton->SetLabelText("Scalar Visibility");
  this->ScalarVisibilityButton->SetBalloonHelpString("set UnstructuredGrid scalar visibility.");
  //this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
  //               this->ScalarVisibilityButton->GetWidgetName() );

  // a menu of the scalar fields available to be set
  this->ScalarMenu = vtkKWMenuButtonWithLabel::New();
  this->ScalarMenu->SetParent ( scalarFrame );
  this->ScalarMenu->Create();
  this->ScalarMenu->SetLabelText("Set Active Scalar:");
  this->ScalarMenu->SetBalloonHelpString("set which scalar field is displayed on the UnstructuredGrid");
  this->ScalarMenu->GetWidget()->SetWidth(20);
  // pack the scalars frame
  this->Script("pack %s %s -side left -anchor w -padx 2 -pady 2 -in %s", 
                this->ScalarVisibilityButton->GetWidgetName(), this->ScalarMenu->GetWidgetName(),
                scalarFrame->GetWidgetName());
  
  // a selector to change the color node associated with this display
  this->ColorSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
  this->ColorSelectorWidget->SetParent ( UnstructuredGridDisplayFrame );
  this->ColorSelectorWidget->Create ( );
  this->ColorSelectorWidget->SetNodeClass("vtkMRMLColorNode", NULL, NULL, NULL);
  this->ColorSelectorWidget->ShowHiddenOn();
  this->ColorSelectorWidget->SetMRMLScene(this->GetMRMLScene());
  this->ColorSelectorWidget->SetBorderWidth(2);
  // this->ColorSelectorWidget->SetReliefToGroove();
  this->ColorSelectorWidget->SetPadX(2);
  this->ColorSelectorWidget->SetPadY(2);
  this->ColorSelectorWidget->GetWidget()->GetWidget()->IndicatorVisibilityOff();
  this->ColorSelectorWidget->GetWidget()->GetWidget()->SetWidth(24);
  this->ColorSelectorWidget->SetLabelText( "Scalar Color Map Select: ");
  this->ColorSelectorWidget->SetBalloonHelpString("select a color node from the current mrml scene.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->ColorSelectorWidget->GetWidgetName());
  // disable this until FreeSurfer nodes are supported
//  this->ColorSelectorWidget->EnabledOff();
  
  this->ClippingButton = vtkKWCheckButtonWithLabel::New();
  this->ClippingButton->SetParent ( UnstructuredGridDisplayFrame );
  this->ClippingButton->Create ( );
  this->ClippingButton->SetLabelText("Clipping");
  this->ClippingButton->SetBalloonHelpString("set UnstructuredGrid clipping with RGB slice planes.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->ClippingButton->GetWidgetName() );
  
  this->OpacityScale = vtkKWScaleWithLabel::New();
  this->OpacityScale->SetParent ( UnstructuredGridDisplayFrame );
  this->OpacityScale->Create ( );
  this->OpacityScale->SetLabelText("Opacity");
  this->OpacityScale->GetWidget()->SetRange(0,1);
  this->OpacityScale->GetWidget()->SetResolution(0.1);
  this->OpacityScale->SetBalloonHelpString("set UnstructuredGrid opacity value.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->OpacityScale->GetWidgetName() );

  this->ChangeColorButton = vtkKWChangeColorButton::New();
  this->ChangeColorButton->SetParent ( UnstructuredGridDisplayFrame );
  this->ChangeColorButton->Create ( );
  this->ChangeColorButton->SetColor(0.0, 1.0, 0.0);
  this->ChangeColorButton->LabelOutsideButtonOn();
  this->ChangeColorButton->SetLabelPositionToRight();
  this->ChangeColorButton->SetBalloonHelpString("set UnstructuredGrid opacity value.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->ChangeColorButton->GetWidgetName() );

  this->SurfaceMaterialPropertyWidget = vtkKWSurfaceMaterialPropertyWidget::New();
  this->SurfaceMaterialPropertyWidget->SetParent ( UnstructuredGridDisplayFrame );
  this->SurfaceMaterialPropertyWidget->Create ( );
  this->SurfaceMaterialPropertyWidget->SetBalloonHelpString("set UnstructuredGrid opacity value.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->SurfaceMaterialPropertyWidget->GetWidgetName() );

  // add observers
  
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  
  this->VisibilityButton->GetWidget()->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ScalarVisibilityButton->GetWidget()->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  this->ScalarMenu->GetWidget()->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand);
  this->ClippingButton->GetWidget()->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  
  this->ChangeColorButton->AddObserver(vtkKWChangeColorButton::ColorChangedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->SurfaceMaterialPropertyWidget->AddObserver(this->SurfaceMaterialPropertyWidget->GetPropertyChangedEvent(), (vtkCommand *)this->GUICallbackCommand );
  this->SurfaceMaterialPropertyWidget->AddObserver(this->SurfaceMaterialPropertyWidget->GetPropertyChangingEvent(), (vtkCommand *)this->GUICallbackCommand );

  this->ColorSelectorWidget->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
   
  UnstructuredGridDisplayFrame->Delete();
    
}
