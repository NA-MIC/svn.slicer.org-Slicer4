#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkProperty.h"

#include "vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget.h"

#include "vtkKWFrameWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWMenu.h"
#include "vtkKWScale.h"
#include "vtkKWMenuButton.h"
#include "vtkKWCheckButton.h"

#include "vtkMRMLFiberBundleNode.h"
#include "vtkMRMLFiberBundleDisplayNode.h"

//---------------------------------------------------------------------------
vtkStandardNewMacro (vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget );
vtkCxxRevisionMacro ( vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget, "$Revision: 1.0 $");


//---------------------------------------------------------------------------
vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget ( )
{

    this->DiffusionTensorVolumeNode = NULL;

    this->ColorSelectorWidget = NULL;
    this->OpacityScale = NULL;
    this->VisibilityButton[0] = NULL;
    this->VisibilityButton[1] = NULL;
    this->VisibilityButton[2] = NULL;
    this->GeometryColorMenu = NULL;

    this->GlyphDisplayWidget = NULL;
    this->DisplayFrame = NULL;


    this->UpdatingMRML = 0;
    this->UpdatingWidget = 0;
}


//---------------------------------------------------------------------------
vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::~vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget ( )
{
  this->RemoveMRMLObservers();
  this->RemoveWidgetObservers();

   if (this->ColorSelectorWidget)
    {
    this->ColorSelectorWidget->SetParent(NULL);
    this->ColorSelectorWidget->Delete();
    this->ColorSelectorWidget = NULL;
    }
  if (this->OpacityScale)
    {
    this->OpacityScale->SetParent(NULL);
    this->OpacityScale->Delete();
    this->OpacityScale = NULL;
    }
  for (int i=0; i<3; i++)
    {
    if (this->VisibilityButton[i])
      {
      this->VisibilityButton[i]->SetParent(NULL);
      this->VisibilityButton[i]->Delete();
      this->VisibilityButton[i] = NULL;
      }
    }

  if (this->GlyphDisplayWidget)
    {
    this->GlyphDisplayWidget->SetParent(NULL);
    this->GlyphDisplayWidget->Delete();
    this->GlyphDisplayWidget = NULL;
    }
    
  if (this->DisplayFrame)
    {
    this->DisplayFrame->SetParent(NULL);
    this->DisplayFrame->Delete();
    this->DisplayFrame = NULL;
    }
  if (this->GeometryColorMenu)
    {
    this->GeometryColorMenu->SetParent(NULL);
    this->GeometryColorMenu->Delete();
    this->GeometryColorMenu = NULL;
    }
  vtkSetAndObserveMRMLNodeMacro(this->DiffusionTensorVolumeNode, NULL);
  this->SetMRMLScene ( NULL );
  
}


//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );

    os << indent << "vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget: " << this->GetClassName ( ) << "\n";
    // print widgets?
}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::SetDiffusionTensorVolumeNode ( vtkMRMLDiffusionTensorVolumeNode *node )
{ 
  if (node == this->DiffusionTensorVolumeNode)
    {
    return;
    }
  vtkSetAndObserveMRMLNodeMacro(this->DiffusionTensorVolumeNode, node);

  if (this->DiffusionTensorVolumeNode )
    {
    this->GlypDisplayNodes = this->DiffusionTensorVolumeNode->GetSliceGlyphDisplayNodes();
    this->UpdateWidget();
    }
}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::ProcessWidgetEvents ( vtkObject *caller,
                                                         unsigned long event, void *callData )
{
  
  if (this->MRMLScene && this->DiffusionTensorVolumeNode != NULL  &&
    !(vtkKWScale::SafeDownCast(caller) == this->OpacityScale->GetWidget() && event == vtkKWScale::ScaleValueChangingEvent) &&
    !(vtkKWScale::SafeDownCast(caller) == this->OpacityScale->GetWidget() && event == vtkKWScale::ScaleValueChangedEvent))
    {
      for (unsigned int i=0; i<this->GlypDisplayNodes.size(); i++)
        {
        this->MRMLScene->SaveStateForUndo(this->GlypDisplayNodes[i]);
        }
    }


  //
  // process color selector events
  //
  vtkSlicerNodeSelectorWidget *colSelector = 
    vtkSlicerNodeSelectorWidget::SafeDownCast(caller);
  if (colSelector == this->ColorSelectorWidget && 
      event == vtkSlicerNodeSelectorWidget::NodeSelectedEvent) 
    {
    vtkMRMLColorNode *color =
      vtkMRMLColorNode::SafeDownCast(this->ColorSelectorWidget->GetSelected());
    if (color != NULL)
      {
      for (unsigned int i=0; i<this->GlypDisplayNodes.size(); i++)
        {
        vtkMRMLDiffusionTensorVolumeSliceDisplayNode* dnode = this->GlypDisplayNodes[i];
        if (dnode != NULL)
          {
          // set and observe it's colour node id
          if (dnode->GetColorNodeID() == NULL ||
              strcmp(dnode->GetColorNodeID(), color->GetID()) != 0)
            {
            dnode->SetAndObserveColorNodeID(color->GetID());
            }
          }           
        }
      }
    return;
    }
  this->UpdateMRML();

}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::UpdateMRML()
{
  if (this->DiffusionTensorVolumeNode == NULL)
    {
    return;
    }

  if (this->UpdatingMRML || this->UpdatingWidget)
    {
    return;
    }

  this->UpdatingMRML = 1;


  vtkMRMLDiffusionTensorDisplayPropertiesNode* propNode = this->GetCurrentDTDisplayPropertyNode();

  if (propNode)
    {
    propNode->SetColorGlyphBy(this->GeometryColorMap[std::string(this->GeometryColorMenu->GetWidget()->GetValue())]);
    }

  for (unsigned int i=0; i<this->GlypDisplayNodes.size(); i++)
    {
    vtkMRMLDiffusionTensorVolumeSliceDisplayNode* dnode = this->GlypDisplayNodes[i];
    dnode->SetOpacity(this->OpacityScale->GetWidget()->GetValue());
    dnode->SetVisibility(this->VisibilityButton[i]->GetWidget()->GetSelectedState());
    }
      
  this->UpdatingMRML = 0;
    
}




//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::ProcessMRMLEvents ( vtkObject *caller,
                                              unsigned long event, void *callData )
{
  vtkDebugWithObjectMacro(this,"Process MRML Events");

  if ( !this->DiffusionTensorVolumeNode )
    {
    return;
    }

  vtkMRMLDiffusionTensorVolumeNode *node = vtkMRMLDiffusionTensorVolumeNode::SafeDownCast(caller);
  
  // if this event comes from our fiberBundleNode, it is not null, and has been modified
  if (node != NULL && event == vtkCommand::ModifiedEvent)
    {
    this->SetDiffusionTensorVolumeNode(node);
    }
  
  this->UpdateWidget();
  
}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::AddMRMLObservers ( )
{
  if (this->DiffusionTensorVolumeNode != NULL)
    {
    this->DiffusionTensorVolumeNode->AddObserver(vtkCommand::ModifiedEvent,
                            (vtkCommand *)this->MRMLCallbackCommand );      
    }
}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::RemoveMRMLObservers ( )
{
  if (this->DiffusionTensorVolumeNode != NULL)
    {
    this->DiffusionTensorVolumeNode->RemoveObservers(vtkCommand::ModifiedEvent,
                            (vtkCommand *)this->MRMLCallbackCommand );      
    }
}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::UpdateWidget()
{
  if (this->UpdatingMRML || this->UpdatingWidget)
    {
    return;
    }
  this->UpdatingWidget = 1;
  

  vtkMRMLDiffusionTensorDisplayPropertiesNode* propNode = this->GetCurrentDTDisplayPropertyNode();

  if (propNode)
    {
    // Set color by value
    this->GeometryColorMenu->GetWidget()->SetValue(propNode->GetColorGlyphByAsString());
    }

  for (unsigned int i=0; i<this->GlypDisplayNodes.size(); i++)
   {
    vtkMRMLDiffusionTensorVolumeSliceDisplayNode* dnode = this->GlypDisplayNodes[i];

    this->VisibilityButton[i]->GetWidget()->SetSelectedState(dnode->GetVisibility());
    if (i == 0)
      {
      this->OpacityScale->GetWidget()->SetValue(dnode->GetOpacity());
      if (dnode->GetColorNode() != NULL)
        {
        vtkMRMLColorNode *color =
          vtkMRMLColorNode::SafeDownCast(this->ColorSelectorWidget->GetSelected());
        if (color == NULL ||
            strcmp(dnode->GetColorNodeID(), color->GetID()) != 0)
          {
          this->ColorSelectorWidget->SetSelected(dnode->GetColorNode());
          }
        }
      }
   }


  vtkMRMLDiffusionTensorDisplayPropertiesNode *dpnode = this->GetCurrentDTDisplayPropertyNode();
  this->GlyphDisplayWidget->SetDiffusionTensorDisplayPropertiesNode(dpnode);
  
  // TODO glyph widget
   this->UpdatingWidget = 0;

}

//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::RemoveWidgetObservers ( ) {
  
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->RemoveObservers(vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );


  this->ColorSelectorWidget->RemoveObservers (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );

  for (int i=0; i<3; i++)
    {
    this->VisibilityButton[i]->GetWidget()->RemoveObservers(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
    
  this->GeometryColorMenu->GetWidget()->GetMenu()->RemoveObservers (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  // TO DO glyph widget
}


//---------------------------------------------------------------------------
void vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::CreateWidget ( )
{
  // Check if already created

  if (this->IsCreated())
    {
    vtkErrorMacro(<< this->GetClassName() << " already created");
    return;
    }
  
  // Call the superclass to create the whole widget
  
  this->Superclass::CreateWidget();
  int i;
  // ---
  // DISPLAY FRAME            
  this->DisplayFrame = vtkKWFrameWithLabel::New ( );
  this->DisplayFrame->SetParent ( this->GetParent() );
  this->DisplayFrame->SetLabelText ("Glyphs on Slices Display");
  this->DisplayFrame->Create ( );
/*
  displayFrame->SetLabelText ("Display");
  displayFrame->CollapseFrame ( );
*/
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->DisplayFrame->GetWidgetName() );  
  for (i=0; i<3; i++)
    {
    this->VisibilityButton[i] = vtkKWCheckButtonWithLabel::New();
    this->VisibilityButton[i]->SetParent ( this->DisplayFrame->GetFrame());
    this->VisibilityButton[i]->Create ( );
    std::stringstream ss;
    ss << "Visibility " << i; 
    this->VisibilityButton[i]->SetLabelText(ss.str().c_str());
    this->VisibilityButton[i]->SetBalloonHelpString("set fiberBundle visibility.");
    this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                   this->VisibilityButton[i]->GetWidgetName() );
    }
  // color by menu

  vtkKWMenuButtonWithLabel *colorMenuButton = 
    vtkKWMenuButtonWithLabel::New();

  this->GeometryColorMenu = colorMenuButton;
  colorMenuButton->SetParent( this->DisplayFrame->GetFrame() );
  colorMenuButton->Create();
  // initialize color menu
  //Create dummy display properties node to init variables
  vtkMRMLDiffusionTensorDisplayPropertiesNode *propNode = 
    vtkMRMLDiffusionTensorDisplayPropertiesNode::New();
  int initIdx = propNode->GetFirstColorGlyphBy();
  int endIdx = propNode->GetLastColorGlyphBy();
  int currentVal = propNode->GetColorGlyphBy();
  this->GeometryColorMap.clear();
  for (int k=initIdx ; k<=endIdx ; k++)
    {
    propNode->SetColorGlyphBy(k);
    const char *tag = propNode->GetColorGlyphByAsString();
    this->GeometryColorMap[std::string(tag)]=k;
    colorMenuButton->GetWidget()->GetMenu()->AddRadioButton(tag);
    }
  // init to class default value
  propNode->SetColorGlyphBy(currentVal);
  colorMenuButton->GetWidget()->SetValue(propNode->GetColorGlyphByAsString());

  propNode->Delete();

  // pack color menu
  colorMenuButton->SetLabelText("Color By Scalar");
  this->Script("pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
               colorMenuButton->GetWidgetName());

  // a selector to change the color node associated with this display
  this->ColorSelectorWidget = vtkSlicerNodeSelectorWidget::New() ;
  this->ColorSelectorWidget->SetParent ( this->DisplayFrame->GetFrame() );
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
  this->ColorSelectorWidget->SetLabelText( "Scalar Color Map");
  this->ColorSelectorWidget->SetBalloonHelpString("select a color node from the current mrml scene.");
  this->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2",
                 this->ColorSelectorWidget->GetWidgetName());
  
  
  this->OpacityScale = vtkKWScaleWithLabel::New();
  this->OpacityScale->SetParent ( this->DisplayFrame->GetFrame() );
  this->OpacityScale->Create ( );
  this->OpacityScale->SetLabelText("Opacity");
  this->OpacityScale->GetWidget()->SetRange(0,1);
  this->OpacityScale->GetWidget()->SetResolution(0.1);
  this->OpacityScale->SetBalloonHelpString("set fiberBundle opacity value.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->OpacityScale->GetWidgetName() );

  this->GlyphDisplayWidget = vtkSlicerDiffusionTensorGlyphDisplayWidget::New();
  this->GlyphDisplayWidget->SetParent ( this->DisplayFrame );
  this->GlyphDisplayWidget->SetMRMLScene(this->GetMRMLScene());
  this->GlyphDisplayWidget->Create ( );
  this->GlyphDisplayWidget->SetBalloonHelpString("set glyph display parameters.");
  this->Script ( "pack %s -side top -anchor nw -expand y -fill x -padx 2 -pady 2",
                 this->GlyphDisplayWidget->GetWidgetName() );

  // add observers
  
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueStartChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangingEvent, (vtkCommand *)this->GUICallbackCommand );
  this->OpacityScale->GetWidget()->AddObserver(vtkKWScale::ScaleValueChangedEvent, (vtkCommand *)this->GUICallbackCommand );
  

  this->ColorSelectorWidget->AddObserver (vtkSlicerNodeSelectorWidget::NodeSelectedEvent, (vtkCommand *)this->GUICallbackCommand );
   
  for (i=0; i<3; i++) 
    {
    this->VisibilityButton[i]->GetWidget()->AddObserver(vtkKWCheckButton::SelectedStateChangedEvent, (vtkCommand *)this->GUICallbackCommand );
    }

  this->GeometryColorMenu->GetWidget()->GetMenu()->AddObserver (vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

    
}

//---------------------------------------------------------------------------
vtkMRMLDiffusionTensorDisplayPropertiesNode* vtkSlicerDiffusionTensorVolumeGlyphDisplayWidget::GetCurrentDTDisplayPropertyNode()
{
  vtkMRMLDiffusionTensorDisplayPropertiesNode *dpnode = NULL;
  if (this->GlypDisplayNodes.size() > 0)
    {
    vtkMRMLDiffusionTensorVolumeSliceDisplayNode *dnode = this->GlypDisplayNodes[0];
    dpnode = vtkMRMLDiffusionTensorDisplayPropertiesNode::SafeDownCast( dnode->GetDTDisplayPropertiesNode() );
    }
  return dpnode;
}
