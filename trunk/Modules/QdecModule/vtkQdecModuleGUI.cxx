/*=auto=========================================================================

Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

See Doc/copyright/copyright.txt
or http://www.slicer.org/copyright/copyright.txt for details.

Program:   3D Slicer
Module:    $RCSfile: vtkQdecModuleGUI.cxx,v $
Date:      $Date: 2006/03/17 15:10:10 $
Version:   $Revision: 1.2 $

=========================================================================auto=*/

#include <string>
#include <iostream>
#include <sstream>

#include "vtkObjectFactory.h"

#include "QdecSubject.h"

#include "vtkQdecModuleGUI.h"

#include "vtkCommand.h"
#include "vtkKWApplication.h"
#include "vtkKWWidget.h"
#include "vtkSlicerApplication.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"
#include "vtkKWScaleWithEntry.h"
#include "vtkKWEntryWithLabel.h"
#include "vtkKWMenuButtonWithLabel.h"
#include "vtkKWMenuButton.h"
#include "vtkKWScale.h"
#include "vtkKWMenu.h"
#include "vtkKWEntry.h"
#include "vtkKWFrame.h"
#include "vtkSlicerApplication.h"
#include "vtkKWFrameWithLabel.h"
#include "vtkKWPushButton.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWMultiColumnList.h"
#include "vtkKWMultiColumnListWithScrollbars.h"
#include "vtkKWLabel.h"
#include "vtkKWListBox.h"
#include "vtkKWListBoxWithScrollbars.h"
#include "vtkKWListBoxWithScrollbarsWithLabel.h"

#include "vtkSlicerModelsGUI.h"

// for pick events
#include "vtkSlicerViewerWidget.h"
#include "vtkSlicerViewerInteractorStyle.h"

// for path manipulation
#include "itksys/SystemTools.hxx"

// for scalar overlays
#include "vtkMRMLProceduralColorNode.h"

//------------------------------------------------------------------------------
vtkQdecModuleGUI* vtkQdecModuleGUI::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkQdecModuleGUI");
  if(ret)
    {
      return (vtkQdecModuleGUI*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkQdecModuleGUI;
}

//----------------------------------------------------------------------------
vtkQdecModuleGUI::vtkQdecModuleGUI()
{
  this->NAMICLabel = NULL;
  this->SubjectsDirectoryButton = NULL;
  this->LoadTableButton = NULL;
  this->LoadResultsButton = NULL;
  this->DesignEntry = NULL;
  this->DiscreteFactorsListBox = NULL;
  this->ContinuousFactorsListBox = NULL;
  this->ApplyButton = NULL;
  this->MultiColumnList = NULL;
  this->MeasureLabel = NULL;
  this->MeasureMenu = NULL;
  this->HemisphereLabel = NULL;
  this->HemisphereMenu = NULL;
  this->SmoothnessLabel = NULL;
  this->SmoothnessMenu = NULL;
  this->QuestionLabel = NULL;
  this->QuestionMenu = NULL;
  this->Logic = NULL;

  // for picking
  this->ViewerWidget = NULL;
  this->InteractorStyle = NULL;
}

//----------------------------------------------------------------------------
vtkQdecModuleGUI::~vtkQdecModuleGUI()
{
  if (this->GetLogic()->GetTclScriptLoaded())
    {
    vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
    if (app)
      {
      app->Script("vtkFreeSurferReadersExit");
      }
    }
  this->RemoveMRMLNodeObservers ( );
  this->RemoveLogicObservers ( );
  this->SetModuleLogic (NULL);
 
  if ( this->NAMICLabel )
    {
    this->NAMICLabel->SetParent ( NULL );
    this->NAMICLabel->Delete();
    this->NAMICLabel = NULL;
    }

  if ( this->SubjectsDirectoryButton )
    {
    this->SubjectsDirectoryButton->SetParent(NULL);
    this->SubjectsDirectoryButton->Delete();
    this->SubjectsDirectoryButton = NULL;
    }
   
  if ( this->LoadTableButton )
    {
    this->LoadTableButton->SetParent(NULL);
    this->LoadTableButton->Delete();
    this->LoadTableButton = NULL;
    }

  if ( this->LoadResultsButton )
    {
    this->LoadResultsButton->SetParent(NULL);
    this->LoadResultsButton->Delete();
    this->LoadResultsButton = NULL;
    }
  
  if ( this->DesignEntry )
    {
    this->DesignEntry->SetParent(NULL);
    this->DesignEntry->Delete();
    this->DesignEntry = NULL;
    }
  
  if ( this->DiscreteFactorsListBox )
    {
    this->DiscreteFactorsListBox->SetParent ( NULL );
    this->DiscreteFactorsListBox->Delete();
    this->DiscreteFactorsListBox = NULL;
    }
 
   if ( this->ContinuousFactorsListBox )
    {
    this->ContinuousFactorsListBox->SetParent ( NULL );
    this->ContinuousFactorsListBox->Delete();
    this->ContinuousFactorsListBox = NULL;
    }

   if ( this->ApplyButton )
    {
    this->ApplyButton->SetParent(NULL);
    this->ApplyButton->Delete();
    this->ApplyButton = NULL;
    }
 
   if (this->MultiColumnList)
     {
     this->MultiColumnList->SetParent(NULL);
     this->MultiColumnList->Delete();
     this->MultiColumnList = NULL;
     }

   if ( this->MeasureLabel )
    {
    this->MeasureLabel->SetParent ( NULL );
    this->MeasureLabel->Delete();
    this->MeasureLabel = NULL;
    }

   if ( this->MeasureMenu )
    {
    this->MeasureMenu->SetParent ( NULL );
    this->MeasureMenu->Delete();
    this->MeasureMenu = NULL;
    }

    if ( this->HemisphereLabel )
    {
    this->HemisphereLabel->SetParent ( NULL );
    this->HemisphereLabel->Delete();
    this->HemisphereLabel = NULL;
    }

   if ( this->HemisphereMenu )
    {
    this->HemisphereMenu->SetParent ( NULL );
    this->HemisphereMenu->Delete();
    this->HemisphereMenu = NULL;
    }

   if ( this->SmoothnessLabel )
    {
    this->SmoothnessLabel->SetParent ( NULL );
    this->SmoothnessLabel->Delete();
    this->SmoothnessLabel = NULL;
    }

   if ( this->SmoothnessMenu )
    {
    this->SmoothnessMenu->SetParent ( NULL );
    this->SmoothnessMenu->Delete();
    this->SmoothnessMenu = NULL;
    }

   if ( this->QuestionLabel )
    {
    this->QuestionLabel->SetParent ( NULL );
    this->QuestionLabel->Delete();
    this->QuestionLabel = NULL;
    }

   if ( this->QuestionMenu )
    {
    this->QuestionMenu->SetParent ( NULL );
    this->QuestionMenu->Delete();
    this->QuestionMenu = NULL;
    }
   
   this->SetViewerWidget(NULL);   
   this->SetInteractorStyle(NULL);
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::CreateModuleEventBindings ( )
{
  vtkDebugMacro("CreateModuleEventBindings");
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::ReleaseModuleEventBindings ( )
{
  vtkDebugMacro("ReleaseModuleEventBindings");
}


//----------------------------------------------------------------------------
void vtkQdecModuleGUI::Enter()
{
  vtkDebugMacro("Enter...");
  if ( this->Built == false )
    {
    this->BuildGUI();
    this->AddGUIObservers();
    }
  this->CreateModuleEventBindings();
  this->UpdateGUI();
}

//----------------------------------------------------------------------------
void vtkQdecModuleGUI::Exit()
{
  vtkDebugMacro("Exit: removing observers on picking");
  this->ReleaseModuleEventBindings();
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::TearDownGUI ( )
{
  this->Exit();
  if ( this->Built )
    {
    this->RemoveGUIObservers();
    }
}

//----------------------------------------------------------------------------
void vtkQdecModuleGUI::PrintSelf(ostream& os, vtkIndent indent)
{
  os << indent << "vtkQdecModuleGUI" << endl;
  os << indent << "vtkQdecModuleLogic: " << this->GetLogic() << endl;
  if (this->GetLogic() != NULL)
    {
    this->GetLogic()->PrintSelf(os, indent.GetNextIndent());
    }
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::AddGUIObservers ( ) 
{
  this->SubjectsDirectoryButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadTableButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->LoadResultsButton->GetWidget()->AddObserver ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
  this->ApplyButton->AddObserver (vtkKWPushButton::InvokedEvent, (vtkCommand *)this->GUICallbackCommand );

  this->QuestionMenu->GetMenu()->AddObserver(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );

}



//---------------------------------------------------------------------------
void vtkQdecModuleGUI::RemoveGUIObservers ( )
{
  if (this->SubjectsDirectoryButton)
    {
    this->SubjectsDirectoryButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }
  
  if (this->LoadTableButton)
    {
    this->LoadTableButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  if (this->LoadResultsButton)
    {
    this->LoadResultsButton->GetWidget()->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );
    }

  this->ApplyButton->RemoveObservers ( vtkKWPushButton::InvokedEvent,  (vtkCommand *)this->GUICallbackCommand );

  if (this->QuestionMenu)
    {
    this->QuestionMenu->GetMenu()->RemoveObservers(vtkKWMenu::MenuItemInvokedEvent, (vtkCommand *)this->GUICallbackCommand );
    }
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::RemoveMRMLNodeObservers ( ) {
    // Fill in.
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::RemoveLogicObservers ( ) {
    // Fill in
}


//---------------------------------------------------------------------------
void vtkQdecModuleGUI::ProcessGUIEvents ( vtkObject *caller,
                                           unsigned long event,
                                           void *callData ) 
{

  
  
  vtkKWPushButton *b = vtkKWPushButton::SafeDownCast ( caller );

  if (b == this->ApplyButton && event == vtkKWPushButton::InvokedEvent ) 
    {
    //this->DebugOn();
    if (this->GetDebug())
      {
      this->GetLogic()->DebugOn();
      }

    vtkDebugMacro("Apply button pushed");
    
    std::string cont1 = "none";
    std::string cont2 = "none";
    std::string dis1 = "none";
    std::string dis2 = "none";
    int factorNum = 0;
    for (int i = 0; (i < this->ContinuousFactorsListBox->GetWidget()->GetWidget()->GetNumberOfItems()) && 
                    (factorNum < 2); i++)
      {
      if ( this->ContinuousFactorsListBox->GetWidget()->GetWidget()->GetSelectState(i))
        {
        vtkDebugMacro("\t" <<  this->ContinuousFactorsListBox->GetWidget()->GetWidget()->GetItem(i));
        if (factorNum == 0)
          {
          cont1 = this->ContinuousFactorsListBox->GetWidget()->GetWidget()->GetItem(i);
          }
        else if (factorNum == 1)
          {
          cont2 = this->ContinuousFactorsListBox->GetWidget()->GetWidget()->GetItem(i);
          }
        factorNum++;
        }
      }
    
    factorNum = 0;
    for (int i = 0; (i < this->DiscreteFactorsListBox->GetWidget()->GetWidget()->GetNumberOfItems()) &&
           (factorNum < 2); i++)
      {
      if ( this->DiscreteFactorsListBox->GetWidget()->GetWidget()->GetSelectState(i))
        {
        vtkDebugMacro("\t" <<  this->DiscreteFactorsListBox->GetWidget()->GetWidget()->GetItem(i));
        if (factorNum == 0)
          {    
          dis1 = this->DiscreteFactorsListBox->GetWidget()->GetWidget()->GetItem(i);
          }
        else if (factorNum == 1)
          {
          dis2 = this->DiscreteFactorsListBox->GetWidget()->GetWidget()->GetItem(i);
          }
        factorNum++;
        }
      }
    vtkDebugMacro("Selected discrete factors = " << dis1.c_str() << " and " << dis2.c_str());
    vtkDebugMacro("Selected continuous factors = " << cont1.c_str() << " and " << cont2.c_str());
    vtkDebugMacro("Design name = " << this->DesignEntry->GetWidget()->GetValue());
    vtkDebugMacro("Measure = " << this->MeasureMenu->GetValue());
    vtkDebugMacro("Hemisphere = " << this->HemisphereMenu->GetValue());
    vtkDebugMacro("Smoothness = " << this->SmoothnessMenu->GetValue());
    
    // now pass it into the QDEC project to create a design
    int err = this->GetLogic()->CreateGlmDesign(this->DesignEntry->GetWidget()->GetValue(),
                                                dis1.c_str(), dis2.c_str(), cont1.c_str(), cont2.c_str(),
                                                this->MeasureMenu->GetValue(),
                                                this->HemisphereMenu->GetValue(),
                                                atoi(this->SmoothnessMenu->GetValue()));
    
    if (err == 0)
      {
      // success!
      vtkDebugMacro("Success in making the GLM design.");
      err = this->GetLogic()->RunGlmFit();
      if (err == 0)
        {
        vtkDebugMacro("Succeeded in running Glm Fit.");
        }
      else
        {
        vtkErrorMacro("Error running GLM Fit...");
        }
      }
    else
      {
      vtkErrorMacro("Error creating the GLM Design...");
      }
    
    // get the models logic to use to load the models and scalars (can't access it in the Logic class)
    vtkSlicerModelsLogic *modelsLogic = vtkSlicerModelsGUI::SafeDownCast(vtkSlicerApplication::SafeDownCast(this->GetApplication())->GetModuleGUIByName("Models"))->GetLogic();
    if (this->GetLogic()->LoadResults(modelsLogic, this->GetApplication()) != 0)
      {
      vtkErrorMacro("Unable to load results of GLM fit processing");
      }
    else
      {
      // clear out the questions menu
      this->QuestionMenu->GetMenu()->DeleteAllItems();
      // update the questions menu

      int numQuestions = this->GetLogic()->GetNumberOfQuestions();
      for (unsigned int i = 0; i < numQuestions; i++)
        {
        this->QuestionMenu->GetMenu()->AddRadioButton(this->GetLogic()->GetQuestion(i).c_str());
        }
      }
    if (this->GetDebug())
      {
      this->GetLogic()->DebugOff();
      }
    this->DebugOff();
    return;
    }

  vtkKWLoadSaveButton *dirbrowse = vtkKWLoadSaveButton::SafeDownCast(caller);

  if (dirbrowse == this->SubjectsDirectoryButton->GetWidget()  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a table file has been selected for loading...
    const char *fileName = this->SubjectsDirectoryButton->GetWidget()->GetFileName();
    if (!fileName || strcmp(fileName,"") == 0)
      {
      vtkDebugMacro("Empty filename");
      this->SubjectsDirectoryButton->GetWidget()->SetText ("None");
      this->GetLogic()->SetSubjectsDirectory("None");
      return;
      }
    vtkDebugMacro("Setting the qdec projects subjects dir to " << fileName);
    this->GetLogic()->SetSubjectsDirectory(fileName);
    return;
    }
  
  vtkKWLoadSaveButton *filebrowse = vtkKWLoadSaveButton::SafeDownCast(caller);

  if (filebrowse == this->LoadTableButton->GetWidget()  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a table file has been selected for loading...
    const char *fileName = this->LoadTableButton->GetWidget()->GetFileName();
    if (!fileName || strcmp(fileName,"") == 0)
      {
      vtkDebugMacro("Empty filename");
      this->LoadTableButton->GetWidget()->SetText ("None");
      return;
      }
    vtkDebugMacro("Trying to load file " << fileName);
    if (this->GetLogic()->LoadDataTable(fileName) == 0)
      {
      // failure
      vtkErrorMacro("Error loading table file " << fileName);
      this->LoadTableButton->GetWidget()->SetText ("None");
      }
    else
      {
      filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
      vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: was able to load file " << fileName);
      if (this->GetDebug())
        {
        this->GetLogic()->QDECProject->DumpDataTable ( stdout );
        }
      this->UpdateGUI();
      
      }
    return;
    }

  if (filebrowse == this->LoadResultsButton->GetWidget()  && event == vtkKWPushButton::InvokedEvent )
    {
    // If a table file has been selected for loading...
    const char *fileName = this->LoadResultsButton->GetWidget()->GetFileName();
    if (!fileName || strcmp(fileName,"") == 0)
      {
      vtkDebugMacro("Empty filename");
      this->LoadResultsButton->GetWidget()->SetText ("None");
      return;
      }
    vtkDebugMacro("Trying to load file " << fileName);
    if (this->GetLogic()->LoadGlmDesign(fileName) == -1)
      {
      // failure
      vtkErrorMacro("Error loading table file " << fileName);
      this->LoadResultsButton->GetWidget()->SetText ("None");
      }
    else
      {
      filebrowse->GetLoadSaveDialog()->SaveLastPathToRegistry("OpenPath");
      vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: was able to load file " << fileName);
      
      
      this->UpdateGUI();

      // load the results
      vtkWarningMacro("Reading results that were listed in file " << fileName);
      // get the models logic to use to load the models and scalars (can't access it in the Logic class)
      vtkSlicerModelsLogic *modelsLogic = vtkSlicerModelsGUI::SafeDownCast(vtkSlicerApplication::SafeDownCast(this->GetApplication())->GetModuleGUIByName("Models"))->GetLogic();
      if (this->GetLogic()->LoadResults(modelsLogic, this->GetApplication()) != 0)
        {
        vtkErrorMacro("Unable to load results of precomputed GLM fit processing from file " << fileName);
        }
      else
        {
        // clear out the questions menu
        this->QuestionMenu->GetMenu()->DeleteAllItems();
        // update the questions menu
        
        int numQuestions = this->GetLogic()->GetNumberOfQuestions();
        for (unsigned int i = 0; i < numQuestions; i++)
          {
          this->QuestionMenu->GetMenu()->AddRadioButton(this->GetLogic()->GetQuestion(i).c_str());
          }
        }
      
      }
    return;
    }
  
  // check for pick events
  if (event == vtkSlicerViewerInteractorStyle::PlotEvent &&
      vtkSlicerViewerInteractorStyle::SafeDownCast(caller) != NULL &&
      callData != NULL)
    {
    vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: Plot event!\n");
    // do the pick
    int x = ((int *)callData)[0];
    int y = ((int *)callData)[1];
    if (this->GetViewerWidget() &&
        this->GetViewerWidget()->Pick(x,y) != 0)
      {
      // check for a valid vertex point
      vtkIdType pointID = this->GetViewerWidget()->GetPickedPointID();
      if (pointID != -1)
        {
        // get the node name
        const char *nodeName = this->GetViewerWidget()->GetPickedNodeName();
        if (strcmp(nodeName, "") != 0)
          {
          vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: got picked node " << nodeName << " with point id = " << pointID);
          double *pickedRAS;
          double RAS[3];
          RAS[0] = RAS[1] = RAS[2] = 0.0;
          pickedRAS = this->GetViewerWidget()->GetPickedRAS();
          if (pickedRAS)
            {
            RAS[0] = pickedRAS[0];
            RAS[1] = pickedRAS[1];
            RAS[2] = pickedRAS[2];
            vtkDebugMacro("Got picked RAS = " << RAS[0] << ", " << RAS[1] << ", " << RAS[2]);
            }
          vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();
          if (app)
            {
            vtkDebugMacro("Calling: vtkFreeSurferReadersPlotPlotData " << pointID << ", " << nodeName << ", and the RAS points");
            std::string plotReturn = app->Script("vtkFreeSurferReadersPlotPlotData %d %s %g %g %g", pointID, nodeName, RAS[0], RAS[1], RAS[2]);
            vtkDebugMacro("Plot call return value = " << plotReturn.c_str());
            // swallow the pick
            if (this->GUICallbackCommand != NULL)
              {
              vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: swallowing the pick");
              this->GUICallbackCommand->SetAbortFlag(1);
              }
            else
              {
              vtkErrorMacro("Unable to get the gui call back command that calls process widget events, event = " << event << " is not swallowed here");
              }
            }
          else
            {
            vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: no application to use to call plot data.");
            }
          }
        else
          {
          vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: invalid node name");
          }
        }
      else
        {
        vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: invalid picked point");
        }
      }
    else
      {
      vtkDebugMacro("vtkQdecModuleGUI:ProcessGUIEvents: invalid pick");
      }
    return;
    }
  
  // changing the scalar overlay when the contrast question menu is invoked
  vtkKWMenu *m = vtkKWMenu::SafeDownCast ( caller );
  if (m == this->QuestionMenu->GetMenu() && event == vtkKWMenu::MenuItemInvokedEvent)
    {
    vtkDebugMacro("Got question menu update: " << this->QuestionMenu->GetValue());
    if (strcmp(this->QuestionMenu->GetValue(), "None") != 0 &&
        this->GetLogic())
      {
      std::string scalarName = this->GetLogic()->GetQuestionScalarName(this->QuestionMenu->GetValue());
      vtkDebugMacro("Got question scalar name from logic: " << scalarName.c_str());
      // trigger display change on the model
      if (this->GetLogic()->GetModelNode() != NULL)
        {
        vtkDebugMacro("Setting the active scalars on " << this->GetLogic()->GetModelNode()->GetName() << " to " << scalarName.c_str());
        this->GetLogic()->GetModelNode()->SetActiveScalars(scalarName.c_str(), NULL);
        this->GetLogic()->GetModelNode()->GetModelDisplayNode()->SetActiveScalarName(scalarName.c_str());
        // the color node has the same name as the scalar array name to facilitate
        // this pairing, find the ID by querying the mrml scene
        std::string colorID = "none";
        if (this->GetApplication() &&
            this->GetApplicationGUI()->GetMRMLScene())
          {
          vtkCollection *colorNodes =  this->GetApplicationGUI()->GetMRMLScene()->GetNodesByName(scalarName.c_str());
          if (colorNodes)
            {
            int numberOfNodes = colorNodes->GetNumberOfItems();
            if (numberOfNodes > 0)
              {
              // take the first one
              colorID = vtkMRMLProceduralColorNode::SafeDownCast(colorNodes->GetItemAsObject(0))->GetID();
              }
            else
              {
              vtkErrorMacro("vtkQdecModuleGUI Cannot find a color node with the name " << scalarName.c_str());
              }
            }
          else
            {
            vtkErrorMacro("vtkQdecModuleGUI cannot find procedural color nodes to check for the one associated with scalar " << scalarName.c_str());
            }         
          } else { vtkErrorMacro("No application or scene, can't find matching color node"); }
        if (strcmp(colorID.c_str(), "none") != 0)
          {
          // use this node id
          if (strcmp(this->GetLogic()->GetModelNode()->GetModelDisplayNode()->GetColorNodeID(), colorID.c_str()) != 0)
            {
            vtkDebugMacro("Setting the model's display node color node id to " << colorID.c_str());
            this->GetLogic()->GetModelNode()->GetModelDisplayNode()->SetAndObserveColorNodeID(colorID.c_str());
            }
          else { vtkDebugMacro("Model's display node color node is already set to " << colorID.c_str()); }
          }
        else
          {
          vtkErrorMacro("Qdec Module gui unable to find matching color node for scalar array " << scalarName.c_str());
          }
        }
      else
        {
        vtkErrorMacro("Qdec Module Logic has no record of a model node, can't switch scalars");
        }
      }
    }
//  this->DebugOff();
  
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::UpdateMRML ()
{

}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::UpdateGUI ()
{
  if (this->GetLogic() && this->GetLogic()->QDECProject)
    {
    // get the discrete and continuous factors
    vector< string > discreteFactors = this->GetLogic()->QDECProject->GetDiscreteFactors();
    this->DiscreteFactorsListBox->GetWidget()->GetWidget()->DeleteAll();
    for (unsigned int i = 0; i < discreteFactors.size(); i++)
      {
      this->DiscreteFactorsListBox->GetWidget()->GetWidget()->Append(discreteFactors[i].c_str());
      }
      
    vector< string > continuousFactors = this->GetLogic()->QDECProject->GetContinousFactors();

    this->ContinuousFactorsListBox->GetWidget()->GetWidget()->DeleteAll();
    for (unsigned int i = 0; i < continuousFactors.size(); i++)
      {
      this->ContinuousFactorsListBox->GetWidget()->GetWidget()->Append(continuousFactors[i].c_str());
      }    
      
    // now populate the table    
    if (this->MultiColumnList)
      {
      if (this->MultiColumnList->GetWidget()->GetNumberOfRows() != 0)
        {
        //clear out the list box
        this->MultiColumnList->GetWidget()->DeleteAllRows();
        }
      // set up the correct number of columns
      if ((1 + discreteFactors.size() + continuousFactors.size()) !=
          this->MultiColumnList->GetWidget()->GetNumberOfColumns())
        {
        this->MultiColumnList->GetWidget()->DeleteAllColumns();
        this->MultiColumnList->GetWidget()->AddColumn("Subject");
        for (unsigned int i = 0; i < discreteFactors.size(); i++)
          {
          vtkDebugMacro("Adding a column for discrete factor: " << discreteFactors[i].c_str());
          this->MultiColumnList->GetWidget()->AddColumn(discreteFactors[i].c_str());
          }
        for (unsigned int i = 0; i < continuousFactors.size(); i++)
          {
          vtkDebugMacro("Adding a column for cont factor: " << continuousFactors[i].c_str());
          this->MultiColumnList->GetWidget()->AddColumn(continuousFactors[i].c_str());
          }
        }
      else
        {
        // just make sure the column names are correct
        int col = 1;
        for (unsigned int i = 0; i < discreteFactors.size(); i++)
          {
          vtkDebugMacro("Setting a column name at " << col << " for discrete factor: " << discreteFactors[i].c_str());
          this->MultiColumnList->GetWidget()->SetColumnName(col, discreteFactors[i].c_str());
          col++;
          }
        for (unsigned int i = 0; i < continuousFactors.size(); i++)
          {
          vtkDebugMacro("Setting a column name at " << col << " for continuous factor: " << continuousFactors[i].c_str());
          this->MultiColumnList->GetWidget()->SetColumnName(col, continuousFactors[i].c_str());
          col++;
          }
        }
      
      vector< string > subjectIDs = this->GetLogic()->QDECProject->GetDataTable()->GetSubjectIDs();
      vector< QdecSubject* > subjects = this->GetLogic()->QDECProject->GetDataTable()->GetSubjects();
      for (unsigned int row = 0; row < subjectIDs.size(); row++)
        {
        this->MultiColumnList->GetWidget()->AddRow();
        int col = 0;
        vtkDebugMacro("Setting cell at " << row << " " << col << " to " << subjectIDs[row].c_str());
        this->MultiColumnList->GetWidget()->SetCellText(row, col, subjectIDs[row].c_str());
        col++;
        // these are strings
        for (unsigned int i = 0; i < discreteFactors.size(); i++)
          {
          vtkDebugMacro("Setting cell at " << row << " " << col << " to factor " << discreteFactors[i].c_str() << " = " << subjects[row]->GetDiscreteFactor(discreteFactors[i].c_str()).c_str());
          this->MultiColumnList->GetWidget()->SetCellText(row, col, subjects[row]->GetDiscreteFactor(discreteFactors[i].c_str()).c_str());
          col++;
          }
        // these are numbers
        for (unsigned int i = 0; i < continuousFactors.size(); i++)
          {
          vtkDebugMacro("Setting cell at " << row << " " << col << " to factor " << continuousFactors[i].c_str() << " = " << subjects[row]->GetContinuousFactor(continuousFactors[i].c_str()));
          this->MultiColumnList->GetWidget()->SetCellTextAsDouble(row, col, subjects[row]->GetContinuousFactor(continuousFactors[i].c_str()));
          col++;
          }
        }
      }
    } // end of valid QdecProject
  else
    {
    vtkErrorMacro("UpdateGUI: invalid Logic or Qdec Project");
    }
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::ProcessMRMLEvents ( vtkObject *caller,
                                            unsigned long event,
                                            void *callData ) 
{
  // if parameter node has been changed externally, update GUI widgets with new values

}




//---------------------------------------------------------------------------
void vtkQdecModuleGUI::BuildGUI ( ) 
{
  vtkSlicerApplication *app = (vtkSlicerApplication *)this->GetApplication();


  this->UIPanel->AddPage ( "QdecModule", "QdecModule", NULL );
  // ---
  // MODULE GUI FRAME 
  // configure a page for a volume loading UI for now.
  // later, switch on the modulesButton in the SlicerControlGUI
  // ---
    
  // HELP FRAME
  const char *help = "The Qdec Module provides support for the QDEC program from MGH: Query, Design, Estimate, Contrast.\nBefore you start: Set up a subjects directory and set your SUBJECTS_DIR environment variable (you'll need to restart Slicer3). In that directory, put the set of subjects you wish to analyse, along with the fsaverage subject. Then make a subdirectory qdec, and in it place your qdec.table.dat file along with any .level files describing the discrete factors (ie gender.levels will have 'male' and 'female' on two separate lines).";
  const char *about = "This work was supported by NA-MIC. See http://na-mic.org/Wiki/index.php/Projects/Slicer3/2007_Project_Week_QDEC_Slicer3_Integration for details on the integration project.";
  vtkKWWidget *page = this->UIPanel->GetPageWidget ( "QdecModule" );
  this->BuildHelpAndAboutFrame ( page, help, about );
  
  this->NAMICLabel = vtkKWLabel::New();
  this->NAMICLabel->SetParent ( this->GetLogoFrame() );
  this->NAMICLabel->Create();
  this->NAMICLabel->SetImageToIcon ( this->GetAcknowledgementIcons()->GetNAMICLogo() );    
  app->Script ( "grid %s -row 0 -column 0 -padx 2 -pady 2 -sticky w", this->NAMICLabel->GetWidgetName());

  // ---
  // Subjects Frame
  vtkSlicerModuleCollapsibleFrame *subjectsFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  subjectsFrame->SetParent ( page );
  subjectsFrame->Create ( );
  subjectsFrame->SetLabelText ("Subjects");
  subjectsFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                subjectsFrame->GetWidgetName(),
                this->UIPanel->GetPageWidget("QdecModule")->GetWidgetName());

  this->SubjectsDirectoryButton = vtkKWLoadSaveButtonWithLabel::New();
  this->SubjectsDirectoryButton->SetParent ( subjectsFrame->GetFrame() );
  this->SubjectsDirectoryButton->Create ( );
  this->SubjectsDirectoryButton->SetLabelText ("Set SUBJECTS_DIR:");
  this->SubjectsDirectoryButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Select the FreeSurfer Subjects directory");
  this->SubjectsDirectoryButton->GetWidget()->GetLoadSaveDialog()->ChooseDirectoryOn();
  if (getenv("SUBJECTS_DIR") == NULL)
    {
    this->SubjectsDirectoryButton->GetWidget()->SetText ("None");
    this->SubjectsDirectoryButton->GetWidget()->GetLoadSaveDialog()->SetInitialFileName( "." );
    }
  else
    {
    this->SubjectsDirectoryButton->GetWidget()->SetText (getenv("SUBJECTS_DIR"));
    this->SubjectsDirectoryButton->GetWidget()->GetLoadSaveDialog()->SetInitialFileName(getenv("SUBJECTS_DIR"));
    }
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4", 
              this->SubjectsDirectoryButton->GetWidgetName());
  
  this->LoadTableButton = vtkKWLoadSaveButtonWithLabel::New();
  this->LoadTableButton->SetParent ( subjectsFrame->GetFrame() );
  this->LoadTableButton->Create ( );
  this->LoadTableButton->SetLabelText ("Load Table Data File:");
  // check to see if there's a qdec table data file in the subjects dir
  
  std::string testDatFile = "None";
  if (getenv("SUBJECTS_DIR") != NULL)
    {
    std::vector<std::string> pathcomponents;
    itksys::SystemTools::SplitPath(getenv("SUBJECTS_DIR"), pathcomponents);
    pathcomponents.push_back("qdec");
    pathcomponents.push_back("qdec.table.dat");
    testDatFile = itksys::SystemTools::JoinPath(pathcomponents);

    if (!itksys::SystemTools::LocateFileInDir(testDatFile.c_str(), testDatFile.c_str(), testDatFile))
      {
      testDatFile = "None";
      }
    }
  this->LoadTableButton->GetWidget()->SetText (testDatFile.c_str());

  this->LoadTableButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Open QDEC table data file");
  this->LoadTableButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
  this->LoadTableButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {All} {.*} } { {Data} {.dat} }");
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4", 
              this->LoadTableButton->GetWidgetName());

  // Load precomputed results
  this->LoadResultsButton = vtkKWLoadSaveButtonWithLabel::New();
  this->LoadResultsButton->SetParent ( subjectsFrame->GetFrame() );
  this->LoadResultsButton->Create ( );
  this->LoadResultsButton->SetLabelText ("Load Results Data File:");
  this->LoadResultsButton->GetWidget()->GetLoadSaveDialog()->SetTitle("Open QDEC file");
  this->LoadResultsButton->GetWidget()->GetLoadSaveDialog()->RetrieveLastPathFromRegistry("OpenPath");
  this->LoadResultsButton->GetWidget()->GetLoadSaveDialog()->SetFileTypes("{ {All} {.*} } { {Data} {.qdec} }");
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4", 
              this->LoadResultsButton->GetWidgetName());

   // add the multicolumn list to show the data table
  this->MultiColumnList = vtkKWMultiColumnListWithScrollbars::New ( );
  this->MultiColumnList->SetParent ( subjectsFrame->GetFrame() );
  this->MultiColumnList->Create ( );
  this->MultiColumnList->SetHeight(4);
  this->MultiColumnList->GetWidget()->SetSelectionTypeToCell();
  this->MultiColumnList->GetWidget()->MovableRowsOff();
  this->MultiColumnList->GetWidget()->MovableColumnsOff();

  this->MultiColumnList->GetWidget()->AddColumn("Subject");
  //not setting up all columns just yet
  app->Script ( "pack %s -fill both -expand true",
                this->MultiColumnList->GetWidgetName());
  this->MultiColumnList->GetWidget()->SetCellUpdatedCommand(this, "UpdateElement");

  // ---
  // Design Frame
  vtkSlicerModuleCollapsibleFrame *designFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  designFrame->SetParent ( page );
  designFrame->Create ( );
  designFrame->SetLabelText ("Design");
  designFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                designFrame->GetWidgetName(),
                this->UIPanel->GetPageWidget("QdecModule")->GetWidgetName());

  this->DesignEntry = vtkKWEntryWithLabel::New();
  this->DesignEntry->SetParent( designFrame->GetFrame());
  this->DesignEntry->SetLabelText ("Design Name:");
  this->DesignEntry->Create();
  this->DesignEntry->GetWidget()->SetValue("Untitled");
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4 -in %s", 
              this->DesignEntry->GetWidgetName(),
              designFrame->GetFrame()->GetWidgetName());
  
  this->DiscreteFactorsListBox = vtkKWListBoxWithScrollbarsWithLabel::New();
  this->DiscreteFactorsListBox->SetParent( designFrame->GetFrame() );
  this->DiscreteFactorsListBox->SetLabelText("Discrete Factors (choose up to two):" );
  this->DiscreteFactorsListBox->Create();
  this->DiscreteFactorsListBox->GetWidget()->GetWidget()->SetSelectionModeToMultiple();
  this->DiscreteFactorsListBox->GetWidget()->GetWidget()->ExportSelectionOff();
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4 -in %s", 
              this->DiscreteFactorsListBox->GetWidgetName(),
              designFrame->GetFrame()->GetWidgetName());
  
  this->ContinuousFactorsListBox = vtkKWListBoxWithScrollbarsWithLabel::New();
  this->ContinuousFactorsListBox->SetParent( designFrame->GetFrame() );
  this->ContinuousFactorsListBox->SetLabelText("Continuous Factors (choose two):");
  this->ContinuousFactorsListBox->Create();
  this->ContinuousFactorsListBox->GetWidget()->GetWidget()->SetSelectionModeToMultiple();
  this->ContinuousFactorsListBox->GetWidget()->GetWidget()->ExportSelectionOff();
  app->Script("pack %s -side top -anchor nw -padx 2 -pady 4 -in %s", 
              this->ContinuousFactorsListBox->GetWidgetName(),
              designFrame->GetFrame()->GetWidgetName());

  vtkKWFrameWithLabel* measuresFrame = vtkKWFrameWithLabel::New();
  measuresFrame->SetParent( designFrame->GetFrame() );
  measuresFrame->Create();
  measuresFrame->SetLabelText( "Measures" );
  this->Script( "pack %s -fill x", measuresFrame->GetWidgetName() );

  
  int nRow = 0;
  this->MeasureLabel = vtkKWLabel::New();
  this->MeasureLabel->SetParent(measuresFrame->GetFrame() );
  this->MeasureLabel->Create();
  this->MeasureLabel->SetText("Measure: ");
  this->MeasureLabel->SetJustificationToRight();
  app->Script( "grid %s -column 0 -row %d -sticky ne -in %s",
               this->MeasureLabel->GetWidgetName(), nRow,
               measuresFrame->GetFrame()->GetWidgetName());

  this->MeasureMenu = vtkKWMenuButton::New();
  this->MeasureMenu->SetParent(measuresFrame->GetFrame() );
  this->MeasureMenu->Create();
  this->MeasureMenu->GetMenu()->AddRadioButton("thickness");
  this->MeasureMenu->GetMenu()->AddRadioButton("sulc");
  this->MeasureMenu->GetMenu()->AddRadioButton("curv");
  this->MeasureMenu->GetMenu()->AddRadioButton("jacobian");
  this->MeasureMenu->GetMenu()->AddRadioButton("area");
  this->MeasureMenu->SetValue("thickness");
  app->Script( "grid %s -column 1 -row %d -sticky nw -in %s",
               this->MeasureMenu->GetWidgetName(), nRow,
               measuresFrame->GetFrame()->GetWidgetName());
  
  nRow++;
  this->HemisphereLabel = vtkKWLabel::New();
  this->HemisphereLabel->SetParent(measuresFrame->GetFrame() );
  this->HemisphereLabel->Create();
  this->HemisphereLabel->SetText("Hemisphere: ");
  this->HemisphereLabel->SetJustificationToRight();
  app->Script( "grid %s -column 0 -row %d -sticky ne -in %s",
               this->HemisphereLabel->GetWidgetName(), nRow,
               measuresFrame->GetFrame()->GetWidgetName());

  this->HemisphereMenu = vtkKWMenuButton::New();
  this->HemisphereMenu->SetParent(measuresFrame->GetFrame() );
  this->HemisphereMenu->Create();
  this->HemisphereMenu->GetMenu()->AddRadioButton("lh");
  this->HemisphereMenu->GetMenu()->AddRadioButton("rh");
  this->HemisphereMenu->SetValue("lh");
  app->Script( "grid %s -column 1 -row %d -sticky nw -in %s",
               this->HemisphereMenu->GetWidgetName(), nRow,
               measuresFrame->GetFrame()->GetWidgetName());
  nRow++;

  this->SmoothnessLabel = vtkKWLabel::New();
  this->SmoothnessLabel->SetParent(measuresFrame->GetFrame());
  this->SmoothnessLabel->Create();
  this->SmoothnessLabel->SetText("Smoothness (FWHM, mm): ");
  this->SmoothnessLabel->SetJustificationToRight();
  app->Script( "grid %s -column 0 -row %d -sticky ne -in %s",
               this->SmoothnessLabel->GetWidgetName(), nRow,
               measuresFrame->GetFrame()->GetWidgetName());

  this->SmoothnessMenu = vtkKWMenuButton::New();
  this->SmoothnessMenu->SetParent(measuresFrame->GetFrame());
  this->SmoothnessMenu->Create();
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "0" );
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "5" );
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "10" );
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "15" );
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "20" );
  this->SmoothnessMenu->GetMenu()->AddRadioButton( "25" );
  this->SmoothnessMenu->SetValue( "10" );
  this->Script( "grid %s -column 1 -row %d -sticky nw -in %s",
                this->SmoothnessMenu->GetWidgetName(), nRow,
                measuresFrame->GetFrame()->GetWidgetName());

  // weight the grid
  for( int nRowConfigure = 0; nRowConfigure <= nRow; nRowConfigure++ )
    {
    this->Script( "grid rowconfigure %s %d -pad 4",
                  measuresFrame->GetFrame()->GetWidgetName(),
                  nRowConfigure );
    }
  
  this->ApplyButton = vtkKWPushButton::New();
  this->ApplyButton->SetParent( designFrame->GetFrame() );
  this->ApplyButton->Create();
  this->ApplyButton->SetText("Run GLM Fit");
  this->ApplyButton->SetWidth ( 12 );
  app->Script("pack %s -side top -anchor e -padx 20 -pady 10 -in %s", 
              this->ApplyButton->GetWidgetName(),
              designFrame->GetFrame()->GetWidgetName());
  

  // ---
  // Display Frame
  vtkSlicerModuleCollapsibleFrame *displayFrame = vtkSlicerModuleCollapsibleFrame::New ( );
  displayFrame->SetParent ( page );
  displayFrame->Create ( );
  displayFrame->SetLabelText ("Display");
  displayFrame->ExpandFrame ( );
  app->Script ( "pack %s -side top -anchor nw -fill x -padx 2 -pady 2 -in %s",
                displayFrame->GetWidgetName(),
                this->UIPanel->GetPageWidget("QdecModule")->GetWidgetName());

  // provide a set of radio boxes that will be populated from the Logic's
  // calculated overlays
  this->QuestionLabel = vtkKWLabel::New();
  this->QuestionLabel->SetParent(displayFrame->GetFrame());
  this->QuestionLabel->Create();
  this->QuestionLabel->SetText("Question: ");
  this->QuestionLabel->SetJustificationToRight();
  app->Script( "grid %s -column 0 -row 1 -sticky ne -in %s",
               this->QuestionLabel->GetWidgetName(),
               displayFrame->GetFrame()->GetWidgetName());

  this->QuestionMenu = vtkKWMenuButton::New();
  this->QuestionMenu->SetParent(displayFrame->GetFrame());
  this->QuestionMenu->Create();
  this->QuestionMenu->GetMenu()->AddRadioButton( "None" );
  this->QuestionMenu->SetValue( "None" );
  this->Script( "grid %s -column 1 -row 1 -sticky nw -in %s",
                this->QuestionMenu->GetWidgetName(),
                displayFrame->GetFrame()->GetWidgetName());
  
  measuresFrame->Delete();
  designFrame->Delete();
  subjectsFrame->Delete();
  displayFrame->Delete();
  

  // for plotting
  if (!this->GetLogic()->GetTclScriptLoaded())
    {
      const char *tclScript = this->GetLogic()->GetPlotTclScript();
      vtkWarningMacro("Loading: " << tclScript);
      if (app->LoadScript(tclScript) == 0)
        {
        vtkErrorMacro("vtkQdecModuleGUI::BuildGUI: unable to load in tcl script " << tclScript);
        }
      else
        {
        this->GetLogic()->SetTclScriptLoaded(1);
        }
    }

  this->Built = true;
}

//---------------------------------------------------------------------------
void vtkQdecModuleGUI::UpdateElement(int row, int col, char * str)
{
  if (!this->MultiColumnList)
    {
    vtkErrorMacro("UpdateElement: cannot update element, multi column list is null");
    return;
    }
  vtkDebugMacro("UpdateElement: row = " << row << ", col = " << col << ", str = " << str << "\n");
  
  // make sure that the row and column exists in the table
  if ((row >= 0) && (row < this->MultiColumnList->GetWidget()->GetNumberOfRows()) &&
      (col >= 0) && (col < this->MultiColumnList->GetWidget()->GetNumberOfColumns()))
    {
    vtkDebugMacro("Valid row " << row << ", column " << col <<  ", valid columns are 0-" << this->MultiColumnList->GetWidget()->GetNumberOfColumns() << "\n");
    } 
  else
    {
    vtkErrorMacro ("Invalid row " << row << " or column " << col <<  ", valid columns are 0-" << this->MultiColumnList->GetWidget()->GetNumberOfColumns() << "\n");
    }
}

//----------------------------------------------------------------------------
void vtkQdecModuleGUI::SetViewerWidget ( vtkSlicerViewerWidget *viewerWidget )
{
  this->ViewerWidget = viewerWidget;
}

//----------------------------------------------------------------------------
void vtkQdecModuleGUI::SetInteractorStyle( vtkSlicerViewerInteractorStyle *interactorStyle )
{
  // note: currently the GUICallbackCommand calls ProcessGUIEvents
  // remove observers
  if (this->InteractorStyle != NULL &&
      this->InteractorStyle->HasObserver(vtkSlicerViewerInteractorStyle::PlotEvent, this->GUICallbackCommand) == 1)
    {
    this->InteractorStyle->RemoveObservers(vtkSlicerViewerInteractorStyle::PlotEvent, (vtkCommand *)this->GUICallbackCommand);
    }
  
  this->InteractorStyle = interactorStyle;

  // add observers
  if (this->InteractorStyle)
    {
    vtkDebugMacro("vtkQdecModuleGUI::SetInteractorStyle: Adding observer on interactor style");
    this->InteractorStyle->AddObserver(vtkSlicerViewerInteractorStyle::PlotEvent, (vtkCommand *)this->GUICallbackCommand);
    }
}
