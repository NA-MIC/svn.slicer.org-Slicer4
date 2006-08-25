#include "vtkObjectFactory.h"
#include "vtkCallbackCommand.h"

#include "vtkSlicerComponentGUI.h"
#include "vtkSlicerApplicationGUI.h"
#include "vtkSlicerApplicationLogic.h"
#include "vtkMRMLScene.h"

#include "vtkKWApplication.h"



vtkCxxRevisionMacro ( vtkSlicerComponentGUI, "$Revision: 1.0 $" );
vtkStandardNewMacro ( vtkSlicerComponentGUI );


//---------------------------------------------------------------------------
vtkSlicerComponentGUI::vtkSlicerComponentGUI ( )
{
    // Set up callbacks
    this->MRMLCallbackCommand = vtkCallbackCommand::New ( );
    this->MRMLCallbackCommand->SetClientData( reinterpret_cast<void *>(this) );
    this->MRMLCallbackCommand->SetCallback( vtkSlicerComponentGUI::MRMLCallback );

    this->LogicCallbackCommand = vtkCallbackCommand::New ( );
    this->LogicCallbackCommand->SetClientData( reinterpret_cast<void *>(this) );
    this->LogicCallbackCommand->SetCallback( vtkSlicerComponentGUI::LogicCallback );

    this->GUICallbackCommand = vtkCallbackCommand::New ( );
    this->GUICallbackCommand->SetClientData( reinterpret_cast<void *>(this) );
    this->GUICallbackCommand->SetCallback( vtkSlicerComponentGUI::GUICallback );
    
    // Set null pointers
    this->MRMLScene = NULL;
    this->ApplicationLogic = NULL;
    this->GUIName = NULL;

    // Instance variable flags for the observers
    this->InGUICallbackFlag = 0;
    this->InMRMLCallbackFlag = 0;
    this->InLogicCallbackFlag = 0;

}


//---------------------------------------------------------------------------
vtkSlicerComponentGUI::~vtkSlicerComponentGUI ( )
{
    // remove observers on MRMLScene or ApplicationLogic,
    this->SetAndObserveMRMLScene ( NULL );
    this->SetAndObserveApplicationLogic ( NULL );

    // unregister and set null pointers.
    if ( this->MRMLCallbackCommand != NULL )
        {
            this->MRMLCallbackCommand->Delete ( );
            this->MRMLCallbackCommand = NULL;
        }
    if ( this->LogicCallbackCommand != NULL )
        {
            this->LogicCallbackCommand->Delete ( );
            this->LogicCallbackCommand = NULL;
        }
    if ( this->GUICallbackCommand != NULL )
        {
            this->GUICallbackCommand->Delete ( );
            this->GUICallbackCommand = NULL;
        }

    // and set null pointers.
    this->GUIName = NULL;
    this->SetApplication ( NULL );
}



//---------------------------------------------------------------------------
void vtkSlicerComponentGUI::PrintSelf ( ostream& os, vtkIndent indent )
{
    this->vtkObject::PrintSelf ( os, indent );
    os << indent << "SlicerComponentGUI: " << this->GetClassName ( ) << "\n";
    os << indent << "MRMLScene: " << this->GetMRMLScene ( ) << "\n";
    os << indent << "ApplicationLogic: " << this->GetApplicationLogic ( ) << "\n";
    os << indent << "GUIName: " << this->GetGUIName ( ) << "\n";
}


//----------------------------------------------------------------------------
// Description:
// the MRMLCallback is a static function to relay modified events from the 
// observed mrml node back into the gui layer for further processing
//
void 
vtkSlicerComponentGUI::MRMLCallback(vtkObject *caller, 
            unsigned long eid, void *clientData, void *callData)
{
  vtkSlicerComponentGUI *self = reinterpret_cast<vtkSlicerComponentGUI *>(clientData);

  if (self->GetInMRMLCallbackFlag())
    {
    vtkErrorWithObjectMacro(self, "In vtkSlicerComponentGUI *********MRMLCallback called recursively?");
    return;
    }

  vtkDebugWithObjectMacro(self, "In vtkSlicerComponentGUI MRMLCallback");

  self->SetInMRMLCallbackFlag(1);
  self->ProcessMRMLEvents(caller, eid, callData);
  self->SetInMRMLCallbackFlag(0);
}

//----------------------------------------------------------------------------
// Description:
// the LogicCallback is a static function to relay modified events from the 
// observed mrml node back into the gui layer for further processing
//
void 
vtkSlicerComponentGUI::LogicCallback(vtkObject *caller, 
            unsigned long eid, void *clientData, void *callData)
{
  vtkSlicerComponentGUI *self = reinterpret_cast<vtkSlicerComponentGUI *>(clientData);

  if (self->GetInLogicCallbackFlag())
    {
    vtkErrorWithObjectMacro(self, "In vtkSlicerComponentGUI *********LogicCallback called recursively?");
    return;
    }

  vtkDebugWithObjectMacro(self, "In vtkSlicerComponentGUI LogicCallback");

  self->SetInLogicCallbackFlag(1);
  self->ProcessLogicEvents(caller, eid, callData);
  self->SetInLogicCallbackFlag(0);
}

//----------------------------------------------------------------------------
// Description:
// the GUICallback is a static function to relay modified events from the 
// observed mrml node back into the gui layer for further processing
//
void 
vtkSlicerComponentGUI::GUICallback(vtkObject *caller, 
            unsigned long eid, void *clientData, void *callData)
{
  vtkSlicerComponentGUI *self = reinterpret_cast<vtkSlicerComponentGUI *>(clientData);

  if (self->GetInGUICallbackFlag())
    {
    vtkErrorWithObjectMacro(self, "In vtkSlicerComponentGUI *********GUICallback called recursively?");
    return;
    }

  vtkDebugWithObjectMacro(self, "In vtkSlicerComponentGUI GUICallback");

  self->SetInGUICallbackFlag(1);
  self->ProcessGUIEvents(caller, eid, callData);
  self->SetInGUICallbackFlag(0);
}



//---------------------------------------------------------------------------
void vtkSlicerComponentGUI::SetMRML ( vtkObject **nodePtr, vtkObject *node )
{
    // Delete 
    if ( *nodePtr )
        {
            ( *nodePtr )->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
            ( *nodePtr )->Delete ( );
            *nodePtr = NULL;
        }

    // Set 
    *nodePtr = node;

    // Register 
    if ( *nodePtr )
        {
            ( *nodePtr )->Register ( this );
        }
}


//---------------------------------------------------------------------------
void vtkSlicerComponentGUI::SetAndObserveMRML ( vtkObject **nodePtr, vtkObject *node )
{
    // Remove observers and delete
    if ( *nodePtr )
        {
            ( *nodePtr )->RemoveObservers ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
            ( *nodePtr )->Delete ( );
            *nodePtr = NULL;
        }

    // Set
    *nodePtr = node;

    // Register and add observers
    if ( *nodePtr )
        {
            ( *nodePtr )->Register ( this );
            ( *nodePtr )->AddObserver ( vtkCommand::ModifiedEvent, this->MRMLCallbackCommand );
        }
}


//---------------------------------------------------------------------------
void vtkSlicerComponentGUI::SetLogic ( vtkObject **logicPtr, vtkObject *logic )
{

    // Delete (check for observers first)
    if ( *logicPtr )
        {
            if ( ( *logicPtr )->HasObserver ( vtkCommand::ModifiedEvent, this->LogicCallbackCommand ) )
                {
                    ( *logicPtr )->RemoveObservers ( vtkCommand::ModifiedEvent, this->LogicCallbackCommand );
                }
            ( *logicPtr )->Delete ( );
            *logicPtr = NULL;
        }

    // Set 
    *logicPtr = logic;

    // Register 
    if ( *logicPtr )
        {
            ( *logicPtr )->Register ( this );
        }

}

//---------------------------------------------------------------------------
void vtkSlicerComponentGUI::SetAndObserveLogic ( vtkObject **logicPtr, vtkObject *logic )
{
    // Remove observers and delete
    if ( *logicPtr )
        {
            ( *logicPtr )->RemoveObservers ( vtkCommand::ModifiedEvent, this->LogicCallbackCommand );
            ( *logicPtr )->Delete ( );
            *logicPtr = NULL;
        }

    // Set
    *logicPtr = logic;

    // Register and add observers
    if ( *logicPtr )
        {
            ( *logicPtr )->Register ( this );
            ( *logicPtr )->AddObserver ( vtkCommand::ModifiedEvent, this->LogicCallbackCommand );
        }

}








