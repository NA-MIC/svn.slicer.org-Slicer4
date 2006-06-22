// .NAME vtkSlicerModelsGUI 
// .SECTION Description
// Main Models GUI and mediator methods for slicer3. 


#ifndef __vtkSlicerModelsGUI_h
#define __vtkSlicerModelsGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkSlicerModuleLogic.h"

//#include "vtkSlicerModelsLogic.h"
//#include "vtkMRMLModelNode.h"

#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWFrame.h"

// Description:
// This class implements Slicer's Models GUI
//
class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerModelsGUI : public vtkSlicerModuleGUI
{
 public:
    // Description:
    // Usual vtk class functions
    static vtkSlicerModelsGUI* New (  );
    vtkTypeRevisionMacro ( vtkSlicerModelsGUI, vtkSlicerModuleGUI );
    void PrintSelf ( ostream& os, vtkIndent indent );
    
    // Description:
    // Get methods on class members ( no Set methods required. )
    vtkGetObjectMacro ( LoadModelButton, vtkKWLoadSaveButton );
    //vtkGetObjectMacro ( Logic, vtkSlicerModelsLogic);
    //vtkGetObjectMacro ( ModelNode, vtkMRMLModelNode );
    
    // Description:
    // API for setting ModelNode, Logic and
    // for both setting and observing them.
    /*
      // classes not yet defined.
    void SetMRMLNode ( vtkMRMLModelNode *node )
        { this->SetMRML ( vtkObjectPointer( &this->MRMLModelNode), node ); }
    void SetAndObserveMRMLNode ( vtkMRMLModelNode *node )
        { this->SetMRML ( vtkObjectPointer( &this->MRMLModelNode), node ); }
    void SetModuleLogic ( vtkSlicerModelsLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ) }
    void SetAndObserveModuleLogic ( vtkSlicerModelsLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ) }
    */

    // Description:
    // This method builds the Models module GUI
    virtual void BuildGUI ( ) ;

    // Description:
    // Add/Remove observers on widgets in the GUI
    virtual void AddGUIObservers ( );
    virtual void RemoveGUIObservers ( );

    // Description:
    // Class's mediator methods for processing events invoked by
    // either the Logic, MRML or GUI.
    virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event, void *callData );
    virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event, void *callData );
    virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event, void *callData );

    // Description:
    // Methods describe behavior at module enter and exit.
    virtual void Enter ( );
    virtual void Exit ( );

 protected:
    vtkSlicerModelsGUI ( );
    ~vtkSlicerModelsGUI ( );

    // Module logic and mrml pointers (classes not yet built!)
    //vtkSlicerModelsLogic *Logic;
    //vtkMRMLModelNode *ModelNode;
    
    // Widgets for the Models module
    vtkKWLoadSaveButton *LoadModelButton;

 private:
    vtkSlicerModelsGUI ( const vtkSlicerModelsGUI& ); // Not implemented.
    void operator = ( const vtkSlicerModelsGUI& ); //Not implemented.
};


#endif
