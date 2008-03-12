// .NAME vtkSlicerUnstructuredGridsGUI 
// .SECTION Description
// Main UGrids GUI and mediator methods for slicer3. 


#ifndef __vtkSlicerUnstructuredGridsGUI_h
#define __vtkSlicerUnstructuredGridsGUI_h

#include "vtkSlicerBaseGUIWin32Header.h"
#include "vtkSlicerModuleGUI.h"
#include "vtkSlicerModuleLogic.h"
#include "vtkSlicerUnstructuredGridsLogic.h"
#include "vtkSlicerNodeSelectorWidget.h"

#include "vtkSlicerUnstructuredGridsLogic.h"
//#include "vtkMRMLUnstructuredGridNode.h"

#include "vtkKWLoadSaveButton.h"
#include "vtkKWLoadSaveDialog.h"
#include "vtkKWLoadSaveButtonWithLabel.h"
#include "vtkKWFrame.h"
#include "vtkKWLabel.h"

// Description:
// This class implements Slicer's UGrids GUI
//
class vtkSlicerUnstructuredGridDisplayWidget;
class vtkSlicerUnstructuredGridHierarchyWidget;
class vtkSlicerModuleCollapsibleFrame;

class VTK_SLICER_BASE_GUI_EXPORT vtkSlicerUnstructuredGridsGUI : public vtkSlicerModuleGUI
{
 public:
    // Description:
    // Usual vtk class functions
    static vtkSlicerUnstructuredGridsGUI* New (  );
    vtkTypeRevisionMacro ( vtkSlicerUnstructuredGridsGUI, vtkSlicerModuleGUI );
    void PrintSelf ( ostream& os, vtkIndent indent );
    
    // Description:
    // Get methods on class members ( no Set methods required. )
    vtkGetObjectMacro ( LoadUnstructuredGridButton, vtkKWLoadSaveButton );
    vtkGetObjectMacro ( LoadScalarsButton, vtkKWLoadSaveButtonWithLabel);
    //vtkGetObjectMacro ( Logic, vtkSlicerUnstructuredGridsLogic);
    //vtkGetObjectMacro ( UnstructuredGridNode, vtkMRMLUnstructuredGridNode );
    
    // Description:
    // API for setting UnstructuredGridNode, Logic and
    // for both setting and observing them.
    /*
      // classes not yet defined.
    void SetMRMLNode ( vtkMRMLUnstructuredGridNode *node )
        { this->SetMRML ( vtkObjectPointer( &this->MRMLUnstructuredGridNode), node ); }
    void SetAndObserveMRMLNode ( vtkMRMLUnstructuredGridNode *node )
        { this->SetAndObserveMRML ( vtkObjectPointer( &this->MRMLUnstructuredGridNode), node ); }

    void SetModuleLogic ( vtkSlicerUnstructuredGridsLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ) }
    void SetAndObserveModuleLogic ( vtkSlicerUnstructuredGridsLogic *logic )
        { this->SetAndObserveLogic ( vtkObjectPointer (&this->Logic), logic ) }
    */

    void SetModuleLogic ( vtkSlicerUnstructuredGridsLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
    void SetAndObserveModuleLogic ( vtkSlicerUnstructuredGridsLogic *logic )
        { this->SetLogic ( vtkObjectPointer (&this->Logic), logic ); }
    vtkGetObjectMacro( Logic, vtkSlicerUnstructuredGridsLogic );

    // Description:
    // This method builds the UGrids module GUI
    virtual void BuildGUI ( ) ;

       // Descripgion:
    // This method releases references and key-bindings,
    // and optionally removes observers.
    virtual void TearDownGUI ( );

        // Description:
    // Methods for adding module-specific key bindings and
    // removing them.
    virtual void CreateModuleEventBindings ( );
    virtual void ReleaseModuleEventBindings ( );

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
    vtkSlicerUnstructuredGridsGUI ( );
    virtual ~vtkSlicerUnstructuredGridsGUI ( );

    // Module logic and mrml pointers (classes not yet built!)
    //vtkSlicerUnstructuredGridsLogic *Logic;
    //vtkMRMLUnstructuredGridNode *UnstructuredGridNode;
    
    // Widgets for the UGrids module
    vtkKWLoadSaveButton *LoadUnstructuredGridButton;
    vtkKWLoadSaveButtonWithLabel *LoadScalarsButton;
    vtkKWLoadSaveButton *LoadUnstructuredGridDirectoryButton;
    vtkKWLoadSaveButton *SaveUnstructuredGridButton;
    vtkSlicerNodeSelectorWidget* UnstructuredGridSelectorWidget;
    vtkSlicerNodeSelectorWidget* UnstructuredGridDisplaySelectorWidget;
    vtkKWLabel *NACLabel;
    vtkKWLabel *NAMICLabel;
    vtkKWLabel *NCIGTLabel;
    vtkKWLabel *BIRNLabel;

    vtkSlicerUnstructuredGridDisplayWidget *UnstructuredGridDisplayWidget;



    vtkSlicerUnstructuredGridsLogic *Logic;

    vtkSlicerModuleCollapsibleFrame *UnstructuredGridDisplayFrame;

 private:
    vtkSlicerUnstructuredGridsGUI ( const vtkSlicerUnstructuredGridsGUI& ); // Not implemented.
    void operator = ( const vtkSlicerUnstructuredGridsGUI& ); //Not implemented.
};


#endif
