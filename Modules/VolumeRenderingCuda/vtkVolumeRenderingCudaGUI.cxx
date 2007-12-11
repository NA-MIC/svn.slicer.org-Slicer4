
#include "vtkVolumeRenderingCudaGUI.h"

static vtkVolumeRenderingCudaModuleGUI* New();


void BuildGUI ( );

void TearDownGUI ( );

void CreateModuleEventBindings ( );
void ReleaseModuleEventBindings ( );

void AddGUIObservers ( );
  
void RemoveGUIObservers ( );
void RemoveMRMLNodeObservers ( );
void RemoveLogicObservers ( );

void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                  void *callData );
  
void ProcessMRMLEvents ( vtkObject *caller, unsigned long event,
                                   void *callData);
  
  
void Enter ( );
void Exit ( );
  
  
void PrintSelf(ostream& os, vtkIndent indent);
  
vtkVolumeRenderingCuda();
~vtkVolumeRenderingCuda();

