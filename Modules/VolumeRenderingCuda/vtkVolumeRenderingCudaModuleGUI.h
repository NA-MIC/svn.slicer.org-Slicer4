#ifndef VTKSLICERVOLUMERENDERINGCUDA_H_
#define VTKSLICERVOLUMERENDERINGCUDA_H_

#include "vtkSlicerModuleGUI.h"
#include "vtkVolumeRenderingCudaModule.h"

class vtkVolumeRenderingCudaModuleLogic;

class vtkVolumeCudaMapper;
class vtkVolume;
class vtkVolumeProperty;


class vtkKWWidget;
class vtkKWTypeChooserBox;
class vtkKWMatrixWidget;
class vtkKWMenuButton;
class vtkKWVolumePropertyWidget;
class vtkKWHistogramSet;
class vtkKWRange;


class VTK_VOLUMERENDERINGCUDAMODULE_EXPORT vtkVolumeRenderingCudaModuleGUI : public vtkSlicerModuleGUI
{
 public:
 //BTX
  typedef vtkSlicerModuleGUI SuperClass;
 //ETX
  static vtkVolumeRenderingCudaModuleGUI* New();
  vtkTypeMacro(vtkVolumeRenderingCudaModuleGUI, vtkSlicerModuleGUI);


  /// Logic part
  vtkGetObjectMacro(Logic, vtkVolumeRenderingCudaModuleLogic);
  // this does not work vtkSetObjectMacro(Logic, vtkVolumeRenderingCudaModuleLogic);
  virtual void SetLogic(vtkVolumeRenderingCudaModuleLogic *logic)
  {
    this->Logic=logic;
  }

  void DeleteWidget(vtkKWWidget* widget);

  // Description:
  // Process events generated by Logic
  virtual void ProcessLogicEvents ( vtkObject *caller, unsigned long event,
                                    void *callData ){};

  /// GUI part
  virtual void BuildGUI ( );
  // This method releases references and key-bindings,
  // and optionally removes observers.
  virtual void TearDownGUI ( );

  // Description:
  // Methods for adding module-specific key bindings and
  // removing them.
  virtual void CreateModuleEventBindings ( );
  virtual void ReleaseModuleEventBindings ( );

  // Description:
  // Add obsereves to GUI widgets
  virtual void AddGUIObservers ( );
  
  // Description:
  // Remove obsereves to GUI widgets
  virtual void RemoveGUIObservers ( );
  virtual void RemoveMRMLNodeObservers ( );
  virtual void RemoveLogicObservers ( );
  

  // Description:
  // Process events generated by GUI widgets
  virtual void ProcessGUIEvents ( vtkObject *caller, unsigned long event,
                                  void *callData );
  
  // Description:
  // Process events generated by MRML
  virtual void ProcessMRMLEvents ( vtkObject *caller, unsigned long event,
                                   void *callData);
  
    // Description:
    // Get/Set the main slicer viewer widget, for picking
    vtkGetObjectMacro(ViewerWidget, vtkSlicerViewerWidget);
    virtual void SetViewerWidget(vtkSlicerViewerWidget *viewerWidget);

    // Description:
    // Get/Set the slicer interactorstyle, for picking
    vtkGetObjectMacro(InteractorStyle, vtkSlicerViewerInteractorStyle);
    virtual void SetInteractorStyle(vtkSlicerViewerInteractorStyle *interactorStyle);

  // Description:
  // Methods describe behavior at module enter and exit.
  virtual void Enter ( );
  virtual void Exit ( );
  
  
  void PrintSelf(ostream& os, vtkIndent indent);
  

  void ScheduleRender();
  bool RenderScheduled;

 protected:
  vtkVolumeRenderingCudaModuleGUI();
  virtual ~vtkVolumeRenderingCudaModuleGUI();


  void UpdateVolume();
  vtkVolumeRenderingCudaModuleGUI(const vtkVolumeRenderingCudaModuleGUI&); // not implemented
  void operator=(const vtkVolumeRenderingCudaModuleGUI&); // not implemented

  // Description:
  // Pointer to the module's logic class
  vtkVolumeRenderingCudaModuleLogic *Logic;

  vtkVolumeCudaMapper* CudaMapper;
  vtkVolume* CudaVolume;
  vtkVolumeProperty* CudaVolumeProperty;
  
  vtkSlicerViewerWidget *ViewerWidget;
  vtkSlicerViewerInteractorStyle *InteractorStyle;


  vtkKWTypeChooserBox*   InputTypeChooser;
  vtkKWMatrixWidget* InputResolutionMatrix;

  vtkKWMenuButton* RenderModeChooser;
  vtkKWVolumePropertyWidget* VolumePropertyWidget;
  vtkKWMatrixWidget* Color;

  vtkKWRange* ThresholdRange;

  vtkKWHistogramSet* Histograms;

  vtkKWPushButton* UpdateButton;
};

#endif /*VTKSLICERVOLUMERENDERINGCUDA_H_*/
