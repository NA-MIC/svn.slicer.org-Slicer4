#include "VolumeRenderingModuleLib.h"

#include "vtkVolumeRenderingModuleLogic.h"
#include "vtkVolumeRenderingModuleGUI.h"

char*
GetLoadableModuleDescription()
{
  return LoadableModuleDescription;
}

void*
GetLoadableModuleGUI()
{
  return vtkVolumeRenderingModuleGUI::New ( );
}


void*
GetLoadableModuleLogic()
{
  return vtkVolumeRenderingModuleLogic::New ( );
}
