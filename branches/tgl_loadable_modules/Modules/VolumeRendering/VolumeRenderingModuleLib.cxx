#include "VolumeRenderingModuleLib.h"

#include "vtkVolumeRenderingModuleLogic.h"
#include "vtkVolumeRenderingModuleGUI.h"

static char VolumeRenderingDescription[] =
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<loadable>\n"
"  <name>Volume Rendering Module</name>\n"
"  <shortname>VRModule</shortname>\n"
"  <guiname>VolumeRendering</guiname>\n"
"  <tclinitname>Volumerenderingmodule_Init</tclinitname>\n"
"  <message>Initializing Volume Rendering Module...</message>\n"
"</loadable>";

char*
GetLoadableModuleDescription()
{
  return VolumeRenderingDescription;
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
