#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#ifdef WIN32
#define Module_EXPORT __declspec(dllexport)
#else
#define Module_EXPORT 
#endif

extern "C" {
  Module_EXPORT char* GetLoadableModuleDescription();
  Module_EXPORT void* GetLoadableModuleGUI();
  Module_EXPORT void* GetLoadableModuleLogic();
}

extern "C" {
  Module_EXPORT char LoadableModuleDescription[] =
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
"<loadable>\n"
"  <name>Volumes</name>\n"
"  <message>Initializing Volumes Module...</message>\n"
"</loadable>";
}
