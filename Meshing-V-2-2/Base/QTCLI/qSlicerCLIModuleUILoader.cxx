#include "qSlicerCLIModuleUILoader.h" 

#include "qSlicerMacros.h"

#include <QDebug>

//-----------------------------------------------------------------------------
struct qSlicerCLIModuleUILoader::qInternal
{
  qInternal()
    {
    }
};

//-----------------------------------------------------------------------------
qSlicerCxxInternalConstructorMacro(qSlicerCLIModuleUILoader);
qSlicerCxxDestructorMacro(qSlicerCLIModuleUILoader);

//-----------------------------------------------------------------------------
void qSlicerCLIModuleUILoader::printAdditionalInfo()
{
  //this->Superclass::dumpObjectInfo(); 
}
