#include "qSlicerCLIModule.h" 

// Libs/ModuleDescriptionParser headers
#include "ModuleDescription.h"
#include "ModuleDescriptionParser.h"

#include <QDebug>

//-----------------------------------------------------------------------------
class qSlicerCLIModule::qInternal
{
public:
  qInternal()
    {
    }
  QString           Title;
  QString           Acknowledgement;
  QString           Help;
};

//-----------------------------------------------------------------------------
qSlicerCxxInternalConstructor1Macro(qSlicerCLIModule, QWidget*);
qSlicerCxxDestructorMacro(qSlicerCLIModule);

//-----------------------------------------------------------------------------
qSlicerGetInternalCxxMacro(qSlicerCLIModule, QString, moduleTitle, Title);
qSlicerGetInternalCxxMacro(qSlicerCLIModule, QString, acknowledgementText, Acknowledgement);
qSlicerGetInternalCxxMacro(qSlicerCLIModule, QString, helpText, Help);

//-----------------------------------------------------------------------------
void qSlicerCLIModule::setXmlModuleDescription(const char* xmlModuleDescription)
{
  //qDebug() << "xmlModuleDescription:" << xmlModuleDescription; 
  
  // Parse module description
  ModuleDescription desc;
  ModuleDescriptionParser parser;
  if (parser.Parse(std::string(xmlModuleDescription), desc) != 0)
    {
    qWarning() << "Failed to parse xml module description";
    return;
    }
    
  // Set properties
  this->Internal->Title = QString::fromStdString(desc.GetTitle());
  this->Internal->Acknowledgement = QString::fromStdString(desc.GetAcknowledgements());
  
  QString help = 
    "%1<br>"
    "For more detailed documentation see:<br>"
    "%2";
  
  this->Internal->Help = help.arg(
    QString::fromStdString(desc.GetDescription())).arg(
    QString::fromStdString(desc.GetDocumentationURL())); 
}

//-----------------------------------------------------------------------------
void qSlicerCLIModule::initializer()
{
  
}

//-----------------------------------------------------------------------------
void qSlicerCLIModule::printAdditionalInfo()
{
  this->Superclass::printAdditionalInfo(); 
}
