/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/

// Qt includes
#include <QDebug>
#include <QFileInfo>
#include <QSettings>
#include <QString>
#include <QStringList>

// SlicerQt includes
#include "qSlicerAbstractModule.h"
#include "qSlicerCoreApplication.h"
#include "qSlicerCoreIOManager.h"
#include "qSlicerIO.h"
#include "qSlicerModuleManager.h"
#include "qSlicerSlicer2SceneReader.h"
#include "qSlicerXcedeCatalogIO.h"

// MRML includes
#include <vtkMRMLScene.h>

// VTK includes
#include <vtkSmartPointer.h>

//-----------------------------------------------------------------------------
class qSlicerSceneIO: public qSlicerIO
{
public: 
  qSlicerSceneIO(QObject* _parent = 0):qSlicerIO(_parent){}
  virtual QString description()const {return "MRML Scene";}
  virtual qSlicerIO::IOFileType fileType()const {return qSlicerIO::SceneFile;}
  virtual QString extensions()const {return "*.mrml";}
  virtual bool load(const qSlicerIO::IOProperties& properties);
};

//-----------------------------------------------------------------------------
bool qSlicerSceneIO::load(const qSlicerIO::IOProperties& properties)
{
  Q_ASSERT(properties.contains("fileName"));
  Q_ASSERT(properties.contains("clear"));
  QString file = properties["fileName"].toString();
  this->mrmlScene()->SetURL(file.toLatin1().data());
  bool clear = properties["clear"].toBool();
  int res = 0;
  if (clear)
    {
    res = this->mrmlScene()->Connect();
    }
  else
    {
    res = this->mrmlScene()->Import();
    }
  return res;
}

//-----------------------------------------------------------------------------
class qSlicerCoreIOManagerPrivate: public ctkPrivate<qSlicerCoreIOManager>
{
public:
  qSlicerCoreIOManagerPrivate();
  ~qSlicerCoreIOManagerPrivate();
  vtkMRMLScene* currentScene()const;
  
  qSlicerIO* reader(const QString& fileName)const;

  QSettings*        ExtensionFileType;
  QList<qSlicerIO*> Readers;
};

//-----------------------------------------------------------------------------
qSlicerCoreIOManagerPrivate::qSlicerCoreIOManagerPrivate()
{
}

//-----------------------------------------------------------------------------
qSlicerCoreIOManagerPrivate::~qSlicerCoreIOManagerPrivate()
{
}

//-----------------------------------------------------------------------------
vtkMRMLScene* qSlicerCoreIOManagerPrivate::currentScene()const
{
  return qSlicerCoreApplication::application()->mrmlScene();
}

//-----------------------------------------------------------------------------
qSlicerIO* qSlicerCoreIOManagerPrivate::reader(const QString& fileName)const
{
  QList<qSlicerIO*> genericReaders;
  foreach(qSlicerIO* reader, this->Readers)
    {
    if (!reader->canLoadFile(fileName))
      {
      continue;
      }
    if (reader->extensions() == "*.*")
      {
      genericReaders << reader;
      continue;
      }
    return reader;
    }
  foreach(qSlicerIO* reader, genericReaders)
    {
    return reader;
    }
  return 0;
}

//-----------------------------------------------------------------------------
qSlicerCoreIOManager::qSlicerCoreIOManager(QObject* _parent)
  :QObject(_parent)
{
  CTK_INIT_PRIVATE(qSlicerCoreIOManager);
  // FIXME move to the application level
  this->registerIO(new qSlicerSceneIO(this));
  this->registerIO(new qSlicerSlicer2SceneReader(this));
  this->registerIO(new qSlicerXcedeCatalogIO(this));
  // end FIXME
}

//-----------------------------------------------------------------------------
qSlicerCoreIOManager::~qSlicerCoreIOManager()
{
}

//-----------------------------------------------------------------------------
qSlicerIO::IOFileType qSlicerCoreIOManager::fileType(const QString& fileName)const
{
  CTK_D(const qSlicerCoreIOManager);
  qSlicerIO* reader = d->reader(fileName);
  if (!reader)
    {
    return  qSlicerIO::NoFile;
    }
  return reader->fileType();
}

//-----------------------------------------------------------------------------
QString qSlicerCoreIOManager::fileDescription(const QString& fileName)const
{
  CTK_D(const qSlicerCoreIOManager);
  qSlicerIO* reader = d->reader(fileName);
  if (!reader)
    {
    return tr("Unknown");
    }
  return reader->description();
}

//-----------------------------------------------------------------------------
qSlicerIOOptions* qSlicerCoreIOManager::fileOptions(const QString& fileName)const
{
  CTK_D(const qSlicerCoreIOManager);
  qSlicerIO* reader = d->reader(fileName);
  if (!reader)
    {
    return 0;
    }
  return reader->options();
}

//-----------------------------------------------------------------------------
bool qSlicerCoreIOManager::loadScene(const QString& fileName, bool clear)
{
  qSlicerIO::IOProperties properties;
  properties["fileName"] = fileName;
  properties["clear"] = clear;
  return this->loadNodes(qSlicerIO::SceneFile, properties);
}

/*
//-----------------------------------------------------------------------------
void qSlicerCoreIOManager::loadScene(vtkMRMLScene* mrmlScene, const QString& filename)
{
  Q_ASSERT(mrmlScene);
  
  // Convert to lowercase
  QString filenameLc = filename.toLower();
    
  if (filenameLc.endsWith(".mrml"))
    {
    mrmlScene->SetURL(filenameLc.toLatin1());
    mrmlScene->Connect();
    }
  else if (filenameLc.endsWith(".xml"))
    {
    qDebug() << "Loading Slicer2Scene... NOT implemented";
    // TODO See ImportSlicer2Scene script
    }
  else if (filenameLc.endsWith(".xcat"))
    {
    qDebug() << "Loading Catalog... NOT implemented";
    // TODO See XcatalogImport script
    }
  else
    {
    qWarning() << "Unknown type of scene file:" << filenameLc; 
    }
  // TODO save last open path
  //this->LoadSceneDialog->SaveLastPathToRegistry("OpenPath");

  if (mrmlScene->GetErrorCode() != 0 )
    {
    qDebug() << "Failed to load scene:" << QString::fromStdString(mrmlScene->GetErrorMessage());
    }
}

//-----------------------------------------------------------------------------
void qSlicerCoreIOManager::importScene(vtkMRMLScene* mrmlScene, const QString& filename)
{
  Q_ASSERT(mrmlScene);
  
  // Convert to lowercase
  QString filenameLc = filename.toLower();

  if (filenameLc.endsWith(".mrml"))
    {
    mrmlScene->SetURL(filenameLc.toLatin1());
    mrmlScene->Import();
    }
  else if (filenameLc.endsWith(".xml"))
    {
    qDebug() << "Importing Slicer2Scene... NOT implemented";
    // TODO See ImportSlicer2Scene script
    }
  else if (filenameLc.endsWith(".xcat"))
    {
    qDebug() << "Importing Catalog... NOT implemented";
    // TODO See XcatalogImport script
    }
  else
    {
    qWarning() << "Unknown type of scene file:" << filenameLc; 
    }
  // TODO save last open path
  //this->LoadSceneDialog->SaveLastPathToRegistry("OpenPath");

  if (mrmlScene->GetErrorCode() != 0 )
    {
    qDebug() << "Failed to load scene:" << QString::fromStdString(mrmlScene->GetErrorMessage());
    }
}

//-----------------------------------------------------------------------------
void qSlicerCoreIOManager::closeScene(vtkMRMLScene* mrmlScene)
{
  Q_ASSERT(mrmlScene);
  
  mrmlScene->Clear(false);
}
*/

//-----------------------------------------------------------------------------
vtkMRMLNode* qSlicerCoreIOManager::loadNode(qSlicerIO::IOFileType fileType, 
                                            const qSlicerIO::IOProperties& parameters)
{ 
  vtkSmartPointer<vtkCollection> loadedNodes = 
    vtkSmartPointer<vtkCollection>::New();
  this->loadNodes(fileType, parameters, loadedNodes);
  vtkMRMLNode* node = 
    vtkMRMLNode::SafeDownCast(loadedNodes->GetItemAsObject(0));
  Q_ASSERT(node);
  return node;
}

//-----------------------------------------------------------------------------
bool qSlicerCoreIOManager::loadNodes(qSlicerIO::IOFileType fileType, 
                                     const qSlicerIO::IOProperties& parameters, 
                                     vtkCollection* loadedNodes)
{ 
  CTK_D(qSlicerCoreIOManager);

  Q_ASSERT(parameters.contains("fileName"));

  QList<qSlicerIO*> readers = this->ios(fileType);

  QStringList nodes;
  foreach (qSlicerIO* reader, readers)
    {
    reader->setMRMLScene(d->currentScene());
    if (!reader->load(parameters))
      {
      continue;
      }
    qDebug() << "reader has read the file" << parameters["fileName"].toString();
    nodes << reader->loadedNodes();
    break;
    }

  if (nodes.count())
    {
    return false;
    }

  if (loadedNodes)
    {
    foreach(const QString& node, nodes)
      {
      loadedNodes->AddItem(
        d->currentScene()->GetNodeByID(node.toLatin1().data()));
      }
    }
  return true;
}

//-----------------------------------------------------------------------------
const QList<qSlicerIO*>& qSlicerCoreIOManager::ios()const
{
  CTK_D(const qSlicerCoreIOManager);
  return d->Readers;
}

//-----------------------------------------------------------------------------
QList<qSlicerIO*> qSlicerCoreIOManager::ios(qSlicerIO::IOFileType fileType)const
{
  CTK_D(const qSlicerCoreIOManager);
  QList<qSlicerIO*> res;
  foreach(qSlicerIO* io, d->Readers)
    {
    if (io->fileType() == fileType)
      {
      res << io;
      }
    }
  return res;
}


//-----------------------------------------------------------------------------
void qSlicerCoreIOManager::registerIO(qSlicerIO* io)
{
  CTK_D(qSlicerCoreIOManager);
  d->Readers << io;
}
