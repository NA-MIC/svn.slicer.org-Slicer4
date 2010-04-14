/*=auto=========================================================================

 Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) 
 All Rights Reserved.

 See Doc/copyright/copyright.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Program:   3D Slicer

=========================================================================auto=*/

#ifndef __qSlicerCoreIOManager_h
#define __qSlicerCoreIOManager_h

/// Qt includes
#include <QObject>
#include <QMap>

/// qCTK includes
#include <qCTKPimpl.h>

/// QtCore includes
#include <qSlicerIO.h>
#include "qSlicerBaseQTCoreExport.h"

class vtkMRMLNode;
class vtkMRMLScene; 
class vtkCollection;
class qSlicerCoreIOManagerPrivate;

class Q_SLICER_BASE_QTCORE_EXPORT qSlicerCoreIOManager:public QObject
{
  Q_OBJECT;
public:
  qSlicerCoreIOManager(QObject* parent = 0);
  virtual ~qSlicerCoreIOManager();

  /*
  /// 
  /// Load/Import scene
  void loadScene(vtkMRMLScene* mrmlScene, const QString& filename);
  void importScene(vtkMRMLScene* mrmlScene, const QString& filename);

  /// 
  /// Close scene
  void closeScene(vtkMRMLScene* mrmlScene);

  bool loadFile(const qSlicerIO::IOProperties& parameters);
  */
  qSlicerIO::IOFileType fileType(const QString& file)const;
  QString fileDescription(const QString& file)const;
  qSlicerIOOptions* fileOptions(const QString& file)const;
  
  ///
  /// attributes are typically: 
  /// All: fileName[s] 
  /// Volume: LabelMap:bool, Center:bool, fileNames:QList<QString>...
  bool loadNodes(qSlicerIO::IOFileType fileType, 
                 const qSlicerIO::IOProperties& parameters, 
                 vtkCollection* loadedNodes= 0);

  ///
  /// Utility function that return the first loaded node
  vtkMRMLNode* loadNode(qSlicerIO::IOFileType fileType, 
                        const qSlicerIO::IOProperties& parameters);

  /// 
  /// Utility function to load/import a scene
  bool loadScene(const QString& fileName, bool clear = true);

  ///
  ///
  void registerIO(qSlicerIO* io);
protected:
  const QList<qSlicerIO*>& ios()const;
  QList<qSlicerIO*> ios(qSlicerIO::IOFileType fileType)const;
private:
  QCTK_DECLARE_PRIVATE(qSlicerCoreIOManager);
};

#endif

