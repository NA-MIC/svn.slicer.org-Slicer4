/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qSlicerScriptedLoadableModule_h
#define __qSlicerScriptedLoadableModule_h

// Qt includes
#include <QVariantMap>

// SlicerQt includes
#include "qSlicerLoadableModule.h"

#include "qSlicerBaseQTGUIExport.h"

class qSlicerScriptedLoadableModulePrivate;

class Q_SLICER_BASE_QTGUI_EXPORT qSlicerScriptedLoadableModule : public qSlicerLoadableModule
{
  Q_OBJECT
  Q_PROPERTY(QString title READ title WRITE setTitle)
  /// Reimplemented to expose the setter setCategories
  Q_PROPERTY(QStringList categories READ categories WRITE setCategories)
  Q_PROPERTY(QStringList contributors READ contributors WRITE setContributors)
  Q_PROPERTY(QString helpText READ helpText WRITE setHelpText)
  Q_PROPERTY(QString acknowledgementText READ acknowledgementText WRITE setAcknowledgementText)
  Q_PROPERTY(QVariantMap extensions READ extensions WRITE setExtensions)
  Q_PROPERTY(QIcon icon READ icon WRITE setIcon)
  Q_PROPERTY(bool hidden READ isHidden WRITE setHidden)
  Q_PROPERTY(QStringList dependencies READ dependencies WRITE setDependencies)
  Q_PROPERTY(int index READ index WRITE setIndex)

public:

  typedef qSlicerLoadableModule Superclass;
  qSlicerScriptedLoadableModule(QObject *parent=0);
  virtual ~qSlicerScriptedLoadableModule();

  bool setPythonSource(const QString& newPythonSource);

  /// Convenient method allowing to retrieve the associated scripted instance
  /// Q_INVOKABLE PyObject* pythonInstance() const;

  virtual QString title()const ;
  void setTitle(const QString& newTitle);

  virtual QStringList categories()const;
  void setCategories(const QStringList& newCategories);

  virtual QStringList contributors()const;
  void setContributors(const QStringList& newContributors);

  virtual QString helpText()const;
  void setHelpText(const QString& newHelpText);

  virtual QString acknowledgementText()const;
  void setAcknowledgementText(const QString& newAcknowledgementText);

  virtual QVariantMap extensions()const;
  void setExtensions(const QVariantMap& extensions);

  virtual QIcon icon()const;
  void setIcon(const QIcon& newIcon);

  virtual QStringList dependencies() const;
  void setDependencies(const QStringList& dependencies);

  virtual int index() const;
  void setIndex(const int index);

  /// Enable/Disable hide state of the module
  /// Needs to be hidden before the module menu is created.
  virtual bool isHidden()const;
  void setHidden(bool hidden);

protected:

  virtual void setup();

  virtual qSlicerAbstractModuleRepresentation* createWidgetRepresentation();

  virtual vtkMRMLAbstractLogic* createLogic();

protected:
  QScopedPointer<qSlicerScriptedLoadableModulePrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerScriptedLoadableModule);
  Q_DISABLE_COPY(qSlicerScriptedLoadableModule);
};

#endif
