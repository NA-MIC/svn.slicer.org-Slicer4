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

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qSlicerModuleFactoryFilterModel_h
#define __qSlicerModuleFactoryFilterModel_h

// Qt includes
#include <QSortFilterProxyModel>
#include <QStringList>

// QtGUI includes
#include "qSlicerBaseQTGUIExport.h"
class qSlicerModuleFactoryFilterModelPrivate;

class Q_SLICER_BASE_QTGUI_EXPORT qSlicerModuleFactoryFilterModel
  : public QSortFilterProxyModel
{
  Q_OBJECT
  /// True by default
  Q_PROPERTY(bool showToLoad READ showToLoad WRITE setShowToLoad)
  /// True by default
  Q_PROPERTY(bool showToIgnore READ showToIgnore WRITE setShowToIgnore)
  /// True by default
  Q_PROPERTY(bool showLoaded READ showLoaded WRITE setShowLoaded)
  /// True by default
  Q_PROPERTY(bool showIgnored READ showIgnored WRITE setShowIgnored)
  /// True by default
  Q_PROPERTY(bool showFailed READ showFailed WRITE setShowFailed)

  /// Don't use in conjunction of setFilter*()
  /// Empty by default
  Q_PROPERTY(QStringList showModules READ showModules WRITE setShowModules NOTIFY showModulesChanged)
  /// false by default
  Q_PROPERTY(bool hideAllWhenShowModulesIsEmpty READ hideAllWhenShowModulesIsEmpty WRITE setHideAllWhenShowModulesIsEmpty)
public:
  /// Superclass typedef
  typedef QSortFilterProxyModel Superclass;

  /// Constructor
  explicit qSlicerModuleFactoryFilterModel(QObject* parent = 0);

  /// Destructor
  virtual ~qSlicerModuleFactoryFilterModel();

  bool showToLoad()const;
  bool showToIgnore()const;
  bool showLoaded()const;
  bool showIgnored()const;
  bool showFailed()const;

  QStringList showModules()const;

  bool hideAllWhenShowModulesIsEmpty()const;
  void setHideAllWhenShowModulesIsEmpty(bool hide);
  virtual Qt::DropActions supportedDropActions()const;
  virtual bool dropMimeData(const QMimeData *data, Qt::DropAction action,
                            int row, int column, const QModelIndex &parent);

public slots:
  void setShowToLoad(bool show);
  void setShowToIgnore(bool show);
  void setShowLoaded(bool show);
  void setShowIgnored(bool show);
  void setShowFailed(bool show);

  void setShowModules(const QStringList& modules);

signals:
  void showModulesChanged(const QStringList&)const;
protected:
  QScopedPointer<qSlicerModuleFactoryFilterModelPrivate> d_ptr;

  virtual bool lessThan(const QModelIndex& leftIndex,
                        const QModelIndex& rightIndex)const;
  virtual bool filterAcceptsRow(int source_row, const QModelIndex& source_parent)const;

private:
  Q_DECLARE_PRIVATE(qSlicerModuleFactoryFilterModel);
  Q_DISABLE_COPY(qSlicerModuleFactoryFilterModel);
};

#endif

