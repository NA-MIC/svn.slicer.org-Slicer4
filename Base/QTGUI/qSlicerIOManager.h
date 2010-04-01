#ifndef __qSlicerIOManager_h
#define __qSlicerIOManager_h

/// Qt includes
#include <QList>
#include <QString>
#include <QUrl>

/// qCTK includes
#include <qCTKPimpl.h>

/// SlicerQT includes
#include "qSlicerCoreIOManager.h"
#include "qSlicerFileDialog.h"

#include "qSlicerBaseQTGUIExport.h"

/// QT declarations
class QWidget;

class qSlicerIOManagerPrivate;

class Q_SLICER_BASE_QTGUI_EXPORT qSlicerIOManager : public qSlicerCoreIOManager
{
  Q_OBJECT;
public:
  typedef qSlicerCoreIOManager Superclass;
  qSlicerIOManager(QObject* parent = 0);
  virtual ~qSlicerIOManager();

  bool openLoadSceneDialog();
  bool openImportSceneDialog();
  bool openLoadVolumeDialog();

  bool openDialog(qSlicerIO::IOFileType fileType,
                  const qSlicerIO::IOProperties& readerProperties
                    = qSlicerIO::IOProperties());

  void addHistory(const QString& path);
  const QStringList& history()const;

  void addFavorite(const QUrl& urls);
  const QList<QUrl>& favorites()const;

  ///
  /// Takes ownership and delete the dialog 
  /// when not needed anymore, it overrides the previously set dialog
  /// corresponding to the same fileType (only 1 dialog per filetype).
  void registerDialog(qSlicerFileDialog* dialog);

protected:
  friend class qSlicerFileDialog;
  using qSlicerCoreIOManager::ios;
private:
  QCTK_DECLARE_PRIVATE(qSlicerIOManager);
};


#endif
