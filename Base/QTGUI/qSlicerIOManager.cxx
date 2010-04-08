/// Qt includes
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QMap>
#include <QString>
#include <QUrl>

/// qSlicer includes
#include "qSlicerIOManager.h"
#include "qSlicerFileDialog.h"
#include "qSlicerDataDialog.h"

/// MRML includes
#include <vtkMRMLScene.h>

//-----------------------------------------------------------------------------
class qSlicerIOManagerPrivate: public qCTKPrivate<qSlicerIOManager>
{
public:
  QCTK_DECLARE_PUBLIC(qSlicerIOManager);
  void init();
  QStringList                   History;
  QList<QUrl>                   Favorites;
  QMap<int, qSlicerFileDialog*> Dialogs;
};

//-----------------------------------------------------------------------------
void qSlicerIOManagerPrivate::init()
{
  QCTK_P(qSlicerIOManager);
  this->Favorites << QUrl::fromLocalFile(QDir::homePath());
  p->registerDialog(new qSlicerStandardFileDialog(p));
  p->registerDialog(new qSlicerDataDialog(p));
}

//-----------------------------------------------------------------------------
qSlicerIOManager::qSlicerIOManager(QObject* _parent):Superclass(_parent)
{
  QCTK_INIT_PRIVATE(qSlicerIOManager);
  qctk_d()->init();
}

//-----------------------------------------------------------------------------
qSlicerIOManager::~qSlicerIOManager()
{
}

//-----------------------------------------------------------------------------
bool qSlicerIOManager::openLoadSceneDialog()
{
  qSlicerIO::IOProperties properties;
  properties["clear"] = true;
  return this->openDialog(qSlicerIO::SceneFile, properties);
}

//-----------------------------------------------------------------------------
bool qSlicerIOManager::openImportSceneDialog()
{
  qSlicerIO::IOProperties properties;
  properties["clear"] = false;
  return this->openDialog(qSlicerIO::SceneFile, properties);
}

//-----------------------------------------------------------------------------
bool qSlicerIOManager::openLoadVolumeDialog()
{
  return this->openDialog(qSlicerIO::VolumeFile);
}

//-----------------------------------------------------------------------------
bool qSlicerIOManager::openDialog(qSlicerIO::IOFileType fileType, 
                                  const qSlicerIO::IOProperties& properties)
{
  QCTK_D(qSlicerIOManager);
  bool deleteDialog = false;
  qSlicerFileDialog* dialog = d->Dialogs[fileType];
  if (dialog == 0)
    {
    deleteDialog = true;
    qSlicerStandardFileDialog* standardDialog = 
      new qSlicerStandardFileDialog(this);
    standardDialog->setFileType(fileType);
    dialog = standardDialog;
    }
  bool res = dialog->exec(properties);
  if (deleteDialog)
   {
    delete dialog;
    }
  return res;
}

//-----------------------------------------------------------------------------
void qSlicerIOManager::addHistory(const QString& path)
{
  QCTK_D(qSlicerIOManager);
  d->History << path;
}

//-----------------------------------------------------------------------------
const QStringList& qSlicerIOManager::history()const
{
  QCTK_D(const qSlicerIOManager);
  return d->History;
}

//-----------------------------------------------------------------------------
void qSlicerIOManager::addFavorite(const QUrl& url)
{
  QCTK_D(qSlicerIOManager);
  d->Favorites << url;
}

//-----------------------------------------------------------------------------
const QList<QUrl>& qSlicerIOManager::favorites()const
{
  QCTK_D(const qSlicerIOManager);
  return d->Favorites;
}

//-----------------------------------------------------------------------------
void qSlicerIOManager::registerDialog(qSlicerFileDialog* dialog)
{
  QCTK_D(qSlicerIOManager);
  if (d->Dialogs[dialog->fileType()])
    {
    delete d->Dialogs[dialog->fileType()];
    }
  d->Dialogs[dialog->fileType()] = dialog;
  dialog->setParent(this);
}
