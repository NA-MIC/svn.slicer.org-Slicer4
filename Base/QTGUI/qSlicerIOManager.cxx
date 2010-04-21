/// Qt includes
#include <QDebug>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QMap>
#include <QString>
#include <QUrl>

/// SlicerQt includes
#include "qSlicerIOManager.h"
#include "qSlicerFileDialog.h"
#include "qSlicerDataDialog.h"

/// MRML includes
#include <vtkMRMLScene.h>

//-----------------------------------------------------------------------------
class qSlicerIOManagerPrivate: public ctkPrivate<qSlicerIOManager>
{
public:
  CTK_DECLARE_PUBLIC(qSlicerIOManager);
  void init();
  QStringList                   History;
  QList<QUrl>                   Favorites;
  QMap<int, qSlicerFileDialog*> Dialogs;
};

//-----------------------------------------------------------------------------
void qSlicerIOManagerPrivate::init()
{
  CTK_P(qSlicerIOManager);
  this->Favorites << QUrl::fromLocalFile(QDir::homePath());
  p->registerDialog(new qSlicerStandardFileDialog(p));
  p->registerDialog(new qSlicerDataDialog(p));
}

//-----------------------------------------------------------------------------
qSlicerIOManager::qSlicerIOManager(QObject* _parent):Superclass(_parent)
{
  CTK_INIT_PRIVATE(qSlicerIOManager);
  ctk_d()->init();
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
  CTK_D(qSlicerIOManager);
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
  CTK_D(qSlicerIOManager);
  d->History << path;
}

//-----------------------------------------------------------------------------
const QStringList& qSlicerIOManager::history()const
{
  CTK_D(const qSlicerIOManager);
  return d->History;
}

//-----------------------------------------------------------------------------
void qSlicerIOManager::addFavorite(const QUrl& url)
{
  CTK_D(qSlicerIOManager);
  d->Favorites << url;
}

//-----------------------------------------------------------------------------
const QList<QUrl>& qSlicerIOManager::favorites()const
{
  CTK_D(const qSlicerIOManager);
  return d->Favorites;
}

//-----------------------------------------------------------------------------
void qSlicerIOManager::registerDialog(qSlicerFileDialog* dialog)
{
  CTK_D(qSlicerIOManager);
  if (d->Dialogs[dialog->fileType()])
    {
    delete d->Dialogs[dialog->fileType()];
    }
  d->Dialogs[dialog->fileType()] = dialog;
  dialog->setParent(this);
}
