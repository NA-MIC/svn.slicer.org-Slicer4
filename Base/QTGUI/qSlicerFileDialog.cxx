/// Qt includes
#include <QDebug>
#include <QFileDialog>
#include <QList>
#include <QUrl>

/// qCTK includes

/// qSlicer includes
#include "qSlicerApplication.h"
#include "qSlicerFileDialog.h"
#include "qSlicerIOManager.h"
/*
//-----------------------------------------------------------------------------
class qSlicerFileDialogPrivate: public qCTKPrivate<qSlicerFileDialog>
{
public:
  QCTK_DECLARE_PUBLIC(qSlicerFileDialog);
};
*/

//-----------------------------------------------------------------------------
qSlicerFileDialog::qSlicerFileDialog(QObject* _parent)
  :QObject(_parent)
{
  //QCTK_INIT_PRIVATE(qSlicerFileDialog);
}

//-----------------------------------------------------------------------------
qSlicerFileDialog::~qSlicerFileDialog()
{
}

//-----------------------------------------------------------------------------
QStringList qSlicerFileDialog::nameFilters(qSlicerIO::IOFileType fileType)
{
  QStringList filters;
  QStringList extensions;
  QList<qSlicerIO*> readers = 
    qSlicerApplication::application()->ioManager()->ios(fileType);
  foreach(const qSlicerIO* reader, readers)
    {
    QString nameFilter= reader->description() + " (" + reader->extensions() + ")";
    filters << nameFilter;
    extensions << reader->extensions();
    }
  filters.insert(0, QString("All (") + extensions.join(" ") + QString(")"));
  return filters;
}

//-----------------------------------------------------------------------------
class qSlicerStandardFileDialogPrivate: public qCTKPrivate<qSlicerStandardFileDialog>
{
public:
  QCTK_DECLARE_PUBLIC(qSlicerStandardFileDialog);
  qSlicerStandardFileDialogPrivate();
  qSlicerIO::IOFileType   FileType;
};

//-----------------------------------------------------------------------------
qSlicerStandardFileDialogPrivate::qSlicerStandardFileDialogPrivate()
{
  this->FileType = qSlicerIO::NoFile;
}

//-----------------------------------------------------------------------------
qSlicerStandardFileDialog::qSlicerStandardFileDialog(QObject* _parent)
  :qSlicerFileDialog(_parent)
{
  QCTK_INIT_PRIVATE(qSlicerStandardFileDialog);
}

//-----------------------------------------------------------------------------
void qSlicerStandardFileDialog::setFileType(qSlicerIO::IOFileType _fileType)
{
  QCTK_D(qSlicerStandardFileDialog);
  d->FileType = _fileType;
}

//-----------------------------------------------------------------------------
qSlicerIO::IOFileType qSlicerStandardFileDialog::fileType()const
{
  QCTK_D(const qSlicerStandardFileDialog);
  return d->FileType;
}

//-----------------------------------------------------------------------------
bool qSlicerStandardFileDialog::exec(const qSlicerIO::IOProperties& readerProperties)
{
  Q_ASSERT(!readerProperties.contains("fileName"));
  qSlicerIOManager* ioManager = qSlicerApplication::application()->ioManager();
  QFileDialog fileDialog(qobject_cast<QWidget*>(this->parent()));
  fileDialog.setNameFilters(
    qSlicerFileDialog::nameFilters(this->fileType()));
  fileDialog.setHistory(ioManager->history());
  if (ioManager->favorites().count())
    {
    fileDialog.setSidebarUrls(ioManager->favorites());
    }
  bool res = fileDialog.exec();
  if (res)
    {
    qSlicerIO::IOProperties properties = readerProperties;
    properties["fileName"] = fileDialog.selectedFiles()[0];
    ioManager->loadNodes(this->fileType(), properties);
    }
  return res;
}
