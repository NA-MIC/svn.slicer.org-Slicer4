/// Qt includes
#include <QDebug>
#include <QCheckBox>
#include <QFileDialog>
#include <QList>
#include <QUrl>

/// qCTK includes

/// qSlicer includes
#include "qSlicerApplication.h"
#include "qSlicerDataDialog.h"
#include "qSlicerDataDialog_p.h"
#include "qSlicerIOManager.h"

//-----------------------------------------------------------------------------
qSlicerDataDialogPrivate::qSlicerDataDialogPrivate(QWidget* _parent)
  :QDialog(_parent)
{
  this->setupUi(this);
  
  connect(this->AddDirectoryButton, SIGNAL(clicked()), this, SLOT(addDirectory()));
  connect(this->AddFilesButton, SIGNAL(clicked()), this, SLOT(addFiles()));
  QPushButton* resetButton = this->ButtonBox->button(QDialogButtonBox::Reset);
  connect(resetButton, SIGNAL(clicked()), this, SLOT(reset()));

}

//-----------------------------------------------------------------------------
qSlicerDataDialogPrivate::~qSlicerDataDialogPrivate()
{
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::addDirectory()
{
  QString directory = QFileDialog::getExistingDirectory(this);
  this->addDirectory(directory);
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::addFiles()
{
  QStringList files = QFileDialog::getOpenFileNames(this);
  foreach(QString file, files)
    {
    this->addFile(file);
    }
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::addDirectory(const QDir& directory)
{
  bool recursive = true;
  QDir::Filters filters = 
    QDir::AllDirs | QDir::Files | QDir::Readable | QDir::NoDotAndDotDot;
  foreach(QFileInfo entry, directory.entryInfoList(filters))
    {
    if (entry.isFile())
      {
      this->addFile(entry);
      }
    else if (entry.isDir() && recursive)
      {
      this->addDirectory(entry.absoluteFilePath());
      }
    }
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::addFile(const QFileInfo& file)
{
  if (!file.isFile() || !file.exists() || !file.isReadable())
    {
    return;
    }
  if (!this->FileWidget->findItems(file.absoluteFilePath(), 
                                   Qt::MatchExactly).isEmpty())
    {// file already exists
    qDebug() <<"already exists";
    return;
    }
  // qSlicerIO::IOFileType fileType = 
  //   qSlicerCoreApplication::application()->coreIOManager()->fileType(
  //     file.absoluteFilePath());
  QString fileDescription = 
    qSlicerCoreApplication::application()->coreIOManager()->fileDescription(
      file.absoluteFilePath());
  if (fileDescription == tr("Unknown"))
    {
    return;
    }
  qDebug() << "file Type: " << fileDescription;
  int row = this->FileWidget->rowCount();
  this->FileWidget->setRowCount(row + 1);
  QTableWidgetItem *fileItem = new QTableWidgetItem(file.absoluteFilePath());
  fileItem->setFlags( (fileItem->flags() | Qt::ItemIsUserCheckable) & ~Qt::ItemIsEditable);
  fileItem->setCheckState(Qt::Checked);
  this->FileWidget->setItem(row, FileColumn, fileItem);
  QTableWidgetItem *descriptionItem = new QTableWidgetItem(fileDescription);
  descriptionItem->setFlags( descriptionItem->flags() & ~Qt::ItemIsEditable);
  this->FileWidget->setItem(row, TypeColumn, descriptionItem);

  // update columns the first time
  if(this->FileWidget->rowCount() == 1)
    {
    this->FileWidget->resizeColumnsToContents();
    }
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::reset()
{
  this->FileWidget->setRowCount(0);
}

//-----------------------------------------------------------------------------
QList<qSlicerIO::IOProperties> qSlicerDataDialogPrivate::selectedFiles()
{
  QList<qSlicerIO::IOProperties> files;
  for (int row = 0; row < this->FileWidget->rowCount(); ++row)
    {
    qSlicerIO::IOProperties properties;
    QTableWidgetItem* fileItem = this->FileWidget->item(row, FileColumn);
    QTableWidgetItem* typeItem = this->FileWidget->item(row, TypeColumn);
    if (fileItem->checkState() != Qt::Checked)
      {
      qDebug() << "unchecked" ;
      continue;
      }
    properties["fileName"] = fileItem->text();
    properties["fileType"] = typeItem->text().toInt();
    files << properties;
    }
  return files;
}

//-----------------------------------------------------------------------------
qSlicerDataDialog::qSlicerDataDialog(QObject* _parent)
  :qSlicerFileDialog(_parent)
{
  // FIXME give qSlicerDataDialog as a parent of qSlicerDataDialogPrivate;
  QCTK_INIT_PRIVATE(qSlicerDataDialog);
}

//-----------------------------------------------------------------------------
qSlicerDataDialog::~qSlicerDataDialog()
{
}

//-----------------------------------------------------------------------------
qSlicerIO::IOFileType qSlicerDataDialog::fileType()const
{
  // FIXME: not really a scene file, but more a collection of files
  return qSlicerIO::NoFile;
}

//-----------------------------------------------------------------------------
bool qSlicerDataDialog::exec(const qSlicerIO::IOProperties& readerProperties)
{
  QCTK_D(qSlicerDataDialog);
  Q_ASSERT(!readerProperties.contains("fileName"));
#ifdef Slicer3_USE_KWWIDGETS
  d->setWindowFlags(d->windowFlags() | Qt::WindowStaysOnTopHint);
#endif
  d->exec();
  bool res = false;
  QList<qSlicerIO::IOProperties> files = d->selectedFiles();
  foreach(qSlicerIO::IOProperties properties, files)
    {
    properties.unite(readerProperties);
    res = qSlicerCoreApplication::application()->coreIOManager()
      ->loadNodes(properties["fileType"].toInt(),
                  properties) || res;
    }
  d->reset();
  return res;
}

