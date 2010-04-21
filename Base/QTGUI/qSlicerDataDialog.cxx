/// Qt includes
#include <QDebug>
#include <QCheckBox>
#include <QFileDialog>
#include <QList>
#include <QUrl>

/// CTK includes
#include <qCTKCheckableHeaderView.h>

/// SlicerQt includes
#include "qSlicerApplication.h"
#include "qSlicerDataDialog.h"
#include "qSlicerDataDialog_p.h"
#include "qSlicerIOManager.h"
#include "qSlicerIOOptionsWidget.h"

//-----------------------------------------------------------------------------
qSlicerDataDialogPrivate::qSlicerDataDialogPrivate(QWidget* _parent)
  :QDialog(_parent)
{
  this->setupUi(this);
  // checkable headers.
  this->FileWidget->model()->setHeaderData(FileColumn, Qt::Horizontal, Qt::Unchecked, Qt::CheckStateRole);
  QHeaderView* previousHeaderView = this->FileWidget->horizontalHeader();
  qCTKCheckableHeaderView* headerView = new qCTKCheckableHeaderView(Qt::Horizontal, this->FileWidget);
  headerView->setClickable(previousHeaderView->isClickable());
  headerView->setMovable(previousHeaderView->isMovable());
  headerView->setHighlightSections(previousHeaderView->highlightSections());
  //headerView->setModel(previousHeaderView->model());
  //headerView->setSelectionModel(previousHeaderView->selectionModel());
  headerView->setPropagateToItems(true);
  this->FileWidget->setHorizontalHeader(headerView);
  /*
  connect(this->FileWidget->model(), SIGNAL(headerDataChanged(Qt::Orientation, int, int)),
          this, SLOT(updateCheckBoxes(Qt::Orientation, int, int)));
  connect(this->FileWidget, SIGNAL(cellChanged(int,int)),
          this, SLOT(updateCheckBoxHeader(int,int)));
  */
  
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
  if (directory.isNull())
    {
    return;
    }
  bool sortingEnabled = this->FileWidget->isSortingEnabled();
  this->FileWidget->setSortingEnabled(false);
  this->addDirectory(directory);
  this->FileWidget->setSortingEnabled(sortingEnabled);
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

  bool sortingEnabled = this->FileWidget->isSortingEnabled();
  this->FileWidget->setSortingEnabled(false);
  int row = this->FileWidget->rowCount();
  this->FileWidget->insertRow(row);
  // File name
  QTableWidgetItem *fileItem = new QTableWidgetItem(file.absoluteFilePath());
  fileItem->setFlags( (fileItem->flags() | Qt::ItemIsUserCheckable) & ~Qt::ItemIsEditable);
  fileItem->setCheckState(Qt::Checked);
  this->FileWidget->setItem(row, FileColumn, fileItem);
  // Description
  QTableWidgetItem *descriptionItem = new QTableWidgetItem();
  descriptionItem->setFlags(descriptionItem->flags() & ~Qt::ItemIsEditable);
  descriptionItem->setText(fileDescription);
  this->FileWidget->setItem(row, TypeColumn, descriptionItem);
  // Options
  qSlicerIOOptionsWidget* optionsWidget = dynamic_cast<qSlicerIOOptionsWidget*>(
    qSlicerCoreApplication::application()->coreIOManager()->fileOptions(
      file.absoluteFilePath()));
  if (optionsWidget)
    {
    this->FileWidget->setCellWidget(row, OptionsColumn, optionsWidget);
    }
  this->FileWidget->setSortingEnabled(sortingEnabled);
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
    qDebug() << "row: " << row;
    qSlicerIO::IOProperties properties;
    QTableWidgetItem* fileItem = this->FileWidget->item(row, FileColumn);
    QTableWidgetItem* typeItem = this->FileWidget->item(row, TypeColumn);
    qSlicerIOOptionsWidget* optionsItem = dynamic_cast<qSlicerIOOptionsWidget*>(
      this->FileWidget->cellWidget(row, OptionsColumn));
    Q_ASSERT(fileItem);
    Q_ASSERT(typeItem);
    if (fileItem->checkState() != Qt::Checked)
      {
      qDebug() << "unchecked" ;
      continue;
      }
    properties["fileName"] = fileItem->text();
    properties["fileType"] = typeItem->text().toInt();
    if (optionsItem)
      {
      properties.unite(optionsItem->options());
      }
    files << properties;
    }
  return files;
}
/*
//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::updateCheckBoxes(Qt::Orientation orientation, int first, int last)
{
  if (orientation != Qt::Horizontal ||
      first > FileColumn || last < FileColumn)
    {
    return;
    }
  Qt::CheckState headerState = this->FileWidget->horizontalHeaderItem(FileColumn)->checkState();
  if (headerState == Qt::PartiallyChecked)
    {
    return;
    }
  for (int row = 0; row < this->FileWidget->rowCount(); ++row)
    {
    this->FileWidget->item(row, FileColumn)->setCheckState(headerState);
    }
}

//-----------------------------------------------------------------------------
void qSlicerDataDialogPrivate::updateCheckBoxHeader(int itemRow, int itemColumn)
{
  if (itemColumn != FileColumn)
    {
    return;
    }
  Qt::CheckState headerCheckState = this->FileWidget->item(itemRow,itemColumn)->checkState();
  for (int row = 0; row < this->FileWidget->rowCount(); ++row)
    {    
    if (this->FileWidget->item(row, FileColumn)->checkState() !=
        headerCheckState)
      {
      this->FileWidget->model()->setHeaderData(FileColumn, Qt::Horizontal, Qt::PartiallyChecked, Qt::CheckStateRole);
      return;
      }
    }
  this->FileWidget->model()->setHeaderData(FileColumn, Qt::Horizontal, headerCheckState, Qt::CheckStateRole);
}*/

//-----------------------------------------------------------------------------
qSlicerDataDialog::qSlicerDataDialog(QObject* _parent)
  :qSlicerFileDialog(_parent)
{
  // FIXME give qSlicerDataDialog as a parent of qSlicerDataDialogPrivate;
  CTK_INIT_PRIVATE(qSlicerDataDialog);
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
  CTK_D(qSlicerDataDialog);
  Q_ASSERT(!readerProperties.contains("fileName"));
#ifdef Slicer3_USE_KWWIDGETS
  d->setWindowFlags(d->windowFlags() | Qt::WindowStaysOnTopHint);
#endif
  bool res = false;
  if (d->exec() != QDialog::Accepted)
    {
    d->reset();
    return res;
    }
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

