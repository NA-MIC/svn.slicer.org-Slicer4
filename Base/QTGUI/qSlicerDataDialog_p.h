#ifndef __qSlicerDataDialog_p_h
#define __qSlicerDataDialog_p_h

/// Qt includes
#include <QDialog>
#include <QDir>
#include <QFileInfo>

/// qCTK includes
#include <qCTKPimpl.h>

// qSlicer includes
#include "qSlicerDataDialog.h"
#include "ui_qSlicerDataDialog.h"
#include "qSlicerBaseQTGUIExport.h"

//-----------------------------------------------------------------------------
class Q_SLICER_BASE_QTGUI_EXPORT qSlicerDataDialogPrivate
  : public QDialog
  , public qCTKPrivate<qSlicerDataDialog>
  , public Ui_qSlicerDataDialog
{
  Q_OBJECT
  QCTK_DECLARE_PUBLIC(qSlicerDataDialog);
public:
  explicit qSlicerDataDialogPrivate(QWidget* _parent=0);
  virtual ~qSlicerDataDialogPrivate();
                                     
  QList<qSlicerIO::IOProperties> selectedFiles();
public slots:
  void addDirectory();
  void addFiles();
  void reset();

protected:
  enum ColumnType
  {
    FileColumn = 0,
    TypeColumn = 1,
    OptionsColumn = 2
  };
  void addDirectory(const QDir& directory);
  void addFile(const QFileInfo& file);
};


#endif
