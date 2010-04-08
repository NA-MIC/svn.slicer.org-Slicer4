#ifndef __qSlicerDataDialog_h
#define __qSlicerDataDialog_h

/// qCTK includes
#include <qCTKPimpl.h>

// qSlicer includes
#include "qSlicerFileDialog.h"
#include "qSlicerBaseQTGUIExport.h"

/// QT declarations
class qSlicerDataDialogPrivate;

//------------------------------------------------------------------------------
class Q_SLICER_BASE_QTGUI_EXPORT qSlicerDataDialog : public qSlicerFileDialog
{
  Q_OBJECT
public:
  typedef QObject Superclass;
  qSlicerDataDialog(QObject* parent =0);
  virtual ~qSlicerDataDialog();
  
  virtual qSlicerIO::IOFileType fileType()const;

  ///
  /// run the dialog to select the file/files/directory
  virtual bool exec(const qSlicerIO::IOProperties& readerProperties =
                    qSlicerIO::IOProperties());

private:
  QCTK_DECLARE_PRIVATE(qSlicerDataDialog);
};

#endif
