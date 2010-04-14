#ifndef __qSlicerVolumesIOOptionsWidget_h
#define __qSlicerVolumesIOOptionsWidget_h

/// qCTK includes
#include <qCTKPimpl.h>

/// SlicerQT includes
#include "qSlicerIOOptionsWidget.h"

/// Volumes includes
#include "qSlicerVolumesModuleExport.h"

class qSlicerVolumesIOOptionsWidgetPrivate;

class Q_SLICER_QTMODULES_VOLUMES_EXPORT qSlicerVolumesIOOptionsWidget :
  public qSlicerIOOptionsWidget
{
  Q_OBJECT
public:
  qSlicerVolumesIOOptionsWidget(QWidget *parent=0);

  virtual qSlicerIO::IOProperties options()const;
  
private:
  QCTK_DECLARE_PRIVATE(qSlicerVolumesIOOptionsWidget);
};

#endif
