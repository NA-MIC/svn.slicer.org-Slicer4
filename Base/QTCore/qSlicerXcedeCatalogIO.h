#ifndef __qSlicerXcedeCatalogIO
#define __qSlicerXcedeCatalogIO

/// qCTK includes
#include <qCTKPimpl.h>

/// qSlicer includes
#include "qSlicerIO.h"

class qSlicerXcedeCatalogIOPrivate;

//-----------------------------------------------------------------------------
class qSlicerXcedeCatalogIO: public qSlicerIO
{
  Q_OBJECT
public: 
  qSlicerXcedeCatalogIO(QObject* parent = 0);
  virtual QString description()const;
  virtual IOFileType fileType()const;
  virtual QString extensions()const;

  virtual bool load(const IOProperties& properties);
private:
  QCTK_DECLARE_PRIVATE(qSlicerXcedeCatalogIO);
};

#endif
