#ifndef __qSlicerVolumesIO
#define __qSlicerVolumesIO

/// qCTK includes
#include <qCTKPimpl.h>

/// qSlicer includes
#include "qSlicerIO.h"

//-----------------------------------------------------------------------------
class qSlicerVolumesIO: public qSlicerIO
{
  Q_OBJECT
public: 
  qSlicerVolumesIO(QObject* parent = 0);
  virtual QString description()const;
  virtual IOFileType fileType()const;
  virtual QString extensions()const;

  virtual bool load(const IOProperties& properties);
};

#endif
