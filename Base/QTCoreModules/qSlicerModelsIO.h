#ifndef __qSlicerModelsIO
#define __qSlicerModelsIO

/// qCTK includes
#include <qCTKPimpl.h>

/// qSlicer includes
#include "qSlicerIO.h"

//-----------------------------------------------------------------------------
class qSlicerModelsIO: public qSlicerIO
{
  Q_OBJECT
public: 
  qSlicerModelsIO(QObject* parent = 0);
  virtual QString description()const;
  virtual IOFileType fileType()const;
  virtual QString extensions()const;

  virtual bool load(const IOProperties& properties);
};

#endif
