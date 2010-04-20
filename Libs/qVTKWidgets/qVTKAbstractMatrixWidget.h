#ifndef __qVTKAbstractMatrixWidget_h
#define __qVTKAbstractMatrixWidget_h

/// CTK includes
#include <ctkMatrixWidget.h>
#include <qCTKPimpl.h>

/// qVTK includes
#include "qVTKObject.h"
#include "qVTKWidgetsExport.h"
 
class vtkMatrix4x4;
class qVTKAbstractMatrixWidgetPrivate;

class QVTK_WIDGETS_EXPORT qVTKAbstractMatrixWidget : public ctkMatrixWidget
{
public:
  /// Self/Superclass typedef
  typedef ctkMatrixWidget   Superclass;
  
  /// Constructors
  qVTKAbstractMatrixWidget(QWidget* parent);
  vtkMatrix4x4* matrix()const;

protected:
  void setMatrixInternal(vtkMatrix4x4* matrix);

private:
  QCTK_DECLARE_PRIVATE(qVTKAbstractMatrixWidget);
}; 

#endif
