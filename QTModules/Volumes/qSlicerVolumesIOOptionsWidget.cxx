/// Qt includes
#include <QDebug>

/// Volumes includes
#include "qSlicerVolumesIOOptionsWidget.h"
#include "ui_qSlicerVolumesIOOptionsWidget.h"

//-----------------------------------------------------------------------------
class qSlicerVolumesIOOptionsWidgetPrivate: 
  public qCTKPrivate<qSlicerVolumesIOOptionsWidget>,
  public Ui_qSlicerVolumesIOOptionsWidget
{
  QCTK_DECLARE_PUBLIC(qSlicerVolumesIOOptionsWidget);
public:
  //void init();
};


//-----------------------------------------------------------------------------
qSlicerVolumesIOOptionsWidget::qSlicerVolumesIOOptionsWidget(QWidget* parentWidget)
{
  QCTK_INIT_PRIVATE(qSlicerVolumesIOOptionsWidget);
  qctk_d()->setupUi(this);
  // FIXME Replace the horizontal layout with a flow layout
}

//-----------------------------------------------------------------------------
qSlicerIO::IOProperties qSlicerVolumesIOOptionsWidget::options()const
{
  QCTK_D(const qSlicerVolumesIOOptionsWidget);
  qSlicerIO::IOProperties properties;
  if (d->NameLineEdit->text().isEmpty())
    {
    properties["name"] = d->NameLineEdit->text();
    }
  properties["labelmap"] = d->LabelMapCheckBox->isChecked();
  properties["center"] = d->CenteredCheckBox->isChecked();
  properties["singleFile"] = d->SingleFileCheckBox->isChecked();
  properties["discardOrientation"] = d->OrientationCheckBox->isChecked();
  return properties;
}
