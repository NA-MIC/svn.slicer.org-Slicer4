/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qMRMLScreenShotDialog_h
#define __qMRMLScreenShotDialog_h

// Qt includes
#include <QDialog>
#include <QMetaType>
#include <QString>

// qMRMLWidget includes
#include "qMRMLWidgetsExport.h"

class qMRMLLayoutManager;
class vtkImageData;

class qMRMLScreenShotDialogPrivate;

class QMRML_WIDGETS_EXPORT qMRMLScreenShotDialog : public QDialog
{
  Q_OBJECT
  Q_ENUMS(WidgetType)
  Q_PROPERTY(WidgetType widgetType READ widgetType WRITE setWidgetType)
public:
  typedef QDialog Superclass;

  enum WidgetType{
    ThreeD = 0,
    Red = 1,
    Yellow = 2,
    Green = 3,
    FullLayout = 4
    };

  qMRMLScreenShotDialog(QWidget *parent = 0);
  ~qMRMLScreenShotDialog();

  void setLayoutManager(qMRMLLayoutManager* newlayoutManager);
  qMRMLLayoutManager* layoutManager()const;

  void setNameEdit(const QString& newName);
  QString nameEdit()const;

  void setDescription(const QString& description);
  QString description()const;

  /// Setting the data prevent the dialog from automatically taking a screenshot
  /// each time the widgettype or scalefactor is changed.
  void setData(const QVariant& newData);
  QVariant data()const;

  void setWidgetType(WidgetType newType);
  WidgetType widgetType()const;

  void setScaleFactor(const double& newScaleFactor);
  double scaleFactor()const;

  void setShowScaleFactorSpinBox(const bool& state);
  bool showScaleFactorSpinBox()const;

  /// set/return the image data of the screenshot
  void setImageData(vtkImageData* screenshot);
  vtkImageData* imageData()const;

  /// Reset the dialog
  void resetDialog();

  /// Grab a screenshot of the 3DView or any sliceView.
  /// The screenshotWindow is Red, Green, Yellow for a sliceView or empty for
  /// a ThreeDView
  void grabScreenShot(int screenshotWindow);

protected slots:

  void grabScreenShot();

private:
  QString enumToString(int type);

protected:
  QScopedPointer<qMRMLScreenShotDialogPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qMRMLScreenShotDialog);
  Q_DISABLE_COPY(qMRMLScreenShotDialog);
};

Q_DECLARE_METATYPE(qMRMLScreenShotDialog::WidgetType)

#endif
