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
  
  This file was originally developed by Johan Andruejol, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1
  
==============================================================================*/

#ifndef __qMRMLLabelComboBox_h
#define __qMRMLLabelComboBox_h

// CTK includes
#include <ctkPimpl.h>
#include <ctkVTKObject.h>

// qMRML includes
#include "qMRMLWidgetsExport.h"
#include "qMRMLWidget.h"

class vtkMRMLNode;
class vtkMRMLColorNode;
class qMRMLLabelComboBoxPrivate;

class QMRML_WIDGETS_EXPORT qMRMLLabelComboBox : public qMRMLWidget
{
  Q_OBJECT
  QVTK_OBJECT
  Q_PROPERTY(bool noneEnabled READ noneEnabled WRITE setNoneEnabled)
  Q_PROPERTY(int currentColor READ currentColor WRITE setCurrentColor USER true)
  Q_PROPERTY(int maximumColorCount READ maximumColorCount WRITE setMaximumColorCount)
  Q_PROPERTY(bool colorNameVisible READ colorNameVisible WRITE setColorNameVisible)
  Q_PROPERTY(bool labelValueVisible READ labelValueVisible WRITE setLabelValueVisible)

public:

  typedef qMRMLWidget Superclass;

  /// Construct an empty qMRMLColorTableComboBox with a null scene,
  /// no nodeType, where the hidden nodes are not forced on display.
  explicit qMRMLLabelComboBox(QWidget* newParent = 0);
  virtual ~qMRMLLabelComboBox();

  /// Set/Get NoneEnabled flags
  /// An additional item is added into the menu list, where the user can select "None".
  bool noneEnabled()const;
  void setNoneEnabled(bool enable);

  ///Display or not the colors names
  bool colorNameVisible() const;
  void setColorNameVisible(bool visible);

  ///Display or not the label values
  bool labelValueVisible() const;
  void setLabelValueVisible(bool visible);

  virtual void printAdditionalInfo();

  vtkMRMLColorNode* mrmlColorNode()const;

  int currentColor()const;

  int maximumColorCount()const;
  void setMaximumColorCount(int maximum);

public slots:

  void setMRMLColorNode(vtkMRMLNode * newMRMLColorNode);

  void setCurrentColor(int index);

  void updateWidgetFromMRML();
  
signals:

  void currentColorChanged(const QColor& color);
  void currentColorChanged(const QString& name);
  void currentColorChanged(int index);

private slots:

  void onCurrentIndexChanged(int index);

protected:
  QScopedPointer<qMRMLLabelComboBoxPrivate> d_ptr;
private:
  Q_DECLARE_PRIVATE(qMRMLLabelComboBox);
  Q_DISABLE_COPY(qMRMLLabelComboBox);
};

#endif
