/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright 2005 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

#ifndef __qMRMLChartViewControllerWidget_p_h
#define __qMRMLChartViewControllerWidget_p_h

// qMRML includes
#include "qMRMLChartViewControllerWidget.h"
#include "qMRMLViewControllerBar_p.h"
#include "ui_qMRMLChartViewControllerWidget.h"

// VTK includes
#include <vtkWeakPointer.h>

class QAction;
class qMRMLSceneViewMenu;
class vtkMRMLChartViewNode;
class vtkMRMLChartNode;
class QString;

//-----------------------------------------------------------------------------
class qMRMLChartViewControllerWidgetPrivate
  : public qMRMLViewControllerBarPrivate
  , public Ui_qMRMLChartViewControllerWidget
{
  Q_OBJECT
  QVTK_OBJECT
  Q_DECLARE_PUBLIC(qMRMLChartViewControllerWidget);

public:
  typedef qMRMLViewControllerBarPrivate Superclass;
  qMRMLChartViewControllerWidgetPrivate(qMRMLChartViewControllerWidget& object);
  virtual ~qMRMLChartViewControllerWidgetPrivate();

  virtual void init();

  vtkWeakPointer<vtkMRMLChartViewNode>  ChartViewNode;
  qMRMLChartView*                       ChartView;

  QString                          ChartViewLabel;

  vtkMRMLChartNode* chartNode();
  
public slots:
  /// Called after a chart node is selected
  /// using the associated qMRMLNodeComboBox
  void onChartNodeSelected(vtkMRMLNode* node);
  
  /// Called after a chart type is selected using the qComboBox
  void onChartTypeSelected(const QString&);

protected:
  virtual void setupPopupUi();

public:

};

#endif
