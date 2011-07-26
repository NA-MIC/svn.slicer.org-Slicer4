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

// Qt includes

// SlicerQt includes
#include "qSlicerEventBrokerModuleWidget.h"
#include "ui_qSlicerEventBrokerModule.h"

//-----------------------------------------------------------------------------
class qSlicerEventBrokerModuleWidgetPrivate: public Ui_qSlicerEventBrokerModule
{
public:
};

//-----------------------------------------------------------------------------
qSlicerEventBrokerModuleWidget::qSlicerEventBrokerModuleWidget(QWidget* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerEventBrokerModuleWidgetPrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerEventBrokerModuleWidget::~qSlicerEventBrokerModuleWidget()
{
}

//-----------------------------------------------------------------------------
void qSlicerEventBrokerModuleWidget::setup()
{
  Q_D(qSlicerEventBrokerModuleWidget);
  d->setupUi(this);
}
