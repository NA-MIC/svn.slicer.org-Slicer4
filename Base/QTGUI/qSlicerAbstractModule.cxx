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

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// Qt includes
#include <QAction>

// SlicerQt includes
#include "qSlicerAbstractModule.h"

class qSlicerAbstractModulePrivate
{
public:
  qSlicerAbstractModulePrivate();
  QAction* Action;
};

//-----------------------------------------------------------------------------
qSlicerAbstractModulePrivate::qSlicerAbstractModulePrivate()
{
  this->Action = 0;
}

//-----------------------------------------------------------------------------
qSlicerAbstractModule::qSlicerAbstractModule(QObject* parentObject)
  :Superclass(parentObject)
  , d_ptr(new qSlicerAbstractModulePrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerAbstractModule::~qSlicerAbstractModule()
{
}

//-----------------------------------------------------------------------------
QIcon qSlicerAbstractModule::icon()const
{
  return QIcon();
}
/*
//-----------------------------------------------------------------------------
QAction* qSlicerAbstractModule::createAction()
{
  QAction* action = new QAction(this->icon(), this->title(), this);
  action->setData(this->name());
  action->setIconVisibleInMenu(true);
  return action;
}
*/
//-----------------------------------------------------------------------------
QAction* qSlicerAbstractModule::action()
{
  Q_D(qSlicerAbstractModule);
  if (d->Action == 0)
    {
    d->Action = new QAction(this->icon(), this->title(), this);
    d->Action->setData(this->name());
    d->Action->setIconVisibleInMenu(true);
    d->Action->setProperty("index", this->index());
    }
  return d->Action;
}
