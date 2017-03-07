/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Laboratory for Percutaneous Surgery (PerkLab)
  Queen's University, Kingston, ON, Canada. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Csaba Pinter, PerkLab, Queen's University
  and was supported through the Applied Cancer Research Unit program of Cancer Care
  Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care

==============================================================================*/

// SubjectHierarchy MRML includes
#include "vtkMRMLSubjectHierarchyNode.h"
#include "vtkMRMLSubjectHierarchyConstants.h"

// SubjectHierarchy Plugins includes
#include "qSlicerSubjectHierarchyPluginHandler.h"
#include "qSlicerSubjectHierarchyMarkupsPlugin.h"
#include "qSlicerSubjectHierarchyDefaultPlugin.h"

// MRML includes
#include <vtkMRMLNode.h>
#include <vtkMRMLScene.h>

// MRML widgets includes
#include "qMRMLNodeComboBox.h"

// VTK includes
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>

// Qt includes
#include <QDebug>
#include <QStandardItem>
#include <QAction>

// SlicerQt includes
#include "qSlicerAbstractModuleWidget.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_SubjectHierarchy_Plugins
class qSlicerSubjectHierarchyMarkupsPluginPrivate: public QObject
{
  Q_DECLARE_PUBLIC(qSlicerSubjectHierarchyMarkupsPlugin);
protected:
  qSlicerSubjectHierarchyMarkupsPlugin* const q_ptr;
public:
  qSlicerSubjectHierarchyMarkupsPluginPrivate(qSlicerSubjectHierarchyMarkupsPlugin& object);
  ~qSlicerSubjectHierarchyMarkupsPluginPrivate();
  void init();
public:
  QIcon MarkupIcon;
};

//-----------------------------------------------------------------------------
// qSlicerSubjectHierarchyMarkupsPluginPrivate methods

//-----------------------------------------------------------------------------
qSlicerSubjectHierarchyMarkupsPluginPrivate::qSlicerSubjectHierarchyMarkupsPluginPrivate(qSlicerSubjectHierarchyMarkupsPlugin& object)
: q_ptr(&object)
{
  this->MarkupIcon = QIcon(":Icons/Markup.png");
}

//-----------------------------------------------------------------------------
qSlicerSubjectHierarchyMarkupsPluginPrivate::~qSlicerSubjectHierarchyMarkupsPluginPrivate()
{
}

//-----------------------------------------------------------------------------
qSlicerSubjectHierarchyMarkupsPlugin::qSlicerSubjectHierarchyMarkupsPlugin(QObject* parent)
 : Superclass(parent)
 , d_ptr( new qSlicerSubjectHierarchyMarkupsPluginPrivate(*this) )
{
  this->m_Name = QString("Markups");

  Q_D(qSlicerSubjectHierarchyMarkupsPlugin);
  d->init();
}

//------------------------------------------------------------------------------
void qSlicerSubjectHierarchyMarkupsPluginPrivate::init()
{
}

//-----------------------------------------------------------------------------
qSlicerSubjectHierarchyMarkupsPlugin::~qSlicerSubjectHierarchyMarkupsPlugin()
{
}

//----------------------------------------------------------------------------
double qSlicerSubjectHierarchyMarkupsPlugin::canAddNodeToSubjectHierarchy(
  vtkMRMLNode* node, vtkIdType parentItemID/*=vtkMRMLSubjectHierarchyNode::INVALID_ITEM_ID*/)const
{
  Q_UNUSED(parentItemID);
  if (!node)
    {
    qCritical() << Q_FUNC_INFO << ": Input node is NULL!";
    return 0.0;
    }
  else if (node->IsA("vtkMRMLMarkupsFiducialNode"))
    {
    // Item is a markup fiducial
    return 0.5;
    }
  return 0.0;
}

//---------------------------------------------------------------------------
double qSlicerSubjectHierarchyMarkupsPlugin::canOwnSubjectHierarchyItem(vtkIdType itemID)const
{
  if (itemID == vtkMRMLSubjectHierarchyNode::INVALID_ITEM_ID)
    {
    qCritical() << Q_FUNC_INFO << ": Invalid input item";
    return 0.0;
    }
  vtkMRMLSubjectHierarchyNode* shNode = qSlicerSubjectHierarchyPluginHandler::instance()->subjectHierarchyNode();
  if (!shNode)
    {
    qCritical() << Q_FUNC_INFO << ": Failed to access subject hierarchy node";
    return 0.0;
    }

  // Markup
  vtkMRMLNode* associatedNode = shNode->GetItemDataNode(itemID);
  if (associatedNode && associatedNode->IsA("vtkMRMLMarkupsFiducialNode"))
    {
    // Item is a markup fiducial
    return 0.5;
    }

  return 0.0;
}

//---------------------------------------------------------------------------
const QString qSlicerSubjectHierarchyMarkupsPlugin::roleForPlugin()const
{
  return "Markup";
}

//---------------------------------------------------------------------------
QIcon qSlicerSubjectHierarchyMarkupsPlugin::icon(vtkIdType itemID)
{
  if (itemID == vtkMRMLSubjectHierarchyNode::INVALID_ITEM_ID)
    {
    qCritical() << Q_FUNC_INFO << ": Invalid input item";
    return QIcon();
    }

  Q_D(qSlicerSubjectHierarchyMarkupsPlugin);

  if (this->canOwnSubjectHierarchyItem(itemID))
    {
    return d->MarkupIcon;
    }

  // Item unknown by plugin
  return QIcon();
}

//---------------------------------------------------------------------------
QIcon qSlicerSubjectHierarchyMarkupsPlugin::visibilityIcon(int visible)
{
  // Have the default plugin (which is not registered) take care of this
  return qSlicerSubjectHierarchyPluginHandler::instance()->defaultPlugin()->visibilityIcon(visible);
}
