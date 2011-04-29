/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

  See Doc/copyright/copyright.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qMRMLSceneViewsModel_h
#define __qMRMLSceneViewsModel_h

#include "qMRMLSceneHierarchyModel.h"

#include "qSlicerSceneViewsModuleExport.h"

class qMRMLSceneViewsModelPrivate;
class vtkMRMLNode;
/// \ingroup Slicer_QtModules_SceneViews
class Q_SLICER_QTMODULES_SCENEVIEWS_EXPORT qMRMLSceneViewsModel : public qMRMLSceneModel
    //public qMRMLSceneHierarchyModel
{
  Q_OBJECT

public:
  qMRMLSceneViewsModel(QObject *parent=0);
  virtual ~qMRMLSceneViewsModel();

  // Enum for the different columns
  enum Columns{
    DummyColumn = 0,
    ThumbnailColumn = 1,
    RestoreColumn = 2,
    NameColumn = 3,
    DescriptionColumn = 4
  };

  virtual void updateItemDataFromNode(QStandardItem* item, vtkMRMLNode* node, int column);

protected:

  virtual void updateNodeFromItemData(vtkMRMLNode* node, QStandardItem* item);
  
  virtual QFlags<Qt::ItemFlag> nodeFlags(vtkMRMLNode* node, int column)const;

private:
  Q_DISABLE_COPY(qMRMLSceneViewsModel);


};

#endif
