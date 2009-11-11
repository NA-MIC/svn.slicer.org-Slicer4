#ifndef __qMRMLNodeSelector_h
#define __qMRMLNodeSelector_h

#include "qCTKAddRemoveComboBox.h"
#include "qVTKObject.h"

#include <QString>

#include "qMRMLWidgetsWin32Header.h"

class vtkMRMLScene; 
class vtkMRMLNode; 

class QMRML_WIDGETS_EXPORT qMRMLNodeSelector : public qCTKAddRemoveComboBox
{
  Q_OBJECT
  QVTK_OBJECT
  Q_PROPERTY(QString NodeType READ nodeType WRITE setNodeType)
  
public:
  // Superclass typedef
  typedef qCTKAddRemoveComboBox Superclass;
  
  // Constructors
  qMRMLNodeSelector(QWidget* parent = 0);
  virtual ~qMRMLNodeSelector();
  
  // Description:
  // Set/Get node type 
  QString nodeType()const ; 
  void setNodeType(const QString& nodeType); 
  
  // Description:
  // Set/Get MRML scene
  vtkMRMLScene* mrmlScene()const;

  // Description:
  // Return the node currently selected
  vtkMRMLNode* currentNode()const;

  // Description:
  // Add a node in the combobox
  void addNode(vtkMRMLNode* node);

public slots:
  // Description:
  // Set the scene the NodeSelector listens to.
  void setMRMLScene(vtkMRMLScene* scene);
  
  // Description:
  // Set the selected node.
  void setCurrentNode(vtkMRMLNode* node);

signals:
  // Description:
  // emit the current displayed node. NULL if
  // the list is empty.
  void currentNodeChanged(vtkMRMLNode* node);

  // Descritpion:
  // Utility function emitted at the same time(right after)
  // then currentNodeChanged(vtkMRMLNode*) signal is emitted
  // emit true when the current node is changed.
  // false when the list is empty. Useful to 
  // enable/disable/show/hide other widgets
  // depending on the validity of the current node.
  void currentNodeChanged(bool);

  // Description:
  // Emit when a node has been added to the list
  void nodeAdded(vtkMRMLNode* node);

  // Description:
  // Emit when a node is about to be removed from a scene.
  // The node can still be found in the mrml scene.
  void nodeAboutToBeRemoved(vtkMRMLNode* node);

  // Description:
  // Emit when a node has been removed from the scene and
  // the list. Warning, the node can't be found in the mrml
  // scene anymore.
  void nodeRemoved(vtkMRMLNode* node);

protected slots:
  virtual void onAdd();
  virtual void onRemove();
  virtual void onEdit();
  
  // Description:
  // Triggered upon MRML scene updates
  void onMRMLSceneNodeAdded(vtkObject * node); 
  void onMRMLSceneNodeRemoved(vtkObject * node); 
  void onMRMLNodeModified(void* call_data, vtkObject * caller);
  
  void onCurrentIndexChanged(int index);
  void onItemAdded(int index);
  void onItemAboutToBeRemoved(int index);
  void onItemRemoved(int index);
  
protected:
  virtual void nodeIdSelected(int index);
  vtkMRMLNode* node(int index)const;
  vtkMRMLNode* node(const QString& id)const;

private:
  struct qInternal; 
  qInternal * Internal;
};

#endif
