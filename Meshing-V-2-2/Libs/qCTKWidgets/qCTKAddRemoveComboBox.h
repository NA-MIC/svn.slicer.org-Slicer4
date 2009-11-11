#ifndef __qCTKAddRemoveComboBox_h
#define __qCTKAddRemoveComboBox_h

#include <QWidget>
#include <QVariant>
#include <QModelIndex>

#include "qCTKWidgetsWin32Header.h"

class QComboBox; 

class QCTK_WIDGETS_EXPORT qCTKAddRemoveComboBox : public QWidget
{
  Q_OBJECT
  Q_PROPERTY(QString emptyText READ emptyText WRITE setEmptyText)
  Q_PROPERTY(bool addEnabled READ addEnabled WRITE setAddEnabled)
  Q_PROPERTY(bool removeEnabled READ removeEnabled WRITE setRemoveEnabled)
  Q_PROPERTY(bool editEnabled READ editEnabled WRITE setEditEnabled)
  
public:
  // Superclass typedef
  typedef QWidget Superclass;
  
  // Constructors
  qCTKAddRemoveComboBox(QWidget* parent = 0);
  virtual ~qCTKAddRemoveComboBox();
  
  // Description:
  // Set text that should be displayed in the comboBox when it is empty
  void setEmptyText(const QString& text); 
  QString emptyText()const;
  
  // Description:
  // Enable/Disable the add button. 
  void setComboBoxEnabled(bool enable);
  bool comboBoxEnabled()const; 
  
  // Description:
  // Enable/Disable the add button. 
  void setAddEnabled(bool enable);
  bool addEnabled()const; 
  
  // Description:
  // Enable/Disable the add button. 
  void setRemoveEnabled(bool enable);
  bool removeEnabled()const; 
  
  // Description:
  // Enable/Disable the edit button. 
  void setEditEnabled(bool enable); 
  bool editEnabled()const;
  
  inline void addItem(const QString &text, const QVariant &userData = QVariant() )
    {this->insertItem(this->count(), text, userData);}
  inline void addItem(const QIcon &icon, const QString &text, const QVariant &userData = QVariant() )
    {this->insertItem(this->count(), icon, text, userData);}
  inline void addItems(const QStringList &texts )
    {this->insertItems(this->count(), texts);}
    
  void insertItem(int index, const QString &text, const QVariant &userData = QVariant() );
  void insertItem(int index, const QIcon &icon, const QString &text, const QVariant &userData = QVariant() );
  void insertItems(int index, const QStringList &texts);  
  
  // Description:
  // Return the number of item
  int count()const;
  bool empty()const;
    
  // Description:
  // Returns the index of the item containing the given text; otherwise returns -1.
  // The flags specify how the items in the combobox are searched.
  int findText(const QString& text, Qt::MatchFlags flags = Qt::MatchExactly | Qt::MatchCaseSensitive ) const;
  int findData(const QVariant & data, int role = Qt::UserRole, Qt::MatchFlags flags = Qt::MatchExactly | Qt::MatchCaseSensitive ) const;

  // Description:
  QString   itemText(int index) const;
  QVariant  itemData(int index, int role = Qt::UserRole) const;

  void setItemText(int index, const QString& text);
  void setItemData(int index, const QVariant& data, int role = Qt::UserRole);

  // Description:
  // Return the current item
  int       currentIndex() const;
  inline QString  currentText() const
    {return this->itemText(this->currentIndex());}
  inline QVariant currentData(int role = Qt::UserRole) const
    {return this->itemData(this->currentIndex(), role);}

  // Description:
  // Remove the item currently selected. See signal 'itemRemoved'
  void removeItem(int index);
  inline void removeCurrentItem()
    {this->removeItem(this->currentIndex());}

signals:
  void currentIndexChanged(int index);
  void activated(int index);

  // Description:
  // This signal is sent after the method 'addItem' has been called programmatically
  void itemAdded(int index);
  
  // Description:
  void itemAboutToBeRemoved(int index);
  void itemRemoved(int index);
    
public slots:
  // Description:
  // Select the current index
  void setCurrentIndex(int index);

protected slots:
  // Description:
  virtual void onAdd();
  virtual void onRemove();
  virtual void onEdit();
private slots:
  //void onRowsAboutToBeInserted(const QModelIndex & parent, int start, int end );
  void onRowsAboutToBeRemoved(const QModelIndex & parent, int start, int end);
  void onRowsInserted(const QModelIndex & parent, int start, int end);
  void onRowsRemoved(const QModelIndex & parent, int start, int end);

private:
  struct qInternal; 
  qInternal * Internal;

};

#endif
