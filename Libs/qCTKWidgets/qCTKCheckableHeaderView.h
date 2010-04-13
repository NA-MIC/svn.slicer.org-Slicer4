/*=========================================================================

   Program: ParaView
   Module:    $RCSfile: pqCheckableHeaderView.h,v $

   Copyright (c) 2005-2008 Sandia Corporation, Kitware Inc.
   All rights reserved.

   ParaView is a free software; you can redistribute it and/or modify it
   under the terms of the ParaView license version 1.2. 

   See License_v1.2.txt for the full ParaView license.
   A copy of this license can be obtained by contacting
   Kitware Inc.
   28 Corporate Drive
   Clifton Park, NY 12065
   USA

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/

#ifndef _qCTKCheckableHeaderView_h
#define _qCTKCheckableHeaderView_h

/// Qt includes
#include <QHeaderView>

/// qCTK includes
#include "qCTKPimpl.h"
#include "qCTKWidgetsExport.h"
class qCTKCheckableHeaderViewPrivate;

class QCTK_WIDGETS_EXPORT qCTKCheckableHeaderView : public QHeaderView
{
  Q_OBJECT;
  Q_PROPERTY(bool propagateToItems READ propagateToItems WRITE setPropagateToItems);
public:
  qCTKCheckableHeaderView(Qt::Orientation orient, QWidget *parent=0);
  virtual ~qCTKCheckableHeaderView();

  virtual void setModel(QAbstractItemModel *model);
  virtual void setRootIndex(const QModelIndex &index);

  /// 
  /// A section is checkable if its CheckStateRole data is non null. 
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole).isEmpty()
  bool isCheckable(int section)const;

  ///
  /// Utility function that returns the checkState of the section. 
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole)
  Qt::CheckState checkState(int section)const;

  ///
  /// Utility function that returns the checkState of the section. 
  /// One can access the same value through the model:
  /// model->headerData(orientation, section, Qt::CheckStateRole)
  bool checkState(int section,Qt::CheckState& checkState )const;

  ///
  ///  Used to listen for focus in/out events.
  /// \param object The object receiving the event.
  /// \param e Event specific data.
  /// \return
  ///   True if the event should be filtered out.
  virtual bool eventFilter(QObject *object, QEvent *e);
                                                      
  void setPropagateToItems(bool propagate);
  bool propagateToItems()const;

public slots:
  ///
  /// if the check state is PartiallyChecked, the section becomes Checked
  void toggleCheckState(int section);
  ///
  /// Warning, setting the check state automatically set the 
  /// header section checkable
  void setCheckState(int section, Qt::CheckState checkState);

private slots:
  void updateHeaderData(Qt::Orientation orient, int first, int last);
  void insertHeaderSection(const QModelIndex &parent, int first, int last);
  inline void updateHeaders();
  void updateHeadersFromItems(const QModelIndex& topLeft, const QModelIndex& bottomRight);

protected:
  virtual void updateHeaders(int first, int last);
  virtual void initStyleSectionOption(QStyleOptionHeader *option, int section, QRect rect)const;
  virtual void mousePressEvent(QMouseEvent *e);
  virtual void mouseReleaseEvent(QMouseEvent *e);
  bool isPointInCheckBox(int section, QPoint pos)const;

private:
  QCTK_DECLARE_PRIVATE(qCTKCheckableHeaderView);
};

//-----------------------------------------------------------------------------
void qCTKCheckableHeaderView::updateHeaders()
{
  this->updateHeaders(0, this->count()-1);
}

#endif
