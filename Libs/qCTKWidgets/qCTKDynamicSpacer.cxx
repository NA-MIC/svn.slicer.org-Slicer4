/*=========================================================================

  Library:   qCTK

  Copyright (c) Kitware Inc. 
  All rights reserved.
  Distributed under a BSD License. See LICENSE.txt file.

  This software is distributed "AS IS" WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the above copyright notice for more information.

=========================================================================*/

#include "qCTKDynamicSpacer.h"

// -----------------------------------------------------------------------------
class qCTKDynamicSpacerPrivate : public ctkPrivate<qCTKDynamicSpacer>
{
  CTK_DECLARE_PUBLIC(qCTKDynamicSpacer);
public:
  void init();

  QSizePolicy ActiveSizePolicy;
  QSizePolicy InactiveSizePolicy;
};

// -----------------------------------------------------------------------------
void qCTKDynamicSpacerPrivate::init()
{
  CTK_P(qCTKDynamicSpacer);
  this->ActiveSizePolicy = p->sizePolicy();
  this->InactiveSizePolicy = p->sizePolicy();
}

// -----------------------------------------------------------------------------
qCTKDynamicSpacer::qCTKDynamicSpacer(QWidget *_parent)
  :QWidget(_parent)
{
  CTK_INIT_PRIVATE(qCTKDynamicSpacer);
  ctk_d()->init();
}

// -----------------------------------------------------------------------------
qCTKDynamicSpacer::~qCTKDynamicSpacer()
{
}

// -----------------------------------------------------------------------------
QSizePolicy qCTKDynamicSpacer::activeSizePolicy() const
{
  CTK_D(const qCTKDynamicSpacer);
  return d->ActiveSizePolicy;
}

// -----------------------------------------------------------------------------
void qCTKDynamicSpacer::setActiveSizePolicy(QSizePolicy newActiveSizePolicy)
{
  CTK_D(qCTKDynamicSpacer);
  d->ActiveSizePolicy = newActiveSizePolicy;
}

// -----------------------------------------------------------------------------
QSizePolicy qCTKDynamicSpacer::inactiveSizePolicy() const
{
  CTK_D(const qCTKDynamicSpacer);
  return d->InactiveSizePolicy;
}

// -----------------------------------------------------------------------------
void qCTKDynamicSpacer::setInactiveSizePolicy(QSizePolicy newInactiveSizePolicy)
{
  CTK_D(qCTKDynamicSpacer);
  d->InactiveSizePolicy = newInactiveSizePolicy;
}

// -----------------------------------------------------------------------------
void qCTKDynamicSpacer::activate(bool enableSizePolicy)
{
  CTK_D(qCTKDynamicSpacer);
  this->setSizePolicy(
    enableSizePolicy ? d->ActiveSizePolicy : d->InactiveSizePolicy);
}
