//Qt includes  
#include <QAbstractButton>
#include <QDebug>
#include <QWeakPointer>

// CTK includes
#include "qCTKButtonGroup.h"

//-----------------------------------------------------------------------------
class qCTKButtonGroupPrivate : public ctkPrivate<qCTKButtonGroup>
{
public:
  CTK_DECLARE_PUBLIC(qCTKButtonGroup);
  bool IsLastButtonPressedChecked;
};

//------------------------------------------------------------------------------
qCTKButtonGroup::qCTKButtonGroup(QObject* _parent)
  :QButtonGroup(_parent)
{
  CTK_INIT_PRIVATE(qCTKButtonGroup);
  connect(this, SIGNAL(buttonClicked(QAbstractButton*)), this, SLOT(onButtonClicked(QAbstractButton*)));
  connect(this, SIGNAL(buttonPressed(QAbstractButton*)), this, SLOT(onButtonPressed(QAbstractButton*)));
}

//------------------------------------------------------------------------------
void qCTKButtonGroup::onButtonClicked(QAbstractButton *clickedButton)
{
  CTK_D(qCTKButtonGroup);
  if (!this->exclusive() || !d->IsLastButtonPressedChecked)
    {
    return;
    }
  this->removeButton(clickedButton);
  clickedButton->setChecked(false);
  this->addButton(clickedButton);
}

//------------------------------------------------------------------------------
void qCTKButtonGroup::onButtonPressed(QAbstractButton *pressedButton)
{
  CTK_D(qCTKButtonGroup);
  Q_ASSERT(pressedButton);
  d->IsLastButtonPressedChecked = pressedButton->isChecked();
}


