/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

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
#include <QDebug>
#include <QLabel>
#include <QListWidget>
#include <QPainter>
#include <QPaintEvent>
#include <QPushButton>
#include <QSignalMapper>
#include <QTextDocument>

// Slicer includes
#include "qSlicerExtensionsManagerModel.h"
#include "qSlicerExtensionsManageWidget.h"
#include "ui_qSlicerExtensionsButtonBox.h"

//-----------------------------------------------------------------------------
class qSlicerExtensionsManageWidgetPrivate
{
  Q_DECLARE_PUBLIC(qSlicerExtensionsManageWidget);
protected:
  qSlicerExtensionsManageWidget* const q_ptr;

public:
  typedef qSlicerExtensionsManageWidgetPrivate Self;
  typedef qSlicerExtensionsManagerModel::ExtensionMetadataType ExtensionMetadataType;


  enum DataRoles
    {
    IconRole = Qt::DecorationRole,
    NameRole = Qt::UserRole,
    };

  qSlicerExtensionsManageWidgetPrivate(qSlicerExtensionsManageWidget& object);
  void init();

  QListWidgetItem * extensionItem(const QString &extensionName) const;

  void addExtensionItem(const ExtensionMetadataType &metadata);

  QSignalMapper LabelLinkMapper;
  QSignalMapper EnableButtonMapper;
  QSignalMapper DisableButtonMapper;
  QSignalMapper ScheduleUninstallButtonMapper;
  QSignalMapper CancelScheduledUninstallButtonMapper;

  qSlicerExtensionsManagerModel * ExtensionsManagerModel;
};

// --------------------------------------------------------------------------
qSlicerExtensionsManageWidgetPrivate::qSlicerExtensionsManageWidgetPrivate(qSlicerExtensionsManageWidget& object)
  :q_ptr(&object)
{
  this->ExtensionsManagerModel = 0;
}

// --------------------------------------------------------------------------
namespace
{

// --------------------------------------------------------------------------
class qSlicerExtensionsButtonBox : public QWidget, public Ui_qSlicerExtensionsButtonBox
{
public:
  typedef QWidget Superclass;
  qSlicerExtensionsButtonBox(QWidget* parent = 0) : Superclass(parent)
  {
    this->setupUi(this);
  }
};

// --------------------------------------------------------------------------
QIcon extensionIcon(const QString& path, bool enabled)
{
  return  QIcon(QIcon(path).pixmap(QSize(64, 64), enabled ? QIcon::Normal : QIcon::Disabled));
}

} // end of anonymous namespace

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidgetPrivate::init()
{
  Q_Q(qSlicerExtensionsManageWidget);

  q->setAlternatingRowColors(true);
  q->setSelectionMode(QAbstractItemView::NoSelection);
  q->setIconSize(QSize(64, 64));
  q->setSpacing(1);

  QObject::connect(&this->EnableButtonMapper, SIGNAL(mapped(QString)),
                   q, SLOT(setExtensionEnabled(QString)));

  QObject::connect(&this->DisableButtonMapper, SIGNAL(mapped(QString)),
                   q, SLOT(setExtensionDisabled(QString)));

  QObject::connect(&this->ScheduleUninstallButtonMapper, SIGNAL(mapped(QString)),
                   q, SLOT(scheduleExtensionForUninstall(QString)));

  QObject::connect(&this->CancelScheduledUninstallButtonMapper, SIGNAL(mapped(QString)),
                   q, SLOT(cancelExtensionScheduledForUninstall(QString)));
}

// --------------------------------------------------------------------------
QListWidgetItem * qSlicerExtensionsManageWidgetPrivate::extensionItem(const QString& extensionName) const
{
  Q_Q(const qSlicerExtensionsManageWidget);

  QAbstractItemModel* model = q->model();
  const QModelIndexList indices =
    model->match(model->index(0, 0, QModelIndex()), Self::NameRole,
                 extensionName, 2, Qt::MatchExactly);

  Q_ASSERT(indices.count() < 2);
  if (indices.count() == 1)
    {
    return q->item(indices.first().row());
    }
  return 0;
}

namespace
{

// --------------------------------------------------------------------------
class qSlicerExtensionsDescriptionLabel : public QLabel
{
public:
  typedef QLabel Superclass;

  enum
  {
    NOWARNING = 0,
    INCOMPATIBLE
  };

  // --------------------------------------------------------------------------
  qSlicerExtensionsDescriptionLabel(const QString& extensionSlicerVersion, const QString& slicerRevision,
                                    const QString& extensionName, const QString& extensionDescription,
                                    int warningType = NOWARNING)
    : QLabel(), ExtensionSlicerVersion(extensionSlicerVersion), SlicerRevision(slicerRevision),
      ExtensionName(extensionName), ExtensionDescription(extensionDescription),
      WarningColor("#bd8530"), WarningType(warningType), LastWidth(0)
  {
    QTextOption textOption = this->Text.defaultTextOption();
    textOption.setWrapMode(QTextOption::NoWrap);
    this->Text.setDefaultTextOption(textOption);

    this->prepareText(extensionDescription); // Ensure reasonable initial height for size hint
    this->setToolTip(extensionDescription);
  }

  // --------------------------------------------------------------------------
  QString incompatibleExtensionText()
  {
    return QString("<p style=\"font-weight: bold; font-size: 80%; color: %2;\">"
                   "<img style=\"float: left\" src=\":/Icons/ExtensionIncompatible.svg\"/> "
                   "Incompatible with Slicer r%3 [built for r%4]</p>").
        arg(this->WarningColor, this->SlicerRevision, this->ExtensionSlicerVersion);
  }

  // --------------------------------------------------------------------------
  QString descriptionAsRichText(const QString& elidedDescription)
  {
    static const QString format = "%1<h2>%2</h2><p>%3%4</p>";

    QString linkText;
    QString warningText;
    if (this->WarningType == INCOMPATIBLE)
      {
      warningText = this->incompatibleExtensionText();
      }
    // linkText = QString(" <a href=\"slicer:%1\">More</a>").arg();
    return format.arg(warningText, this->ExtensionName, elidedDescription, linkText);
  }

  // --------------------------------------------------------------------------
  void prepareText(const QString& elidedDescription)
  {
    this->Text.setHtml(this->descriptionAsRichText(elidedDescription));
  }

  // --------------------------------------------------------------------------
  virtual QSize sizeHint() const
  {
    QSize hint = this->Superclass::sizeHint();
    hint.setHeight(qRound(this->Text.size().height() + 0.5) +
                   this->margin() * 2);
    return hint;
  }

  // --------------------------------------------------------------------------
  virtual void paintEvent(QPaintEvent * event)
  {
    QPainter painter(this);
    const QRect cr = this->contentsRect();

    if (this->LastWidth != cr.width())
      {
      int margin = this->margin() * 2;
      if (!this->ExtensionId.isEmpty())
        {
        margin += this->fontMetrics().width(" More");
        }
      this->prepareText(
        this->fontMetrics().elidedText(this->ExtensionDescription,
                                       Qt::ElideRight, cr.width() - margin));
      this->Text.setTextWidth(this->LastWidth = cr.width());
      }

    painter.translate(cr.topLeft());
    painter.setPen(this->palette().text().color());
    this->Text.drawContents(&painter);
  }

  QString ExtensionSlicerVersion;
  QString SlicerRevision;

  bool ExtensionIncompatible;
  QString ExtensionName;
  QString ExtensionDescription;
  QString WarningColor;
  int WarningType;

  QTextDocument Text;

  int LastWidth;
};

// --------------------------------------------------------------------------
class qSlicerExtensionsItemWidget : public QWidget
{
public:
  qSlicerExtensionsItemWidget(qSlicerExtensionsDescriptionLabel * label, QWidget* parent = 0)
    : QWidget(parent), Label(label)
  {
    QHBoxLayout * layout = new QHBoxLayout;
    layout->addWidget(label, 1);
    layout->addWidget(this->ButtonBox = new qSlicerExtensionsButtonBox);
    layout->setContentsMargins(0, 0, 0, 0);
    this->setLayout(layout);

    label->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
  }

  qSlicerExtensionsDescriptionLabel * Label;
  qSlicerExtensionsButtonBox * ButtonBox;
};

} // end of anonymous namespace

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidgetPrivate::addExtensionItem(const ExtensionMetadataType& metadata)
{
  Q_Q(qSlicerExtensionsManageWidget);

  QString extensionName = metadata.value("extensionname").toString();
  if (extensionName.isEmpty())
    {
    qCritical() << "Missing metadata identified with 'extensionname' key";
    return;
    }
  Q_ASSERT(this->extensionItem(extensionName) == 0);
  QString description = metadata.value("description").toString();
  QString extensionSlicerRevision = metadata.value("slicer_revision").toString();
  bool enabled = QVariant::fromValue(metadata.value("enabled")).toBool();

  QListWidgetItem * item = new QListWidgetItem();

  item->setIcon(extensionIcon(":/Icons/ExtensionDefaultIcon.png", enabled));

  item->setData(Self::NameRole, extensionName); // See extensionItem(...)

  q->addItem(item);

  bool isExtensionCompatible =
      q->extensionsManagerModel()->isExtensionCompatible(extensionName).isEmpty();

  int warningType = qSlicerExtensionsDescriptionLabel::NOWARNING;
  if (!isExtensionCompatible)
    {
    warningType = qSlicerExtensionsDescriptionLabel::INCOMPATIBLE;
    }

  qSlicerExtensionsDescriptionLabel * label = new qSlicerExtensionsDescriptionLabel(
        extensionSlicerRevision,
        q->extensionsManagerModel()->slicerRevision(),
        extensionName, description, warningType);
  label->setOpenExternalLinks(true);
  label->setMargin(6);
  this->LabelLinkMapper.setMapping(label, extensionName);
  QObject::connect(label, SIGNAL(linkActivated(QString)), &this->LabelLinkMapper, SLOT(map()));

  QSize hint = label->sizeHint();
  hint.setWidth(hint.width() + 64);
  item->setSizeHint(hint);

  qSlicerExtensionsItemWidget * widget = new qSlicerExtensionsItemWidget(label);
  q->setItemWidget(item, widget);

  this->EnableButtonMapper.setMapping(widget->ButtonBox->EnableButton, extensionName);
  QObject::connect(widget->ButtonBox->EnableButton, SIGNAL(clicked()), &this->EnableButtonMapper, SLOT(map()));
  widget->ButtonBox->EnableButton->setVisible(!enabled);
  widget->ButtonBox->EnableButton->setEnabled(isExtensionCompatible);

  this->DisableButtonMapper.setMapping(widget->ButtonBox->DisableButton, extensionName);
  QObject::connect(widget->ButtonBox->DisableButton, SIGNAL(clicked()), &this->DisableButtonMapper, SLOT(map()));
  widget->ButtonBox->DisableButton->setVisible(enabled);

  bool scheduledForUninstall = this->ExtensionsManagerModel->isExtensionScheduledForUninstall(extensionName);

  this->ScheduleUninstallButtonMapper.setMapping(widget->ButtonBox->ScheduleForUninstallButton, extensionName);
  QObject::connect(widget->ButtonBox->ScheduleForUninstallButton, SIGNAL(clicked()), &this->ScheduleUninstallButtonMapper, SLOT(map()));
  widget->ButtonBox->ScheduleForUninstallButton->setVisible(!scheduledForUninstall);

  this->CancelScheduledUninstallButtonMapper.setMapping(widget->ButtonBox->CancelScheduledForUninstallButton, extensionName);
  QObject::connect(widget->ButtonBox->CancelScheduledForUninstallButton, SIGNAL(clicked()), &this->CancelScheduledUninstallButtonMapper, SLOT(map()));
  widget->ButtonBox->CancelScheduledForUninstallButton->setVisible(scheduledForUninstall);
}

// --------------------------------------------------------------------------
qSlicerExtensionsManageWidget::qSlicerExtensionsManageWidget(QWidget* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerExtensionsManageWidgetPrivate(*this))
{
  Q_D(qSlicerExtensionsManageWidget);
  d->init();
}

// --------------------------------------------------------------------------
qSlicerExtensionsManageWidget::~qSlicerExtensionsManageWidget()
{
}

// --------------------------------------------------------------------------
qSlicerExtensionsManagerModel* qSlicerExtensionsManageWidget::extensionsManagerModel()const
{
  Q_D(const qSlicerExtensionsManageWidget);
  return d->ExtensionsManagerModel;
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::setExtensionsManagerModel(qSlicerExtensionsManagerModel* model)
{
  Q_D(qSlicerExtensionsManageWidget);

  if (this->extensionsManagerModel() == model)
    {
    return;
    }

  disconnect(this, SLOT(onModelUpdated()));
  disconnect(this, SLOT(onExtensionInstalled(QString)));
  disconnect(this, SLOT(onExtensionScheduledForUninstall(QString)));
  disconnect(this, SLOT(onExtensionCancelledScheduleForUninstall(QString)));
  disconnect(this, SLOT(onModelExtensionEnabledChanged(QString,bool)));

  d->ExtensionsManagerModel = model;

  if (d->ExtensionsManagerModel)
    {
    this->onModelUpdated();
    connect(d->ExtensionsManagerModel, SIGNAL(modelUpdated()),
            this, SLOT(onModelUpdated()));
    connect(d->ExtensionsManagerModel, SIGNAL(extensionInstalled(QString)),
            this, SLOT(onExtensionInstalled(QString)));
    connect(d->ExtensionsManagerModel, SIGNAL(extensionScheduledForUninstall(QString)),
            this, SLOT(onExtensionScheduledForUninstall(QString)));
    connect(d->ExtensionsManagerModel, SIGNAL(extensionCancelledScheduleForUninstall(QString)),
            this, SLOT(onExtensionCancelledScheduleForUninstall(QString)));
    connect(d->ExtensionsManagerModel, SIGNAL(extensionEnabledChanged(QString,bool)),
            this, SLOT(onModelExtensionEnabledChanged(QString,bool)));
    }
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::displayExtensionDetails(const QString& extensionName)
{
  Q_UNUSED(extensionName);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::setExtensionEnabled(const QString& extensionName)
{
  if (!this->extensionsManagerModel())
    {
    return;
    }
  this->extensionsManagerModel()->setExtensionEnabled(extensionName, true);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::setExtensionDisabled(const QString& extensionName)
{
  if (!this->extensionsManagerModel())
    {
    return;
    }
  this->extensionsManagerModel()->setExtensionEnabled(extensionName, false);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::scheduleExtensionForUninstall(const QString& extensionName)
{
  if (!this->extensionsManagerModel())
    {
    return;
    }
  this->extensionsManagerModel()->scheduleExtensionForUninstall(extensionName);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::cancelExtensionScheduledForUninstall(const QString& extensionName)
{
  if (!this->extensionsManagerModel())
    {
    return;
    }
  this->extensionsManagerModel()->cancelExtensionScheduledForUninstall(extensionName);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::onExtensionInstalled(const QString& extensionName)
{
  Q_D(qSlicerExtensionsManageWidget);
  d->addExtensionItem(d->ExtensionsManagerModel->extensionMetadata(extensionName));
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::onExtensionScheduledForUninstall(const QString& extensionName)
{
  Q_D(qSlicerExtensionsManageWidget);
  QListWidgetItem * item = d->extensionItem(extensionName);
  Q_ASSERT(item);
  qSlicerExtensionsItemWidget * widget =
      dynamic_cast<qSlicerExtensionsItemWidget*>(this->itemWidget(item));
  Q_ASSERT(widget);
  widget->ButtonBox->CancelScheduledForUninstallButton->setVisible(true);
  widget->ButtonBox->ScheduleForUninstallButton->setVisible(false);
}

// -------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::onExtensionCancelledScheduleForUninstall(const QString& extensionName)
{
  Q_D(qSlicerExtensionsManageWidget);
  QListWidgetItem * item = d->extensionItem(extensionName);
  Q_ASSERT(item);
  qSlicerExtensionsItemWidget * widget =
      dynamic_cast<qSlicerExtensionsItemWidget*>(this->itemWidget(item));
  Q_ASSERT(widget);
  widget->ButtonBox->CancelScheduledForUninstallButton->setVisible(false);
  widget->ButtonBox->ScheduleForUninstallButton->setVisible(true);
}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::onModelExtensionEnabledChanged(const QString &extensionName, bool enabled)
{
  Q_D(qSlicerExtensionsManageWidget);
  QListWidgetItem * item = d->extensionItem(extensionName);
  Q_ASSERT(item);
  qSlicerExtensionsItemWidget * widget =
      dynamic_cast<qSlicerExtensionsItemWidget*>(this->itemWidget(item));
  Q_ASSERT(widget);
  widget->ButtonBox->EnableButton->setVisible(!enabled);
  widget->ButtonBox->DisableButton->setVisible(enabled);
  item->setIcon(extensionIcon(":/Icons/ExtensionDefaultIcon.png", enabled));

}

// --------------------------------------------------------------------------
void qSlicerExtensionsManageWidget::onModelUpdated()
{
  Q_D(qSlicerExtensionsManageWidget);
  this->clear();
  foreach(const QString& extensionName, d->ExtensionsManagerModel->installedExtensions())
    {
    d->addExtensionItem(d->ExtensionsManagerModel->extensionMetadata(extensionName));
    }
}
