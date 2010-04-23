#ifndef __qCTKWidgetsPlugin_h
#define __qCTKWidgetsPlugin_h

#include "qCTKWidgetsPluginExport.h"

#include <QDesignerCustomWidgetCollectionInterface>

// \class Group the plugins in one library
class QCTK_WIDGETS_PLUGIN_EXPORT qCTKWidgetsPlugin : public QObject,
                           public QDesignerCustomWidgetCollectionInterface
{
  Q_OBJECT
  Q_INTERFACES(QDesignerCustomWidgetCollectionInterface);

public:
  QList<QDesignerCustomWidgetInterface*> customWidgets() const
    {
    QList<QDesignerCustomWidgetInterface *> plugins;
    
    return plugins;
    }
};

#endif
