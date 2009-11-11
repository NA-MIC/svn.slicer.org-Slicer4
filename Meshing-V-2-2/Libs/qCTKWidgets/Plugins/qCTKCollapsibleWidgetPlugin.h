#ifndef __qCTKCollapsibleWidgetPlugin_h
#define __qCTKCollapsibleWidgetPlugin_h

#include "qCTKWidgetsBasePlugin.h"
#include "qCTKWidgetsPluginWin32Header.h"

class QCTK_WIDGETS_PLUGIN_EXPORT qCTKCollapsibleWidgetPlugin : public QObject,
                                    public qCTKWidgetsBasePlugin
{
  Q_OBJECT

public:
  qCTKCollapsibleWidgetPlugin(QObject *_parent = 0);
  
  QWidget *createWidget(QWidget *_parent);
  QString domXml() const;
  QString includeFile() const;
  bool isContainer() const;
  QString name() const;
  void initialize(QDesignerFormEditorInterface *formEditor);
};

#endif
