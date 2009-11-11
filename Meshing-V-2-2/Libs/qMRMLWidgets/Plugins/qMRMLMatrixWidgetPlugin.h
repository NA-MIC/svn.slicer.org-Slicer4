#ifndef __qMRMLMatrixWidgetPlugin_h
#define __qMRMLMatrixWidgetPlugin_h

#include "qMRMLWidgetsBasePlugin.h"
#include "qMRMLWidgetsPluginWin32Header.h"

class QMRML_WIDGETS_PLUGIN_EXPORT qMRMLMatrixWidgetPlugin : public QObject,
                                public qMRMLWidgetsBasePlugin
{
  Q_OBJECT

public:
  qMRMLMatrixWidgetPlugin(QObject *_parent = 0);
  
  QWidget *createWidget(QWidget *_parent);
  QString domXml() const;
  QString includeFile() const;
  bool isContainer() const;
  QString name() const;
  
};

#endif
