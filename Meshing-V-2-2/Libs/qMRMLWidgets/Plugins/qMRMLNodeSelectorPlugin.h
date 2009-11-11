#ifndef __qMRMLNodeSelectorPlugin_h
#define __qMRMLNodeSelectorPlugin_h

#include "qMRMLWidgetsBasePlugin.h"
#include "qMRMLWidgetsPluginWin32Header.h"

class QMRML_WIDGETS_PLUGIN_EXPORT qMRMLNodeSelectorPlugin : public QObject,
                                         public qMRMLWidgetsBasePlugin
{
  Q_OBJECT

public:
  qMRMLNodeSelectorPlugin(QObject *_parent = 0);
  
  QWidget *createWidget(QWidget *_parent);
  QString domXml() const;
  QString includeFile() const;
  bool isContainer() const;
  QString name() const;
  
};

#endif
