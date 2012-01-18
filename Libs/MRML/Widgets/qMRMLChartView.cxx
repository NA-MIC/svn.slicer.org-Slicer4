

// Qt includes
#include <QDebug>
#include <QEvent>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QToolButton>

// CTK includes
#include <ctkAxesWidget.h>
#include <ctkLogger.h>
#include <ctkPopupWidget.h>

// qMRML includes
#include "qMRMLColors.h"
#include "qMRMLChartView_p.h"

// MRMLDisplayableManager includes
#include <vtkMRMLAbstractDisplayableManager.h>
#include <vtkMRMLDisplayableManagerGroup.h>
#include <vtkMRMLChartViewDisplayableManagerFactory.h>
#include <vtkChartViewInteractorStyle.h>

// MRML includes
#include <vtkMRMLDoubleArrayNode.h>
#include <vtkMRMLChartNode.h>
#include <vtkMRMLChartViewNode.h>
#include <vtkMRMLScene.h>

// VTK includes
#include <vtkSmartPointer.h>
#include <vtkStringArray.h>

// Convenient macro
#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

//--------------------------------------------------------------------------
static ctkLogger logger("org.slicer.libs.qmrmlwidgets.qMRMLChartView");
//--------------------------------------------------------------------------

const char *examplePlot = 
  "<!DOCTYPE html>"
  "<html>"
  "<head>"
  "<title>Line Charts and Options</title>"
  "<link class=\"include\" rel=\"stylesheet\" type=\"text/css\" href=\"qrc:/jqPlot/jquery.jqplot.min.css\" />"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/jquery.min.js\"></script>"
  "</head>"
  "<body>"
  "<div id=\"chart1\"></div>"
  "<script class=\"code\" type=\"text/javascript\">"
  "$(document).ready(function(){"
  "var plot1 = $.jqplot ('chart1', [[3,7,9,1,5,3,8,2,5]],"
  "{title: 'An example plot', highlighter: {show: true}, cursor: {show: false}}"
  ");"
  "});"
  "</script>"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/jquery.jqplot.min.js\"></script>"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.canvasTextRenderer.min.js\"></script>"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.canvasAxisLabelRenderer.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.highlighter.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.cursor.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.dateAxisRenderer.min.js\"></script>"
  "</body>"
  "</html>";

const char *plotPreamble = 
  "<!DOCTYPE html>"
  "<html>"
  "<head>"
  "<title>Line Charts and Options</title>"
  "<link class=\"include\" rel=\"stylesheet\" type=\"text/css\" href=\"qrc:/jqPlot/jquery.jqplot.min.css\" />"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/jquery.min.js\"></script>"
  "</head>"
  "<body>";

const char *plotPostscript =
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/jquery.jqplot.min.js\"></script>"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.canvasTextRenderer.min.js\"></script>"
  "<script class=\"include\" type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.canvasAxisLabelRenderer.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.highlighter.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.cursor.min.js\"></script>"
  "<script type=\"text/javascript\" src=\"qrc:/jqPlot/plugins/jqplot.dateAxisRenderer.min.js\"></script>"
  "</body>"
  "</html>";



//--------------------------------------------------------------------------
// qMRMLChartViewPrivate methods

//---------------------------------------------------------------------------
qMRMLChartViewPrivate::qMRMLChartViewPrivate(qMRMLChartView& object)
  : q_ptr(&object)
{
  this->DisplayableManagerGroup = 0;
  this->MRMLScene = 0;
  this->MRMLChartViewNode = 0;
  this->PinButton = 0;
  this->PopupWidget = 0;
}

//---------------------------------------------------------------------------
qMRMLChartViewPrivate::~qMRMLChartViewPrivate()
{
  if (this->DisplayableManagerGroup)
    {
    this->DisplayableManagerGroup->Delete();
    }
}

//---------------------------------------------------------------------------
void qMRMLChartViewPrivate::init()
{
  Q_Q(qMRMLChartView);
//  q->setRenderEnabled(this->MRMLScene != 0);
  q->setEnabled(this->MRMLScene != 0);

  this->PopupWidget = new ctkPopupWidget;
  QHBoxLayout* popupLayout = new QHBoxLayout;
  popupLayout->addWidget(new QToolButton);
  this->PopupWidget->setLayout(popupLayout);

  // VTK_CREATE(vtkChartViewInteractorStyle, interactorStyle);
  // q->interactor()->SetInteractorStyle(interactorStyle);

//  this->initDisplayableManagers();

//  q->setHtml(examplePlot); 
  q->setHtml("");
  q->show();
}

//---------------------------------------------------------------------------
void qMRMLChartViewPrivate::initDisplayableManagers()
{
  // Q_Q(qMRMLChartView);
  // vtkMRMLChartViewDisplayableManagerFactory* factory
  //   = vtkMRMLChartViewDisplayableManagerFactory::GetInstance();

  // QStringList displayableManagers;
  // displayableManagers << "vtkMRMLChartViewDisplayableManager";
  //                     //<< "vtkMRMLCameraDisplayableManager"
  //                     //<< "vtkMRMLViewDisplayableManager"
  //                     //<< "vtkMRMLModelDisplayableManager";
  // foreach(const QString& displayableManager, displayableManagers)
  //   {
  //   if(!factory->IsDisplayableManagerRegistered(displayableManager.toLatin1()))
  //     {
  //     factory->RegisterDisplayableManager(displayableManager.toLatin1());
  //     }
  //   }

  // this->DisplayableManagerGroup
  //   = factory->InstantiateDisplayableManagers(q->renderer());
  // // Observe displayable manager group to catch RequestRender events
  // this->qvtkConnect(this->DisplayableManagerGroup, vtkCommand::UpdateEvent,
  //                   q, SLOT(scheduleRender()));
}

//---------------------------------------------------------------------------
void qMRMLChartViewPrivate::setMRMLScene(vtkMRMLScene* newScene)
{
  Q_Q(qMRMLChartView);
  if (newScene == this->MRMLScene)
    {
    return;
    }

  this->qvtkReconnect(
    this->mrmlScene(), newScene,
    vtkMRMLScene::StartBatchProcessEvent, this, SLOT(startProcessing()));

  this->qvtkReconnect(
    this->mrmlScene(), newScene,
    vtkMRMLScene::EndBatchProcessEvent, this, SLOT(endProcessing()));

  this->MRMLScene = newScene;
//  q->setRenderEnabled(this->MRMLScene != 0);
  q->setEnabled(this->MRMLScene != 0);
}


// --------------------------------------------------------------------------
void qMRMLChartViewPrivate::startProcessing()
{
  logger.trace("startProcessing");
  Q_Q(qMRMLChartView);
//  q->setRenderEnabled(false);
  q->setEnabled(false);
}

//
// --------------------------------------------------------------------------
void qMRMLChartViewPrivate::endProcessing()
{
  logger.trace("endProcessing");
  Q_Q(qMRMLChartView);
//  q->setRenderEnabled(true);
  q->setEnabled(true);
}

// --------------------------------------------------------------------------
vtkMRMLScene* qMRMLChartViewPrivate::mrmlScene()
{
  return this->MRMLScene;
}

// --------------------------------------------------------------------------
void qMRMLChartViewPrivate::updateWidgetFromMRML()
{
  Q_Q(qMRMLChartView);
  if (!this->MRMLScene || !this->MRMLChartViewNode)
    {
    return;
    }

  if (!q->isEnabled())
    {
    return;
    }

  // Get the ChartNode
  char *chartnodeid = this->MRMLChartViewNode->GetChartNodeID();

  if (!chartnodeid)
    {
    return;
    }

  vtkMRMLChartNode* cn = vtkMRMLChartNode::SafeDownCast(this->MRMLScene->GetNodeByID(chartnodeid));

  if (!cn)
    {
    return;
    }

  vtkStringArray *arrayIDs = cn->GetArrays();
  vtkStringArray *arrayNames = cn->GetArrayNames();

  // data to plot - represented in javascript
  //
  //
  QStringList plotData;
  plotData << "var data = [";

  // for each curve
  for (int idx = 0; idx < arrayIDs->GetNumberOfValues(); idx++)
    {
    vtkMRMLDoubleArrayNode *dn = vtkMRMLDoubleArrayNode::SafeDownCast(this->MRMLScene->GetNodeByID( arrayIDs->GetValue(idx).c_str() ));

    if (dn)
      {
      double x, y;

      plotData << "[";
      
      // for each value
      for (unsigned int j = 0; j < dn->GetSize(); ++j)
        {
        dn->GetXYValue(j, &x, &y);
        plotData << "[" << QString("%1").arg(x) << ", " << QString("%1").arg(y) << "]";
        if (j < dn->GetSize()-1)
          {
          plotData << ",";
          }
        }

      plotData<< "]";

      if (idx < arrayIDs->GetNumberOfValues()-1)
        {
        plotData << ",";
        }
      }
    }

  plotData << "];";
  

  // properties for the plot - represented in javascript
  //
  //
  QStringList plotOptions;
  plotOptions << "var options = {";

  // plot level properties: title, axis labels, grid, ...
  plotOptions << 
    "highlighter: {show: true}, cursor: {show: false}, legend: {show: true}";
  
  // if (cn->GetTitle() && cn->ShowTitle())
  //   {
  //   plotOptions << ", title: " << cn->GetTitle();
  //   }

  // series level properties
  plotOptions << ", series: [";
  for (int idx = 0; idx < arrayNames->GetNumberOfValues(); idx++)
    {
    // for each series
    plotOptions << "{";
    // legend
    plotOptions << "label: '" << arrayNames->GetValue(idx).c_str() << "'";

    // markers
    plotOptions << ", showMarker: false";

    // lines

    // end of a series
    plotOptions << "}";
    if (idx < arrayNames->GetNumberOfValues()-1)
      {
      plotOptions << ",";
      }
    }
  // end of series properties
  plotOptions << "]";
  // end of properties
  plotOptions << "};";


  // resize slot - represented in javascript
  // pass in resetAxes: true option to get rid of old ticks and axis properties
  QStringList plotResizeSlot;
  plotResizeSlot << 
    "var resizeSlot = function() {"
    "$('#chart').css('width', 0.95*$(window).width());"
    "$('#chart').css('height', 0.95*$(window).height());"
    "plot1.replot( {resetAxes: true} );"
    "};";

  // an initial call to the resize slot - represented in javascript
  QStringList plotInitialResize;
  plotInitialResize <<
    "resizeSlot();";

  // resize hook - represented in javascript
  // resize function should be bound to the #chart not to the window
  QStringList plotResizeHook;
  plotResizeHook <<
    "$(window).resize( resizeSlot );";


  // Assemble the plot
  //
  // 1. HTML page preamble
  // 2. Div container for the chart
  // 3. Script container
  // 4. Definition of the "ready" function inside the script container
  // 5. HTML page poscript
  //
  QStringList plot;
  plot << plotPreamble;       // 1. page header, css, javascript

  plot << 
    "<div id=\"chart\"></div>"                       // 2. container for the chart
    "<script class=\"code\" type=\"text/javascript\">"    // 3. container for js
    "$(document).ready(function(){";                 // 4. ready function     
  plot << plotData;     // insert data
  plot << plotOptions;  // insert options
  plot << 
    "var plot1 = $.jqplot ('chart', data, options);";  // call the plot
  plot << plotResizeSlot;        // insert definition of the resizeSlot
  plot << plotInitialResize;     // insert an initial call to resizeSlot 
  plot << plotResizeHook;        // insert hook to call resizeSlot on page resize

  plot << 
    "});"                   // end of function and end of call to ready()
    "</script>";            // end of the javascript
      
  plot << plotPostscript;   // 5. page postscript, additional javascript
  
  // qDebug() << plot.join("");

  // show the plot
  q->setHtml(plot.join("")); 
  q->show();

}

// --------------------------------------------------------------------------
// qMRMLChartView methods

// --------------------------------------------------------------------------
qMRMLChartView::qMRMLChartView(QWidget* _parent) : Superclass(_parent)
  , d_ptr(new qMRMLChartViewPrivate(*this))
{
  Q_D(qMRMLChartView);
  d->init();
}

// --------------------------------------------------------------------------
qMRMLChartView::~qMRMLChartView()
{
}

//------------------------------------------------------------------------------
void qMRMLChartView::addDisplayableManager(const QString&) //displayableManagerName)
{
  // Q_D(qMRMLChartView);
  // vtkSmartPointer<vtkMRMLAbstractDisplayableManager> displayableManager;
  // displayableManager.TakeReference(
  //   vtkMRMLDisplayableManagerGroup::InstantiateDisplayableManager(
  //     displayableManagerName.toLatin1()));
  // d->DisplayableManagerGroup->AddDisplayableManager(displayableManager);
}

//------------------------------------------------------------------------------
void qMRMLChartView::setMRMLScene(vtkMRMLScene* newScene)
{
  Q_D(qMRMLChartView);
  if (newScene == d->MRMLScene)
    {
    return;
    }

  d->setMRMLScene(newScene);

  if (d->MRMLChartViewNode && newScene != d->MRMLChartViewNode->GetScene())
    {
    this->setMRMLChartViewNode(0);
    }

  emit mrmlSceneChanged(newScene);
}

//---------------------------------------------------------------------------
void qMRMLChartView::setMRMLChartViewNode(vtkMRMLChartViewNode* newChartViewNode)
{
  Q_D(qMRMLChartView);
  if (d->MRMLChartViewNode == newChartViewNode)
    {
    return;
    }

  d->qvtkReconnect(
    d->MRMLChartViewNode, newChartViewNode,
    vtkCommand::ModifiedEvent, d, SLOT(updateWidgetFromMRML()));

  d->MRMLChartViewNode = newChartViewNode;
  // d->DisplayableManagerGroup->SetMRMLDisplayableNode(newChartViewNode);

  d->updateWidgetFromMRML();
}

//---------------------------------------------------------------------------
vtkMRMLChartViewNode* qMRMLChartView::mrmlChartViewNode()const
{
  Q_D(const qMRMLChartView);
  return d->MRMLChartViewNode;
}

//---------------------------------------------------------------------------
vtkMRMLScene* qMRMLChartView::mrmlScene()const
{
  Q_D(const qMRMLChartView);
  return d->MRMLScene;
}


