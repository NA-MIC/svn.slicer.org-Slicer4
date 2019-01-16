/*==============================================================================

  Copyright (c) Kapteyn Astronomical Institute
  University of Groningen, Groningen, Netherlands. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Davide Punzo, Kapteyn Astronomical Institute,
  and was supported through the European Research Council grant nr. 291531.

==============================================================================*/

#ifndef __vtkMRMLPlotViewNode_h
#define __vtkMRMLPlotViewNode_h

#include "vtkMRMLAbstractViewNode.h"

class vtkMRMLPlotChartNode;

/// \brief MRML node to represent Plot view parameters.
///
/// PlotViewNodes are associated one to one with a PlotWidget.
class VTK_MRML_EXPORT vtkMRMLPlotViewNode : public vtkMRMLAbstractViewNode
{
public:
  static vtkMRMLPlotViewNode *New();
  vtkTypeMacro(vtkMRMLPlotViewNode, vtkMRMLAbstractViewNode);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  /// Enum of the available plot types
  enum {
    InteractionModePanView,
    InteractionModeSelectPoints,
    InteractionModeFreehandSelectPoints,
    InteractionModeMovePoints,
    InteractionMode_Last // must be last
  };

  //--------------------------------------------------------------------------
  /// MRMLNode methods
  //--------------------------------------------------------------------------

   virtual vtkMRMLNode* CreateNodeInstance()  VTK_OVERRIDE;

  ///
  /// Read node attributes from XML file.
  virtual void ReadXMLAttributes( const char** atts) VTK_OVERRIDE;

  ///
  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent) VTK_OVERRIDE;

  ///
  /// Copy the node's attributes to this object.
  virtual void Copy(vtkMRMLNode *node) VTK_OVERRIDE;

  ///
  /// Get node XML tag name (like Volume, Model).
  virtual const char* GetNodeTagName() VTK_OVERRIDE { return "PlotView"; };

  ///
  /// Set and Update the PlotChart node id displayed in this PlotView.
  virtual void SetPlotChartNodeID(const char *PlotChartNodeID);

  ///
  /// Get the PlotChart node id displayed in this PlotView.
  const char* GetPlotChartNodeID();

  ///
  /// Get the PlotChart node displayed in this PlotView.
  vtkMRMLPlotChartNode* GetPlotChartNode();

  ///
  /// Configures the behavior of PropagatePlotChartSelection().
  /// If DoPropagatePlotChartSelection set to false then this
  /// view will not be affected by PropagatePlotChartSelection.
  /// Default value is true.
  vtkSetMacro (DoPropagatePlotChartSelection, bool );
  vtkGetMacro (DoPropagatePlotChartSelection, bool );

  /// Get/Set interaction mode for mouse left-click and drag.
  /// InteractionModePanView: pans chart view
  /// InteractionModeSelectPoints: select points with rectangular shape
  /// InteractionModeFreehandSelectPoints: select points with free-hand drawn shape
  /// InteractionModeMovePoints: move plot data points
  vtkGetMacro(InteractionMode, int);
  vtkSetMacro(InteractionMode, int);

  /// Enable moving of data points along X axis by click-and-drag in plot views.
  /// Points may only be moved in the view if moving is enabled in the plot chart node as well.
  vtkBooleanMacro(EnablePointMoveAlongX, bool);
  vtkGetMacro(EnablePointMoveAlongX, bool);
  vtkSetMacro(EnablePointMoveAlongX, bool);

  /// Enable moving of data points along Y axis by click-and-drag in plot views.
  /// Points may only be moved in the view if moving is enabled in the plot chart node as well.
  vtkBooleanMacro(EnablePointMoveAlongY, bool);
  vtkGetMacro(EnablePointMoveAlongY, bool);
  vtkSetMacro(EnablePointMoveAlongY, bool);

  ///
  /// Convert between interaction mode ID and name
  static const char *GetInteractionModeAsString(int id);
  static int GetInteractionModeFromString(const char *name);

  ///
  /// Method to propagate events generated in mrml.
  virtual void ProcessMRMLEvents(vtkObject *caller,
                                 unsigned long event,
                                 void *callData) VTK_OVERRIDE;

  /// PlotModifiedEvent is fired when:
  ///  - a new PlotChart node is observed
  ///  - a PlotChart node is not longer observed
  ///  - an associated PlotChart node is modified
  /// Note that when SetAndObserve(Nth)NodeID() is called with an ID that
  /// has not yet any associated plot node in the scene, then
  /// plotModifiedEvent is not fired until found for the first time in
  /// the scene, e.g. UpdateScene()...
  enum
  {
    PlotChartNodeChangedEvent = 18000
  };

  virtual const char* GetPlotChartNodeReferenceRole();

protected:
  vtkMRMLPlotViewNode();
  ~vtkMRMLPlotViewNode();
  vtkMRMLPlotViewNode(const vtkMRMLPlotViewNode&);
  void operator=(const vtkMRMLPlotViewNode&);

  virtual const char* GetPlotChartNodeReferenceMRMLAttributeName();

  static const char* PlotChartNodeReferenceRole;
  static const char* PlotChartNodeReferenceMRMLAttributeName;

  ///
  /// Called when a node reference ID is added (list size increased).
  virtual void OnNodeReferenceAdded(vtkMRMLNodeReference *reference) VTK_OVERRIDE;

  ///
  /// Called when a node reference ID is modified.
  virtual void OnNodeReferenceModified(vtkMRMLNodeReference *reference) VTK_OVERRIDE;

  ///
  /// Called after a node reference ID is removed (list size decreased).
  virtual void OnNodeReferenceRemoved(vtkMRMLNodeReference *reference) VTK_OVERRIDE;

  bool DoPropagatePlotChartSelection;

  int InteractionMode;
  bool EnablePointMoveAlongX;
  bool EnablePointMoveAlongY;
};

#endif
