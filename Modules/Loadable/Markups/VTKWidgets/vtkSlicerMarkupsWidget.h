/*=========================================================================

 Copyright (c) ProxSim ltd., Kwun Tong, Hong Kong. All Rights Reserved.

 See COPYRIGHT.txt
 or http://www.slicer.org/copyright/copyright.txt for details.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 This file was originally developed by Davide Punzo, punzodavide@hotmail.it,
 and development was supported by ProxSim ltd.

=========================================================================*/

/**
 * @class   vtkSlicerMarkupsWidget
 * @brief   Process interaction events to update state of markup widget nodes
 *
 * @sa
 * vtkMRMLAbstractWidget vtkSlicerWidgetRepresentation vtkSlicerWidgetEventTranslator
 *
*/

#ifndef vtkSlicerMarkupsWidget_h
#define vtkSlicerMarkupsWidget_h

#include "vtkSlicerMarkupsModuleVTKWidgetsExport.h"
#include "vtkMRMLAbstractWidget.h"
#include "vtkWidgetCallbackMapper.h"

#include "vtkMRMLMarkupsNode.h"

class vtkMRMLAbstractViewNode;
class vtkMRMLApplicationLogic;
class vtkMRMLInteractionEventData;
class vtkMRMLInteractionNode;
class vtkIdList;
class vtkPolyData;
class vtkSlicerMarkupsWidgetRepresentation;

class VTK_SLICER_MARKUPS_MODULE_VTKWIDGETS_EXPORT vtkSlicerMarkupsWidget : public vtkMRMLAbstractWidget
{
public:
  /// Standard methods for a VTK class.
  vtkTypeMacro(vtkSlicerMarkupsWidget, vtkMRMLAbstractWidget);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// Create the default widget representation and initializes the widget and representation.
  virtual void CreateDefaultRepresentation(vtkMRMLMarkupsDisplayNode* markupsDisplayNode, vtkMRMLAbstractViewNode* viewNode, vtkRenderer* renderer) = 0;

  /// Widget states
  enum
  {
    WidgetStateDefine = WidgetStateUser, // click in empty area will place a new point
    WidgetStateTranslateControlPoint, // translating the active point by mouse move
  };

  /// Widget events
  enum
  {
    WidgetEventControlPointPlace = WidgetEventUser,
    WidgetEventStopPlace,
    WidgetEventControlPointMoveStart,
    WidgetEventControlPointMoveEnd,
    WidgetEventControlPointDelete,
    WidgetEventControlPointInsert
  };

  // Returns true if one of the markup points are just being previewed and not placed yet.
  bool IsPointPreviewed();

  /// Add/update a point preview to the current active Markup at the specified position.
  void UpdatePreviewPoint(const int displayPos[2], const double worldPos[3], const char* associatedNodeID);

  /// Remove the point preview to the current active Markup.
  /// Returns true is preview point existed and now it is removed.
  bool RemovePreviewPoint();

  // Places a new markup point.
  // Reuses current preview point, if possible.
  // Returns true if the event is processed.
  bool PlacePoint(vtkMRMLInteractionEventData* eventData);

  /// Add a point to the current active Markup at input World coordiantes.
  virtual int AddPointFromWorldCoordinate(const double worldCoordinates[3]);

  /// Given a specific X, Y pixel location, add a new node
  /// on the widget at this location.
  virtual int AddNodeOnWidget(const int displayPos[2]);

  /// Return true if the widget can process the event.
  bool CanProcessInteractionEvent(vtkMRMLInteractionEventData* eventData, double &distance2) override;

  /// Process interaction event.
  bool ProcessInteractionEvent(vtkMRMLInteractionEventData* eventData) override;

  /// Called when the the widget loses the focus.
  void Leave() override;

  // Allows the widget to request interactive mode (faster updates)
  bool GetInteractive() override;

  vtkMRMLMarkupsNode* GetMarkupsNode();
  vtkMRMLMarkupsDisplayNode* GetMarkupsDisplayNode();
  int GetActiveControlPoint();

  vtkSlicerMarkupsWidgetRepresentation* GetMarkupsRepresentation();

protected:
  vtkSlicerMarkupsWidget();
  ~vtkSlicerMarkupsWidget() override;

  void StartWidgetInteraction(vtkMRMLInteractionEventData* eventData);
  void EndWidgetInteraction();

  virtual void TranslatePoint(double eventPos[2]);
  virtual void TranslateWidget(double eventPos[2]);
  virtual void ScaleWidget(double eventPos[2]);
  virtual void RotateWidget(double eventPos[2]);

  bool IsAnyControlPointLocked();

  // Get accurate world position.
  // World position that comes in the event data may be inaccurate, this method computes a more reliable position.
  // Returns true on success.
  // refWorldPos is an optional reference position: if point distance from camera cannot be determined then
  // depth of this reference position is used.
  bool ConvertDisplayPositionToWorld(const int displayPos[2], double worldPos[3], double worldOrientationMatrix[9],
    double* refWorldPos = nullptr);

  /// Index of the control point that is currently being previewed (follows the mouse pointer).
  /// If <0 it means that there is currently no point being previewed.
  int PreviewPointIndex;

  // Callback interface to capture events when
  // placing the widget.
  // Return true if the event is processed.
  virtual bool ProcessMouseMove(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessControlPointDelete(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessControlPointInsert(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessControlPointMoveStart(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessWidgetTranslateStart(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessWidgetRotateStart(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessWidgetScaleStart(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessEndMouseDrag(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessWidgetReset(vtkMRMLInteractionEventData* eventData);
  virtual bool ProcessWidgetJumpCursor(vtkMRMLInteractionEventData* eventData);

  // Variables for translate/rotate/scale
  double LastEventPosition[2];
  double StartEventOffsetPosition[2];

  // True if mouse button pressed since a point was placed.
  // This is used to filter out "click" events that started before the point was placed.
  bool MousePressedSinceMarkupPlace;

private:
  vtkSlicerMarkupsWidget(const vtkSlicerMarkupsWidget&) = delete;
  void operator=(const vtkSlicerMarkupsWidget&) = delete;
};

#endif
