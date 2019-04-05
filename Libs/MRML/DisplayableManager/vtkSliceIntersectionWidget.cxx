/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkSliceIntersectionWidget.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkSliceIntersectionWidget.h"

#include "vtkMRMLAbstractSliceViewDisplayableManager.h"
#include "vtkMRMLApplicationLogic.h"
#include "vtkMRMLCrosshairDisplayableManager.h"
#include "vtkMRMLCrosshairNode.h"
#include "vtkMRMLInteractionEventData.h"
#include "vtkMRMLScalarVolumeDisplayNode.h"
#include "vtkMRMLScene.h"
#include "vtkMRMLSegmentationDisplayNode.h"
#include "vtkMRMLSliceCompositeNode.h"
#include "vtkMRMLSliceLayerLogic.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkSliceIntersectionRepresentation2D.h"

#include "vtkCommand.h"
#include "vtkCallbackCommand.h"
#include "vtkEvent.h"
#include "vtkGeneralTransform.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkTransform.h"
#include "vtkWidgetEvent.h"

#include <deque>

vtkStandardNewMacro(vtkSliceIntersectionWidget);

//----------------------------------------------------------------------------------
vtkSliceIntersectionWidget::vtkSliceIntersectionWidget()
{
  this->StartEventPosition[0] = 0.0;
  this->StartEventPosition[1] = 0.0;

  this->PreviousRotationAngleRad = 0.0;

  this->PreviousEventPosition[0] = 0;
  this->PreviousEventPosition[1] = 0;

  this->StartRotationCenter[0] = 0.0;
  this->StartRotationCenter[1] = 0.0;

  this->StartRotationCenter_RAS[0] = 0.0;
  this->StartRotationCenter_RAS[1] = 0.0;
  this->StartRotationCenter_RAS[2] = 0.0;
  this->StartRotationCenter_RAS[3] = 1.0; // to allow easy homogeneous tranformations

  this->StartActionFOV[0] = 0.;
  this->StartActionFOV[1] = 0.;
  this->StartActionFOV[2] = 0.;
  this->VolumeScalarRange[0] = 0;
  this->VolumeScalarRange[1] = 0;

  this->LastForegroundOpacity = 0.;
  this->LastLabelOpacity = 0.;
  this->StartActionSegmentationDisplayNode = nullptr;

  this->ModifierKeyPressedSinceLastMouseButtonRelease = true;

  this->SliceLogicsModifiedCommand->SetClientData(this);
  this->SliceLogicsModifiedCommand->SetCallback(vtkSliceIntersectionWidget::SliceLogicsModifiedCallback);

  this->ActionsEnabled = ActionAll;
  this->UpdateInteractionEventMapping();
}

//----------------------------------------------------------------------------------
vtkSliceIntersectionWidget::~vtkSliceIntersectionWidget()
{
  this->SetMRMLApplicationLogic(nullptr);
}

//----------------------------------------------------------------------
void vtkSliceIntersectionWidget::UpdateInteractionEventMapping()
{
  this->EventTranslators.clear();

  if (this->GetActionEnabled(ActionRotateSliceIntersection))
    {
    // MacOSX touchpad slice rotate
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::RotateEvent, vtkEvent::AnyModifier, WidgetEventTouchpadRotateSliceIntersection);

    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::LeftButtonPressEvent,
      vtkEvent::AltModifier+vtkEvent::ControlModifier, WidgetStateRotate, WidgetEventRotateStart, WidgetEventRotateEnd);
    }
  if (this->GetActionEnabled(ActionTranslateSliceIntersection))
    {
    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::LeftButtonPressEvent,
      vtkEvent::AltModifier+vtkEvent::ShiftModifier, WidgetStateTranslate, WidgetEventTranslateStart, WidgetEventTranslateEnd);
    }
  if (this->GetActionEnabled(ActionSetCrosshairPosition))
    {
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::MouseMoveEvent, vtkEvent::ShiftModifier, WidgetEventSetCrosshairPosition);
    }
  if (this->GetActionEnabled(ActionBlend))
    {
    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::LeftButtonPressEvent,
      vtkEvent::ControlModifier, WidgetStateBlend, WidgetEventBlendStart, WidgetEventBlendEnd);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "g", WidgetEventToggleLabelOpacity);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "t", WidgetEventToggleForegroundOpacity);
    }
  if (this->GetActionEnabled(ActionBrowseSlice))
    {
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "Right", WidgetEventIncrementSlice);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "Left", WidgetEventDecrementSlice);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "Up", WidgetEventIncrementSlice);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "Down", WidgetEventDecrementSlice);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "f", WidgetEventIncrementSlice);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "b", WidgetEventDecrementSlice);
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::MouseWheelForwardEvent, vtkEvent::NoModifier, WidgetEventIncrementSlice);
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::MouseWheelBackwardEvent, vtkEvent::NoModifier, WidgetEventDecrementSlice);
    }
  if (this->GetActionEnabled(ActionShowSlice))
    {
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "v", WidgetEventToggleSliceVisibility);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::AnyModifier, 0, 0, "V", WidgetEventToggleAllSlicesVisibility);
    }
  if (this->GetActionEnabled(ActionSelectVolume))
    {
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::AnyModifier, 0, 0, "bracketleft", WidgetEventShowPreviousBackgroundVolume);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::AnyModifier, 0, 0, "bracketright", WidgetEventShowNextBackgroundVolume);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::AnyModifier, 0, 0, "braceleft", WidgetEventShowPreviousForegroundVolume);
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::AnyModifier, 0, 0, "braceright", WidgetEventShowNextForegroundVolume);
    }
  if (this->GetActionEnabled(ActionTranslate) && this->GetActionEnabled(ActionZoom))
    {
    this->SetKeyboardEventTranslation(WidgetStateIdle, vtkEvent::NoModifier, 0, 0, "r", WidgetEventResetFieldOfView);
    }
  if (this->GetActionEnabled(ActionZoom))
    {
    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::RightButtonPressEvent, vtkEvent::NoModifier,
      WidgetStateZoomSlice, WidgetEventZoomSliceStart, WidgetEventZoomSliceEnd);
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::MouseWheelForwardEvent, vtkEvent::ControlModifier, WidgetEventZoomOutSlice);
    this->SetEventTranslation(WidgetStateIdle, vtkCommand::MouseWheelBackwardEvent, vtkEvent::ControlModifier, WidgetEventZoomInSlice);
    }

  if (this->GetActionEnabled(ActionTranslate))
    {
    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::LeftButtonPressEvent, vtkEvent::ShiftModifier,
      WidgetStateTranslateSlice, WidgetEventTranslateSliceStart, WidgetEventTranslateSliceEnd);
    this->SetEventTranslationClickAndDrag(WidgetStateIdle, vtkCommand::MiddleButtonPressEvent, vtkEvent::NoModifier,
      WidgetStateTranslateSlice, WidgetEventTranslateSliceStart, WidgetEventTranslateSliceEnd);
    }
}

//----------------------------------------------------------------------
void vtkSliceIntersectionWidget::CreateDefaultRepresentation()
{
  if (this->WidgetRep)
    {
    // already created
    return;
    }
  vtkMRMLApplicationLogic *mrmlAppLogic = this->GetMRMLApplicationLogic();
  if (!mrmlAppLogic)
    {
    vtkWarningMacro("vtkSliceIntersectionWidget::CreateDefaultRepresentation failed: application logic invalid");
    return;
    }
  vtkNew<vtkSliceIntersectionRepresentation2D> newRep;
  newRep->SetMRMLApplicationLogic(mrmlAppLogic);
  newRep->SetSliceNode(this->SliceNode);
  this->WidgetRep = newRep;
}

//-----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::CanProcessInteractionEvent(vtkMRMLInteractionEventData* eventData, double &distance2)
{
  if (!this->SliceLogic)
    {
    return false;
    }

  if (eventData->GetType() == vtkCommand::LeftButtonReleaseEvent
    || eventData->GetType() == vtkCommand::MiddleButtonReleaseEvent
    || eventData->GetType() == vtkCommand::RightButtonReleaseEvent)
    {
    this->ModifierKeyPressedSinceLastMouseButtonRelease = false;
    }
  if (eventData->GetType() == vtkCommand::LeaveEvent)
    {
    // We cannot capture keypress events until the user clicks in the view
    // so when we are outside then we should assume that modifier
    // is not just "stuck".
    this->ModifierKeyPressedSinceLastMouseButtonRelease = true;
    }
  if (eventData->GetType() == vtkCommand::KeyPressEvent)
    {
    if (eventData->GetKeySym().find("Shift") != std::string::npos)
      {
      this->ModifierKeyPressedSinceLastMouseButtonRelease = true;
      }
    }

  unsigned long widgetEvent = this->TranslateInteractionEventToWidgetEvent(eventData);
  if (widgetEvent == WidgetEventNone)
    {
    return false;
    }
  if (!this->GetRepresentation())
    {
    return false;
    }

  // If we are currently dragging a point then we interact everywhere
  if (this->WidgetState == WidgetStateTranslate
    || this->WidgetState == WidgetStateRotate
    || this->WidgetState == WidgetStateBlend
    || this->WidgetState == WidgetStateTranslateSlice
    || this->WidgetState == WidgetStateZoomSlice)
    {
    distance2 = 0.0;
    return true;
    }

  // We could do something like this if wa wanted to restrict mouse drag action to where the widget appears
  //if (this->GetRepresentation()->IsPositionOnWidget(eventData->GetDisplayPosition(), eventData->GetWorldPosition(), distance2))...

  distance2 = 1e10; // we can process this event but we let more specific widgets to claim it (if they are closer)
  return true;
}

//-----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessInteractionEvent(vtkMRMLInteractionEventData* eventData)
{
  if (!this->SliceLogic)
    {
    return false;
    }

  unsigned long widgetEvent = this->TranslateInteractionEventToWidgetEvent(eventData);

  bool processedEvent = true;

  switch (widgetEvent)
    {
    case WidgetEventMouseMove:
    // click-and-dragging the mouse cursor
      processedEvent = this->ProcessMouseMove(eventData);
      break;
    case WidgetEventTouchpadRotateSliceIntersection:
      // TODO: save state when the gesture starts. Need to get an event from Qt via VTK.
      this->Rotate(-1.0*vtkMath::RadiansFromDegrees(eventData->GetRotation() - eventData->GetLastRotation()));
      break;
    case WidgetEventTranslateStart:
      this->SetWidgetState(WidgetStateTranslate);
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      processedEvent = this->ProcessStartMouseDrag(eventData);
      break;
    case WidgetEventTranslateEnd:
      processedEvent = this->ProcessEndMouseDrag(eventData);
      break;
    case WidgetEventRotateStart:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      processedEvent = this->ProcessRotateStart(eventData);
      break;
    case WidgetEventRotateEnd:
      processedEvent = this->ProcessEndMouseDrag(eventData);
      break;
    case WidgetEventTranslateSliceStart:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->SliceLogic->StartSliceNodeInteraction(vtkMRMLSliceNode::XYZOriginFlag);
      this->SetWidgetState(WidgetStateTranslateSlice);
      processedEvent = this->ProcessStartMouseDrag(eventData);
      break;
    case WidgetEventTranslateSliceEnd:
      processedEvent = this->ProcessEndMouseDrag(eventData);
      this->SliceLogic->EndSliceNodeInteraction();
      break;
    case WidgetEventZoomSliceStart:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->SetWidgetState(WidgetStateZoomSlice);
      this->SliceLogic->StartSliceNodeInteraction(vtkMRMLSliceNode::FieldOfViewFlag);
      this->SliceLogic->GetSliceNode()->GetFieldOfView(this->StartActionFOV);
      processedEvent = this->ProcessStartMouseDrag(eventData);
      break;
    case WidgetEventZoomSliceEnd:
      processedEvent = this->ProcessEndMouseDrag(eventData);
      this->SliceLogic->EndSliceNodeInteraction();
      break;
    case WidgetEventBlendStart:
      {
      this->SetWidgetState(WidgetStateBlend);
      vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->LastForegroundOpacity = sliceCompositeNode->GetForegroundOpacity();
      this->LastLabelOpacity = this->GetLabelOpacity();
      this->StartActionSegmentationDisplayNode = this->GetVisibleSegmentationDisplayNode();
      processedEvent = this->ProcessStartMouseDrag(eventData);
      }
      break;
    case WidgetEventBlendEnd:
      processedEvent = this->ProcessEndMouseDrag(eventData);
      break;
    case WidgetEventSetCrosshairPosition:
      processedEvent = this->ProcessSetCrosshair(eventData);
      break;
    case WidgetEventToggleLabelOpacity:
      {
      this->StartActionSegmentationDisplayNode = nullptr;
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->StartActionSegmentationDisplayNode = nullptr;      double opacity = this->GetLabelOpacity();
      if (opacity != 0.0)
        {
        this->LastLabelOpacity = opacity;
        this->SetLabelOpacity(0.0);
        }
      else
        {
        this->SetLabelOpacity(this->LastLabelOpacity);
        }
      }
      break;
    case WidgetEventToggleForegroundOpacity:
      {
      vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      double opacity = sliceCompositeNode->GetForegroundOpacity();
      if (opacity != 0.0)
        {
        this->LastForegroundOpacity = opacity;
        sliceCompositeNode->SetForegroundOpacity(0.0);
        }
      else
        {
        sliceCompositeNode->SetForegroundOpacity(this->LastForegroundOpacity);
        }
      }
      break;
    case WidgetEventIncrementSlice:
      this->IncrementSlice();
      break;
    case WidgetEventDecrementSlice:
      this->DecrementSlice();
      break;
    case WidgetEventToggleSliceVisibility:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->GetSliceNode()->SetSliceVisible(!this->GetSliceNode()->GetSliceVisible());
      break;
    case WidgetEventToggleAllSlicesVisibility:
      // TODO: need to set all slices visible
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->GetSliceNode()->SetSliceVisible(!this->GetSliceNode()->GetSliceVisible());
      break;
    case WidgetEventResetFieldOfView:
      {
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->SliceLogic->StartSliceNodeInteraction(vtkMRMLSliceNode::ResetFieldOfViewFlag);
      this->SliceLogic->FitSliceToAll();
      this->GetSliceNode()->UpdateMatrices();
      this->SliceLogic->EndSliceNodeInteraction();
      }
      break;
    case WidgetEventZoomInSlice:
      this->ScaleZoom(1.2, eventData);
      break;
    case WidgetEventZoomOutSlice:
      this->ScaleZoom(0.8, eventData);
      break;
    break;
    case WidgetEventShowPreviousBackgroundVolume:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->CycleVolumeLayer(LayerBackground, -1);
      break;
    case WidgetEventShowNextBackgroundVolume:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->CycleVolumeLayer(LayerBackground, 1);
      break;
    case WidgetEventShowPreviousForegroundVolume:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->CycleVolumeLayer(LayerForeground, -1);
      break;
    case WidgetEventShowNextForegroundVolume:
      this->SliceLogic->GetMRMLScene()->SaveStateForUndo();
      this->CycleVolumeLayer(LayerForeground, 1);
      break;


    default:
      processedEvent = false;
    }

  return processedEvent;
}

//-------------------------------------------------------------------------
void vtkSliceIntersectionWidget::Leave()
{
  this->Superclass::Leave();
}

//-------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessMouseMove(vtkMRMLInteractionEventData* eventData)
{
  if (!this->WidgetRep || !eventData)
    {
    return false;
    }

  switch (this->WidgetState)
    {
    case WidgetStateRotate:
      this->ProcessRotate(eventData);
      break;
    case WidgetStateTranslate:
      {
      const double* worldPos = eventData->GetWorldPosition();
      this->GetSliceNode()->JumpAllSlices(worldPos[0], worldPos[1], worldPos[2]);
      }
      break;
    case WidgetStateBlend:
      this->ProcessBlend(eventData);
      break;
    case WidgetStateTranslateSlice:
      this->ProcessTranslateSlice(eventData);
      break;
    case WidgetStateZoomSlice:
      this->ProcessZoomSlice(eventData);
      break;
    }

  return true;
}

//-------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessStartMouseDrag(vtkMRMLInteractionEventData* eventData)
{
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(this->WidgetRep);
  if (!rep)
    {
    return false;
    }

  const int* displayPos = eventData->GetDisplayPosition();

  this->StartEventPosition[0] = displayPos[0];
  this->StartEventPosition[1] = displayPos[1];

  this->PreviousEventPosition[0] = this->StartEventPosition[0];
  this->PreviousEventPosition[1] = this->StartEventPosition[1];

  this->ProcessMouseMove(eventData);
  return true;
}

//----------------------------------------------------------------------------
double vtkSliceIntersectionWidget::GetSliceRotationAngleRad(double eventPos[2])
{
  return atan2(eventPos[1] - this->StartRotationCenter[1],
    eventPos[0] - this->StartRotationCenter[0]);
}


//-------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessEndMouseDrag(vtkMRMLInteractionEventData* vtkNotUsed(eventData))
{
  if (this->WidgetState == WidgetStateIdle)
    {
    return false;
    }
  this->SetWidgetState(WidgetStateIdle);
  return true;
}

//----------------------------------------------------------------------------------
void vtkSliceIntersectionWidget::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SetMRMLApplicationLogic(vtkMRMLApplicationLogic* appLogic)
{
  if (appLogic == this->ApplicationLogic)
    {
    return;
    }
  this->Superclass::SetMRMLApplicationLogic(appLogic);
  vtkCollection* sliceLogics = nullptr;
  this->SliceLogic = nullptr;
  if (appLogic)
    {
    sliceLogics = appLogic->GetSliceLogics();
    }
  if (sliceLogics != this->SliceLogics)
    {
    if (this->SliceLogics)
      {
      this->SliceLogics->RemoveObserver(this->SliceLogicsModifiedCommand);
      }
    if (sliceLogics)
      {
      sliceLogics->AddObserver(vtkCommand::ModifiedEvent, this->SliceLogicsModifiedCommand);
      }
    this->SliceLogics = sliceLogics;
    if (this->GetSliceNode())
      {
      this->SliceLogic = this->GetMRMLApplicationLogic()->GetSliceLogic(this->GetSliceNode());
      }
    }
}

//----------------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SliceLogicsModifiedCallback(vtkObject* vtkNotUsed(caller), unsigned long vtkNotUsed(eid), void* clientData, void* vtkNotUsed(callData))
{
  vtkSliceIntersectionWidget* self = vtkSliceIntersectionWidget::SafeDownCast((vtkObject*)clientData);
  if (!self)
    {
    return;
    }
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(self->WidgetRep);
  if (rep)
    {
    rep->UpdateIntersectingSliceNodes();
    }
  self->SliceLogic = nullptr;
  if (self->GetMRMLApplicationLogic())
    {
    self->SliceLogic = self->GetMRMLApplicationLogic()->GetSliceLogic(self->GetSliceNode());
    }
}

//----------------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SetSliceNode(vtkMRMLSliceNode* sliceNode)
{
  if (this->SliceNode == sliceNode)
    {
    // no change
    return;
    }
  this->SliceNode = sliceNode;

  // Update slice logic
  this->SliceLogic = nullptr;
  if (this->GetMRMLApplicationLogic())
    {
    this->SliceLogic = this->GetMRMLApplicationLogic()->GetSliceLogic(this->SliceNode);
    }

  // Update representation
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(this->WidgetRep);
  if (rep)
    {
    rep->SetSliceNode(sliceNode);
    }
}

//----------------------------------------------------------------------------------
vtkMRMLSliceNode* vtkSliceIntersectionWidget::GetSliceNode()
{
  return this->SliceNode;
}

//----------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessRotateStart(vtkMRMLInteractionEventData* eventData)
{
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(this->WidgetRep);
  if (!rep)
    {
    return false;
    }

  this->SetWidgetState(WidgetStateRotate);

  // Save rotation center
  double* sliceIntersectionPoint = rep->GetSliceIntersectionPoint();
  this->StartRotationCenter[0] = sliceIntersectionPoint[0];
  this->StartRotationCenter[1] = sliceIntersectionPoint[1];
  double startRotationCenterXY[4] = { sliceIntersectionPoint[0] , sliceIntersectionPoint[1], 0.0, 1.0 };
  this->GetSliceNode()->GetXYToRAS()->MultiplyPoint(startRotationCenterXY, this->StartRotationCenter_RAS);

  // Save initial rotation angle
  const int* displayPos = eventData->GetDisplayPosition();
  double displayPosDouble[2] = { static_cast<double>(displayPos[0]), static_cast<double>(displayPos[1]) };
  this->PreviousRotationAngleRad = this->GetSliceRotationAngleRad(displayPosDouble);

  return this->ProcessStartMouseDrag(eventData);
}

//----------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessRotate(vtkMRMLInteractionEventData* eventData)
{
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(this->WidgetRep);
  if (!rep)
    {
    return false;
    }

  const int* displayPos = eventData->GetDisplayPosition();
  double displayPosDouble[2] = { static_cast<double>(displayPos[0]), static_cast<double>(displayPos[1]) };
  double sliceRotationAngleRad = this->GetSliceRotationAngleRad(displayPosDouble);

  vtkMatrix4x4* sliceToRAS = this->GetSliceNode()->GetSliceToRAS();
  vtkNew<vtkTransform> rotatedSliceToSliceTransform;

  rotatedSliceToSliceTransform->Translate(this->StartRotationCenter_RAS[0], this->StartRotationCenter_RAS[1], this->StartRotationCenter_RAS[2]);
  double rotationDirection = vtkMath::Determinant3x3(sliceToRAS->Element[0], sliceToRAS->Element[1], sliceToRAS->Element[2]) >= 0 ? 1.0 : -1.0;
  rotatedSliceToSliceTransform->RotateWXYZ(rotationDirection*vtkMath::DegreesFromRadians(sliceRotationAngleRad - this->PreviousRotationAngleRad),
    sliceToRAS->GetElement(0, 2), sliceToRAS->GetElement(1, 2), sliceToRAS->GetElement(2, 2));
  rotatedSliceToSliceTransform->Translate(-this->StartRotationCenter_RAS[0], -this->StartRotationCenter_RAS[1], -this->StartRotationCenter_RAS[2]);

  this->PreviousRotationAngleRad = sliceRotationAngleRad;
  this->PreviousEventPosition[0] = displayPos[0];
  this->PreviousEventPosition[1] = displayPos[1];

  rep->TransformIntersectingSlices(rotatedSliceToSliceTransform->GetMatrix());

  return true;
}

//----------------------------------------------------------------------
bool vtkSliceIntersectionWidget::Rotate(double sliceRotationAngleRad)
{
  vtkSliceIntersectionRepresentation2D* rep = vtkSliceIntersectionRepresentation2D::SafeDownCast(this->WidgetRep);
  if (!rep)
    {
    return false;
    }
  vtkMatrix4x4* sliceToRAS = this->GetSliceNode()->GetSliceToRAS();
  vtkNew<vtkTransform> rotatedSliceToSliceTransform;

  double* sliceIntersectionPoint = rep->GetSliceIntersectionPoint();
  double startRotationCenterXY[4] = { sliceIntersectionPoint[0] , sliceIntersectionPoint[1], 0.0, 1.0 };
  this->GetSliceNode()->GetXYToRAS()->MultiplyPoint(startRotationCenterXY, this->StartRotationCenter_RAS);

  rotatedSliceToSliceTransform->Translate(this->StartRotationCenter_RAS[0], this->StartRotationCenter_RAS[1], this->StartRotationCenter_RAS[2]);
  double rotationDirection = vtkMath::Determinant3x3(sliceToRAS->Element[0], sliceToRAS->Element[1], sliceToRAS->Element[2]) >= 0 ? 1.0 : -1.0;
  rotatedSliceToSliceTransform->RotateWXYZ(rotationDirection*vtkMath::DegreesFromRadians(sliceRotationAngleRad),
    sliceToRAS->GetElement(0, 2), sliceToRAS->GetElement(1, 2), sliceToRAS->GetElement(2, 2));
  rotatedSliceToSliceTransform->Translate(-this->StartRotationCenter_RAS[0], -this->StartRotationCenter_RAS[1], -this->StartRotationCenter_RAS[2]);

  this->PreviousRotationAngleRad = sliceRotationAngleRad;

  rep->TransformIntersectingSlices(rotatedSliceToSliceTransform->GetMatrix());
  return true;
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessBlend(vtkMRMLInteractionEventData* eventData)
{
  const int* eventPosition = eventData->GetDisplayPosition();

  int* windowSize = this->Renderer->GetRenderWindow()->GetSize();
  double windowMinSize = std::min(windowSize[0], windowSize[1]);

  int deltaY = eventPosition[1] - this->PreviousEventPosition[1];
  double offsetY = (2.0 * deltaY) / windowMinSize;
  double newForegroundOpacity =
    this->LastForegroundOpacity + offsetY;
  newForegroundOpacity = std::min(std::max(newForegroundOpacity, 0.), 1.);
  vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
  if (sliceCompositeNode->GetForegroundVolumeID() != nullptr)
    {
    sliceCompositeNode->SetForegroundOpacity(newForegroundOpacity);
    this->LastForegroundOpacity = newForegroundOpacity;
    }
  int deltaX = eventPosition[0] - this->PreviousEventPosition[0];
  double offsetX = (2.0 * deltaX) / windowMinSize;
  double newLabelOpacity = this->LastLabelOpacity + offsetX;
  newLabelOpacity = std::min(std::max(newLabelOpacity, 0.), 1.);
  if (sliceCompositeNode->GetLabelVolumeID() != nullptr || this->StartActionSegmentationDisplayNode != nullptr)
    {
    this->SetLabelOpacity(newLabelOpacity);
    this->LastLabelOpacity = newLabelOpacity;
    }

  this->PreviousEventPosition[0] = eventPosition[0];
  this->PreviousEventPosition[1] = eventPosition[1];

  return true;
}


//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SetLabelOpacity(double opacity)
{
  // If a labelmap node is selected then adjust opacity of that
  vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
  if (sliceCompositeNode->GetLabelVolumeID())
  {
    sliceCompositeNode->SetLabelOpacity(opacity);
    return;
  }
  // No labelmap node is selected, adjust segmentation node instead
  vtkMRMLSegmentationDisplayNode* displayNode = this->StartActionSegmentationDisplayNode;
  if (!displayNode)
  {
    displayNode = this->GetVisibleSegmentationDisplayNode();
  }
  if (!displayNode)
  {
    return;
  }
  displayNode->SetOpacity(opacity);
}

//----------------------------------------------------------------------------
double vtkSliceIntersectionWidget::GetLabelOpacity()
{
  // If a labelmap node is selected then get opacity of that
  vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
  if (sliceCompositeNode->GetLabelVolumeID())
  {
    return sliceCompositeNode->GetLabelOpacity();
  }
  // No labelmap node is selected, use segmentation node instead
  vtkMRMLSegmentationDisplayNode* displayNode = this->StartActionSegmentationDisplayNode;
  if (!displayNode)
  {
    displayNode = this->GetVisibleSegmentationDisplayNode();
  }
  if (!displayNode)
  {
    return 0;
  }
  return displayNode->GetOpacity();
}

//----------------------------------------------------------------------------
vtkMRMLSegmentationDisplayNode* vtkSliceIntersectionWidget::GetVisibleSegmentationDisplayNode()
{
  vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();
  vtkMRMLScene* scene = this->SliceLogic->GetMRMLScene();
  std::vector<vtkMRMLNode*> displayNodes;
  int nnodes = scene ? scene->GetNodesByClass("vtkMRMLSegmentationDisplayNode", displayNodes) : 0;
  for (int i = 0; i < nnodes; i++)
  {
    vtkMRMLSegmentationDisplayNode* displayNode = vtkMRMLSegmentationDisplayNode::SafeDownCast(displayNodes[i]);
    if (displayNode
      && displayNode->GetVisibility(sliceNode->GetID())
      && (displayNode->GetVisibility2DOutline() || displayNode->GetVisibility2DFill()))
    {
      return displayNode;
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------------
double vtkSliceIntersectionWidget::GetSliceSpacing()
{
  double spacing;
  vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();

  if (sliceNode->GetSliceSpacingMode() == vtkMRMLSliceNode::PrescribedSliceSpacingMode)
    {
    spacing = sliceNode->GetPrescribedSliceSpacing()[2];
    }
  else
    {
    spacing = this->SliceLogic->GetLowestVolumeSliceSpacing()[2];
    }
  return spacing;
}

//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::IncrementSlice()
{
  this->MoveSlice(this->GetSliceSpacing());
}
//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::DecrementSlice()
{
  this->MoveSlice(-1. * this->GetSliceSpacing());
}
//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::MoveSlice(double delta)
{
  double offset = this->SliceLogic->GetSliceOffset();
  double newOffset = offset + delta;

  double sliceBounds[6] = {0, -1, 0, -1, 0, -1};
  this->SliceLogic->GetSliceBounds(sliceBounds);
  if (newOffset >= sliceBounds[4] && newOffset <= sliceBounds[5])
    {
    this->SliceLogic->StartSliceNodeInteraction(vtkMRMLSliceNode::SliceToRASFlag);
    this->SliceLogic->SetSliceOffset(newOffset);
    this->SliceLogic->EndSliceNodeInteraction();
    }
}

//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::CycleVolumeLayer(int layer, int direction)
{
  // first, find the current volume index for the given layer (can be nullptr)
  vtkMRMLScene *scene = this->SliceLogic->GetMRMLScene();
  vtkMRMLSliceCompositeNode *sliceCompositeNode = this->SliceLogic->GetSliceCompositeNode();
  char *volumeID = nullptr;
  switch (layer)
    {
    case 0: { volumeID = sliceCompositeNode->GetBackgroundVolumeID(); } break;
    case 1: { volumeID = sliceCompositeNode->GetForegroundVolumeID(); } break;
    case 2: { volumeID = sliceCompositeNode->GetLabelVolumeID(); } break;
    }
  vtkMRMLVolumeNode *volumeNode = vtkMRMLVolumeNode::SafeDownCast(scene->GetNodeByID(volumeID));

  // now figure out which one it is in the list
  int volumeCount = scene->GetNumberOfNodesByClass("vtkMRMLVolumeNode");
  int volumeIndex;
  for (volumeIndex = 0; volumeIndex < volumeCount; volumeIndex++)
    {
    if (volumeNode == scene->GetNthNodeByClass(volumeIndex, "vtkMRMLVolumeNode"))
      {
      break;
      }
    }

  // now increment by direction, and clamp to number of nodes
  volumeIndex += direction;
  if (volumeIndex >= volumeCount)
    {
    volumeIndex = 0;
    }
  if (volumeIndex < 0)
    {
    volumeIndex = volumeCount - 1;
    }

  // if we found a node, set it in the given layer
  volumeNode = vtkMRMLVolumeNode::SafeDownCast(
                  scene->GetNthNodeByClass(volumeIndex, "vtkMRMLVolumeNode"));
  if (volumeNode)
    {
    switch (layer)
      {
      case 0: { sliceCompositeNode->SetBackgroundVolumeID(volumeNode->GetID()); } break;
      case 1: { sliceCompositeNode->SetForegroundVolumeID(volumeNode->GetID()); } break;
      case 2: { sliceCompositeNode->SetLabelVolumeID(volumeNode->GetID()); } break;
      }
    }
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessTranslateSlice(vtkMRMLInteractionEventData* eventData)
{
  vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();

  double xyz[3];
  sliceNode->GetXYZOrigin(xyz);

  const int* eventPosition = eventData->GetDisplayPosition();

  // account for zoom using XYToSlice matrix
  vtkMatrix4x4* xyToSlice = sliceNode->GetXYToSlice();
  double deltaX = xyToSlice->GetElement(0, 0)*(this->PreviousEventPosition[0] - eventPosition[0]);
  double deltaY = xyToSlice->GetElement(1, 1)*(this->PreviousEventPosition[1] - eventPosition[1]);

  sliceNode->SetSliceOrigin(xyz[0] + deltaX, xyz[1] + deltaY, 0);

  this->PreviousEventPosition[0] = eventPosition[0];
  this->PreviousEventPosition[1] = eventPosition[1];
  return true;
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessZoomSlice(vtkMRMLInteractionEventData* eventData)
{
  const int* eventPosition = eventData->GetDisplayPosition();

  int* windowSize = this->GetRenderer()->GetRenderWindow()->GetSize();

  int deltaY = eventPosition[1] - this->StartEventPosition[1];
  double percent = (windowSize[1] + deltaY) / (1.0 * windowSize[1]);

  // the factor operation is so 'z' isn't changed and the
  // slider can still move through the full range
  if (percent > 0.)
    {
    double newFOVx = this->StartActionFOV[0] * percent;
    double newFOVy = this->StartActionFOV[1] * percent;
    double newFOVz = this->StartActionFOV[2];
    vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();
    sliceNode->SetFieldOfView(newFOVx, newFOVy, newFOVz);
    sliceNode->UpdateMatrices();
    }
  return true;
}

//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::ScaleZoom(double zoomScaleFactor, vtkMRMLInteractionEventData* eventData)
{
  // the factor operation is so 'z' isn't changed and the
  // slider can still move through the full range
  if (zoomScaleFactor <= 0)
    {
    vtkWarningMacro("vtkSliceViewInteractorStyle::ScaleZoom: invalid zoom scale factor (" << zoomScaleFactor);
    return;
    }
  vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();

  // Get distance of event position from slice center
  const int* eventPosition = eventData->GetDisplayPosition();
  int* windowSize = this->GetRenderer()->GetRenderWindow()->GetSize();
  vtkMatrix4x4* xyToSlice = sliceNode->GetXYToSlice();
  double evenPositionDistanceFromOrigin[2] =
    {
    (eventPosition[0] - windowSize[0] / 2) * xyToSlice->GetElement(0, 0),
    (eventPosition[1] - windowSize[1] / 2) * xyToSlice->GetElement(1, 1)
    };

  // Adjust field of view
  double fov[3] = { 1.0 };
  sliceNode->GetFieldOfView(fov);
  fov[0] *= zoomScaleFactor;
  fov[1] *= zoomScaleFactor;
  sliceNode->SetFieldOfView(fov[0], fov[1], fov[2]);

  // Keep the mouse position at the same place on screen
  double sliceOrigin[3] = { 0 };
  sliceNode->GetXYZOrigin(sliceOrigin);
  sliceNode->SetSliceOrigin(
    sliceOrigin[0] + evenPositionDistanceFromOrigin[0] * (1.0 - zoomScaleFactor),
    sliceOrigin[1] + evenPositionDistanceFromOrigin[1] * (1.0 - zoomScaleFactor),
    sliceOrigin[2]);

  sliceNode->UpdateMatrices();
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::ProcessSetCrosshair(vtkMRMLInteractionEventData* eventData)
{
  if (!this->ModifierKeyPressedSinceLastMouseButtonRelease)
    {
    // this event was caused by a "stuck" modifier key
    return false;
    }
  vtkMRMLSliceNode* sliceNode = this->GetSliceNode();
  vtkMRMLCrosshairNode* crosshairNode = vtkMRMLCrosshairDisplayableManager::FindCrosshairNode(sliceNode->GetScene());
  if (!crosshairNode)
    {
    return false;
    }
  const double* worldPos = eventData->GetWorldPosition();
  crosshairNode->SetCrosshairRAS(const_cast<double*>(worldPos));
  if (crosshairNode->GetCrosshairBehavior() != vtkMRMLCrosshairNode::NoAction)
    {
    int viewJumpSliceMode = vtkMRMLSliceNode::OffsetJumpSlice;
    if (crosshairNode->GetCrosshairBehavior() == vtkMRMLCrosshairNode::CenteredJumpSlice)
      {
      viewJumpSliceMode = vtkMRMLSliceNode::CenteredJumpSlice;
      }
    sliceNode->JumpAllSlices(sliceNode->GetScene(),
      worldPos[0], worldPos[1], worldPos[2],
      viewJumpSliceMode, sliceNode->GetViewGroup(), sliceNode);
    }
  return true;
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::IsEventInsideVolume(bool background, vtkMRMLInteractionEventData* eventData)
{
  vtkMRMLSliceNode *sliceNode = this->SliceLogic->GetSliceNode();
  if (!sliceNode)
  {
    return false;
  }
  vtkMRMLSliceLayerLogic* layerLogic = background ?
    this->SliceLogic->GetBackgroundLayer() : this->SliceLogic->GetForegroundLayer();
  if (!layerLogic)
  {
    return false;
  }
  vtkMRMLVolumeNode* volumeNode = layerLogic->GetVolumeNode();
  if (!volumeNode || !volumeNode->GetImageData())
  {
    return false;
  }
  const int* eventPosition = eventData->GetDisplayPosition();
  double xyz[3] = { 0 };
  vtkMRMLAbstractSliceViewDisplayableManager::ConvertDeviceToXYZ(this->GetRenderer(), sliceNode, eventPosition[0], eventPosition[1], xyz);
  vtkGeneralTransform* xyToBackgroundIJK = layerLogic->GetXYToIJKTransform();
  double mousePositionIJK[3] = { 0 };
  xyToBackgroundIJK->TransformPoint(xyz, mousePositionIJK);
  int volumeExtent[6] = { 0 };
  volumeNode->GetImageData()->GetExtent(volumeExtent);
  for (int i = 0; i < 3; i++)
  {
    if (mousePositionIJK[i]<volumeExtent[i * 2] || mousePositionIJK[i]>volumeExtent[i * 2 + 1])
    {
      return false;
    }
  }
  return true;
}

//----------------------------------------------------------------------------
bool vtkSliceIntersectionWidget::GetActionEnabled(int actionsMask)
{
  return (this->ActionsEnabled & actionsMask) == actionsMask;
}

//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SetActionEnabled(int actionsMask, bool enable /*=true*/)
{
  if (enable)
    {
    this->ActionsEnabled |= actionsMask;
    }
  else
    {
    this->ActionsEnabled &= (~actionsMask);
    }
  this->UpdateInteractionEventMapping();
}

//----------------------------------------------------------------------------
int vtkSliceIntersectionWidget::GetActionsEnabled()
{
  return this->ActionsEnabled;
}

//----------------------------------------------------------------------------
void vtkSliceIntersectionWidget::SetActionsEnabled(int actions)
{
  this->ActionsEnabled = actions;
  this->UpdateInteractionEventMapping();
}
