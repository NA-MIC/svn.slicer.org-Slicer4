
package require Itcl

#########################################################
#
if {0} { ;# comment

  EffectSWidget a superclass for editor effects

  - subclasses of this need to define their specialness

# TODO : 

}
#
#########################################################

#
#########################################################
# ------------------------------------------------------------------
#                             EffectSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class EffectSWidget] == "" } {

  itcl::class EffectSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    public variable scope "all"

    variable _startPosition "0 0 0"
    variable _currentPosition "0 0 0"
    variable _cursorActors ""
    variable _outputLabel ""
    variable _observerRecords "" ;# list of the observers so we can clean up

    # methods
    method processEvent {} {}
    method preProcessEvent {} {}
    method positionCursor {} {}
    method createCursor {} {}
    method setCursor {imageData} { $o(cursorMapper) SetInput $imageData }
    method preview {} {}
    method apply {} {}
    method postApply {} {}
    method getInputBackground {} {}
    method getInputLabel {} {}
    method getOutputLabel {} {}
    method getOptionsFrame {} {}
    method buildOptions {} {}
    method tearDownOptions {} {}
    method previewOptions {} {}
    method applyOptions {} {}
    method flashCursor { {repeat 1} {delay 50} } {}
    method errorDialog { message } {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body EffectSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI
 
  $this createCursor

  set _startPosition "0 0 0"
  set _currentPosition "0 0 0"

  $this processEvent

  set _guiObserverTags ""
  lappend _guiObserverTags [$sliceGUI AddObserver DeleteEvent "::SWidget::ProtectedDelete $this"]
  foreach event "LeftButtonPressEvent LeftButtonReleaseEvent MouseMoveEvent EnterEvent LeaveEvent" {
    lappend _guiObserverTags \
              [$sliceGUI AddObserver $event "::SWidget::ProtectedCallback $this processEvent"]
  }
  set node [[$sliceGUI GetLogic] GetSliceNode]
  lappend _nodeObserverTags [$node AddObserver DeleteEvent "::SWidget::ProtectedDelete $this"]
  lappend _nodeObserverTags [$node AddObserver AnyEvent "::SWidget::ProtectedCallback $this processEvent"]
}

itcl::body EffectSWidget::destructor {} {

  if { [info command $sliceGUI] != "" } {
    foreach tag $_guiObserverTags {
      $sliceGUI RemoveObserver $tag
    }
  }

  if { [info command $_sliceNode] != "" } {
    foreach tag $_nodeObserverTags {
      $_sliceNode RemoveObserver $tag
    }
  }

  foreach record $_observerRecords {
    foreach {obj tag} $record {
      if { [info command $obj] != "" } {
        $obj RemoveObserver $tag
      }
    }
  }

  if { [info command $_renderer] != "" } {
    foreach a $_cursorActors {
      $_renderer RemoveActor2D $a
    }
  }
}

itcl::configbody EffectSWidget::scope {
  if { [lsearch "all visible" $scope] == -1 } {
    set scope "all"
    error "invalid scope for EffectSWidget $this"
  }
}

# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body EffectSWidget::createCursor {} {

  set o(cursorDummyImage) [vtkNew vtkImageData]
  $o(cursorDummyImage) AllocateScalars
  set o(cursorMapper) [vtkNew vtkImageMapper]
  $o(cursorMapper) SetInput $o(cursorDummyImage)
  set o(cursorActor) [vtkNew vtkActor2D]
  $o(cursorActor) SetMapper $o(cursorMapper)
  $o(cursorMapper) SetColorWindow 255
  $o(cursorMapper) SetColorLevel 128

  lappend _cursorActors $o(cursorActor)

  foreach actor $_cursorActors {
    [$_renderWidget GetRenderer] AddActor2D $o(cursorActor)
  }

  $o(cursorActor) VisibilityOff
}

itcl::body EffectSWidget::positionCursor {} {
  set xyzw [$this rasToXY $_currentPosition]
  foreach {x y z w} $xyzw {}
  set x [expr $x + 16]
  set y [expr $y - 32]
  foreach actor $_cursorActors {
    eval $actor SetPosition $x $y
  }
}

itcl::body EffectSWidget::getOptionsFrame { } {
  EditorSelectModule
  return [EditorGetOptionsFrame $::Editor(singleton)]
}

itcl::body EffectSWidget::buildOptions { } {
  # default implementation, there is nothing
}

itcl::body EffectSWidget::tearDownOptions { } {
  # default implementation, there is nothing
}



itcl::body EffectSWidget::flashCursor { {repeat 1} {delay 50} } {

  for {set i 0} {$i < $repeat} {incr i} {
    set oldVisibility [$o(cursorActor) GetVisibility]
    $o(cursorActor) SetVisibility [expr !$oldVisibility]
    [$sliceGUI GetSliceViewer] RequestRender
    update
    after $delay
    $o(cursorActor) SetVisibility $oldVisibility
    [$sliceGUI GetSliceViewer] RequestRender
    update
  }
}


#
# returns 1 if the event is 'swallowed' by
# the superclass, otherwise returns 0 and the
# subclass should do normal processing
#
itcl::body EffectSWidget::preProcessEvent { } {

  if { [info command $sliceGUI] == "" } {
    # the sliceGUI was deleted behind our back, so we need to 
    # self destruct
    itcl::delete object $this
    return 1
  }

  set grabID [$sliceGUI GetGrabID]
  if { ($grabID != "") && ($grabID != $this) } {
    # some other widget wants these events
    # -- we can position wrt the current slice node
    $this positionCursor
    [$sliceGUI GetSliceViewer] RequestRender
    return 1
  }

  return 0
}

#
# manage the data to work on:
# - input is always the current contents of the 
#   background and label layers
#   -- can be either the whole volume or just visible portion
# - output is a temp buffer that gets re-inserted into
#   the label layer using the postApply method
# 

itcl::body EffectSWidget::getInputBackground {} {
  set logic [[$sliceGUI GetLogic]  GetBackgroundLayer]
  switch $scope {
    "all" {
      set node [$logic GetVolumeNode]
      if { $node != "" } {
        return [$node GetImageData]
      }
    }
    "visible" {
      #return [$logic GetImageData]
      return [[$logic GetReslice] GetOutput]
    }
  }
  return ""
}

itcl::body EffectSWidget::getInputLabel {} {
  set logic [[$sliceGUI GetLogic]  GetLabelLayer]
  switch $scope {
    "all" {
      set node [$logic GetVolumeNode]
      if { $node != "" } {
        return [$node GetImageData]
      }
    }
    "visible" {
      return [$logic GetImageData]
    }
  }
  return ""
}

itcl::body EffectSWidget::getOutputLabel {} {
  if { $_outputLabel == "" } {
    set _outputLabel [vtkImageData New]
  }
  return $_outputLabel
}

itcl::body EffectSWidget::postApply {} {
  set logic [[$sliceGUI GetLogic]  GetLabelLayer]
  switch $scope {
    "all" {
      set node [$logic GetVolumeNode]
      $node SetAndObserveImageData $_outputLabel
    }
    "visible" {
      # TODO: need to use vtkImageSlicePaint to insert visible
      # paint back into the label volume
      error "not yet supported"
    }
  }
  $_outputLabel Delete
  set _outputLabel ""
}


#
# default implementations of methods to be overridden by subclass
#

itcl::body EffectSWidget::preview {} {
  # to be overridden by subclass
}


itcl::body EffectSWidget::processEvent { } {
  # to be overridden by subclass
  # - should include call to superclass preProcessEvent
  #   to handle 'friendly' interaction with other SWidgets

  $this preProcessEvent

  # your event processing can replace the dummy code below...

  set event [$sliceGUI GetCurrentGUIEvent] 
  set _currentPosition [$this xyToRAS [$_interactor GetEventPosition]]

  switch $event {
    "LeftButtonPressEvent" {
      $this apply
      $sliceGUI SetGUICommandAbortFlag 1
      $sliceGUI SetGrabID $this
    }
    "EnterEvent" {
      $o(cursorActor) VisibilityOn
    }
    "LeaveEvent" {
      $o(cursorActor) VisibilityOff
    }
  }

  $this positionCursor
  [$sliceGUI GetSliceViewer] RequestRender

}

itcl::body EffectSWidget::apply {} {
  # default behavior, just flash...
  $this flashCursor 3
}

itcl::body EffectSWidget::errorDialog { message } {
  # TODO: convert this to a kww dialog
  tk_messageBox -message $message
}


#
# helper procs to manage effects
#

proc EffectSWidget::Add {effect} {
  foreach sw [itcl::find objects -class SliceSWidget] {
    set sliceGUI [$sw cget -sliceGUI]
    if { [info command $sliceGUI] != "" } {
      $effect #auto $sliceGUI
    }
  }
}

proc EffectSWidget::RemoveAll {} {
  foreach ew [itcl::find objects -isa EffectSWidget] {
    itcl::delete object $ew
  }
}

proc EffectSWidget::Remove {effect} {
  foreach pw [itcl::find objects -class $effect] {
    itcl::delete object $pw
  }
}

proc EffectSWidget::Toggle {effect} {
  if { [itcl::find objects -class $effect] == "" } {
    EffectSWidget::Add $effect
  } else {
    EffectSWidget::Remove $effect
  }
}

proc EffectSWidget::ConfigureAll { effect args } {
  foreach pw [itcl::find objects -class $effect] {
    eval $pw configure $args
  }
}

proc EffectSWidget::SetCursorAll { effect imageData } {

  foreach ew [itcl::find objects -class $effect] {
    $ew setCursor $imageData
  }
}
