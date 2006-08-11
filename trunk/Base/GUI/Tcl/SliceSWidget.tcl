
#########################################################
#
if {0} { ;# comment

  SliceSWidget  - manages slice plane interactions

# TODO : 

}
#
#########################################################
# ------------------------------------------------------------------
#                             SliceSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class SliceSWidget] == "" } {

  itcl::class SliceSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    variable _actionStartRAS "0 0 0"
    variable _actionStartXY "0 0"
    variable _actionStartFOV "250 250 250"

    # methods
    method processEvent {} {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body SliceSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI
 
  set o(storeXYToRAS) [$this vtkNew vtkMatrix4x4]
  set o(storeSliceToRAS) [$this vtkNew vtkMatrix4x4]
  set o(scratchMatrix) [$this vtkNew vtkMatrix4x4]

  [$_annotation GetTextProperty] SetColor 1 1 1
  [$_annotation GetTextProperty] SetShadow 1

  $this processEvent

  set _guiObserverTags ""
  lappend _guiObserverTags [$sliceGUI AddObserver DeleteEvent "itcl::delete object $this"]
  lappend _guiObserverTags [$sliceGUI AddObserver AnyEvent "$this processEvent"]
  set node [[$sliceGUI GetLogic] GetSliceNode]
  lappend _nodeObserverTags [$node AddObserver DeleteEvent "itcl::delete object $this"]
  lappend _nodeObserverTags [$node AddObserver AnyEvent "$this processEvent"]
}

itcl::body SliceSWidget::destructor {} {

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

  if { [info command $_renderer] != "" } {
    foreach a $_actors {
      $_renderer RemoveActor2D $a
    }
  }
}


# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body SliceSWidget::processEvent { } {

  if { [info command $sliceGUI] == "" } {
    # the sliceGUI was deleted behind our back, so we need to 
    # self destruct
    itcl::delete object $this
    return
  }

  set grabID [$sliceGUI GetGrabID]
  if { ($grabID != "" && $grabID != $this) } {
    return ;# some other widget wants these events
  }

  #
  # get the current event info
  # - the event type
  # - the actual x y location of the event
  # - fill the layers info with local info
  # - get the RAS space location of the event
  #
  set event [$sliceGUI GetCurrentGUIEvent] 
  if { $event != "ConfigureEvent" } {
    # this call is needed (shouldn't be)
    eval $_interactor UpdateSize [$_renderer GetSize]
  }
  foreach {x y} [$_interactor GetEventPosition] {}
  $this queryLayers $x $y
  set xyToRAS [$_sliceNode GetXYToRAS]
  set ras [$xyToRAS MultiplyPoint $x $y 0 1]
  foreach {r a s t} $ras {}


  switch $event {

    "MouseMoveEvent" {
      #
      # Mouse move behavior governed by _actionState mode
      # - first update the annotation
      # - then handle modifying the view
      #


      $_annotation SetText 0 "Lb: $_layers(label,pixel)\nFg: $_layers(foreground,pixel)\nBg: $_layers(background,pixel)"
      $_annotation SetText 1 [format "Bg I: %d\nBg J: %d\nBg K: %d" \
                    $_layers(background,i) $_layers(background,j) $_layers(background,k)]
      $_annotation SetText 2 "X: $x\nY:$y"
      set rasText [format "R: %.1f\nA: %.1f\nS: %.1f" $r $a $s]
      $_annotation SetText 3 $rasText

      switch $_actionState {
        "Translate" {
          #
          # Translate
          #
          set currentRAS [$o(storeXYToRAS) MultiplyPoint $x $y 0 1]
          foreach d {dr da ds} start $_actionStartRAS current $currentRAS {
            set $d [expr $current - $start]
          }
          $o(scratchMatrix) DeepCopy $o(storeSliceToRAS)
          foreach d {dr da ds} i {0 1 2} {
            set v [$o(scratchMatrix) GetElement $i 3]
            $o(scratchMatrix) SetElement $i 3 [expr $v - [set $d]]
          }
          [$_sliceNode GetSliceToRAS] DeepCopy $o(scratchMatrix)
          $_sliceNode UpdateMatrices
          $sliceGUI SetGUICommandAbortFlag 1
        }
        "Zoom" {
          #
          # Zoom
          #
          set deltay [expr $y - [lindex $_actionStartXY 1]]
          set tkwindow [$_renderWidget  GetWidgetName]
          set h [winfo height $tkwindow]
          set percent [expr ($h + $deltay) / (1.0 * $h)]

          # the factor operation is so 'z' isn't changed and the 
          # slider can still move through the full range
          set newFOV ""
          foreach f $_actionStartFOV factor "$percent $percent 1" {
            lappend newFOV [expr $f * $factor]
          }
          eval $_sliceNode SetFieldOfView $newFOV

          $_sliceNode UpdateMatrices
          $sliceGUI SetGUICommandAbortFlag 1
        }
        default {
          # need to render to show the annotation
          [$sliceGUI GetSliceViewer] RequestRender
        }
      }
    }

    "RightButtonPressEvent" {
      set _actionState "Zoom"
      set _actionStartXY "$x $y"
      $sliceGUI SetGrabID $this
      $sliceGUI SetGUICommandAbortFlag 1
      set _actionStartFOV [$_sliceNode GetFieldOfView]
      $::slicer3::MRMLScene SaveStateForUndo $_sliceNode
    }
    "RightButtonReleaseEvent" { 
      set _actionState ""
      $sliceGUI SetGrabID ""
      $sliceGUI SetGUICommandAbortFlag 1
    }
    "LeftButtonPressEvent" {
      if { [info command SeedSWidget] != "" } {
        if { [$_interactor GetControlKey] } {
          set seedSWidget [SeedSWidget #auto $sliceGUI]
          $seedSWidget place $r $a $s
        }
      }
    }
    "LeftButtonReleaseEvent" { 
    }
    "MiddleButtonPressEvent" {
      set _actionState "Translate"
      set _actionStartRAS $ras
      $sliceGUI SetGrabID $this
      $sliceGUI SetGUICommandAbortFlag 1
      $o(storeXYToRAS) DeepCopy [$_sliceNode GetXYToRAS]
      $o(storeSliceToRAS) DeepCopy [$_sliceNode GetSliceToRAS]
      $::slicer3::MRMLScene SaveStateForUndo $_sliceNode
    }
    "MiddleButtonReleaseEvent" { 
      set _actionState ""
      $sliceGUI SetGrabID ""
      $sliceGUI SetGUICommandAbortFlag 1
    }
    "MouseWheelForwardEvent" { }
    "MouseWheelBackwardEvent" { }
    "ExposeEvent" { }
    "ConfigureEvent" {
      set tkwindow [$_renderWidget  GetWidgetName]
      set w [winfo width $tkwindow]
      set h [winfo height $tkwindow]

      if { $w == "10" && $h == "10" } {
        puts "ignoring bogus resize"
      } else {
        set oldFOV [$_sliceNode GetFieldOfView]
        set oldDim [$_sliceNode GetDimensions]
        set oldPixelSize [expr [lindex $oldFOV 0] / (1. * [lindex $oldDim 0])]
        $_sliceNode SetDimensions $w $h [lindex $oldDim 2]
        $_sliceNode SetFieldOfView \
            [expr $oldPixelSize * $w] [expr $oldPixelSize * $h] [lindex $oldFOV 2]
      }
    }
    "EnterEvent" { 
      $_renderWidget CornerAnnotationVisibilityOn
      [$::slicer3::ApplicationGUI GetMainSlicerWin]  SetStatusText "Middle Button: Pan; Left Button: Zoom"
    }
    "LeaveEvent" { 
      $_renderWidget CornerAnnotationVisibilityOff
      [$::slicer3::ApplicationGUI GetMainSlicerWin]  SetStatusText ""
    }
    "TimerEvent" { }
    "KeyPressEvent" { 
      $sliceGUI SetCurrentGUIEvent "" ;# reset event so we don't respond again
      switch [$_interactor GetKeyCode] {
        "v" {
          $sliceGUI SetGUICommandAbortFlag 1
          $_sliceNode SetSliceVisible [expr ![$_sliceNode GetSliceVisible]]
        }
        "p" {
          PaintSWidget::TogglePaint
        }
        "f" {
          #
          # fit the view to the background volume
          # - figure out the image dimensions mapped to view space
          # - set the field of view to include the full dimensions
          # - reset the pan/zoom/slice to origin
          #
          set ijkToRAS [vtkMatrix4x4 New]
          set dims [$_layers(background,image) GetDimensions]
          $_layers(background,node) GetIJKToRASMatrix $ijkToRAS
          set rasDims [eval $ijkToRAS MultiplyPoint $dims 0]
          $ijkToRAS Delete

          set rasToSlice [vtkMatrix4x4 New]
          $rasToSlice DeepCopy [$_sliceNode GetSliceToRAS]
          foreach r {0 1 2} {$rasToSlice SetElement $r 3  0}
          $rasToSlice Invert
          set sliceDims [eval $rasToSlice MultiplyPoint $rasDims ]
          $rasToSlice Delete


          set absSliceDims ""
          foreach d $sliceDims {
            lappend absSliceDims [expr abs($d)]
          }

          set tkwindow [$_renderWidget  GetWidgetName]
          set w [winfo width $tkwindow]
          set h [winfo height $tkwindow]
          set aspect [expr $w / ($h * 1.0)]
          foreach {fx fy fz fw} $absSliceDims {}
          if { $w > $h } {
            set fx [expr $fx * $aspect]
          } else {
            set fy [expr $fy / $aspect]
          }
          $_sliceNode SetFieldOfView $fx $fy $fz

          # reset to origin
          $o(scratchMatrix) DeepCopy [$_sliceNode GetSliceToRAS]
          foreach i {0 1 2} {
            $o(scratchMatrix) SetElement $i 3  0
          }
          [$_sliceNode GetSliceToRAS] DeepCopy $o(scratchMatrix)

          $_sliceNode UpdateMatrices
          $sliceGUI SetGUICommandAbortFlag 1
        }
        default {
          puts "[$_interactor GetKeyCode]"
        }
      }
    }
    "KeyReleaseEvent" { 
    }
    "CharEvent" {
      if { 0 } { 
        puts -nonewline "char event [$_interactor GetKeyCode]"
        if { [$_interactor GetControlKey] } {
          puts -nonewline " with control"
        }
        if { [$_interactor GetShiftKey] } {
          puts -nonewline " with shift"
        }
        puts ""
      }
    }
    "ExitEvent" { }

  }

}

