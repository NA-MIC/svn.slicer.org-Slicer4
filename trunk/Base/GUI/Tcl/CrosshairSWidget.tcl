
#########################################################
#
if {0} { ;# comment

  CrosshairSWidget  - displays a crosshair on a sliceGUI

# TODO : 

}
#
#########################################################
# ------------------------------------------------------------------
#                             CrosshairSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class CrosshairSWidget] == "" } {

  itcl::class CrosshairSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    #public variable rgba ".5 .9 .5 .6"  ;# crosshair color
    #public variable rgba "1.0 0.8 0.1 .6"  ;# crosshair color
    public variable rgba "1.0 0.8 0.1 1.0"  ;# crosshair color

    variable _compositeNode

    # methods
    method processEvent {{caller ""} {event ""}} {}
    method updateCrosshair { } {}
    method resetCrosshair { } {}
    method addCrosshairLine { startPoint endPoint } {}

    method setPosition { r a s } {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body CrosshairSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI

  # cache the color of the viewer
  set rgba "[lindex [[$sliceGUI GetSliceViewer] GetHighlightColor] 0] [lindex [[$sliceGUI GetSliceViewer] GetHighlightColor] 1] [lindex [[$sliceGUI GetSliceViewer] GetHighlightColor] 2] [lindex $rgba 3]"
 
  # create crosshair display parts
  set o(crosshairPolyData) [vtkNew vtkPolyData]
  set o(crosshairLines) [vtkNew vtkCellArray]
  $o(crosshairPolyData) SetLines $o(crosshairLines)
  set o(crosshairPoints) [vtkNew vtkPoints]
  $o(crosshairPolyData) SetPoints $o(crosshairPoints)

  set o(crosshairMapper) [vtkNew vtkPolyDataMapper2D]
  set o(crosshairActor) [vtkNew vtkActor2D]
  $o(crosshairMapper) SetInput $o(crosshairPolyData)
  $o(crosshairActor) SetMapper $o(crosshairMapper)
  eval [$o(crosshairActor) GetProperty] SetColor [lrange $rgba 0 2]
  eval [$o(crosshairActor) GetProperty] SetOpacity [lindex $rgba 3]
  [$_renderWidget GetRenderer] AddActor2D $o(crosshairActor)
  lappend _actors $o(crosshairActor)

  #
  # set up observers on sliceGUI and on sliceNode
  #

  $::slicer3::Broker AddObservation $sliceGUI DeleteEvent "::SWidget::ProtectedDelete $this"

  set events {  "MouseMoveEvent" "UserEvent" "EnterEvent" "LeaveEvent"}
  foreach event $events {
    $::slicer3::Broker AddObservation $sliceGUI $event "::SWidget::ProtectedCallback $this processEvent $sliceGUI $event"
  }

  set node [[$sliceGUI GetLogic] GetSliceNode]
  $::slicer3::Broker AddObservation $node DeleteEvent "::SWidget::ProtectedDelete $this"
  $::slicer3::Broker AddObservation $node AnyEvent "::SWidget::ProtectedCallback $this processEvent $node AnyEvent"

  set _compositeNode [[$sliceGUI GetLogic] GetSliceCompositeNode]
  $::slicer3::Broker AddObservation $node DeleteEvent "::SWidget::ProtectedDelete $this"
  $::slicer3::Broker AddObservation $node AnyEvent "::SWidget::ProtectedCallback $this processEvent $node AnyEvent"

  $this updateCrosshair
}


itcl::body CrosshairSWidget::destructor {} {

  if { [info command $_renderer] != "" } {
    foreach a $_actors {
      $_renderer RemoveActor2D $a
    }
  }
}

itcl::configbody CrosshairSWidget::rgba {
  eval [$o(crosshairActor) GetProperty] SetColor [lrange $rgba 0 2]
  eval [$o(crosshairActor) GetProperty] SetOpacity [lindex $rgba 3]
}


# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

#
# handle interactor events
#
itcl::body CrosshairSWidget::processEvent { {caller ""} {event ""} } {

  if { [info command $sliceGUI] == "" } {
    # the sliceGUI was deleted behind our back, so we need to 
    # self destruct
    ::SWidget::ProtectedDelete $this
    return
  }

  if { $caller == $_sliceNode || $caller == $_compositeNode } {
    $this updateCrosshair
    return
  }

  if { $caller == $sliceGUI } {

    switch $event {

      "MouseMoveEvent" {
          #
          # TODO: what do we want to do on mouse move?
          #
          
          # update the actors...
          foreach {windowx windowy} [$_interactor GetEventPosition] {}
          foreach {x y z} [$this dcToXYZ $windowx $windowy] {}  
          
          set xyToRAS [$_sliceNode GetXYToRAS]
          set ras [$xyToRAS MultiplyPoint $x $y $z 1]
          
          # position this crosshair actor at the viewport position
          $o(crosshairActor) SetPosition $x $y

          # set other crosshairs to this ras
          foreach {r a s w} $ras {}
          set itclobjects [itcl::find objects]
          foreach cw $itclobjects {
            if {[$cw isa CrosshairSWidget] && $cw != $this} {
              $cw setPosition $r $a $s
              if { [[[[$cw cget -sliceGUI] GetLogic] GetSliceCompositeNode] GetCrosshairBehavior] != 0 } {
                [[[$cw cget -sliceGUI] GetLogic] GetSliceNode] JumpSlice $r $a $s
              }
              [[$cw cget -sliceGUI] GetSliceViewer] RequestRender
            }
          }

        return
      }
      "EnterEvent" {
#         puts "EnterEvent"
#         if { [$_sliceCompositeNode GetCrosshairMode] != 0 } {
#           # set the cursor to a dot (can't figure out how to hide it)
#           set tkutil [vtkKWTkUtilities New]
#           $tkutil SetTopLevelMouseCursor [[$sliceGUI GetSliceViewer] GetParentToplevel] "rightbutton"
#           $tkutil Delete
#         }

        return
      }
      "LeaveEvent" {
#         puts "LeaveEvent"
#         if { [$_sliceCompositeNode GetCrosshairMode] != 0 } {
#           # show the cursor
#           set tkutil [vtkKWTkUtilities New]
#           $tkutil SetTopLevelMouseCursor [[$sliceGUI GetSliceViewer] GetParentToplevel] "arrow"
#           $tkutil Delete
#         }

        return
      }
    }
  }
}

itcl::body CrosshairSWidget::resetCrosshair { } {

  set idArray [$o(crosshairLines) GetData]
  $idArray Reset
  $idArray InsertNextTuple1 0
  $o(crosshairPoints) Reset
  $o(crosshairLines) SetNumberOfCells 0

}

itcl::body CrosshairSWidget::addCrosshairLine { startPoint endPoint } {

  set startPoint [lrange $startPoint 0 2]
  set endPoint [lrange $endPoint 0 2]
  set startIndex [eval $o(crosshairPoints) InsertNextPoint $startPoint]
  set endIndex [eval $o(crosshairPoints) InsertNextPoint $endPoint]

  set cellCount [$o(crosshairLines) GetNumberOfCells]
  set idArray [$o(crosshairLines) GetData]
  $idArray InsertNextTuple1 2
  $idArray InsertNextTuple1 $startIndex
  $idArray InsertNextTuple1 $endIndex
  $o(crosshairLines) SetNumberOfCells [expr $cellCount + 1]

}


#
# make the crosshair object
#
itcl::body CrosshairSWidget::updateCrosshair { } {

  $this resetCrosshair

  set logic [$sliceGUI GetLogic]
  set sliceCompositeNode [$logic GetSliceCompositeNode]
  if { $sliceCompositeNode == "" } {
    return
  }

  set tkwindow [$_renderWidget  GetWidgetName]
  set w [winfo width $tkwindow]
  set h [winfo height $tkwindow]

  set negW [expr -1.0*$w]
  set negWminus -5
  set negWminus2 -10
  set posWplus 5
  set posWplus2 10
  set posW $w

  set negH [expr -1.0*$h]
  set negHminus -5
  set negHminus2 -10
  set posHplus 5
  set posHplus2 10
  set posH $h



  switch [$sliceCompositeNode GetCrosshairMode] {
    "0" {
      # do nothing
    }
    "1" {
      # show basic
      $this addCrosshairLine "0 $negH 0 0" "0 $negHminus 0 0"
      $this addCrosshairLine "0 $posHplus 0 0" "0 $posH 0 0"
      $this addCrosshairLine "$negW 0 0 0" "$negWminus 0 0 0"
      $this addCrosshairLine "$posWplus 0 0 0" "$posW 0 0 0"
    }
    "2" {
      # show intersection
      $this addCrosshairLine "$negW 0 0 0" "$posW 0 0 0"
      $this addCrosshairLine "0 $negH 0 0" "0 $posH 0 0"
    }
    "3" {
      # show hashmarks
      # TODO: add hashmarks (in cm?)
      $this addCrosshairLine "0 $negH 0 0" "0 $negHminus 0 0"
      $this addCrosshairLine "0 $posHplus 0 0" "0 $posH 0 0"
      $this addCrosshairLine "$negW 0 0 0" "$negWminus 0 0 0"
      $this addCrosshairLine "$posWplus 0 0 0" "$posW 0 0 0"
    }
    "4" {
      # show all
      # TODO: add hashmarks (in cm?)
      # show intersection
      $this addCrosshairLine "$negW 0 0 0" "$posW 0 0 0"
      $this addCrosshairLine "0 $negH 0 0" "0 $posH 0 0"
    }
    "5" {
      # small open cross
      $this addCrosshairLine "0 $negHminus2 0 0" "0 $negHminus 0 0"
      $this addCrosshairLine "0 $posHplus 0 0" "0 $posHplus2 0 0"
      $this addCrosshairLine "$negWminus2 0 0 0" "$negWminus 0 0 0"
      $this addCrosshairLine "$posWplus 0 0 0" "$posWplus2 0 0 0"
    }
    "6" {
      # small closed cross (intersection)
      $this addCrosshairLine "0 $negHminus2 0 0" "0 $posHplus2 0 0"
      $this addCrosshairLine "$negWminus2 0 0 0" "$posWplus2 0 0 0"
    }
  }

  if { [$sliceCompositeNode GetCrosshairThickness] == 1 } {
    [$o(crosshairActor) GetProperty] SetLineWidth 1
  } elseif  { [$sliceCompositeNode GetCrosshairThickness] == 2 } {
    [$o(crosshairActor) GetProperty] SetLineWidth 3
  } elseif  { [$sliceCompositeNode GetCrosshairThickness] == 3 } {
    [$o(crosshairActor) GetProperty] SetLineWidth 5
  } 

  [$sliceGUI GetSliceViewer] RequestRender
}


itcl::body CrosshairSWidget::setPosition { r a s } {

  foreach {x y z } [rasToXYZ "$r $a $s"] {}

  $o(crosshairActor) SetPosition $x $y
}

proc CrosshairSWidget::AddCrosshair {} {
  foreach sw [itcl::find objects -class SliceSWidget] {
    set sliceGUI [$sw cget -sliceGUI]
    if { [info command $sliceGUI] != "" } {
      CrosshairSWidget #auto [$sw cget -sliceGUI]
    }
  }
}

proc CrosshairSWidget::RemoveCrosshair {} {
  foreach pw [itcl::find objects -class CrosshairSWidget] {
    itcl::delete object $pw
  }
}

proc CrosshairSWidget::ToggleCrosshair {} {
  if { [itcl::find objects -class CrosshairSWidget] == "" } {
    CrosshairSWidget::AddCrosshair
  } else {
    CrosshairSWidget::RemoveCrosshair
  }
}

