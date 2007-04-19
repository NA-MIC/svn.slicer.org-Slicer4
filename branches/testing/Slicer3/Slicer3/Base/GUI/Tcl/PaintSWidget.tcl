
package require Itcl

#########################################################
#
if {0} { ;# comment

  PaintSWidget a class for slicer painting


# TODO : 

}
#
#########################################################

#
#########################################################
# ------------------------------------------------------------------
#                             PaintSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class PaintSWidget] == "" } {

  itcl::class PaintSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    public variable paintColor 1
    public variable thresholdPaint 0
    public variable thresholdMin 1
    public variable thresholdMax 1
    public variable radius 10
    public variable paintDropper 0
    public variable paintOver 1
    public variable delayedPaint 1

    variable _startPosition "0 0 0"
    variable _currentPosition "0 0 0"
    variable _paintCoordinates ""
    variable _feedbackActors ""

    # methods
    method processEvent {} {}
    method positionActors {} {}
    method highlight {} {}
    method createGlyph { {polyData ""} } {}
    method paintBrush {x y} {}
    method paintAddPoint {x y} {}
    method paintFeedback {} {}
    method paintApply {} {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body PaintSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI
 
  set o(brush) [vtkNew vtkPolyData]
  $this createGlyph $o(brush)
  set o(mapper) [vtkNew vtkPolyDataMapper2D]
  set o(actor) [vtkNew vtkActor2D]
  $o(mapper) SetInput $o(brush)
  $o(actor) SetMapper $o(mapper)
  $o(actor) VisibilityOff
  [$_renderWidget GetRenderer] AddActor2D $o(actor)
  lappend _actors $o(actor)

  set _startPosition "0 0 0"
  set _currentPosition "0 0 0"

  $this processEvent

  set _guiObserverTags ""
  lappend _guiObserverTags [$sliceGUI AddObserver DeleteEvent "::SWidget::ProtectedDelete $this"]
  foreach event {LeftButtonPressEvent LeftButtonReleaseEvent MouseMoveEvent EnterEvent LeaveEvent} {
    lappend _guiObserverTags [$sliceGUI AddObserver $event "::SWidget::ProtectedCallback $this processEvent"]    }
  set node [[$sliceGUI GetLogic] GetSliceNode]
  lappend _nodeObserverTags [$node AddObserver DeleteEvent "::SWidget::ProtectedDelete $this"]
  lappend _nodeObserverTags [$node AddObserver AnyEvent "::SWidget::ProtectedCallback $this processEvent"]
}

itcl::body PaintSWidget::destructor {} {

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

itcl::configbody PaintSWidget::radius {
  $this createGlyph $o(brush)
}

# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body PaintSWidget::createGlyph { {polyData ""} } {
  # make a circle paint brush
  if { $polyData == "" } {
    set polyData [vtkNew vtkPolyData]
  }
  set points [vtkPoints New]
  set lines [vtkCellArray New]
  $polyData SetPoints $points
  $polyData SetLines $lines
  set PI 3.1415926
  set TWOPI [expr $PI * 2]
  set PIoverFOUR [expr $PI / 4]
  set PIoverEIGHT [expr $PI / 8]
  set PIoverSIXTEEN [expr $PI / 16]
  set prevPoint ""
  set firstPoint ""
  for { set angle 0 } { $angle <= $TWOPI } { set angle [expr $angle + $PIoverSIXTEEN] } {
    set x [expr $radius * cos($angle)]
    set y [expr $radius * sin($angle)]
    set p [$points InsertNextPoint $x $y 0]
    if { $prevPoint != "" } {
      set idList [vtkIdList New]
      $idList InsertNextId $prevPoint
      $idList InsertNextId $p
      $polyData InsertNextCell 3 $idList  ;# 3 is a VTK_LINE
      $idList Delete
    }
    set prevPoint $p
    if { $firstPoint == "" } {
      set firstPoint $p
    }
  }
  # make the last line in the circle
  set idList [vtkIdList New]
  $idList InsertNextId $p
  $idList InsertNextId $firstPoint
  $polyData InsertNextCell 3 $idList  ;# 3 is a VTK_LINE
  $idList Delete

  $points Delete
  $lines Delete
  return $polyData
}

itcl::body PaintSWidget::positionActors { } {

  set xyzw [$this rasToXY $_currentPosition]
  eval $o(actor) SetPosition [lrange $xyzw 0 1]
}

itcl::body PaintSWidget::highlight { } {

  set property [$o(actor) GetProperty]
  $property SetColor 1 1 0
  $property SetLineWidth 1
  set _description ""
  switch $_actionState {
    "dragging" {
      $property SetColor 0 1 0
      set _description "Move mouse with left button down to drag"
    }
    default {
      switch $_pickState {
        "over" {
          $property SetColor 1 1 0
          $property SetLineWidth 2
        }
      }
    }
  }
}

itcl::body PaintSWidget::processEvent { } {

  if { [info command $sliceGUI] == "" } {
    # the sliceGUI was deleted behind our back, so we need to 
    # self destruct
    itcl::delete object $this
    return
  }

  set grabID [$sliceGUI GetGrabID]
  if { ($grabID != "") && ($grabID != $this) } {
    # some other widget wants these events
    # -- we can position wrt the current slice node
    $this positionActors
    [$sliceGUI GetSliceViewer] RequestRender
    return 
  }

  set event [$sliceGUI GetCurrentGUIEvent] 

  switch $event {
    "LeftButtonPressEvent" {
      set _actionState "painting"
      foreach {x y} [$_interactor GetEventPosition] {}
      if { $paintDropper } {
        # in Dropper mode, set the paint color to be the first pixel you touch
        $this queryLayers $x $y
        set paintColor [$this getPixel $_layers(label,image) \
          $_layers(label,i) $_layers(label,j) $_layers(label,k)]
      }
      $this paintAddPoint $x $y
      $sliceGUI SetGUICommandAbortFlag 1
      $sliceGUI SetGrabID $this
      [$_renderWidget GetRenderWindow] HideCursor
    }
    "MouseMoveEvent" {
      $o(actor) VisibilityOn
      set _currentPosition [$this xyToRAS [$_interactor GetEventPosition]]
      switch $_actionState {
        "painting" {
          foreach {x y} [$_interactor GetEventPosition] {}
          $this paintAddPoint $x $y
        }
        default {
        }
      }
    }
    "LeftButtonReleaseEvent" {
      $this paintApply
      [$_renderWidget GetRenderWindow] ShowCursor
      $_layers(label,node) Modified
      set _actionState ""
      $sliceGUI SetGrabID ""
      set _description ""
    }
    "EnterEvent" {
      set _description "Ready to Paint!"
      $o(actor) VisibilityOn
    }
    "LeaveEvent" {
      set _description ""
      $o(actor) VisibilityOff
    }
  }

  $this highlight
  $this positionActors
  [$sliceGUI GetSliceViewer] RequestRender
}

itcl::body PaintSWidget::paintFeedback {} {

  set renderer [$_renderWidget GetRenderer]

  foreach a $_feedbackActors {
    $renderer RemoveActor2D $a
    $a Delete
  }
  set _feedbackActors ""

  foreach xy $_paintCoordinates {
    set a [vtkActor2D New]
    lappend _feedbackActors $a
    $a SetMapper $o(mapper)
    eval $a SetPosition $xy
    set property [$a GetProperty]
    $property SetColor .7 .7 0
    $property SetOpacity .5
    $renderer AddActor2D $a
  }
}

itcl::body PaintSWidget::paintAddPoint {x y} {

  lappend _paintCoordinates "$x $y"
  if { $delayedPaint } {
    paintFeedback 
  } else {
    $this paintApply
  }
}

itcl::body PaintSWidget::paintApply {} {

  foreach xy $_paintCoordinates {
    eval $this paintBrush $xy
  }
  set _paintCoordinates ""
  paintFeedback
}


## Note: there is a pure-tcl implementation of the painting
##  in the subversion history - it is slower (of course) but
##  useful for debugging (and was remarkably terse)
itcl::body PaintSWidget::paintBrush {x y} {

  #
  # paint with a brush that is circular in XY space 
  # (could be streched or rotate when transformed to IJK)
  # - make sure to hit ever pixel in IJK space 
  # - apply the threshold if selected
  #

  $this queryLayers $x $y

  if { $_layers(label,node) == "" } {
    # if there's no label, we can't paint
    return
  }

  #
  # get the brush bounding box in ijk coordinates
  # - get the xy bounds
  # - transform to ijk
  # - clamp the bounds to the dimensions of the label image
  #
  set bounds [[$o(brush) GetPoints] GetBounds]
  set left [expr $x + [lindex $bounds 0]]
  set right [expr $x + [lindex $bounds 1]]
  set bottom [expr $y + [lindex $bounds 2]]
  set top [expr $y + [lindex $bounds 3]]

  set xyToIJK [[$_layers(label,logic) GetXYToIJKTransform] GetMatrix]
  set tlIJK [$xyToIJK MultiplyPoint $left $top 0 1]
  set trIJK [$xyToIJK MultiplyPoint $right $top 0 1]
  set blIJK [$xyToIJK MultiplyPoint $left $bottom 0 1]
  set brIJK [$xyToIJK MultiplyPoint $right $bottom 0 1]

  set dims [$_layers(label,image) GetDimensions]
  foreach v {i j k} c [lrange $tlIJK 0 2] d $dims {
    set tl($v) [expr int(round($c))]
    if { $tl($v) < 0 } { set tl($v) 0 }
    if { $tl($v) >= $d } { set tl($v) [expr $d - 1] }
  }
  foreach v {i j k} c [lrange $trIJK 0 2] d $dims {
    set tr($v) [expr int(round($c))]
    if { $tr($v) < 0 } { set tr($v) 0 }
    if { $tr($v) >= $d } { set tr($v) [expr $d - 1] }
  }
  foreach v {i j k} c [lrange $blIJK 0 2] d $dims {
    set bl($v) [expr int(round($c))]
    if { $bl($v) < 0 } { set bl($v) 0 }
    if { $bl($v) >= $d } { set bl($v) [expr $d - 1] }
  }
  foreach v {i j k} c [lrange $brIJK 0 2] d $dims {
    set br($v) [expr int(round($c))]
    if { $br($v) < 0 } { set br($v) 0 }
    if { $br($v) >= $d } { set br($v) [expr $d - 1] }
  }


  #
  # get the ijk to ras matrices 
  #
  set backgroundIJKToRAS [vtkMatrix4x4 New]
  $_layers(background,node) GetIJKToRASMatrix $backgroundIJKToRAS
  set labelIJKToRAS [vtkMatrix4x4 New]
  $_layers(label,node) GetIJKToRASMatrix $labelIJKToRAS

  #
  # get the brush center and radius in World
  # - assume uniform scaling between XY and RAS
  #
  set xyToRAS [$_sliceNode GetXYToRAS]
  set brushCenter [lrange [$xyToRAS MultiplyPoint $x $y 0 1] 0 2]
  set worldScale [lrange [$xyToRAS MultiplyPoint 1 1 1 0] 0 2]
  set scale 1
  foreach c $worldScale {
    if { $c != 1 } { 
      set scale $c
      break
    }
  }
  set brushRadius [expr $scale * $radius]

  #
  # set up the painter class and let 'r rip!
  #
  set extractImage [vtkImageData New]
  set painter [vtkImageSlicePaint New]
  $painter SetBackgroundImage $_layers(background,image)
  $painter SetBackgroundIJKToWorld $backgroundIJKToRAS
  $painter SetWorkingImage $_layers(label,image)
  $painter SetWorkingIJKToWorld $labelIJKToRAS
  $painter SetTopLeft $tl(i) $tl(j) $tl(k)
  $painter SetTopRight $tr(i) $tr(j) $tr(k)
  $painter SetBottomLeft $bl(i) $bl(j) $bl(k)
  $painter SetBottomRight $br(i) $br(j) $br(k)
  eval $painter SetBrushCenter $brushCenter
  $painter SetBrushRadius $brushRadius
  $painter SetPaintLabel $paintColor
  $painter SetPaintOver $paintOver
  $painter SetThresholdPaint $thresholdPaint
  $painter SetThresholdPaintRange $thresholdMin $thresholdMax
  $painter Paint

  $extractImage Delete
  $painter Delete
  $labelIJKToRAS Delete
  $backgroundIJKToRAS Delete


  return
}

proc PaintSWidget::AddPaint {} {
  foreach sw [itcl::find objects -class SliceSWidget] {
    set sliceGUI [$sw cget -sliceGUI]
    if { [info command $sliceGUI] != "" } {
      PaintSWidget #auto [$sw cget -sliceGUI]
    }
  }
}

proc PaintSWidget::RemovePaint {} {
  foreach pw [itcl::find objects -class PaintSWidget] {
    itcl::delete object $pw
  }
}

proc PaintSWidget::TogglePaint {} {
  if { [itcl::find objects -class PaintSWidget] == "" } {
    PaintSWidget::AddPaint
  } else {
    PaintSWidget::RemovePaint
  }
}

proc PaintSWidget::ConfigureAll { args } {
  foreach pw [itcl::find objects -class PaintSWidget] {
    eval $pw configure $args
  }
}
