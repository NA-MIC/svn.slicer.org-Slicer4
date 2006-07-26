
package require Itcl

#########################################################
#
if {0} { ;# comment

  a class for collecting information about a slicer widget 
  including it's vtk class instances and it's interaction
  state

# TODO : 

}
#
#########################################################


#
# The partent class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class SWidget] == "" } {

  itcl::class SWidget {

    constructor {} {
      # make a unique name associated with this object
      set _name [namespace tail $this]
    }

    destructor {
      vtkDelete  
    }

    # configure options
    public variable state ""  ;# the interaction state of the SWidget
    public variable description ""  ;# a status string describing the current state
    public variable sliceGUI ""  ;# the sliceGUI on which the SWidget lives

    variable _name ""
    variable _vtkObjects ""
    variable _pickState "outside"
    variable _actionState ""

    # methods
    method processEvent {} {}
    method pick {} {}
    method highlight {} {}

    # make a new instance of a class and add it to the list for cleanup
    method vtkNew {class} {
      set o [$class New]
      set _vtkObjects "$o $_vtkObjects"
      return $o
    }

    # clean up the vtk classes instanced by this SWidget
    method vtkDelete {} {
      foreach o $_vtkObjects {
        $o Delete
      }
      set _vtkObjects ""
    }

  }
}



# ------------------------------------------------------------------
#                             CsysSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class CsysSWidget] == "" } {

  itcl::class CsysSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    variable o ;# array of the objects for this widget, for convenient cleanup
    variable _actors "" ;# list of actors for removing from the renderer
    variable _observerTag ;# save so destructor can remove observer

    variable _startPosition "0 0 0"
    variable _currentPosition "0 0 0"

    # methods
    method processEvent {} {}
    method positionActors {} {}
    method pick {} {}
    method highlight {} {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body CsysSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI
  set renderWidget [[$sliceGUI GetSliceViewer] GetRenderWidget]
 
  set o(sphere) [vtkNew vtkSphereSource]
  set o(mapper) [vtkNew vtkPolyDataMapper2D]
  set o(actor) [vtkNew vtkActor2D]
  $o(mapper) SetInput [$o(sphere) GetOutput]
  $o(actor) SetMapper $o(mapper)
  [$renderWidget GetRenderer] AddActor2D $o(actor)
  lappend _actors $o(actor)

  set size [[$renderWidget GetRenderWindow]  GetSize]
  foreach {w h} $size {}
  foreach d {w h} c {cx cy} { set $c [expr [set $d] / 2.0] }

  set _startPosition "$cx $cy 0"
  set _currentPosition "$cx $cy 0"

  $o(sphere) SetRadius 5
  
  $this processEvent
  set _observerTag [$sliceGUI AddObserver AnyEvent "$this processEvent"]
}

itcl::body CsysSWidget::destructor {} {

  $sliceGUI RemoveObserver $_observerTag

  set renderer [[[$sliceGUI GetSliceViewer] GetRenderWidget] GetRenderer]
  foreach a $_actors {
    $renderer RemoveActor2D $a
  }
}

# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body CsysSWidget::pick {} {

  set renderWidget [[$sliceGUI GetSliceViewer] GetRenderWidget]
  set interactor [$renderWidget GetRenderWindowInteractor]

  foreach {cx cy cz} $_currentPosition {}
  foreach {ex ey} [$interactor GetEventPosition] {}
  if { [expr abs($ex - $cx) < 15] && [expr abs($ey - $cy) < 15] } {
    set _pickState "over"
  } else {
    set _pickState "outside"
  }

}

itcl::body CsysSWidget::positionActors { } {
  eval $o(sphere) SetCenter $_currentPosition
}

itcl::body CsysSWidget::highlight { } {

  set property [$o(actor) GetProperty]
  $property SetColor 1 1 1
  switch $_actionState {
    "dragging" {
      $property SetColor 0 1 0
    }
    default {
      switch $_pickState {
        "over" {
          $property SetColor 0 1 1
        }
      }
    }
  }
}

itcl::body CsysSWidget::processEvent { } {


  set grabID [$sliceGUI GetGrabID]
  if { ! ($grabID == "" || $grabID == $this) } {
    return ;# some other widget wants these events
  }
  if { $grabID != $this } {
    # only check pick if we haven't grabbed (avoid 'dropping' the widget
    # when the mouse moves quickly)
    $this pick
  }

  set event [$sliceGUI GetCurrentGUIEvent] 

  switch $_pickState {
    "outside" {
      set _actionState ""
      set _description ""
    }
    "over" {
      $sliceGUI SetGUICommandAbortFlag 1
      switch $event {
        "LeftButtonPressEvent" {
          set _actionState "dragging"
          set _description "Move mouse with left button down to drag"
          $sliceGUI SetGrabID $this
        }
        "MouseMoveEvent" {
          switch $_actionState {
            "dragging" {
              set renderWidget [[$sliceGUI GetSliceViewer] GetRenderWidget]
              set interactor [$renderWidget GetRenderWindowInteractor]
              set _currentPosition "[$interactor GetEventPosition] 0"
            }
            default {
              set _description "Press left mouse button to begin dragging"
            }
          }
        }
        "LeftButtonReleaseEvent" {
          set _actionState ""
          $sliceGUI SetGrabID ""
          set _description ""
        }
      }
    }
  }

  $this highlight
  $this positionActors
  [$sliceGUI GetSliceViewer] RequestRender
}





# ------------------------------------------------------------------
#    slicer interface -- create and remove the widgets
# ------------------------------------------------------------------

proc CsysWidgetRemove {} {

  foreach csys [itcl::find objects -class CsysSWidget] {
    set sliceGUI [$csys cget -sliceGUI]
    itcl::delete object $csys
    [$sliceGUI GetSliceViewer] RequestRender
  }
}

proc CsysWidgetAdd {} {

  CsysWidgetRemove

  set n [[$::slicer3::SlicesGUI GetSliceGUICollection] GetNumberOfItems]
  for {set i 0} {$i < $n} {incr i} {
    set sliceGUI [$::slicer3::SlicesGUI GetSliceGUI $i]
    CsysSWidget #auto $sliceGUI 
  }
}

