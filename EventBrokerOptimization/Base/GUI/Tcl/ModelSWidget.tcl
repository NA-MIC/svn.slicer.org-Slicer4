package require Itcl

#########################################################
#
if {0} { ;# comment

  ModelSWidget a class for slicer fiducials in 2D


# TODO : 

}
#
#########################################################

#
#########################################################
# ------------------------------------------------------------------
#                             ModelSWidget
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class ModelSWidget] == "" } {

  itcl::class ModelSWidget {

    inherit SWidget

    constructor {args} {}
    destructor {}

    public variable modelID ""
    public variable opacity "0.5"
    public variable visibility "1"

    variable _modelNode ""
    variable _modelNodeObservation ""
    variable _modelDisplayNodeObservation ""
    variable _sliceCompositeNode ""

    # methods
    method processEvent {{caller ""} {event ""}} {}
    method positionActors {} {}
    method highlight {} {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body ModelSWidget::constructor {sliceGUI} {

  $this configure -sliceGUI $sliceGUI
 
  set o(cutter) [vtkNew vtkCutter]
  set o(plane) [vtkNew vtkPlane]
  $o(cutter) SetCutFunction $o(plane)
  set o(cutTransform) [vtkNew vtkTransform]
  set o(cutTransformFilter) [vtkNew vtkTransformPolyDataFilter]
  $o(cutTransformFilter) SetInputConnection [$o(cutter) GetOutputPort]
  $o(cutTransformFilter) SetTransform $o(cutTransform)

  set o(mapper) [vtkNew vtkPolyDataMapper2D]
  set o(actor) [vtkNew vtkActor2D]
  $o(mapper) SetInputConnection [$o(cutTransformFilter) GetOutputPort]
  $o(actor) SetMapper $o(mapper)
  set _renderer [$_renderWidget GetRenderer]
  $_renderer AddActor2D $o(actor)
  lappend _actors $o(actor)

  set _sliceCompositeNode [[$sliceGUI GetLogic] GetSliceCompositeNode]
  $this configure -visibility [$_sliceCompositeNode GetSliceIntersectionVisibility]

  $this processEvent

  # observe the slice GUI for user input events
  # TODO: no mouse events until we start interacting with the slice nodes
  $::slicer3::Broker AddObservation $sliceGUI DeleteEvent "::SWidget::ProtectedDelete $this"

  # observe the slice node for direct manipulations of MRML
  set node [[$sliceGUI GetLogic] GetSliceNode]
  $::slicer3::Broker AddObservation $node DeleteEvent "::SWidget::ProtectedDelete $this"
  $::slicer3::Broker AddObservation $node AnyEvent "::SWidget::ProtectedCallback $this processEvent $node AnyEvent"

  # observe the composite node for slice plane visibility requests
  $::slicer3::Broker AddObservation $_sliceCompositeNode DeleteEvent "::SWidget::ProtectedDelete $this"
  $::slicer3::Broker AddObservation $_sliceCompositeNode AnyEvent "::SWidget::ProtectedCallback $this processEvent $_sliceCompositeNode AnyEvent"

}

itcl::body ModelSWidget::destructor {} {

  $o(cutTransformFilter) SetInput ""
  $o(cutTransformFilter) SetTransform ""

  if { [info command $_renderer] != "" } {
    foreach a $_actors {
      $_renderer RemoveActor2D $a
    }
  }

}

#
# when told what model to observe...
#
itcl::configbody ModelSWidget::modelID {
  # find the model node
  set modelNode [$::slicer3::MRMLScene GetNodeByID $modelID]
  if { $modelNode == "" } {
    #error "no node for id $modelID"
    return
  }
  set displayNode [$modelNode GetDisplayNode]
  if { $displayNode == "" } {
    #error "no display node for id $modelID"
    return
  }

  # remove observation from old node and add to new node
  # then set input to pipeline
  if { $modelNode != $_modelNode } {
    if { $_modelNodeObservation != "" } {
      $::slicer3::Broker RemoveObservation $_modelNodeObservation
    }
    if { $_modelDisplayNodeObservation != "" } {
      $::slicer3::Broker RemoveObservation $_modelDisplayNodeObservation
    }
    set _modelNode $modelNode
    if { $_modelNode != "" } {
      $o(cutter) SetInput [$_modelNode GetPolyData]
      set _modelNodeObservation [$::slicer3::Broker AddObservation $_modelNode AnyEvent "::SWidget::ProtectedCallback $this processEvent $_modelNode AnyEvent"]
      if { $displayNode != "" } {
        set _modelDisplayNodeObservation [$::slicer3::Broker AddObservation $displayNode AnyEvent "::SWidget::ProtectedCallback $this processEvent $displayNode AnyEvent"]
      }
    }
  }

  $this highlight
  [$sliceGUI GetSliceViewer] RequestRender
}

itcl::configbody ModelSWidget::opacity {
  $this highlight
  [$sliceGUI GetSliceViewer] RequestRender
}

itcl::configbody ModelSWidget::visibility {
  $this highlight
  [$sliceGUI GetSliceViewer] RequestRender
}

# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body ModelSWidget::positionActors { } {

  $o(actor) SetPosition 0 0
  return
}

itcl::body ModelSWidget::highlight { } {

  set property [$o(actor) GetProperty]

  $o(actor) SetVisibility $visibility

  #
  # set color (extracted from the display node)
  #
  set color "0.5 0.5 0.5"
  set modelNode [$::slicer3::MRMLScene GetNodeByID $modelID]
  if { $modelNode != "" } {
    set displayNode [$modelNode GetDisplayNode]
    if { $displayNode != "" } {
      set color [$displayNode GetColor]
    }
  }

  eval $property SetColor $color
  $property SetLineWidth 3
  $property SetOpacity $opacity

  return

}

itcl::body ModelSWidget::processEvent { {caller ""} {event ""} } {


  if { [info command $sliceGUI] == "" || [$sliceGUI GetLogic] == "" } {
    # the sliceGUI was deleted behind our back, so we need to 
    # self destruct
    itcl::delete object $this
    return
  }

  # control visibility based on ModelDisplayNode
  if { $_modelNode != "" } { 
    $o(cutter) SetInput [$_modelNode GetPolyData]
    set displayNode [$_modelNode GetDisplayNode]
    if { $caller == $displayNode } {
      $this configure -visibility [$displayNode GetSliceIntersectionVisibility]
    }
  }

  #
  # update the transform from world to screen space
  # for the extracted cut plane
  #
  $this queryLayers 0 0 0
  set rasToXY [vtkMatrix4x4 New]
  $rasToXY DeepCopy [$_sliceNode GetXYToRAS]
  $rasToXY Invert
  $o(cutTransform) SetMatrix $rasToXY
  $rasToXY Delete

  #
  # update the plane equation for the current slice cutting plane
  # - extract from the slice matrix
  # - normalize the normal
  #
  foreach row {0 1 2} {
    lappend normal [[$_sliceNode GetXYToRAS] GetElement $row 2]
    lappend origin [[$_sliceNode GetXYToRAS] GetElement $row 3]
  }
  set sum 0.
  foreach ele $normal {
    set sum [expr $sum + $ele * $ele]
  }
  set lenInv [expr 1./sqrt($sum)]
  foreach ele $normal {
    lappend norm [expr $ele * $lenInv]
  }
  eval $o(plane) SetNormal $norm
  eval $o(plane) SetOrigin $origin

  set grabID [$sliceGUI GetGrabID]
  if { ! ($grabID == "" || $grabID == $this) } {
    # some other widget wants these events
    # -- we can position wrt the current slice node
    $this positionActors
    $o(cutter) Modified
    [$sliceGUI GetSliceViewer] RequestRender
    return 
  }

  set _actionState ""
  $sliceGUI SetGrabID ""

  $this highlight
  $this positionActors
  $o(cutter) Modified
  [$sliceGUI GetSliceViewer] RequestRender
}

#
# The Rasterize proc is an experiment for creating label maps
# from models.  This works well for simple shapes, but unfortunately 
# does not work for more complex shapes (i.e. with multple contours
# intersecting the slice plane)
#
namespace eval ModelSWidget {
  proc Rasterize { modelSW } {

    set sliceGUI [$modelSW cget -sliceGUI]
    set labeler [Labeler #auto $sliceGUI]
    array set o [$modelSW getObjects]

    set stripper [vtkStripper New]
    set cleaner [vtkCleanPolyData New]
    $stripper SetInput [$o(cutTransformFilter) GetOutput]
    $cleaner SetInput [$stripper GetOutput]
    set polyData [$cleaner GetOutput]
    $polyData Update

    set maskResult [$labeler makeMaskImage $polyData]
    foreach {maskIJKToRAS mask} $maskResult {}
    [$polyData GetPoints] Modified
    set bounds [$polyData GetBounds]
    $labeler applyImageMask $maskIJKToRAS $mask $bounds

    itcl::delete object $labeler
  }
}
