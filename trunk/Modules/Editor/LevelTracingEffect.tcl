
package require Itcl

#########################################################
#
if {0} { ;# comment

  LevelTracingEffect an editor effect


# TODO : 

}
#
#########################################################

#
#########################################################
# ------------------------------------------------------------------
#                             LevelTracingEffect
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class LevelTracingEffect] == "" } {

  itcl::class LevelTracingEffect {

    inherit Labeler

    constructor {sliceGUI} {Labeler::constructor $sliceGUI} {}
    destructor {}

    # methods
    method processEvent {} {}
    method preview {} {}
    method apply {} {}
    method buildOptions {} {}
    method tearDownOptions {} {}
  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body LevelTracingEffect::constructor {sliceGUI} {
}

itcl::body LevelTracingEffect::destructor {} {
}

# ------------------------------------------------------------------
#                             METHODS
# ------------------------------------------------------------------

itcl::body LevelTracingEffect::processEvent { } {

  if { [$this preProcessEvent] } {
    # superclass processed the event, so we don't
    return
  }

  set event [$sliceGUI GetCurrentGUIEvent] 
  set _currentPosition [$this xyToRAS [$_interactor GetEventPosition]]

  switch $event {
    "LeftButtonPressEvent" {
      $this apply
      $sliceGUI SetGUICommandAbortFlag 1
    }
    "MouseMoveEvent" {
      $this preview
    }
    "EnterEvent" {
      $o(cursorActor) VisibilityOn
      if { [info exists o(tracingActor)] } {
       $o(tracingActor) VisibilityOn
      }
    }
    "LeaveEvent" {
      $o(cursorActor) VisibilityOff
      if { [info exists o(tracingActor)] } {
       $o(tracingActor) VisibilityOff
      }
    }
  }

  $this positionCursor
  [$sliceGUI GetSliceViewer] RequestRender
}

itcl::body LevelTracingEffect::apply {} {

  if { [$this getInputLabel] == "" || [$this getOutputLabel] == "" } {
    $this flashCursor 3
    return
  }

  $this applyMaskImage $o(tracingPolyData)

}

itcl::body LevelTracingEffect::preview {} {

  foreach {x y} [$_interactor GetEventPosition] {}
  $this queryLayers $x $y

  if { ![info exists o(ijkToXY)] } {
    set o(tracingFilter) [vtkNew vtkITKLevelTracingImageFilter]

    set o(ijkToXY) [vtkNew vtkTransform]
    set o(xyPoints) [vtkNew vtkPoints]

    set o(tracingPolyData) [vtkNew vtkPolyData]
    set o(tracingMapper) [vtkNew vtkPolyDataMapper2D]
    set o(tracingActor) [vtkNew vtkActor2D]
    $o(tracingActor) SetMapper $o(tracingMapper)
    $o(tracingMapper) SetInput $o(tracingPolyData)
    set property [$o(tracingActor) GetProperty]
    $property SetColor [expr 107/255.] [expr 190/255.] [expr 99/255.]
    $property SetLineWidth 1
    [$_renderWidget GetRenderer] AddActor2D $o(tracingActor)
    lappend _actors $o(tracingActor)
  }

  $o(tracingFilter) SetInput [$this getInputBackground]
  $o(tracingFilter) SetSeed $_layers(background,i) $_layers(background,j) $_layers(background,k) 

  # figure out which plane to use
  foreach {i0 j0 k0 l0} [$_layers(background,xyToIJK) MultiplyPoint $x $y 0 1] {}
  set x1 [expr $x + 1]; set y1 [expr $y + 1]
  foreach {i1 j1 k1 l1} [$_layers(background,xyToIJK) MultiplyPoint $x1 $y1 0 1] {}
  if { $i0 == $i1 } { $o(tracingFilter) SetPlaneToJK }
  if { $j0 == $j1 } { $o(tracingFilter) SetPlaneToIK }
  if { $k0 == $k1 } { $o(tracingFilter) SetPlaneToIJ }

  $o(tracingFilter) Update
  set polyData [$o(tracingFilter) GetOutput]
  
  $o(xyPoints) Reset
  $o(ijkToXY) SetMatrix $_layers(background,xyToIJK)
  $o(ijkToXY) Inverse
  $o(ijkToXY) TransformPoints [$polyData GetPoints] $o(xyPoints)

  $o(tracingPolyData) DeepCopy $polyData
  [$o(tracingPolyData) GetPoints] DeepCopy $o(xyPoints)
}
  
itcl::body LevelTracingEffect::buildOptions {} {


  #
  # a cancel button
  #
  set o(cancel) [vtkNew vtkKWPushButton]
  $o(cancel) SetParent [$this getOptionsFrame]
  $o(cancel) Create
  $o(cancel) SetText "Cancel"
  $o(cancel) SetBalloonHelpString "Cancel level tracing without applying to label map."
  pack [$o(cancel) GetWidgetName] \
    -side right -anchor e -padx 2 -pady 2 

  #
  # a help button
  #
  set o(help) [vtkNew vtkSlicerPopUpHelpWidget]
  $o(help) SetParent [$this getOptionsFrame]
  $o(help) Create
  $o(help) SetHelpTitle "Level Tracing"
  $o(help) SetHelpText "Use this tool to track around similar intensity levels."
  $o(help) SetBalloonHelpString "Bring up help window."
  pack [$o(help) GetWidgetName] \
    -side right -anchor sw -padx 2 -pady 2 

  #
  # event observers - TODO: if there were a way to make these more specific, I would...
  #
  set tag [$o(cancel) AddObserver AnyEvent "after idle ::EffectSWidget::RemoveAll"]
  lappend _observerRecords "$o(cancel) $tag"

  if { [$this getInputBackground] == "" || [$this getOutputLabel] == "" } {
    $this errorDialog "Background and Label map needed for Threshold"
    after idle ::EffectSWidget::RemoveAll
  }
}

itcl::body LevelTracingEffect::tearDownOptions { } {
  foreach w "help cancel" {
    if { [info exists o($w)] } {
      $o($w) SetParent ""
      pack forget [$o($w) GetWidgetName] 
    }
  }
}
