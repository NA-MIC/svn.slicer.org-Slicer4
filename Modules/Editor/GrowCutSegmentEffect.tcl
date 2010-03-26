
package require Itcl

#########################################################
#
if {0} { ;# comment

   GrowCutSegmentEffect an editor effect
# TODO : 

}

#
#########################################################

#
#########################################################
# ------------------------------------------------------------------
#                             GrowCutSegmentEffect
# ------------------------------------------------------------------
#
# The class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class GrowCutSegmentEffect] == "" } {

    itcl::class GrowCutSegmentEffect {

  inherit PaintEffect 

  constructor {sliceGUI} {PaintEffect::constructor $sliceGUI} {}
  destructor {}
  
  public variable maxIterations 10
  public variable contrastNoiseRatio 80
  public variable priorStrength 0.0003
  public variable overlay 0
  public variable segmented 2

  variable _segmentedImage ""
  variable _gestureImage ""
  
  variable _labelColors ""

  variable _setGestures 0
  variable _changedImage 0
  
  # methods
  method processEvent {{caller ""} {event ""} } {}
  method buildOptions {} {}
  method tearDownOptions {} {}
  method setMRMLDefaults {} {}
  method updateMRMLFromGUI {} {}
  method updateGUIFromMRML {} {}
  method apply {} {} 
  method initialize {} {}
  method overlayGestures {} {}
  method extractGestures {} {}
  method setLabelMapToGestures {} {}
  method setLabelMapToSegmented {} {}
    }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body GrowCutSegmentEffect::constructor {sliceGUI} {
    
   $this configure -sliceGUI $sliceGUI

   puts "GrowCutSegmentEffect"
   
   set _segmentedImage [vtkNew vtkImageData]
   set _gestureImage [vtkNew vtkImageData]

   $this initialize

   # remove observers 
#    set _guiObserverTags ""
#    foreach event {MouseEvent LeftButtonReleaseEvent EnterEvent LeaveEvent} {
       
#    }
}

itcl::body GrowCutSegmentEffect::destructor {} {

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

itcl::body GrowCutSegmentEffect::initialize {} {
    
    puts "Initializing... "

    # remove observers that we don't use
    

    set dim [[$this getInputLabel] GetDimensions]
    set x [lindex $dim 0]
    set y [lindex $dim 1]
    set z [lindex $dim 2]
    puts "image dimensions : $x $y $z"
    $_gestureImage SetDimensions $x $y $z
    
    set extent [[$this getInputLabel] GetExtent]
    $_gestureImage SetExtent [lindex $extent 0] [lindex $extent 1] \
  [lindex $extent 2] [lindex $extent 3] \
  [lindex $extent 4] [lindex $extent 5]
    
    puts "image extent : $extent"

    set origin [[$this getInputLabel] GetOrigin]
    $_gestureImage SetOrigin [lindex $origin 0] [lindex $origin 1] [lindex $origin 2]
    
    set spacing [[$this getInputLabel] GetSpacing]
    $_gestureImage SetSpacing [lindex $spacing 0] [lindex $spacing 1] [lindex $spacing 2]

    $_gestureImage SetScalarType [[$this getInputLabel] GetScalarType]
    $_gestureImage AllocateScalars
    $_gestureImage DeepCopy [$this getInputLabel]


    $_segmentedImage SetDimensions $x $y $z
    
    $_segmentedImage SetExtent [lindex $extent 0] [lindex $extent 1] \
  [lindex $extent 2] [lindex $extent 3] \
  [lindex $extent 4] [lindex $extent 5]
    
    $_segmentedImage SetOrigin [lindex $origin 0] [lindex $origin 1] [lindex $origin 2]
    
    $_segmentedImage SetSpacing [lindex $spacing 0] [lindex $spacing 1] [lindex $spacing 2]
    
    $_segmentedImage SetScalarType [[$this getInputLabel] GetScalarType]
    $_segmentedImage AllocateScalars
    
    $_segmentedImage DeepCopy [$this getInputLabel]
    
#    $_segmentedImage Register
}

# ------------------------------------------------------------------
#                             HELPER METHODS
# ------------------------------------------------------------------


itcl::body GrowCutSegmentEffect::overlayGestures {} {
    

    puts "overlaying the gestures on the segmented image "
    
    catch {
    # method to overlay the old gestures on the segmentation
    set imageLogic [vtkNew vtkITKLogicImageFilter]
    $imageLogic SetInput1 $_gestureImage
    $imageLogic SetInput2 [$this getInputLabel]
    $imageLogic SetOperationToOr 
    
    $_layers(label,node) SetAndObserveImageData [$imageLogic GetOutput]
    $_layers(label,node) Modified
    
    $imageLogic Update
   } foo
    if { $foo != "" } { puts "error in overlay : $foo" }
   
}



itcl::body GrowCutSegmentEffect::extractGestures {} {
    
    puts "extracting gestures..."
    # method to get the gestures alone
    catch {
  set imageLogic [vtkNew vtkITKLogicImageFilter]
  $imageLogic SetInput1 [$this getInputLabel]
  $imageLogic SetInput2 $_segmentedImage
  $imageLogic SetOperationToXor

  $imageLogic Update
    
#  puts " doing an OR operation for copying the gestures alone"
  catch {
      set imageLogic1 [vtkNew vtkITKLogicImageFilter]
      $imageLogic1 SetInput1 [$imageLogic GetOutput]
      $imageLogic1 SetInput2 $_gestureImage
      $imageLogic1 SetOperationToOr
      $imageLogic1 Update
      catch { $_gestureImage DeepCopy [$imageLogic1 GetOutput] } err3
      if { $err3 != "" } { puts "error in copying gesture image : $err3" }
      
  } err2
  if { $err2 != "" } {puts "error in extracting gestures 2: $err2" }
    } err1  
    if { $err1 != "" } {puts "error in extracting gestures: $err1" }


}


itcl::body GrowCutSegmentEffect::setLabelMapToGestures {} {
    
    puts "setting the label map to gestures "
    $_layers(label,node) SetAndObserveImageData $_gestureImage
    $_layers(label,node) Modified
}


itcl::body GrowCutSegmentEffect::setLabelMapToSegmented {} {

    puts "setting the label map to segmented "
    $_layers(label,node) SetAndObserveImageData $_segmentedImage
    $_layers(label,node) Modified 
}



itcl::body GrowCutSegmentEffect::processEvent { {caller ""} {event ""} } {

    puts "In ProcessEvent : GrowCutSegmentEffect "

  # chain to superclass
    chain $caller $event

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
    
   if { [info exists o(overlay)] } {
       if { $overlay } {
     if { $_changedImage } {
         puts "overlaying gestures "
         
         $this overlayGestures
         set segmented 1
         set _changedImage 0
     }
       } else {
     
     if { $segmented == 1} {
         puts "segmented is : $segmented"
         $this setLabelMapToSegmented
         set segmented 0
         set _changedImage 1
     }
       }
   }


  set event [$sliceGUI GetCurrentGUIEvent] 
  
  catch {
  switch $event {
    "LeftButtonPressEvent" {

      set _changedImage 1
      
      foreach {x y} [$_interactor GetEventPosition] {}              
      if { $smudge } {
    continue
      } else {
    set activeColor [EditorGetPaintLabel]
    if { $activeColor != 0 } {
        if { [lsearch $_labelColors $activeColor] == -1 } {
      lappend _labelColors $activeColor
        }
    }
      }
  }
  }
  } catchString
  set catch [string first "vtkObj" $catchString]
  if { $catchString != "" & $catch == -1 } { puts " error in switching in processEvent : $catchString "}

  $this highlight
  $this positionActors
  [$sliceGUI GetSliceViewer] RequestRender

}


itcl::body GrowCutSegmentEffect::apply {} {

  puts "in Apply method"
  
  set _changedImage 1
  
  if { [$this getInputLabel] == "" || [$this getInputBackground] == "" } {
    $this flashCursor 3
    return
  }

  puts "Applying to run the Grow Cut Segmentation Algorithm"

  if { ![info exists growCutFilter] } {
      set growCutFilter [vtkNew vtkITKGrowCutSegmentationImageFilter]
  }
  
  set gestureColors [vtkPoints New]
  for { set i 0 } {$i < [llength $_labelColors]} {incr i} {
      set j [expr $i + 1]
      set t 1
      set l [lindex $_labelColors $i]
      $gestureColors InsertNextPoint $l $j $t
  }
#  set _labelColors [lreplace _labelColors 0 [llength $_labelColors]]
  catch { 
      $this extractGestures 
  } checkGestures
  if { $checkGestures != "" } {puts "cannot copy gestures : $checkGestures"}
  
  catch {
      puts "$_labelColors"
      
      $growCutFilter SetInput 0 [$this getInputBackground] 
      if { $_setGestures } {
    $growCutFilter SetInput 2 [$this getInputLabel]
    $growCutFilter SetInput 1 $_gestureImage 
      } else {
    $growCutFilter SetInput 2 [$this getInputLabel]
    $growCutFilter SetInput 1 [$this getInputLabel]
      }

      #       $growCutFilter SetInput 0 [$this getInputBackground]
      #       $growCutFilter SetInput 1 [$this getInputLabel]
      puts "max Iterations $maxIterations"
      $growCutFilter SetMaxIterations $maxIterations
      $growCutFilter SetContrastNoiseRatio $contrastNoiseRatio
      $growCutFilter SetGestureColors $gestureColors
      $growCutFilter SetPriorSegmentConfidence  $priorStrength 
      
      $_layers(label,node) SetAndObserveImageData [$growCutFilter GetOutput]
      $_layers(label,node) Modified

      $growCutFilter Update

      $_segmentedImage DeepCopy [$growCutFilter GetOutput]
      
      set _setGestures 1
      # copy the filter output to the label image
  } checkFilter
#  if { $checkFilter != "" } { puts "error in applying : $checkFilter" }

  gestureColors Delete
  
}

itcl::body GrowCutSegmentEffect::buildOptions {} {
   

  puts "Build options before chaining"
 
  # call superclass version of buildOptions
  chain

  # forget the options set by the labeler
  foreach w "paintThreshold paintRange cancel" {
      if { [info exists o($w)] } {
    $o($w) SetParent ""
    pack forget [$o($w) GetWidgetName]
      }
  }

  puts "bulding options.... "

  #
  # set max Iterations control
  #
  catch {
  set o(maxIterations) [vtkNew vtkKWThumbWheel]
  $o(maxIterations) SetParent [$this getOptionsFrame]
  $o(maxIterations) PopupModeOn
  $o(maxIterations) Create
  $o(maxIterations) DisplayEntryAndLabelOnTopOn
  $o(maxIterations) DisplayEntryOn
  $o(maxIterations) DisplayLabelOn
  $o(maxIterations) SetValue [[EditorGetParameterNode] GetParameter GrowCutSegment,maxIterations]
  $o(maxIterations) SetMinimumValue 1
  $o(maxIterations) ClampMinimumValueOn
  [$o(maxIterations) GetLabel] SetText "Max Iterations: "
  $o(maxIterations) SetBalloonHelpString "Set the maximum number of iterations for the algorithm"
  pack [$o(maxIterations) GetWidgetName] \
      -side top -anchor e -fill x -padx 2 -pady 2 
  
#  puts "max Iterations "
  } test0
  if { $test0 != ""} { puts "after max  Iterations : $test0" }
  
   #
   # set the contrast noise ratio for the selected class 
   #
  catch {
  set o(contrastNoiseRatio) [vtkNew vtkKWThumbWheel]
  $o(contrastNoiseRatio) SetParent [$this getOptionsFrame]
  $o(contrastNoiseRatio) PopupModeOn
  $o(contrastNoiseRatio) Create
  $o(contrastNoiseRatio) DisplayEntryAndLabelOnTopOn
  $o(contrastNoiseRatio) DisplayEntryOn
  $o(contrastNoiseRatio) DisplayLabelOn
  $o(contrastNoiseRatio) SetValue [[EditorGetParameterNode] GetParameter GrowCutSegment,contrastNoiseRatio] 
  $o(contrastNoiseRatio) SetMinimumValue 1
  $o(contrastNoiseRatio) SetMaximumValue 100
  $o(contrastNoiseRatio) ClampMinimumValueOn
  $o(contrastNoiseRatio) ClampMaximumValueOn
  [$o(contrastNoiseRatio) GetLabel] SetText "Contrast Ratio: "
 $o(contrastNoiseRatio) SetBalloonHelpString "Set the relative contrast between foreground class and its surrounding background \[1 to 100\]."
  pack [$o(contrastNoiseRatio) GetWidgetName] \
     -side top -anchor e -fill x -padx 2 -pady 2 

#  puts "contrast Ratio "
  } test1 
  if { $test1 != "" } {puts " after maxIter and cNR : $test1" }

   #  
  # set the prior strength if using a segmented label image to start segmentation
  #
 catch {
 set o(priorStrength) [vtkNew vtkKWThumbWheel]
 $o(priorStrength) SetParent [$this getOptionsFrame]
 $o(priorStrength) PopupModeOn
 $o(priorStrength) Create
 $o(priorStrength) DisplayEntryAndLabelOnTopOn
 $o(priorStrength) DisplayEntryOn
 $o(priorStrength) DisplayLabelOn
 $o(priorStrength) SetValue [[EditorGetParameterNode] GetParameter GrowCutSegment,priorStrength] 
 $o(priorStrength) SetMinimumValue 0
 $o(priorStrength) SetMaximumValue 1
 $o(priorStrength) ClampMinimumValueOn
 $o(priorStrength) ClampMaximumValueOn
 [$o(priorStrength) GetLabel] SetText "Prev Segment Confidence: "
 $o(priorStrength) SetBalloonHelpString "Set the confidence in the initial segmentation if using an existing segmentation for correction \[0 to 1\]."
 pack [$o(priorStrength) GetWidgetName] \
     -side top -anchor e -fill x -padx 2 -pady 2 

# puts "prior Strength "

 } testPrior
 if { $testPrior != ""} { puts " after priorStrength: $testPrior" }
 
  #
  # set display to overlay gestures
  #
  catch {
  set o(overlay) [vtkKWCheckButtonWithLabel New]
  $o(overlay) SetParent [$this getOptionsFrame]
  $o(overlay) Create
  $o(overlay) SetLabelText "Overlay Gestures: "
  $o(overlay) SetBalloonHelpString "Set the Gestures input from previous interactions to be overlayed on the segmented label map. You may add additional gestures to correct segmentation."
  [$o(overlay) GetWidget] SetSelectedState [[EditorGetParameterNode] GetParameter GrowCutSegment,overlay] 
  pack [$o(overlay) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 
  } testOverlay
  if { $testOverlay != "" } {puts " after overlay : $testOverlay" }
  
  
  #
  # set display to segmented
  #
#   catch {
#   set o(segmented) [vtkKWCheckButtonWithLabel New]
#   $o(segmented) SetParent [$this getOptionsFrame]
#   $o(segmented) Create
#   $o(segmented) SetLabelText "Display segmented: "
#   $o(segmented) SetBalloonHelpString "Set to show the segmented image."
#   [$o(segmented) GetWidget] SetSelectedState [[EditorGetParameterNode] GetParameter GrowCutSegment,segmented] 
#   pack [$o(segmented) GetWidgetName] \
#     -side top -anchor e -fill x -padx 2 -pady 2 
#   } testSeg
#   if { $testSeg != "" } {puts " after segmented : $testSeg" }

  #
  # a help button
  #
  set o(help) [vtkNew vtkSlicerPopUpHelpWidget]
  $o(help) SetParent [$this getOptionsFrame]
  $o(help) Create
  $o(help) SetHelpTitle "Grow Cut Segmentation"
  $o(help) SetHelpText "Use this tool to interactively segment the object of interest. \nFirst select the number of classes. \nThen select each class 1 to N, then paint strokes on the region corresponding to that label. \nOptionally, edit an already existing segmentation by loading a label map, and editing that map. To segment, 
add gestures or draw inside regions of interest. Place background gestures anywhere around object. Paint objects with the lower paint values and use highest paint value for background. Ex if 1,2,3 are object paints, 4 or higher is background."
  $o(help) SetBalloonHelpString "Bring up help window."
  pack [$o(help) GetWidgetName] \
    -side right -anchor sw -padx 2 -pady 2 

  #
  # an apply button
  #
   set o(apply) [vtkNew vtkKWPushButton]
   $o(apply) SetParent [$this getOptionsFrame]
   $o(apply) Create
   $o(apply) SetText "Apply"
   $o(apply) SetBalloonHelpString "Apply to run segmentation.\n Use the 'a' hotkey to apply segmentation"
   pack [$o(apply) GetWidgetName] \
      -side right -anchor e -padx 2 -pady 2 

  #
  # a cancel button
  #
  set o(cancel) [vtkNew vtkKWPushButton]
  $o(cancel) SetParent [$this getOptionsFrame]
  $o(cancel) Create
  $o(cancel) SetText "Cancel"
  $o(cancel) SetBalloonHelpString "Cancel current segmentation."
  pack [$o(cancel) GetWidgetName] \
      -side right -anchor e -padx 2 -pady 2 


  if { [info exists o(overlay)] } {
      puts "o(overlay) exists"
  } else {
      puts "o(overlay) doesn't exist"
  }

  #
  # event observers - TODO: if there were a way to make these more specific, I would...
  set tag [$o(maxIterations) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(maxIterations) $tag"
  
  set tag [$o(contrastNoiseRatio) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(contrastNoiseRatio) $tag"

  set tag [$o(priorStrength) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(priorStrength) $tag"

  set tag [[$o(overlay) GetWidget] AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(overlay) $tag"

#   set tag [[$o(segmented) GetWidget] AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
#   lappend _observerRecords "$o(segmented) $tag"
  
  set tag [$o(apply) AddObserver AnyEvent "$this apply"]
  lappend _observerRecords "$o(apply) $tag"
  set tag [$o(cancel) AddObserver AnyEvent "after idle ::EffectSWidget::RemoveAll"]
  lappend _observerRecords "$o(cancel) $tag"

  if { [$this getInputBackground] == "" || [$this getInputLabel] == "" } {
    $this errorDialog "Background and Label map needed for Grow Cut Segmentation"
    after idle ::EffectSWidget::RemoveAll
  }
 
  $this updateGUIFromMRML
}


itcl::body GrowCutSegmentEffect::tearDownOptions { } {

  # call superclass version of tearDownOptions
  chain

  foreach w "help cancel apply maxIterations contrastNoiseRatio priorStrength overlay" {
      if { [info exists o($w)] } {
    $o($w) SetParent ""
    pack forget [$o($w) GetWidgetName] 
      }
  }
  
  if { $segmented != 2 & $_setGestures } {
      $this setLabelMapToSegmented
  }

}


itcl::body GrowCutSegmentEffect::updateMRMLFromGUI { } {
    
    chain
    set node [EditorGetParameterNode]

    $this configure -maxIterations [$o(maxIterations) GetValue]
    $node SetParameter "GrowCutSegment,maxIterations" $maxIterations   

    $this configure -contrastNoiseRatio [$o(contrastNoiseRatio) GetValue]
    $node SetParameter "GrowCutSegment,contrastNoiseRatio" $contrastNoiseRatio   

    $this configure -priorStrength [$o(priorStrength) GetValue]
    $node SetParameter "GrowCutSegment,priorStrength" $priorStrength   
    
    $this configure -overlay [[$o(overlay) GetWidget] GetSelectedState]
    $node SetParameter "GrowCutSegment,overlay" $overlay
    
}


itcl::body GrowCutSegmentEffect::setMRMLDefaults { } {
    
    chain
    
    puts "setting MRML defaults in Grow Cut"
    
    set node [EditorGetParameterNode]
    foreach {param default} {
  maxIterations 5
  contrastNoiseRatio 80
  priorStrength 0.00003
  overlay 0
  segmented 2
    } {
        set pvalue [$node GetParameter Paint,$param] 
  if { $pvalue == "" } {
      $node SetParameter GrowCutSegment,$param $default
  }
    }
}



itcl::body GrowCutSegmentEffect::updateGUIFromMRML { } {
  #
  # get the parameter from the node
  # - set default value if it doesn't exist
  #
  chain

  set node [EditorGetParameterNode]
  # set the GUI and effect parameters to match node
  # (only if this is the instance that "owns" the GUI
  
  set maxIterations [$node GetParameter GrowCutSegment,maxIterations] 
  $this configure -maxIterations $maxIterations
  if { [info exists o(maxIterations)] } {
      $o(maxIterations) SetValue $maxIterations
  }

  set contrastNoiseRatio [$node GetParameter GrowCutSegment,contrastNoiseRatio] 
  $this configure -contrastNoiseRatio $contrastNoiseRatio
  if { [info exists o(contrastNoiseRatio)] } {
      $o(contrastNoiseRatio) SetValue $contrastNoiseRatio
  }

 set priorStrength [$node GetParameter GrowCutSegment,priorStrength] 
  $this configure -priorStrength $priorStrength
  if { [info exists o(priorStrength)] } {
    $o(priorStrength) SetValue $priorStrength
  }

 set overlay [$node GetParameter GrowCutSegment,overlay] 
 $this configure -overlay $overlay
 if { [info exists o(overlay)] } {
   [$o(overlay) GetWidget] SetSelectedState $overlay
 }

#  set segmented [$node GetParameter GrowCutSegment,segmented] 
#  $this configure -segmented $segmented
#  if { [info exists o(segmented)] } {
#    [$o(segmented) GetWidget] SetSelectedState $segmented
#  }

  $this preview
}

