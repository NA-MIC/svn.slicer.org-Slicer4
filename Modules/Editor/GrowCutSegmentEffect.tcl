
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
  
#  public variable maxIterations 1
  public variable objectSize 100
  public variable contrastNoiseRatio 0.8
  public variable priorStrength 0.0003
  public variable overlay 0
  public variable segmented 2
  

  variable _maxObjectSize 0
  variable _segmentedImage ""
  variable _gestureImage ""
  
  variable _labelColors ""

  variable _setGestures 0
  variable _changedImage 0
  variable _iterations 0
  
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

   set dim [[$this getInputLabel] GetDimensions]
   set x [lindex $dim 0]
   set y [lindex $dim 1]
   set z [lindex $dim 2]
   
   set _maxObjectSize [expr $x * [expr $y * $z]]
   puts "max object size $_maxObjectSize $x $y $z"

   $this initialize

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

    #[$_sliceNode SetOrientationToAxial()]

    # remove observers that we don't use
    

    
    set dim [[$this getInputLabel] GetDimensions]
    set x [lindex $dim 0]
    set y [lindex $dim 1]
    set z [lindex $dim 2]
    puts "image dimensions : $x $y $z"
    $_gestureImage SetDimensions $x $y $z
    
    #catch {
#    set diagonal [expr [expr $x * $x] + [expr $y * $y] + [expr $z * $z]]
#    set diagonal [expr sqrt ( $diagonal)]
#    set _iterations [expr round ( $diagonal)]
#    set maxIterations $_iterations
    #} err
    #if { $err != "" } {puts "error in setting default max Iterations: $err"}
#    puts "max Iterations : $_iterations"

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

    puts "Done Initializing"
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

  set sliceController [$sliceGUI GetSliceController] 

  #set offsetScale [$sliceController OffsetScaleMin]

  set logic [$sliceGUI GetLogic]
  
  set offset [$logic GetSliceOffset]  

   set redGUI [$::slicer3::ApplicationGUI GetMainSliceGUI Red]
   set redLogic [$redGUI GetLogic]
   set redNode [$redLogic GetSliceNode]
   set redOrient [$redNode GetOrientationString]

   if { [$redNode GetOrientationString] == "Reformat" } {
       $redNode SetOrientationToAxial          
   }

   set greenGUI [$::slicer3::ApplicationGUI GetMainSliceGUI Green]
   set greenLogic [$greenGUI GetLogic]
   set greenNode [$greenLogic GetSliceNode]
   set greenOrient [$greenNode GetOrientationString]
   
   if { [$greenNode GetOrientationString] == "Reformat" } {
       $greenNode SetOrientationToCoronal          
   }

   set yellowGUI [$::slicer3::ApplicationGUI GetMainSliceGUI Yellow]
   set yellowLogic [$yellowGUI GetLogic]
   set yellowNode [$yellowLogic GetSliceNode]
   set yellowOrient [$yellowNode GetOrientationString]
   
   if { [$yellowNode GetOrientationString] == "Reformat" } {
       $yellowNode SetOrientationToSagittal          
   }
   
   set orientString [$_sliceNode GetOrientationString]  
   
 #[$logic [$SliceNode SetOrientationToAxial]]
  set sliceSpacing [$logic GetLowestVolumeSliceSpacing]
  
 # [$logic GetBackgroundSliceBounds bounds]

  puts "slice offset is : $offset, orientation is : $orientString"
  puts "slice spacing : $sliceSpacing"
   
  # set sliceBounds [double 6] 
 #[$logic GetBackgroundSliceBounds]

  
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

#  puts "Done process event"
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

#      convert size into max iterations
      set cubeIndex [expr 1.0/3.0]
      puts "object size $objectSize"
      set div [expr 1.0/$objectSize]
      
      set objectSize [expr round([expr [expr pow([expr $_maxObjectSize * $div],$cubeIndex)] / 2])]
      
#      set maxIterations [expr round([expr $maxIterations + [expr $maxIterations * 0.25]])]
#      puts "max Iterations $maxIterations"
      puts "object size $objectSize"

 #     $growCutFilter SetMaxIterations $maxIterations
      $growCutFilter SetObjectSize $objectSize
      $growCutFilter SetContrastNoiseRatio $contrastNoiseRatio
      $growCutFilter SetGestureColors $gestureColors
      $growCutFilter SetPriorSegmentConfidence  $priorStrength 
      
      $_layers(label,node) SetAndObserveImageData [$growCutFilter GetOutput]
      $_layers(label,node) Modified

      $growCutFilter Update

      $_segmentedImage DeepCopy [$growCutFilter GetOutput]
      
      set _setGestures 1
  } checkFilter
 if { $checkFilter != "" } { puts "error in applying : $checkFilter" }

  gestureColors Delete
  
}

itcl::body GrowCutSegmentEffect::buildOptions {} {
   

  puts "Build options before chaining"
 
#  call superclass version of buildOptions
  chain

#  forget the options set by the labeler

  foreach w "paintThreshold paintRange cancel" {
      if { [info exists o($w)] } {
    $o($w) SetParent ""
    pack forget [$o($w) GetWidgetName]
      }
  }

  puts "bulding options.... "

  catch {
  
  puts "max Object Size : $_maxObjectSize"

  catch {
  set o(objectSize) [vtkKWScaleWithEntry New]
  $o(objectSize) SetParent [$this getOptionsFrame]
  $o(objectSize) PopupModeOn
  $o(objectSize) Create
  $o(objectSize) SetRange 0 100 
  $o(objectSize) SetResolution 1
  $o(objectSize) SetLabelText "object size: " 
  $o(objectSize) SetValue 100
  $o(objectSize) SetBalloonHelpString "Set the size of object being segmented. For multiple objects, use the size of largest object"
  pack [$o(objectSize) GetWidgetName] \
     -side top -anchor e -fill x -padx 2 -pady 2 
  } test0
  
  if { $test0 != "" } {puts "after objectSize : $test0"}

  puts "object Size built "

#    set the contrast noise ratio for the selected class 
  catch {
  set o(contrastNoiseRatio) [vtkKWScaleWithEntry New]
  $o(contrastNoiseRatio) SetParent [$this getOptionsFrame]
  $o(contrastNoiseRatio) PopupModeOn
  $o(contrastNoiseRatio) Create
  $o(contrastNoiseRatio) SetRange 0.0 1.0
  $o(contrastNoiseRatio) SetResolution 0.01
  $o(contrastNoiseRatio) SetLabelText "Contrast Ratio: " 
  $o(contrastNoiseRatio) SetValue $contrastNoiseRatio    
 $o(contrastNoiseRatio) SetBalloonHelpString "Set the relative contrast between foreground class and its surrounding background \[0 to 1\]."
  pack [$o(contrastNoiseRatio) GetWidgetName] \
     -side top -anchor e -fill x -padx 2 -pady 2 
  } test1

  puts "contrast noise ratio built "

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

  puts "prior Strength "

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
  set tag [$o(objectSize) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(objectSize) $tag"
  
  set tag [$o(contrastNoiseRatio) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(contrastNoiseRatio) $tag"

  set tag [$o(priorStrength) AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(priorStrength) $tag"

  set tag [[$o(overlay) GetWidget] AddObserver AnyEvent "after idle $this updateMRMLFromGUI"]
  lappend _observerRecords "$o(overlay) $tag"

  set tag [$o(apply) AddObserver AnyEvent "$this apply"]
  lappend _observerRecords "$o(apply) $tag"
  set tag [$o(cancel) AddObserver AnyEvent "after idle ::EffectSWidget::RemoveAll"]
  lappend _observerRecords "$o(cancel) $tag"

  if { [$this getInputBackground] == "" || [$this getInputLabel] == "" } {
    $this errorDialog "Background and Label map needed for Grow Cut Segmentation"
    after idle ::EffectSWidget::RemoveAll
  }
 
  $this updateGUIFromMRML

    } buildCatch
    if {$buildCatch != "" } { puts "error in building : $buildCatch" }

    puts "done build options"
}


itcl::body GrowCutSegmentEffect::tearDownOptions { } {

  # call superclass version of tearDownOptions
  chain

  foreach w "help cancel apply objectSize contrastNoiseRatio priorStrength overlay" {
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
    
#    puts "updating MRML from GUI"

    chain
    set node [EditorGetParameterNode]

    $this configure -objectSize [$o(objectSize) GetValue]
    $node SetParameter "GrowCutSegment,objectSize" $objectSize   

    $this configure -contrastNoiseRatio [$o(contrastNoiseRatio) GetValue]
    $node SetParameter "GrowCutSegment,contrastNoiseRatio" $contrastNoiseRatio   

    $this configure -priorStrength [$o(priorStrength) GetValue]
    $node SetParameter "GrowCutSegment,priorStrength" $priorStrength   
    
    $this configure -overlay [[$o(overlay) GetWidget] GetSelectedState]
    $node SetParameter "GrowCutSegment,overlay" $overlay

#    puts "done updating MRML from GUI"
    
}


itcl::body GrowCutSegmentEffect::setMRMLDefaults { } {
    
    puts "setting MRML defaults in Grow Cut"

    chain
    
    set node [EditorGetParameterNode]
#    $node SetParameter "GrowCutSegment,maxIterations" $_maxObjectSize
#    $node SetParameter "GrowCutSegment,contrastNoiseRatio" 0.8
#    $node SetParameter "GrowCutSegment,priorStrength" 0.0003
#    $node SetParameter "GrowCutSegment,overlay" 0
#    $node SetParameter "GrowCutSegment,segmented" 2
    

    foreach {param default} {
      objectSize 1
      contrastNoiseRatio 0.8
      priorStrength 0.00003
      overlay 0
      segmented 2
    } {
        set pvalue [$node GetParameter Paint,$param] 
 if { $pvalue == "" } {
     $node SetParameter GrowCutSegment,$param $default
 }
    }

    $node SetParameter "GrowCutSegment,objectSize" 100 
    puts "done setting MRML defaults in Grow Cut"
}



itcl::body GrowCutSegmentEffect::updateGUIFromMRML { } {
  #
  # get the parameter from the node
  # - set default value if it doesn't exist
  #
  chain

#  puts "updating gui from MRML"
    

  set node [EditorGetParameterNode]
  # set the GUI and effect parameters to match node
  # (only if this is the instance that "owns" the GUI
  
  catch {
  set objectSize [$node GetParameter GrowCutSegment,objectSize] 
  $this configure -objectSize $objectSize
  if { [info exists o(objectSize)] } {
      $o(objectSize) SetValue $objectSize
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

  } err1
  if { $err1 != ""} {puts $err1}

#  set segmented [$node GetParameter GrowCutSegment,segmented] 
#  $this configure -segmented $segmented
#  if { [info exists o(segmented)] } {
#    [$o(segmented) GetWidget] SetSelectedState $segmented
#  }

  $this preview

#  puts "done updating gui from MRML"
}

