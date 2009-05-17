#
# FastMarchingSegmentation GUI Procs
# - the 'this' argument to all procs is a vtkScriptedModuleGUI
#

proc FastMarchingSegmentationConstructor {this} {
}

proc FastMarchingSegmentationDestructor {this} {
}

proc FastMarchingSegmentationTearDownGUI {this} {


  # nodeSelector  ;# disabled for now
  set widgets {
    runButton inputSelector outputLabelText
    outputCreateButton currentOutputText
    fiducialsSelector segVolumeThumbWheel
    timeScrollScale labelColorSpin
    timeScrollText initFrame outputVolumeFrame outputParametersFrame
  }

  foreach w $widgets {
    $::FastMarchingSegmentation($this,$w) SetParent ""
    $::FastMarchingSegmentation($this,$w) Delete
  }

  $::FastMarchingSegmentation($this,cast) Delete
  $::FastMarchingSegmentation($this,fastMarchingFilter) Delete

  if { [[$this GetUIPanel] GetUserInterfaceManager] != "" } {
    set pageWidget [[$this GetUIPanel] GetPageWidget "FastMarchingSegmentation"]
    [$this GetUIPanel] RemovePage "FastMarchingSegmentation"
  }

  unset ::FastMarchingSegmentation(singleton)

}

proc FastMarchingSegmentationBuildGUI {this} {
  load $::Slicer3_HOME/bin/libPichonFastMarching.so
  package require PichonFastMarching

  if { [info exists ::FastMarchingSegmentation(singleton)] } {
    error "FastMarchingSegmentation singleton already created"
  }
  set ::FastMarchingSegmentation(singleton) $this

  #
  # create and register the node class
  # - since this is a generic type of node, only do it if 
  #   it hasn't already been done by another module
  #

  set mrmlScene [[$this GetLogic] GetMRMLScene]
  set tag [$mrmlScene GetTagByClassName "vtkMRMLScriptedModuleNode"]
  if { $tag == "" } {
    set node [vtkMRMLScriptedModuleNode New]
    $mrmlScene RegisterNodeClass $node
    $node Delete
  }


  $this SetCategory "Segmentation"
  [$this GetUIPanel] AddPage "FastMarchingSegmentation" "FastMarchingSegmentation" ""
  set pageWidget [[$this GetUIPanel] GetPageWidget "FastMarchingSegmentation"]

  #
  # help frame
  #
  set helptext "This module performs segmentation using fast marching method with automatic estimation of region statistics. The core C++ classes were contributed by Eric Pichon in slicer2.\n\nIn order to benefit from this module, please use the following steps:\n(1) specify input scalar volume to be segmented\n(2) put fiducial seeds within the region you want to segment\n(3) specify the expected volume of the structure to be segmented. Note, overestimation of this volume is OK, because you will be able to adjust the actual volume once the segmentation is complete\n(4) specify the color for the segmentation, and create the output label volume\n(5) push \"Run\" button to segment\n(6) use volume control slider to adjust segmentation result."
  set abouttext "This module was developed by Andriy Fedorov based on the original implementation of Eric Pichon in Slicer2.\nThis work was funded by Brain Science Foundation, and supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details."
  $this BuildHelpAndAboutFrame $pageWidget $helptext $abouttext

  #
  # FastMarchingSegmentation input and initialization parameters
  #
  set ::FastMarchingSegmentation($this,initFrame) [vtkSlicerModuleCollapsibleFrame New]
  set initFrame $::FastMarchingSegmentation($this,initFrame)
  $initFrame SetParent $pageWidget
  $initFrame Create
  $initFrame SetLabelText "Input/initialization parameters"
  pack [$initFrame GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  #
  # FastMarchingSegmentation output
  #
  set ::FastMarchingSegmentation($this,outputVolumeFrame) [vtkSlicerModuleCollapsibleFrame New]
  set outputVolumeFrame $::FastMarchingSegmentation($this,outputVolumeFrame)
  $outputVolumeFrame SetParent $pageWidget
  $outputVolumeFrame Create
  $outputVolumeFrame SetLabelText "Output"
  pack [$outputVolumeFrame GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  #
  # FastMarchingSegmentation output parameters
  #
  set ::FastMarchingSegmentation($this,outputParametersFrame) [vtkSlicerModuleCollapsibleFrame New]
  set outputParametersFrame $::FastMarchingSegmentation($this,outputParametersFrame)
  $outputParametersFrame SetParent $pageWidget
  $outputParametersFrame Create
  $outputParametersFrame SetLabelText "Segmentation result adjustment"
  pack [$outputParametersFrame GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  # Input frame widgets

  set ::FastMarchingSegmentation($this,inputSelector) [vtkSlicerNodeSelectorWidget New]
  set select $::FastMarchingSegmentation($this,inputSelector)
  $select SetParent [$initFrame GetFrame]
  $select Create
  $select SetNodeClass "vtkMRMLScalarVolumeNode" "" "" ""
  $select SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $select UpdateMenu
  $select SetLabelText "Input volume:"
  $select SetBalloonHelpString "The Source Volume to operate on"
  pack [$select GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::FastMarchingSegmentation($this,fiducialsSelector) [vtkSlicerNodeSelectorWidget New]
  set fiducials $::FastMarchingSegmentation($this,fiducialsSelector)
  $fiducials SetParent [$initFrame GetFrame]
  $fiducials Create
  $fiducials NewNodeEnabledOn
  $fiducials SetNodeClass "vtkMRMLFiducialListNode" "" "" ""
  $fiducials SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $fiducials UpdateMenu
  $fiducials SetLabelText "Input seeds:"
  $fiducials SetBalloonHelpString "List of fiducials to be used as seeds for segmentation"
  pack [$fiducials GetWidgetName] -side top -anchor e -padx 2 -pady 2
  
  set ::FastMarchingSegmentation($this,segVolumeThumbWheel) [vtkKWThumbWheel New]
  set segvolume $::FastMarchingSegmentation($this,segVolumeThumbWheel)
  $segvolume SetParent [$initFrame GetFrame]
  $segvolume PopupModeOn
  $segvolume Create
  $segvolume DisplayEntryAndLabelOnTopOn
  $segvolume DisplayEntryOn
  $segvolume DisplayLabelOn
  $segvolume SetResolution 1
  $segvolume SetMinimumValue 0
  $segvolume SetClampMinimumValue 0
  $segvolume SetValue 0
  [$segvolume GetLabel] SetText "Target segmented volume (mm^3):"
  $segvolume SetBalloonHelpString "Overestimate of the segmented structure volume"
  pack [$segvolume GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  # Output volume definition
  set ::FastMarchingSegmentation($this,currentOutputText) [vtkKWLabel New]
  set outName $::FastMarchingSegmentation($this,currentOutputText)
  $outName SetParent [$outputVolumeFrame GetFrame]
  $outName Create
  $outName SetText "Output label volume not defined!"
  pack [$outName GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::FastMarchingSegmentation($this,labelColorSpin) [vtkKWSpinBoxWithLabel New]
  set outColor $::FastMarchingSegmentation($this,labelColorSpin)
  $outColor SetParent [$outputVolumeFrame GetFrame]
  $outColor SetLabelText "Output label value:"
  $outColor Create
  $outColor SetWidth 3
  [$outColor GetWidget] SetValue 1.0
  $outColor SetBalloonHelpString "Specify color for the output label"
  pack [$outColor GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::FastMarchingSegmentation($this,outputLabelText) [vtkKWEntryWithLabel New]
  set outLabelName $::FastMarchingSegmentation($this,outputLabelText)
  $outLabelName SetParent [$outputVolumeFrame GetFrame]
  $outLabelName Create
  $outLabelName SetLabelText "Output volume name:"
  $outLabelName SetBalloonHelpString "Name of the output label volume"
  $outLabelName SetLabelPositionToLeft


  set ::FastMarchingSegmentation($this,outputCreateButton) [vtkKWPushButton New]
  set outCreateButton $::FastMarchingSegmentation($this,outputCreateButton)
  $outCreateButton SetParent [$outputVolumeFrame GetFrame]
  $outCreateButton Create
  $outCreateButton SetText "Create output volume"
  $outCreateButton SetBalloonHelpString "Push to create output label volume"
  $outCreateButton EnabledOff
  pack [$outLabelName GetWidgetName] [$outCreateButton GetWidgetName] \
    -side left -anchor e -padx 2 -pady 2 


  set ::FastMarchingSegmentation($this,runButton) [vtkKWPushButton New]
  set run $::FastMarchingSegmentation($this,runButton)
  $run SetParent $outputVolumeFrame
  $run Create
  $run SetText "Run Segmentation"
  $run EnabledOff
  pack [$run GetWidgetName] -side top -anchor e -padx 2 -pady 2 -fill x

  # FastMarching output parameters
  set ::FastMarchingSegmentation($this,timeScrollText) [vtkKWLabel New]
  set timeScrollText $::FastMarchingSegmentation($this,timeScrollText)
  $timeScrollText SetParent [$outputParametersFrame GetFrame]
  $timeScrollText Create
  $timeScrollText SetText "Output segmentation volume:"

  set ::FastMarchingSegmentation($this,timeScrollScale) [vtkKWScale New]
  set timescroll $::FastMarchingSegmentation($this,timeScrollScale)
  $timescroll SetParent [$outputParametersFrame GetFrame]
  $timescroll Create
  $timescroll SetRange 0.0 1.0
  $timescroll SetValue 1.0
  $timescroll SetResolution 0.001
  $timescroll SetLength 150
  $timescroll SetBalloonHelpString "Scroll back in segmentation process"
  $timescroll EnabledOff
  pack [$timeScrollText GetWidgetName] [$timescroll GetWidgetName] -side left -anchor e -padx 2 -pady 2


  set ::FastMarchingSegmentation($this,cast) [vtkImageCast New]
  set ::FastMarchingSegmentation($this,fastMarchingFilter) [vtkPichonFastMarching New]
}

proc FastMarchingSegmentationAddGUIObservers {this} {
  $this AddObserverByNumber $::FastMarchingSegmentation($this,runButton) 10000 
  $this AddObserverByNumber $::FastMarchingSegmentation($this,timeScrollScale) 10000
  $this AddObserverByNumber $::FastMarchingSegmentation($this,inputSelector)  11000
  $this AddObserverByNumber $::FastMarchingSegmentation($this,outputCreateButton)  10000
  $this AddObserverByNumber $::FastMarchingSegmentation($this,fiducialsSelector)  11000
  $this AddObserverByNumber $::FastMarchingSegmentation($this,segVolumeThumbWheel)  10000
  
#  $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
    
}

proc FastMarchingSegmentationRemoveGUIObservers {this} {
}

proc FastMarchingSegmentationRemoveLogicObservers {this} {
}

proc FastMarchingSegmentationRemoveMRMLNodeObservers {this} {
}

proc FastMarchingSegmentationProcessLogicEvents {this caller event} {
}

proc FastMarchingSegmentationProcessGUIEvents {this caller event} {
  
  if { $caller == $::FastMarchingSegmentation($this,runButton) } {
  
    set inputFiducials [$::FastMarchingSegmentation($this,fiducialsSelector) GetSelected]
    if {$inputFiducials == ""} {
      FastMarchingSegmentationErrorDialog $this "Please specify input fiducial list!"
      return
    }
    if { [$inputFiducials GetNumberOfFiducials] == 0} {
      FastMarchingSegmentationErrorDialog $this "Input fiducial list must be non-empty!"
      return
    } 
    if { [$::FastMarchingSegmentation($this,segVolumeThumbWheel) GetValue] == 0} {
      FastMarchingSegmentationErrorDialog $this "Target segmentation volume cannot be 0!"
      return
    }
    FastMarchingSegmentationExpand $this
    $::FastMarchingSegmentation($this,timeScrollScale) EnabledOn
    $::FastMarchingSegmentation($this,runButton) EnabledOff
    set timescroll $::FastMarchingSegmentation($this,timeScrollScale)
    set segmentedVolume [$::FastMarchingSegmentation($this,segVolumeThumbWheel) GetValue]
    set knownpoints [$::FastMarchingSegmentation($this,fastMarchingFilter) nKnownPoints]
    $timescroll SetRange 0.0 $segmentedVolume
    $timescroll SetValue $segmentedVolume
    $timescroll SetResolution [ expr [expr double($segmentedVolume)] / [expr double($knownpoints)] ]
  } 

  if { $caller == $::FastMarchingSegmentation($this,timeScrollScale) } {
    FastMarchingSegmentationUpdateTime $this    
  } 

  if { $caller == $::FastMarchingSegmentation($this,segVolumeThumbWheel) } {
  } 

  if {$caller == $::FastMarchingSegmentation($this,inputSelector) } {
    $::FastMarchingSegmentation($this,currentOutputText) SetText "Output label volume not defined!"
    $::FastMarchingSegmentation($this,outputCreateButton) EnabledOn
    $::FastMarchingSegmentation($this,timeScrollScale) EnabledOff
    $::FastMarchingSegmentation($this,runButton) EnabledOff
  }

  if {$caller == $::FastMarchingSegmentation($this,fiducialsSelector) } {
  }

  if {$caller == $::FastMarchingSegmentation($this,outputCreateButton) } {
    FastMarchingSegmentationInitializeFilter $this
    $::FastMarchingSegmentation($this,runButton) EnabledOn
  }

# FastMarchingSegmentationUpdateMRML $this
}

#
# Accessors to FastMarchingSegmentation state
#


# get the FastMarchingSegmentation parameter node, or create one if it doesn't exist
proc FastMarchingSegmentationCreateParameterNode {} {
  set node [vtkMRMLScriptedModuleNode New]
  $node SetModuleName "FastMarchingSegmentation"

  # set node defaults
  $node SetParameter label 1

  $::slicer3::MRMLScene AddNode $node
  $node Delete
}

# get the FastMarchingSegmentation parameter node, or create one if it doesn't exist
proc FastMarchingSegmentationGetParameterNode {} {

  set node ""
  set nNodes [$::slicer3::MRMLScene GetNumberOfNodesByClass "vtkMRMLScriptedModuleNode"]
  for {set i 0} {$i < $nNodes} {incr i} {
    set n [$::slicer3::MRMLScene GetNthNodeByClass $i "vtkMRMLScriptedModuleNode"]
    if { [$n GetModuleName] == "FastMarchingSegmentation" } {
      set node $n
      break;
    }
  }

  if { $node == "" } {
    FastMarchingSegmentationCreateParameterNode
    set node [FastMarchingSegmentationGetParameterNode]
  }

  return $node
}


proc FastMarchingSegmentationGetLabel {} {
  set node [FastMarchingSegmentationGetParameterNode]
  if { [$node GetParameter "label"] == "" } {
    $node SetParameter "label" 1
  }
  return [$node GetParameter "label"]
}

proc FastMarchingSegmentationSetLabel {index} {
  set node [FastMarchingSegmentationGetParameterNode]
  $node SetParameter "label" $index
}

#
# MRML Event processing
#

proc FastMarchingSegmentationUpdateMRML {this} {
}

proc FastMarchingSegmentationProcessMRMLEvents {this callerID event} {

    set caller [[[$this GetLogic] GetMRMLScene] GetNodeByID $callerID]
    if { $caller == "" } {
        return
    }
}

proc FastMarchingSegmentationEnter {this} {
}

proc FastMarchingSegmentationExit {this} {
}

proc FastMarchingSegmentationProgressEventCallback {filter} {

  set mainWindow [$::slicer3::ApplicationGUI GetMainSlicerWindow]
  set progressGauge [$mainWindow GetProgressGauge]
  set renderWidget [[$::slicer3::ApplicationGUI GetViewControlGUI] GetNavigationWidget]

  if { $filter == "" } {
    $mainWindow SetStatusText ""
    $progressGauge SetValue 0
    $renderWidget SetRendererGradientBackground 0
  } else {
    # TODO: this causes a tcl 'update' which re-triggers the module (possibly changing
    # values while it is executing!  Talk about evil...
    #$mainWindow SetStatusText [$filter GetClassName]
    #$progressGauge SetValue [expr 100 * [$filter GetProgress]]

    set progress [$filter GetProgress]
    set remaining [expr 1.0 - $progress]

    #$renderWidget SetRendererGradientBackground 1
    #$renderWidget SetRendererBackgroundColor $progress $progress $progress
    #$renderWidget SetRendererBackgroundColor2 $remaining $remaining $remaining
  }
}

proc FastMarchingSegmentationApply {this} {

  if { ![info exists ::FastMarchingSegmentation($this,processing)] } { 
    set ::FastMarchingSegmentation($this,processing) 0
  }

  if { $::FastMarchingSegmentation($this,processing) } {
    return
  }

  set volumeNode [$::FastMarchingSegmentation($this,volumesSelect) GetSelected]
  set ijkToRAS [vtkMatrix4x4 New]
  $volumeNode GetIJKToRASMatrix $ijkToRAS
  set outVolumeNode [$::FastMarchingSegmentation($this,volumeOutputSelect) GetSelected]
  set outModelNode [$::FastMarchingSegmentation($this,modelOutputSelect) GetSelected]
  set value [$::FastMarchingSegmentation($this,thresholdSelect) GetValue]
  set minFactor [$::FastMarchingSegmentation($this,minFactor) GetValue]
  set cast $::FastMarchingSegmentation($this,cast)
  set dt $::FastMarchingSegmentation($this,distanceTransform)
  set resample $::FastMarchingSegmentation($this,resample)
  set cubes $::FastMarchingSegmentation($this,marchingCubes)
  set changeIn $::FastMarchingSegmentation($this,changeIn)
  set changeOut $::FastMarchingSegmentation($this,changeOut)
  set polyTransformFilter $::FastMarchingSegmentation($this,polyTransformFilter)
  set polyTransform $::FastMarchingSegmentation($this,polyTransform)

  #
  # check that inputs are valid
  #
  set errorText ""
  if { $volumeNode == "" || [$volumeNode GetImageData] == "" } {
    set errorText "No input volume data..."
  }
  if { $outVolumeNode == "" } {
    set errorText "No output volume node..."
  }
  if { $outModelNode == "" } {
    set errorText "No output model node..."
  }

  if { $errorText != "" } {
    set dialog [vtkKWMessageDialog New]
    $dialog SetParent [$::slicer3::ApplicationGUI GetMainSlicerWindow]
    $dialog SetMasterWindow [$::slicer3::ApplicationGUI GetMainSlicerWindow]
    $dialog SetStyleToMessage
    $dialog SetText $errorText
    $dialog Create
    $dialog Invoke
    $dialog Delete
    return
  }

  set ::FastMarchingSegmentation($this,processing) 1

  #
  # configure the pipeline
  #
  $changeIn SetInput [$volumeNode GetImageData]
  eval $changeIn SetOutputSpacing [$volumeNode GetSpacing]
  $cast SetInput [$changeIn GetOutput]
  $cast SetOutputScalarTypeToFloat
  $dt SetInput [$cast GetOutput]
  $dt SetSquaredDistance 0
  $dt SetUseImageSpacing 1
  $dt SetInsideIsPositive 0
  $changeOut SetInput [$dt GetOutput]
  $changeOut SetOutputSpacing 1 1 1
  $resample SetInput [$dt GetOutput]
  $resample SetAxisMagnificationFactor 0 $minFactor
  $resample SetAxisMagnificationFactor 1 $minFactor
  $resample SetAxisMagnificationFactor 2 $minFactor
  $cubes SetInput [$resample GetOutput]
  $cubes SetValue 0 $value
  $polyTransformFilter SetInput [$cubes GetOutput]
  $polyTransformFilter SetTransform $polyTransform
  set magFactor [expr 1.0 / $minFactor]
  $polyTransform Identity
  $polyTransform Concatenate $ijkToRAS
  foreach sp [$volumeNode GetSpacing] {
    lappend invSpacing [expr 1. / $sp]
  }
  eval $polyTransform Scale $invSpacing

  #
  # set up progress observers
  #
  set observerRecords ""
  set filters "$changeIn $resample $dt $changeOut $cubes"
  foreach filter $filters {
    set tag [$filter AddObserver ProgressEvent "FastMarchingSegmentationProgressEventCallback $filter"]
    lappend observerRecords "$filter $tag"
  }

  #
  # activate the pipeline
  #
  $polyTransformFilter Update

  # remove progress observers
  foreach record $observerRecords {
    foreach {filter tag} $record {
      $filter RemoveObserver $tag
    }
  }
  FastMarchingSegmentationProgressEventCallback ""

  #
  # create a mrml model display node if needed
  #
  if { [$outModelNode GetDisplayNode] == "" } {
    set modelDisplayNode [vtkMRMLModelDisplayNode New]
    $outModelNode SetScene $::slicer3::MRMLScene
    eval $modelDisplayNode SetColor .5 1 1
    $::slicer3::MRMLScene AddNode $modelDisplayNode
    $outModelNode SetAndObserveDisplayNodeID [$modelDisplayNode GetID]
  }

  #
  # set the output into the MRML scene
  #
  $outModelNode SetAndObservePolyData [$polyTransformFilter GetOutput]

  $outVolumeNode SetAndObserveImageData [$changeOut GetOutput]
  $outVolumeNode SetIJKToRASMatrix $ijkToRAS
  $ijkToRAS Delete


  set ::FastMarchingSegmentation($this,processing) 0
}

proc FastMarchingSegmentationErrorDialog {this errorText} {
  set dialog [vtkKWMessageDialog New]
  $dialog SetParent [$::slicer3::ApplicationGUI GetMainSlicerWindow]
  $dialog SetMasterWindow [$::slicer3::ApplicationGUI GetMainSlicerWindow]
  $dialog SetStyleToMessage
  $dialog SetText $errorText
  $dialog Create
  $dialog Invoke
  $dialog Delete
}

