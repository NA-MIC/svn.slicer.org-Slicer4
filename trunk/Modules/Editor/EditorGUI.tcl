#
# Editor GUI Procs
#

proc EditorConstructor {this} {

}

proc EditorDestructor {this} {
}


# Note: not a method - this is invoked directly by the GUI
# - remove the GUI from the current Application
# - re-source the code (this file)
# - re-build the GUI
proc EditorReload { {this ""} } {

  if { $this == "" } {
    set tag [lindex [array names ::Editor] 0]
    regsub -all "," $tag " " tag
    set this [lindex $tag 0]
  }
  EditorRemoveGUIObservers $this
  EditorTearDownGUI $this
  foreach n [array names ::Editor ${this}*] {
    unset ::Editor($n)
  }
  # TODO: figure this out from the CMakeCache and only offer the 
  # reload button if the source files exist
  source c:/pieper/bwh/slicer3/debug/Slicer3/Modules/Editor/EditorGUI.tcl
  EditorBuildGUI $this
  EditorAddGUIObservers $this

}

# Note: not a method - this is invoked directly by the GUI
proc EditorTearDownGUI {this} {

  set widgets {
    nodeSelector volumesCreate volumeName volumesSelect
    volumesFrame paintThreshold paintOver paintDropper
    paintRadius paintRange paintEnable paintLabel
    paintPaint paintDraw 
      paintFrame rebuildButton colorsColor colorsFrame
  }

  foreach w $widgets {
    $::Editor($this,$w) SetParent ""
    $::Editor($this,$w) Delete
  }

  if { [[$this GetUIPanel] GetUserInterfaceManager] != "" } {
    set pageWidget [[$this GetUIPanel] GetPageWidget "Editor"]
    [$this GetUIPanel] RemovePage "Editor"
  }

}

proc EditorBuildGUI {this} {

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

  # TODO: make a node to store our parameters

  [$this GetUIPanel] AddPage "Editor" "Editor" ""
  set pageWidget [[$this GetUIPanel] GetPageWidget "Editor"]

  #
  # help frame
  #
  set helptext "The Editor allows label maps to be created and edited. The active label map will be modified by the Editor. This module is currently a prototype and will be under active development throughout 3DSlicer's Beta release."
  set abouttext "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details."
  $this BuildHelpAndAboutFrame $pageWidget $helptext $abouttext

  set ::Editor($this,nodeSelector) [vtkSlicerNodeSelectorWidget New]
  $::Editor($this,nodeSelector) SetNodeClass "vtkMRMLScriptedModuleNode" "ModuleName" "Editor" "EditorParameter"
  $::Editor($this,nodeSelector) SetNewNodeEnabled 1
  $::Editor($this,nodeSelector) NoneEnabledOn
  $::Editor($this,nodeSelector) NewNodeEnabledOn
  $::Editor($this,nodeSelector) SetParent $pageWidget
  $::Editor($this,nodeSelector) Create
  $::Editor($this,nodeSelector) SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $::Editor($this,nodeSelector) UpdateMenu
  $::Editor($this,nodeSelector) SetLabelText "Editor Parameters"
  $::Editor($this,nodeSelector) SetBalloonHelpString "Select Editor parameters from current scene or create new ones"
  pack [$::Editor($this,nodeSelector) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  #
  # Editor Volumes
  #
  set ::Editor($this,volumesFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,volumesFrame) SetParent $pageWidget
  $::Editor($this,volumesFrame) Create
  $::Editor($this,volumesFrame) SetLabelText "Volumes"
  pack [$::Editor($this,volumesFrame) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  set ::Editor($this,volumesSelect) [vtkSlicerNodeSelectorWidget New]
  $::Editor($this,volumesSelect) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumesSelect) Create
  $::Editor($this,volumesSelect) SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $::Editor($this,volumesSelect) SetNodeClass "vtkMRMLScalarVolumeNode" "" "" ""
  $::Editor($this,volumesSelect) UpdateMenu
  $::Editor($this,volumesSelect) SetLabelText "Source Volume:"
  $::Editor($this,volumesSelect) SetBalloonHelpString "The Source Volume will define the dimensions and directions for the new label map"
  pack [$::Editor($this,volumesSelect) GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::Editor($this,volumeName) [vtkKWEntryWithLabel New]
  $::Editor($this,volumeName) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumeName) Create
  $::Editor($this,volumeName) SetLabelText "Name for label map volume: "
  $::Editor($this,volumeName) SetBalloonHelpString \
    "Leave blank for automatic label name based on input name."
  pack [$::Editor($this,volumeName) GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::Editor($this,volumesCreate) [vtkKWPushButton New]
  $::Editor($this,volumesCreate) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumesCreate) Create
  $::Editor($this,volumesCreate) SetText "Create Label Map"
  $::Editor($this,volumesCreate) SetBalloonHelpString "Create a new label map based on the source."
  pack [$::Editor($this,volumesCreate) GetWidgetName] -side top -anchor e -padx 2 -pady 2 


  #
  # Editor Paint
  #
  set ::Editor($this,paintFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,paintFrame) SetParent $pageWidget
  $::Editor($this,paintFrame) Create
  $::Editor($this,paintFrame) SetLabelText "Tool"
  pack [$::Editor($this,paintFrame) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]


  set ::Editor($this,paintEnable) [vtkKWCheckButtonWithLabel New]
  $::Editor($this,paintEnable) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintEnable) Create
  $::Editor($this,paintEnable) SetLabelText "Enable: "
  pack [$::Editor($this,paintEnable) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintPaint) [vtkKWRadioButton New]
  $::Editor($this,paintPaint) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintPaint) Create
  $::Editor($this,paintPaint) SetText "Paint"
  $::Editor($this,paintPaint) SetValue "Paint"
  pack [$::Editor($this,paintPaint) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintDraw) [vtkKWRadioButton New]
  $::Editor($this,paintDraw) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintDraw) Create
  $::Editor($this,paintDraw) SetText "Draw"
  $::Editor($this,paintDraw) SetValue "Draw"
  pack [$::Editor($this,paintDraw) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  $::Editor($this,paintPaint) SetSelectedState 1
  set ::Editor($this,paintMode) "Paint"
  $::Editor($this,paintPaint) SetVariableName ::Editor($this,paintMode)
  $::Editor($this,paintDraw) SetVariableName ::Editor($this,paintMode)


  set ::Editor($this,paintRadius) [vtkKWThumbWheel New]
  $::Editor($this,paintRadius) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintRadius) PopupModeOn
  $::Editor($this,paintRadius) Create
  $::Editor($this,paintRadius) DisplayEntryAndLabelOnTopOn
  $::Editor($this,paintRadius) DisplayEntryOn
  $::Editor($this,paintRadius) DisplayLabelOn
  $::Editor($this,paintRadius) SetValue 10
  [$::Editor($this,paintRadius) GetLabel] SetText "Radius: "
  $::Editor($this,paintRadius) SetBalloonHelpString "Set the radius of the paint brush in screen space pixels"
  pack [$::Editor($this,paintRadius) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintLabel) [vtkKWThumbWheel New]
  $::Editor($this,paintLabel) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintLabel) PopupModeOn
  $::Editor($this,paintLabel) Create
  $::Editor($this,paintLabel) DisplayEntryAndLabelOnTopOn
  $::Editor($this,paintLabel) DisplayEntryOn
  $::Editor($this,paintLabel) DisplayLabelOn
  $::Editor($this,paintLabel) SetValue 1
  [$::Editor($this,paintLabel) GetLabel] SetText "Label: "
#  pack [$::Editor($this,paintLabel) GetWidgetName] \
#    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintOver) [vtkKWCheckButtonWithLabel New]
  $::Editor($this,paintOver) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintOver) Create
  $::Editor($this,paintOver) SetLabelText "Paint Over: "
  $::Editor($this,paintOver) SetBalloonHelpString "Allow brush to paint over non-zero labels."
  pack [$::Editor($this,paintOver) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintDropper) [vtkKWCheckButtonWithLabel New]
  $::Editor($this,paintDropper) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintDropper) Create
  $::Editor($this,paintDropper) SetLabelText "Eye Dropper: "
  $::Editor($this,paintDropper) SetBalloonHelpString "Set the label number automatically by sampling the pixel location where the brush stroke starts."
  pack [$::Editor($this,paintDropper) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintThreshold) [vtkKWCheckButtonWithLabel New]
  $::Editor($this,paintThreshold) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintThreshold) Create
  $::Editor($this,paintThreshold) SetLabelText "Threshold Painting: "
  $::Editor($this,paintThreshold) SetBalloonHelpString "Enable/Disable threshold mode for painting."
  pack [$::Editor($this,paintThreshold) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 

  set ::Editor($this,paintRange) [vtkKWRange New]
  $::Editor($this,paintRange) SetParent [$::Editor($this,paintFrame) GetFrame]
  $::Editor($this,paintRange) Create
  $::Editor($this,paintRange) SetLabelText "Min/Max for Threshold Paint"
  $::Editor($this,paintRange) SetWholeRange 0 2000
  $::Editor($this,paintRange) SetRange 50 2000
  $::Editor($this,paintRange) SetReliefToGroove
  $::Editor($this,paintRange) SetBalloonHelpString "In threshold mode, the label will only be set if the background value is within this range."
  # don't pack this, it gets conditionally packed below

  #
  # Rebuild Button
  #
  set ::Editor($this,rebuildButton) [vtkKWPushButton New]
  $::Editor($this,rebuildButton) SetParent $pageWidget
  $::Editor($this,rebuildButton) Create
  $::Editor($this,rebuildButton) SetText "Reload"
  #pack [$::Editor($this,rebuildButton) GetWidgetName] -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]
  #
  # Editor Colors
  #
  set ::Editor($this,colorsFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,colorsFrame) SetParent $pageWidget
  $::Editor($this,colorsFrame) Create
  $::Editor($this,colorsFrame) SetLabelText "Colors"
  pack [$::Editor($this,colorsFrame) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  set ::Editor($this,colorsColor) [vtkSlicerColorDisplayWidget New]
  $::Editor($this,colorsColor) SetParent [$::Editor($this,colorsFrame) GetFrame]
  $::Editor($this,colorsColor) SetMRMLScene  [[$this GetLogic] GetMRMLScene]
  $::Editor($this,colorsColor) Create
  # set it up with the labels color node. 
  # TODO: here and in vtkSlicerVolumesLogic::CreateLabelVolume, use the color logic to get a default color node id
  set colorNode [vtkMRMLColorTableNode New]
  $colorNode SetTypeToLabels
  $::Editor($this,colorsColor) SetColorNode [[[$this GetLogic] GetMRMLScene] GetNodeByID [$colorNode GetTypeAsIDString]]
  $colorNode Delete
  pack [$::Editor($this,colorsColor) GetWidgetName] \
    -side top -anchor e -fill x -padx 2 -pady 2 
}

proc EditorAddGUIObservers {this} {
  $this AddObserverByNumber $::Editor($this,rebuildButton) 10000 
  $this AddObserverByNumber $::Editor($this,volumesCreate) 10000 
  $this AddObserverByNumber [$::Editor($this,paintEnable) GetWidget] 10000 
  $this AddObserverByNumber $::Editor($this,paintDraw) 10000 
  $this AddObserverByNumber $::Editor($this,paintPaint) 10000 
  $this AddObserverByNumber $::Editor($this,paintLabel) 10001 
  $this AddObserverByNumber $::Editor($this,colorsColor) 30000
  $this AddObserverByNumber $::Editor($this,colorsColor) 30001
  $this AddObserverByNumber $::Editor($this,paintRange) 10001 
  $this AddObserverByNumber [$::Editor($this,paintThreshold) GetWidget] 10000 
  $this AddObserverByNumber [$::Editor($this,paintOver) GetWidget] 10000 
  $this AddObserverByNumber [$::Editor($this,paintDropper) GetWidget] 10000 
  $this AddObserverByNumber $::Editor($this,paintRadius) 10001 
    
#    $this DebugOn
    if {[$this GetDebug]} {
        puts "Adding mrml observer to selection node, modified event"
    
    $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
    }
}

proc EditorRemoveGUIObservers {this} {
    if {[$this GetDebug]} {
        puts "Removing mrml observer on selection node, modified event"
        $this RemoveMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
    }
}

proc EditorRemoveLogicObservers {this} {
}

proc EditorRemoveMRMLNodeObservers {this} {
}

proc EditorProcessLogicEvents {this caller event} {
}

proc EditorProcessGUIEvents {this caller event} {
  
  if { $caller == $::Editor($this,rebuildButton) } {
    switch $event {
      "10000" {
        EditorReload $this
      }
    }
  } elseif { $caller == $::Editor($this,volumesCreate) } {
    switch $event {
      "10000" {
        EditorCreateLabelVolume $this
      }
    }
  } elseif { $caller == [$::Editor($this,paintEnable) GetWidget] ||
             $caller == $::Editor($this,paintPaint) ||
             $caller == $::Editor($this,paintDraw) } {
    switch $event {
      "10000" {
        ::PaintSWidget::RemovePaint
        ::DrawSWidget::RemoveDraw
        set checkButton [$::Editor($this,paintEnable) GetWidget]
        if { [$checkButton GetSelectedState] } {
          switch $::Editor($this,paintMode) {
            "Paint" {
              ::PaintSWidget::AddPaint
              $::Editor($this,paintDropper) SetEnabled 1
              $::Editor($this,paintRadius) SetEnabled 1
            }
            "Draw" {
              ::DrawSWidget::AddDraw
              $::Editor($this,paintDropper) SetEnabled 0
              $::Editor($this,paintRadius) SetEnabled 0
            }
          }
        } 
      }
    }
    EditorUpdateSWidgets $this
  } elseif { $caller == $::Editor($this,paintLabel) } {
    switch $event {
      "10001" {
        EditorUpdateSWidgets $this
      }
    }
  } elseif { $caller == $::Editor($this,colorsColor) } {
      switch $event {
          "30000" {
              # update the label volume's display node so that it will show up with the correct colours
              set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
              set labelVolume [[[$this GetLogic] GetMRMLScene] GetNodeByID [$selectionNode GetActiveLabelVolumeID]]
              if {$labelVolume != ""} {
                  set displayNode [$labelVolume GetDisplayNode] 
                  if {$displayNode != ""} {
                      $displayNode SetAndObserveColorNodeID [[$::Editor($this,colorsColor) GetColorNode] GetID]
                  } else {
                      puts "ERROR: display node not found"
                  }
              } else {
                  puts "ERROR: active label volume not found"
              }
              EditorUpdateSWidgets $this
          }
          "30001" {
              # Got selected color modified event
              EditorUpdateSWidgets $this
          }
      }
  } elseif { $caller == $::Editor($this,paintRadius) } {
    switch $event {
      "10001" {
        EditorUpdateSWidgets $this
      }
    }
  } elseif { $caller == [$::Editor($this,paintOver) GetWidget] } {
    switch $event {
      "10000" {
        EditorUpdateSWidgets $this
      }
    }
  } elseif { $caller == [$::Editor($this,paintDropper) GetWidget] } {
    switch $event {
      "10000" {
        EditorUpdateSWidgets $this
      }
    }
  } elseif { $caller == [$::Editor($this,paintThreshold) GetWidget] } {
    switch $event {
      "10000" {
        EditorUpdateSWidgets $this
        if { [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState] } {
          pack [$::Editor($this,paintRange) GetWidgetName] -side top -anchor e -fill x -padx 2 -pady 2 
        } else {
          pack forget [$::Editor($this,paintRange) GetWidgetName]
        }
      }
    }
  } elseif { $caller == $::Editor($this,paintRange) } {
    switch $event {
      "10001" {
        EditorUpdateSWidgets $this
      }
    }
  }

  EditorUpdateMRML $this
}

proc EditorUpdateSWidgets {this} {

  ::PaintSWidget::ConfigureAll -radius [$::Editor($this,paintRadius) GetValue]
  foreach {lo hi} [$::Editor($this,paintRange) GetRange] {}

  set cmd ::PaintSWidget::ConfigureAll
    $cmd -paintColor [$::Editor($this,colorsColor) GetSelectedColorIndex]
    $cmd -paintOver [[$::Editor($this,paintOver) GetWidget] GetSelectedState]
    $cmd -paintDropper [[$::Editor($this,paintDropper) GetWidget] GetSelectedState]
    $cmd -thresholdPaint [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState]
    $cmd -thresholdMin $lo -thresholdMax $hi

  set cmd ::DrawSWidget::ConfigureAll
    $cmd -drawColor [$::Editor($this,colorsColor) GetSelectedColorIndex]
    $cmd -drawOver [[$::Editor($this,paintOver) GetWidget] GetSelectedState]
    $cmd -thresholdPaint [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState]
    $cmd -thresholdMin $lo -thresholdMax $hi
}

proc EditorUpdateMRML {this} {
}

proc EditorProcessMRMLEvents {this callerID event} {
    if {[$this GetDebug]} {
        puts "EditorProcessMRMLEvents: event = $event, callerID = $callerID"
    }
    set caller [[[$this GetLogic] GetMRMLScene] GetNodeByID $callerID]
    set selectionNode  [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
    if {$caller == $selectionNode && $event == 31} {
        if {[$this GetDebug]} {
            puts "...caller is selection node, with modified event"
        }
        # get the active label volume
        set labelVolume [[[$this GetLogic] GetMRMLScene] GetNodeByID [$selectionNode GetActiveLabelVolumeID]]
        # check it's display node colour node
        set displayNode [$labelVolume GetDisplayNode] 
        # is it the one we're showing?
        if {$displayNode != "" && [$displayNode GetColorNodeID] != [[$::Editor($this,colorsColor) GetColorNode] GetID]} {
            if {[$this GetDebug]} {
                puts "Resetting the color node"
            }
            $::Editor($this,colorsColor) SetColorNode [$displayNode GetColorNode]
        }
    }
}

proc EditorEnter {this} {
    if {[$this GetDebug]} {
        puts "EditorEnter: Adding mrml observer on selection node, modified event"
    }
    $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode] 31
}

proc EditorExit {this} {
    if {[$this GetDebug]} {
        puts "EditorExit: Removing mrml observer on selection node modified event"
    }
    $this RemoveMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode] 31
}

# TODO: there might be a better place to put this for general use...  
proc EditorCreateLabelVolume {this} {

  set volumeNode [$::Editor($this,volumesSelect) GetSelected]
  if { $volumeNode == "" } {
    return;
  }

  set name [[$::Editor($this,volumeName) GetWidget] GetValue]
  if { $name == "" } {
    set name "[$volumeNode GetName]-label"
  }

  set scene [[$this GetLogic] GetMRMLScene]

  set volumesLogic [$::slicer3::VolumesGUI GetLogic]
  set labelNode [$volumesLogic CreateLabelVolume $scene $volumeNode $name]

  # make the source node the active background, and the label node the active label
  set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
  $selectionNode SetReferenceActiveVolumeID [$volumeNode GetID]
  $selectionNode SetReferenceActiveLabelVolumeID [$labelNode GetID]
  [[$this GetLogic] GetApplicationLogic]  PropagateVolumeSelection

  # update the color node to show the label volume's display node's color node
  $::Editor($this,colorsColor) SetColorNode [[$labelNode GetDisplayNode] GetColorNode]

  $labelNode Delete

  # update the editor range to be the full range of the background image
  set range [[$volumeNode GetImageData] GetScalarRange]
  eval $::Editor($this,paintRange) SetWholeRange $range
  eval $::Editor($this,paintRange) SetRange $range

  
}

proc EditorSetRandomLabelColormap { {size 255} } {

  # TODO: once the mrml label map functionality is set up this will migrate
  # into the vtkSlicerVolumesDisplay -- the label map should be part of the DisplayNode
  # get a lut with:
  # [[[$::slicer3::ApplicationGUI GetMainSliceLogic0] GetLabelLayer] GetMapToColors] GetLookupTable
    foreach g {0 1 2} {
        puts "Setting logic $g label map to label colors..."
        [[[[$::slicer3::ApplicationGUI GetMainSliceLogic$g] GetLabelLayer] GetVolumeDisplayNode] GetColorNode] SetTypeToRandom
    }
}

proc EditorSetLabelColormap {} {
    foreach g {0 1 2} {
        puts "EditorGUI: Setting logic $g label map to label colors..."
        [[[[$::slicer3::ApplicationGUI GetMainSliceLogic$g] GetLabelLayer] GetVolumeDisplayNode] GetColorNode] SetTypeToLabels
    }
}

proc EditorLabelSelectDialog {this} {
  set pageWidget [[$this GetUIPanel] GetPageWidget "Editor"]

  set dialog [vtkKWDialog New]
  $dialog SetParent $pageWidget
  $dialog Create
  set frame [$dialog GetFrame]
  set color [vtkSlicerColorDisplayWidget New]
  $color SetMRMLScene $::slicer3::MRMLScene
  $color SetParent $frame
  $color Create

  pack [$color GetWidgetName] -side top -anchor nw -fill x -padx 2 -pady 2 

  $dialog Display
  $dialog Invoke
}

