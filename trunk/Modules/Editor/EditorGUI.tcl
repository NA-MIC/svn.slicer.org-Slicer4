#
# Editor GUI Procs
#

proc EditorConstructor {this} {
}

proc EditorDestructor {this} {

  set boxes [itcl::find objects -isa Box]
  foreach b $boxes {
    itcl::delete object $b
  }
}


# Note: not a method - this is invoked directly by the GUI
# - remove the GUI from the current Application
# - re-source the code (this file)
# - re-build the GUI
# Note: not a method - this is invoked directly by the GUI
proc EditorTearDownGUI {this} {

  # nodeSelector  ;# disabled for now
  set widgets {
    volumesCreate volumeName volumesSelect
    volumesFrame paintThreshold paintOver paintDropper
    paintRadius paintRange paintEnable paintLabel
    paintPaint paintDraw 
    optionsSpacer optionsFrame
    paintFrame 
    toolsActiveTool toolsEditFrame toolsColorFrame
    toolsFrame 
  }

  itcl::delete object $::Editor($this,editColor)
  itcl::delete object $::Editor($this,editBox)

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

  if { [info exists ::Editor(singleton)] } {
    error "editor singleton already created"
  }
  set ::Editor(singleton) $this

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


  [$this GetUIPanel] AddPage "Editor" "Editor" ""
  set pageWidget [[$this GetUIPanel] GetPageWidget "Editor"]

  #
  # help frame
  #
  set helptext "The Editor allows label maps to be created and edited. The active label map will be modified by the Editor. This module is currently a prototype and will be under active development throughout 3DSlicer's Beta release."
  set abouttext "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See http://www.slicer.org for details."
  $this BuildHelpAndAboutFrame $pageWidget $helptext $abouttext

  if { 0 } { 
    # leave out the node selector for now - we'll use one global node and save GUI space
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
  }

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
  $::Editor($this,volumesSelect) SetNodeClass "vtkMRMLVolumeNode" "" "" ""
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
  # Tool Frame
  #
  set ::Editor($this,toolsFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,toolsFrame) SetParent $pageWidget
  $::Editor($this,toolsFrame) Create
  $::Editor($this,toolsFrame) SetLabelText "Tools"
  pack [$::Editor($this,toolsFrame) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  set ::Editor($this,toolsEditFrame) [vtkKWFrame New]
  $::Editor($this,toolsEditFrame) SetParent [$::Editor($this,toolsFrame) GetFrame]
  $::Editor($this,toolsEditFrame) Create
  pack [$::Editor($this,toolsEditFrame) GetWidgetName] \
    -side right -anchor ne -padx 2 -pady 2

  set ::Editor($this,toolsColorFrame) [vtkKWFrame New]
  $::Editor($this,toolsColorFrame) SetParent [$::Editor($this,toolsFrame) GetFrame]
  $::Editor($this,toolsColorFrame) Create
  $::Editor($this,toolsColorFrame) SetBackgroundColor [expr 232/255.] [expr 230/255.] [expr 235/255.]
  pack [$::Editor($this,toolsColorFrame) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2

  set ::Editor($this,toolsActiveTool) [vtkKWLabelWithLabel New]
  $::Editor($this,toolsActiveTool) SetParent [$::Editor($this,toolsFrame) GetFrame]
  $::Editor($this,toolsActiveTool) Create
  $::Editor($this,toolsActiveTool) SetBackgroundColor [expr 232/255.] [expr 230/255.] [expr 235/255.]
  [$::Editor($this,toolsActiveTool) GetWidget] SetBackgroundColor [expr 232/255.] [expr 230/255.] [expr 235/255.]
  [$::Editor($this,toolsActiveTool) GetLabel] SetBackgroundColor [expr 232/255.] [expr 230/255.] [expr 235/255.]
  $::Editor($this,toolsActiveTool) SetLabelText "Active Tool: "
  pack [$::Editor($this,toolsActiveTool) GetWidgetName] \
    -side top -anchor nw -fill x -padx 2 -pady 2
  
  
  # create the edit box and the color picker - note these aren't kwwidgets
  #  but helper classes that create kwwidgets in the given frame
  set ::Editor($this,editColor) [::EditColor #auto]
  $::Editor($this,editColor) configure -frame $::Editor($this,toolsColorFrame)
  $::Editor($this,editColor) create
  set ::Editor($this,editBox) [::EditBox #auto]
  $::Editor($this,editBox) configure -frame $::Editor($this,toolsEditFrame)
  $::Editor($this,editBox) create

  #
  # Tool Options
  #
  set ::Editor($this,optionsFrame) [vtkKWFrame New]
  $::Editor($this,optionsFrame) SetParent [$::Editor($this,toolsFrame) GetFrame]
  $::Editor($this,optionsFrame) Create
  $::Editor($this,optionsFrame) SetBorderWidth 1
  $::Editor($this,optionsFrame) SetReliefToSolid
  pack [$::Editor($this,optionsFrame) GetWidgetName] \
    -side left -anchor nw -fill both -padx 2 -pady 2 

  # one pixel label to keep the option panel expanded
  set ::Editor($this,optionsSpacer) [vtkKWLabel New]
  $::Editor($this,optionsSpacer) SetParent $::Editor($this,optionsFrame)
  $::Editor($this,optionsSpacer) Create
  pack [$::Editor($this,optionsSpacer) GetWidgetName] \
    -fill both -expand true 

}

proc EditorAddGUIObservers {this} {
  $this AddObserverByNumber $::Editor($this,volumesCreate) 10000 
  $this AddObserverByNumber [$::Editor($this,paintEnable) GetWidget] 10000 
  $this AddObserverByNumber $::Editor($this,paintDraw) 10000 
  $this AddObserverByNumber $::Editor($this,paintPaint) 10000 
  $this AddObserverByNumber $::Editor($this,paintLabel) 10001 
  $this AddObserverByNumber $::Editor($this,paintRange) 10001 
  $this AddObserverByNumber [$::Editor($this,paintThreshold) GetWidget] 10000 
  $this AddObserverByNumber [$::Editor($this,paintOver) GetWidget] 10000 
  $this AddObserverByNumber [$::Editor($this,paintDropper) GetWidget] 10000 
  $this AddObserverByNumber $::Editor($this,paintRadius) 10001 
    
# $this DebugOn
    if {[$this GetDebug]} {
        puts "Adding mrml observer to selection node, modified event"
    }
    $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
    
}

proc EditorRemoveGUIObservers {this} {
    if {[$this GetDebug]} {
        puts "Removing mrml observer on selection node, modified event"
    }
    $this RemoveMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
}

proc EditorRemoveLogicObservers {this} {
}

proc EditorRemoveMRMLNodeObservers {this} {
}

proc EditorProcessLogicEvents {this caller event} {
}

proc EditorProcessGUIEvents {this caller event} {
  
  if { $caller == $::Editor($this,volumesCreate) } {
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
    $cmd -paintColor [EditorGetPaintLabel]
    $cmd -paintOver [[$::Editor($this,paintOver) GetWidget] GetSelectedState]
    $cmd -paintDropper [[$::Editor($this,paintDropper) GetWidget] GetSelectedState]
    $cmd -thresholdPaint [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState]
    $cmd -thresholdMin $lo -thresholdMax $hi

  set cmd ::DrawSWidget::ConfigureAll
    $cmd -drawColor [EditorGetPaintLabel]
    $cmd -drawOver [[$::Editor($this,paintOver) GetWidget] GetSelectedState]
    $cmd -thresholdPaint [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState]
    $cmd -thresholdMin $lo -thresholdMax $hi
}


#
# Accessors to editor state
#


# get the editor parameter node, or create one if it doesn't exist
proc EditorCreateParameterNode {} {
  set node [vtkMRMLScriptedModuleNode New]
  $node SetModuleName "Editor"

  # set node defaults
  $node SetParameter label 1

  $::slicer3::MRMLScene AddNode $node
  $node Delete
}

# get the editor parameter node, or create one if it doesn't exist
proc EditorGetParameterNode {} {

  set node ""
  set nNodes [$::slicer3::MRMLScene GetNumberOfNodesByClass "vtkMRMLScriptedModuleNode"]
  for {set i 0} {$i < $nNodes} {incr i} {
    set n [$::slicer3::MRMLScene GetNthNodeByClass $i "vtkMRMLScriptedModuleNode"]
    if { [$n GetModuleName] == "Editor" } {
      set node $n
      break;
    }
  }

  if { $node == "" } {
    EditorCreateParameterNode
    set node [EditorGetParameterNode]
  }

  return $node
}

proc EditorSetActiveToolLabel {name} {
  [$::Editor($::Editor(singleton),toolsActiveTool) GetWidget] SetText $name
  [$::slicer3::ApplicationGUI GetMainSlicerWindow]  SetStatusText $name
}

proc EditorGetPaintLabel {} {
  set node [EditorGetParameterNode]
  if { [$node GetParameter "label"] == "" } {
    $node SetParameter "label" 1
  }
  return [$node GetParameter "label"]
}

proc EditorSetPaintLabel {index} {
  set node [EditorGetParameterNode]
  $node SetParameter "label" $index
}

proc EditorGetPaintColor {this} {

  set sliceLogic [$::slicer3::ApplicationGUI GetMainSliceLogic0]
  if { $sliceLogic != "" } {
    set logic [$sliceLogic GetLabelLayer]
    if { $logic != "" } {
      set volumeDisplayNode [$logic GetVolumeDisplayNode]
      if { $volumeDisplayNode != "" } {
        set node [$volumeDisplayNode GetColorNode]
        set lut [$node GetLookupTable]
        set index [EditorGetPaintLabel]
        return [$lut GetTableValue $index]
      }
    }
  }
  return "0 0 0 0"
}

proc EditorGetPaintThreshold {this} {
  return [$::Editor($this,paintRange) GetRange]
}

proc EditorSetPaintThreshold {this min max} {
  $::Editor($this,paintRange) SetRange $min $max
  EditorUpdateSWidgets $this
}

proc EditorGetPaintThresholdState {this onOff} {
  return [[$::Editor($this,paintThreshold) GetWidget] GetSelectedState]
}

proc EditorSetPaintThresholdState {this onOff} {
  [$::Editor($this,paintThreshold) GetWidget] SetSelectedState $onOff
  EditorUpdateSWidgets $this
}

proc EditorGetOptionsFrame {this} {
  return $::Editor($this,optionsFrame)
}

proc EditorSelectModule {} {
  set toolbar [$::slicer3::ApplicationGUI GetApplicationToolbar]
  [$toolbar GetModuleChooseGUI] SelectModule "Editor"
}


#
# MRML Event processing
#

proc EditorUpdateMRML {this} {
}

proc EditorProcessMRMLEvents {this callerID event} {

    if { [$this GetDebug] } {
        puts "EditorProcessMRMLEvents: event = $event, callerID = $callerID"
    }
    set caller [[[$this GetLogic] GetMRMLScene] GetNodeByID $callerID]
    if { $caller == "" } {
        return
    }
    set selectionNode  [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
    # get the active label volume
    set labelVolume [[[$this GetLogic] GetMRMLScene] GetNodeByID [$selectionNode GetActiveLabelVolumeID]]
    if { $labelVolume == "" } {
        if { [$this GetDebug] } { puts "No labelvolume..." }
        return
    }
    # check it's display node colour node
    set displayNode [$labelVolume GetDisplayNode] 

    if { $caller == $selectionNode && $event == 31 } {
        if { [$this GetDebug] } {
            puts "...caller is selection node, with modified event"
        }
        if {[info exists ::Editor($this,observedNodeID)] && $::Editor($this,observedNodeID) != "" &&
        $::Editor($this,observedNodeID) != [$selectionNode GetActiveLabelVolumeID]} {
            # remove the observer on the old volume's display node
            if { [$this GetDebug] } {
                puts "Removing mrml observer on last node: $::Editor($this,observedNodeID) and $::Editor($this,observedEvent)"
            }
            $this RemoveMRMLObserverByNumber [[[$this GetLogic] GetMRMLScene] GetNodeByID $::Editor($this,observedNodeID)] $::Editor($this,observedEvent)
        }
        

        # is it the one we're showing?
        if { $displayNode != "" } {
            if { [$displayNode GetColorNodeID] != [[$::Editor($this,colorsColor) GetColorNode] GetID] } {
                if { [$this GetDebug] } {
                    puts "Resetting the color node"
                }
                $::Editor($this,colorsColor) SetColorNode [$displayNode GetColorNode]
            }
            # add an observer on the volume node for display modified events
            if { [$this GetDebug] } {
                puts "Adding display node observer on label volume [$labelVolume GetID]"
            }
            $this AddMRMLObserverByNumber $labelVolume 18000
            set ::Editor($this,observedNodeID) [$labelVolume GetID]
            set ::Editor($this,observedEvent) 18000
        } else {
            if { [$this GetDebug] } {
                puts "Not resetting the color node, resetting the observed node id,  not adding display node observers, display node is null"
            }
            set ::Editor($this,observedNodeID) ""
            set ::Editor($this,observedEvent) -1
        }
        return
    } 

    if { $caller == $labelVolume && $event == 18000 } {
        if { [$this GetDebug] } {
            puts "... caller is label volume, got a display modified event, display node = $displayNode"
        }
        if { $displayNode != "" && [$displayNode GetColorNodeID] != [[$::Editor($this,colorsColor) GetColorNode] GetID] } {
            if { [$this GetDebug] } {
                puts "...resetting the color node"
            }
            $::Editor($this,colorsColor) SetColorNode [$displayNode GetColorNode]
        }
        return
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

  $labelNode Delete

  # update the editor range to be the full range of the background image
  set range [[$volumeNode GetImageData] GetScalarRange]
  eval $::Editor($this,paintRange) SetWholeRange $range
  eval $::Editor($this,paintRange) SetRange $range

  
}

