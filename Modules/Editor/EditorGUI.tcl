#
# Editor GUI Procs
#

proc EditorConstructor {this} {
}

proc EditorDestructor {this} {
}

proc EditorTearDownGUI {this} {

  # nodeSelector  ;# disabled for now
  set widgets {
      labelMapDerive labelMapSelect activeLabelMap
      volumesCreate volumeName volumesSelect
      volumesFrame 
      optionsSpacer optionsFrame
      toolsActiveTool toolsEditFrame toolsColorFrame
      toolsFrame enableCheckPoint
  }

  itcl::delete object $::Editor($this,editColor)
  itcl::delete object $::Editor($this,editBox)

  # kill any remaining boxes (popup)
  set boxes [itcl::find objects -isa Box]
  foreach b $boxes {
    itcl::delete object $b
  }

  foreach w $widgets {
    $::Editor($this,$w) SetParent ""
    $::Editor($this,$w) Delete
  }

  if { [[$this GetUIPanel] GetUserInterfaceManager] != "" } {
    set pageWidget [[$this GetUIPanel] GetPageWidget "Editor"]
    [$this GetUIPanel] RemovePage "Editor"
  }

  unset ::Editor(singleton)

  EditorFreeCheckPointVolumes

}



proc UpdateCurrentLabelMapLabel {this} {

    #--- check to see if there is an active label map. if so,
    #--- select that in the labelMapSelect selector.
    
    set labelmap [ $::Editor($this,labelMapSelect) GetSelected ]
    if { $labelmap != "" } {
        set mapname  [  $labelmap GetName ]
        if { $mapname != "" } {
            $::Editor($this,activeLabelMap) SetText "Editing $mapname..."
        } else {
            set id [ $labelmap GetID ]
            $::Editor($this,activeLabelMap) SetText "Editing $id..."
        }
    } else {
        $::Editor($this,activeLabelMap) SetText "No label map selected for editing..."
    }

    #--- clean out status text.
    [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText ""
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
  set helptext "The Editor allows label maps to be created and edited. The active label map will be modified by the Editor.  See <a>http://www.slicer.org/slicerWiki/index.php/Modules:Editor-Documentation</a>."
  set abouttext "This work is supported by NA-MIC, NAC, BIRN, NCIGT, and the Slicer Community. See <a>http://www.slicer.org</a> for details."
  $this BuildHelpAndAboutFrame $pageWidget $helptext $abouttext

  #
  # Editor Volumes
  #
  set ::Editor($this,volumesFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,volumesFrame) SetParent $pageWidget
  $::Editor($this,volumesFrame) Create
  $::Editor($this,volumesFrame) SetLabelText "Create & Select Label Maps"
  pack [$::Editor($this,volumesFrame) GetWidgetName] \
    -side top -anchor nw -fill x -expand true -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  set ::Editor($this,volumesSelect) [vtkSlicerNodeSelectorWidget New]
  $::Editor($this,volumesSelect) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumesSelect) Create
  $::Editor($this,volumesSelect) SetNodeClass "vtkMRMLScalarVolumeNode" "" "" ""
  $::Editor($this,volumesSelect) ChildClassesEnabledOff
  $::Editor($this,volumesSelect) SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $::Editor($this,volumesSelect) UpdateMenu
  $::Editor($this,volumesSelect) SetLabelText "Source Volume:"
  $::Editor($this,volumesSelect) SetBalloonHelpString "The Source Volume will define the dimensions and directions for the new label map"
  #--- new test
  if { 0 } {
      pack [$::Editor($this,volumesSelect) GetWidgetName] -side top -anchor e -padx 2 -pady 2
  }
  #--- end new test

  set ::Editor($this,volumeName) [vtkKWEntryWithLabel New]
  $::Editor($this,volumeName) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumeName) Create
  $::Editor($this,volumeName) SetLabelText "Name for label map volume: "
  [$::Editor($this,volumeName) GetWidget] SetValue "Working"
  $::Editor($this,volumeName) SetBalloonHelpString \
    "Leave blank for automatic label name based on input name."
  #--- new test
  if { 0 } {
      pack [$::Editor($this,volumeName) GetWidgetName] -side top -anchor e -padx 2 -pady 2
  }
  #--- end new test

  set ::Editor($this,volumesCreate) [vtkKWPushButton New]
  $::Editor($this,volumesCreate) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,volumesCreate) Create
  $::Editor($this,volumesCreate) SetText "Create Label Map"
  $::Editor($this,volumesCreate) SetBalloonHelpString "Create a new label map based on the source."
  #--- new test
  if { 0 } {
      pack [$::Editor($this,volumesCreate) GetWidgetName] -side top -anchor e -padx 2 -pady 2 
  }
  #--- end new test

  #--- when user selects a source volume,
  #--- new label map is created, named and selected
  #--- in the labelMap Selector.
  set ::Editor($this,labelMapDerive) [vtkSlicerNodeSelectorWidget New]
  $::Editor($this,labelMapDerive) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,labelMapDerive) Create
  $::Editor($this,labelMapDerive) SetNodeClass "vtkMRMLScalarVolumeNode" "" "" ""
  $::Editor($this,labelMapDerive) NewNodeEnabledOff
  $::Editor($this,labelMapDerive) ChildClassesEnabledOff
  $::Editor($this,labelMapDerive) NoneEnabledOn
  $::Editor($this,labelMapDerive) SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $::Editor($this,labelMapDerive) UpdateMenu
  $::Editor($this,labelMapDerive) SetLabelText "Create Label Map From:"
  $::Editor($this,labelMapDerive) SetBalloonHelpString "Create a new label map wth dimensions and directions from an existing volume."
  pack [$::Editor($this,labelMapDerive) GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  set ::Editor($this,labelMapSelect) [vtkSlicerNodeSelectorWidget New]
  $::Editor($this,labelMapSelect) SetParent [$::Editor($this,volumesFrame) GetFrame]
  $::Editor($this,labelMapSelect) Create
  $::Editor($this,labelMapSelect) SetNodeClass "vtkMRMLScalarVolumeNode" "LabelMap" "1" ""
  $::Editor($this,labelMapSelect) ChildClassesEnabledOff
  $::Editor($this,labelMapSelect) NoneEnabledOn
  $::Editor($this,labelMapSelect) SetMRMLScene [[$this GetLogic] GetMRMLScene]
  $::Editor($this,labelMapSelect) UpdateMenu
  $::Editor($this,labelMapSelect) SetLabelText "Select Label Map to Edit:"
  $::Editor($this,labelMapSelect) SetBalloonHelpString "Choose from existing label maps in the scene."
  pack [$::Editor($this,labelMapSelect) GetWidgetName] -side top -anchor e -padx 2 -pady 2 

  #
  # Tool Frame
  #
  set ::Editor($this,toolsFrame) [vtkSlicerModuleCollapsibleFrame New]
  $::Editor($this,toolsFrame) SetParent $pageWidget
  $::Editor($this,toolsFrame) Create
  $::Editor($this,toolsFrame) SetLabelText "Edit Selected Label Map"
  pack [$::Editor($this,toolsFrame) GetWidgetName] \
    -side top -anchor nw -fill x -expand true -padx 2 -pady 2 -in [$pageWidget GetWidgetName]

  set ::Editor($this,activeLabelMap) [ vtkKWLabel New ]
  $::Editor($this,activeLabelMap) SetParent [$::Editor($this,toolsFrame) GetFrame]
  $::Editor($this,activeLabelMap) Create
  $::Editor($this,activeLabelMap) SetForegroundColor 0.2 0.7 0.1
  UpdateCurrentLabelMapLabel $this
  pack [ $::Editor($this,activeLabelMap) GetWidgetName ] -side top -fill x -expand y -anchor w -padx 2 -pady 2


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
    -side left -anchor nw -fill both -expand true -padx 2 -pady 2 

  # one pixel label to keep the option panel expanded
    # TODO: doesn't work as intended
  set ::Editor($this,optionsSpacer) [vtkKWLabel New]
  $::Editor($this,optionsSpacer) SetParent $::Editor($this,optionsFrame)
  $::Editor($this,optionsSpacer) Create
  pack [$::Editor($this,optionsSpacer) GetWidgetName] \
    -fill both -expand true 

  # Enable check point check button
  set ::Editor($this,enableCheckPoint) [vtkKWCheckButton New]
  $::Editor($this,enableCheckPoint) SetParent $::Editor($this,toolsFrame)
  $::Editor($this,enableCheckPoint) Create
  $::Editor($this,enableCheckPoint) SetText "Enable Volume Check Points"
  $::Editor($this,enableCheckPoint) SetBalloonHelpString "Volume Check Points allow you to move back and forth through recent edits.\n\nNote: for large volumes, you may run out of system memory when this is enabled."
  $::Editor($this,enableCheckPoint) SetSelectedState 0
  pack [$::Editor($this,enableCheckPoint) GetWidgetName] \
    -side left
  set ::Editor(checkPointsEnabled) 0
}

proc EditorAddGUIObservers {this} {
    $this AddObserverByNumber $::Editor($this,volumesCreate) 10000 
    $this AddObserverByNumber $::Editor($this,enableCheckPoint) 10000 
    #--- new test
    $this AddObserverByNumber $::Editor($this,labelMapSelect) 11000 
    $this AddObserverByNumber $::Editor($this,labelMapSelect) 11002
    $this AddObserverByNumber $::Editor($this,labelMapDerive) 11000 
    #--- end new test    

# $this DebugOn
    if {[$this GetDebug]} {
        puts "Adding mrml observer to selection node, modified event"
    }
    $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic] GetSelectionNode] 31
}

proc EditorRemoveGUIObservers {this} {
    #--- new test
  $this RemoveObserverByNumber $::Editor($this,volumesCreate) 10000 
  $this RemoveObserverByNumber $::Editor($this,enableCheckPoint) 10000 
  $this RemoveObserverByNumber $::Editor($this,labelMapSelect) 11000 
  $this RemoveObserverByNumber $::Editor($this,labelMapSelect) 11002
  $this RemoveObserverByNumber $::Editor($this,labelMapDerive) 11000 
    #--- end new test
  
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
  } 
  if { $caller == $::Editor($this,labelMapSelect) } {
    switch $event {
      "11000" {
          #--- give users some feedback
          [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText "Selecting label map..."
          update idletasks
          #--- put this selected label map into the label layer (make it active).
          set labelNode [$::Editor($this,labelMapSelect) GetSelected ]
          if {  $labelNode != ""  &&  [$this GetLogic] != ""  &&  [[$this GetLogic] GetApplicationLogic] != "" } {
              set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
          } else {
              EditorErrorDialog "The selected label map cannot be used for editing. Please derive a new, or select a different one."
          }
          if { $selectionNode != "" } {
              $selectionNode SetReferenceActiveLabelVolumeID [$labelNode GetID]
              [[$this GetLogic] GetApplicationLogic]  PropagateVolumeSelection
          } else {
              EditorErrorDialog "Error trying to display label map in the lable layer. Please try to do this manually."
          }
          #--- then update the label to communicate 
          #--- which label map user is currently editing
          UpdateCurrentLabelMapLabel $this
          [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText ""
          update idletasks
          return
      } "11002" {
          #--- new test
          UpdateCurrentLabelMapLabel $this
          return
          #--- end new test
      }
    }
  }

  if { $caller == $::Editor($this,labelMapDerive) } {
    switch $event {
      "11000" {
          #--- add some text feedback to user
          set vName [[[$::Editor($this,labelMapDerive) GetWidget] GetWidget] GetValue ]
          if { $vName != "" } {
              [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText "Creating new label map based on $vName..."
          } else {
              [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText "Creating new label map..."
          }
          update idletasks
          #--- derive new label map
          set mapID [ EditorCreateLabelVolume $this ]
          set mapNode [ $::slicer3::MRMLScene GetNodeByID $mapID ]
          if { $mapNode != "" } {
              $::Editor($this,labelMapSelect) SetSelected $mapNode
              #--- then update the label to communicate 
              #--- which label map user is currently editing
              UpdateCurrentLabelMapLabel $this
          } else {
              EditorErrorDialog "Error trying to either derive label map, or in trying to select it for display and editing."
          }
          #--- clear out status text.
          [$::slicer3::ApplicationGUI GetMainSlicerWindow] SetStatusText ""
      }
    }
  }

  if { $caller == $::Editor($this,enableCheckPoint) } {
    switch $event {
      "10000" {
        set onoff [$::Editor($this,enableCheckPoint) GetSelectedState]
        EditorSetCheckPointEnabled $onoff
      }
    }
  } 

  EditorUpdateMRML $this
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
  return [expr int([$node GetParameter "label"])]
}

proc EditorSetPaintLabel {index} {
  set node [EditorGetParameterNode]
  set index [expr int($index)]
  $node SetParameter "label" $index
}

proc EditorToggleErasePaintLabel {} {
  # if in erase mode (label is 0), set to stored color
  # if in color, store current and set to 0
  if { [EditorGetPaintLabel] == 0 } {
    if { [info exists ::Editor(savedLabelValue)] } {
      EditorSetPaintLabel $::Editor(savedLabelValue)
    }
  } else {
    set ::Editor(savedLabelValue) [EditorGetPaintLabel]
    EditorSetPaintLabel 0
  }
}


proc EditorGetPaintColor {this} {

  set sliceLogic [$::slicer3::ApplicationLogic GetSliceLogic "Red"]
  if { $sliceLogic != "" } {
    set logic [$sliceLogic GetLabelLayer]
    if { $logic != "" } {
      set volumeDisplayNode [$logic GetVolumeDisplayNode]
      if { $volumeDisplayNode != "" } {
        set node [$volumeDisplayNode GetColorNode]
        set lut [$node GetLookupTable]
        set index [EditorGetPaintLabel]
        set index [expr int($index)]
        return [$lut GetTableValue $index]
      }
    }
  }
  return "0 0 0 0"
}

proc EditorGetPaintThreshold {} {
  set node [EditorGetParameterNode]
  return [list \
    [$node GetParameter "Labeler,paintThresholdMin"] \
    [$node GetParameter "Labeler,paintThresholdMax"]]
}

proc EditorSetPaintThreshold {min max} {
  set node [EditorGetParameterNode]
  $node SetParameter "Labeler,paintThresholdMin" $min
  $node SetParameter "Labeler,paintThresholdMax" $max
}

proc EditorGetPaintThresholdState {} {
  set node [EditorGetParameterNode]
  return [$node GetParameter "Labeler,paintThreshold"]
}

proc EditorSetPaintThresholdState {onOff} {
  set node [EditorGetParameterNode]
  $node SetParameter "Labeler,paintThreshold" $onOff
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
            if {0} {
                # deprecated, Editor(x,colorsColor) no longer used
                if { [$displayNode GetColorNodeID] != [[$::Editor($this,colorsColor) GetColorNode] GetID] } {
                    if { [$this GetDebug] } {
                        puts "Resetting the color node"
                    }
                    $::Editor($this,colorsColor) SetColorNode [$displayNode GetColorNode]
                }
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
        if {0} {
            # deprecated, not using colorsColor
            if { $displayNode != "" && [$displayNode GetColorNodeID] != [[$::Editor($this,colorsColor) GetColorNode] GetID] } {
                if { [$this GetDebug] } {
                    puts "...resetting the color node"
                }
                $::Editor($this,colorsColor) SetColorNode [$displayNode GetColorNode]
            }
        }
        return
    }
}

proc EditorEnter {this} {
    if {[$this GetDebug]} {
        puts "EditorEnter: Adding mrml observer on selection node, modified event"
    }

  $this AddMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode] 31

  $::Editor($this,volumesSelect) UpdateMenu
    #--- new test
  UpdateCurrentLabelMapLabel $this
    #--- end new test
}

proc EditorExit {this} {

  EffectSWidget::RemoveAll
  if {[$this GetDebug]} {
    puts "EditorExit: Removing mrml observer on selection node modified event"
  }
  $this RemoveMRMLObserverByNumber [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode] 31
  
  # delete the current effect - users were getting confused that the editor was still
  # active when the module wasn't visible
  after idle ::EffectSWidget::RemoveAll
}

# TODO: there might be a better place to put this for general use...  
proc EditorCreateLabelVolume {this} {
  set volumeNode [$::Editor($this,volumesSelect) GetSelected]

  #--- new test
  set volumeNode [ $::Editor($this,labelMapDerive) GetSelected]
  #--- end new test

  if { $volumeNode == "" } {
    EditorErrorDialog "Select Scalar Source Volume for Label Map"
    return;
  }

  #--- new test
  #  set name [[$::Editor($this,volumeName) GetWidget] GetValue]
  #  if { $name == "" } {
  #   set name "[$volumeNode GetName]-label"
  #  }
   set name "[$volumeNode GetName]-label"
  #--- end new test  

  set scene [[$this GetLogic] GetMRMLScene]

  set volumesLogic [$::slicer3::VolumesGUI GetLogic]
  set labelNode [$volumesLogic CreateLabelVolume $scene $volumeNode $name]

  # make the source node the active background, and the label node the active label
  set selectionNode [[[$this GetLogic] GetApplicationLogic]  GetSelectionNode]
  $selectionNode SetReferenceActiveVolumeID [$volumeNode GetID]
  $selectionNode SetReferenceActiveLabelVolumeID [$labelNode GetID]
  [[$this GetLogic] GetApplicationLogic]  PropagateVolumeSelection
  
  #--- new test
  set id [ $labelNode GetID ]
  #--- end new test
  $labelNode Delete

  # update the editor range to be the full range of the background image
  set range [[$volumeNode GetImageData] GetScalarRange]
  eval ::Labler::SetPaintRange $range

  #--- new test
  return $id
  #--- end new test
}

#
# store/restore a fixed number of volumes
# - for now, limit to 10 check point volumes total
# - keep two lists - check point and redo
# -- adding to nextCheckPoint invalidates redo list
# - each list entry is a pair of vtkImageData and mrml Node ID
#
proc EditorFreeVolumes {volumeList} {
  foreach volume $volumeList {
    foreach {imageData nodeID} $volume {}
    if { [info command $imageData] != "" } {
      $imageData Delete
    }
  }
}

# called by checkbox in GUI
proc EditorSetCheckPointEnabled {onoff} {
  if { !$onoff } {
    EditorFreeCheckPointVolumes
    EditorUpdateCheckPointButtons
  }
  set ::Editor(checkPointsEnabled) $onoff
}

# called at tear down time to free local instances
proc EditorFreeCheckPointVolumes {} {
  foreach list {previousCheckPointImages nextCheckPointImages} {
    if { [info exists ::Editor($list)] } {
      EditorFreeVolumes $::Editor($list)
      set ::Editor($list) ""
    }
  }
}

proc EditorUpdateCheckPointButtons {} {
  if { $::Editor(previousCheckPointImages) != "" } {
    EditBox::SetButtonState PreviousCheckPoint ""
  } else {
    EditBox::SetButtonState PreviousCheckPoint Disabled
  }
  if { $::Editor(nextCheckPointImages) != "" } {
    EditBox::SetButtonState NextCheckPoint ""
  } else {
    EditBox::SetButtonState NextCheckPoint Disabled
  }
}

# called by editor effects
proc EditorStoreCheckPoint {node} {

  if { !$::Editor(checkPointsEnabled) } {
    return
  }

  EditorStoreCheckPointVolume [$node GetID] previousCheckPointImages

  # invalidate NextCheckPoint list
  EditorFreeVolumes $::Editor(nextCheckPointImages)

  EditorUpdateCheckPointButtons 
}


# unsed internally to manage nodes
proc EditorStoreCheckPointVolume {nodeID listID} {
  
  # create lists if needed
  if { ![info exists ::Editor(previousCheckPointImages)] } {
    set ::Editor(previousCheckPointImages) ""
    set ::Editor(nextCheckPointImages) ""
    set ::Editor(numberOfCheckPoints) 10
  }

  # trim oldest previousCheckPoint image if needed
  if { [llength $::Editor($listID)] >= $::Editor(numberOfCheckPoints) } {
    set disposeImage [lindex $::Editor($listID) 0]
    foreach {imageData nodeID} $disposeImage {}
    $imageData Delete
    set ::Editor($listID) [lrange $::Editor($listID) 1 end]
  }

  # add new 
  set node [$::slicer3::MRMLScene GetNodeByID $nodeID]
  set imageData [vtkImageData New]
  if { $node != "" } {
    $imageData DeepCopy [$node GetImageData]
  } else {
    error "no node for $nodeID"
  }
  set nodeID [$node GetID]
  lappend ::Editor($listID) "$imageData $nodeID"
}

proc EditorRestoreData {imageData nodeID} {
  # restore the volume data
  set node [$::slicer3::MRMLScene GetNodeByID $nodeID]
  if { $node != "" } {
    [$node GetImageData] DeepCopy $imageData
  } else {
    error "no node for $nodeID"
  }
  $node SetModifiedSinceRead 1
  $node Modified
}

# called by button presses or keystrokes
proc EditorPerformPreviousCheckPoint {} {
  if { ![info exists ::Editor(previousCheckPointImages)] || [llength $::Editor(previousCheckPointImages)] == 0 } {
    return
  }

  # get the volume to restore
  set restore [lindex $::Editor(previousCheckPointImages) end]
  foreach {imageData nodeID} $restore {}

  # save the current state as a redo point
  EditorStoreCheckPointVolume $nodeID nextCheckPointImages

  # now pop the next item on the previousCheckPoint stack
  set ::Editor(previousCheckPointImages) [lrange $::Editor(previousCheckPointImages) 0 end-1]

  EditorRestoreData $imageData $nodeID
  EditorUpdateCheckPointButtons 
}

# called by button presses or keystrokes
proc EditorPerformNextCheckPoint {} {
  if { ![info exists ::Editor(nextCheckPointImages)] || [llength $::Editor(nextCheckPointImages)] == 0 } {
    return
  }

  # get the volume to restore
  set restore [lindex $::Editor(nextCheckPointImages) end]
  foreach {imageData nodeID} $restore {}

  # save the current state as an previousCheckPoint Point
  EditorStoreCheckPointVolume $nodeID previousCheckPointImages

  # now pop the next item on the redo stack
  set ::Editor(nextCheckPointImages) [lrange $::Editor(nextCheckPointImages) 0 end-1]
  EditorRestoreData $imageData $nodeID
  EditorUpdateCheckPointButtons 
}

#
# helper to display error
#
proc EditorErrorDialog {errorText} {
  set dialog [vtkKWMessageDialog New]
  $dialog SetParent [$::slicer3::ApplicationGUI GetMainSlicerWindow]
  $dialog SetMasterWindow [$::slicer3::ApplicationGUI GetMainSlicerWindow]
  $dialog SetStyleToMessage
  $dialog SetText $errorText
  $dialog Create
  $dialog Invoke
  $dialog Delete
}

