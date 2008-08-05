package require Itcl

#########################################################
#
if {0} { ;# comment

  LoadVolume is a wrapper around a set of kwwidgets and 
  file manipulation logic load a single volume file

# TODO : 
  - part of this should be split out into logic helper methods
  - this could be reimplemented in C++ pretty easily
  - it would be nice if vtkITKArchetype* could be used
    to tell us the filenames for a given archetype without
    actually reading.  That would allow the add method
    to be intelligent about dicom files and other series

}
#
#########################################################


namespace eval LoadVolume {

  # 
  # utility to bring up the current window or create a new one
  # - optional path is added to dialog
  #
  proc ShowDialog {} {

    set loaders [itcl::find objects -class LoadVolume]
    if { $loaders != "" } {
      set loader [lindex $loaders 0]
      array set o [$loader objects]
      raise [$o(toplevel) GetWidgetName]
    } else {
      set loader [LoadVolume #auto]
    }
  }
}


#
# The partent class definition - define if needed (not when re-sourcing)
#
if { [itcl::find class LoadVolume] == "" } {

  itcl::class LoadVolume {

    constructor  {} {
    }

    destructor {
      vtkDelete  
    }

    variable _vtkObjects ""

    variable o ;# array of the objects for this widget, for convenient access

    variable _volumeExtensions ".hdr .nhdr .nrrd .mhd .mha .vti .mgz"
    variable _observerRecords ""
    variable _processingEvents 0
    variable _DICOM ;# map from group/element to name
    variable _dicomColumn ;# keep track of columns
    variable _dicomTree ;# currently loaded dicom directory

    # methods
    method processEvent {{caller ""} {event ""}} {}
    method selectArchetype {path name} {}
    method apply {} {}
    method status { message } {}
    method errorDialog {errorText} {}
    method loadDICOMDictionary {} {}
    method parseDICOMHeader {fileName arrayName} {}
    method populateDICOMTable {fileName} {}
    method parseDICOMDirectory {directoryName arrayName} {}
    method populateDICOMTree {directoryName arrayName} {}
    method safeNodeName {name} {}

    method objects {} {return [array get o]}

    # make a new instance of a class and add it to the list for cleanup
    method vtkNew {class} {
      set object [$class New]
      set _vtkObjects "$object $_vtkObjects"
      return $object
    }

    # clean up the vtk classes instanced by this LoadVolume
    method vtkDelete {} {
      foreach object $_vtkObjects {
        catch "$object Delete"
      }
      set _vtkObjects ""
    }

  }
}

# ------------------------------------------------------------------
#                        CONSTRUCTOR/DESTRUCTOR
# ------------------------------------------------------------------
itcl::body LoadVolume::constructor {} {

  #
  # make the toplevel 
  #
  set o(toplevel) [vtkNew vtkKWTopLevel]
  $o(toplevel) SetApplication $::slicer3::Application
  $o(toplevel) SetTitle "Add Volume"
  $o(toplevel) Create
  $o(toplevel) SetGeometry 800x800

  # delete this instance when the window is closed
  wm protocol [$o(toplevel) GetWidgetName] \
    WM_DELETE_WINDOW "itcl::delete object $this"

  #
  # top frame
  #
  set o(topFrame) [vtkNew vtkKWFrame]
  $o(topFrame) SetParent $o(toplevel)
  $o(topFrame) Create
  pack [$o(topFrame) GetWidgetName] -side top -anchor nw -fill x

  #
  # the file browser widget
  #
  set o(browser) [vtkNew vtkKWFileBrowserWidget]
  $o(browser) SetParent $o(topFrame)
  $o(browser) Create

  $::slicer3::Application RequestRegistry "OpenPath"
  set path [$::slicer3::Application GetRegistryHolder]
  $o(browser) OpenDirectory $path

  set fileTable [$o(browser) GetFileListTable]
  $fileTable SetSelectionModeToSingle
  set tag [$fileTable AddObserver DeleteEvent "itcl::delete $this"]
  lappend _observerRecords [list $fileTable $tag]
  set tag [$fileTable AddObserver AnyEvent "$this processEvent $o(browser)"]
  lappend _observerRecords [list $fileTable $tag]

  set directoryExplorer [$o(browser) GetDirectoryExplorer]
  set tag [$directoryExplorer AddObserver DeleteEvent "itcl::delete $this"]
  lappend _observerRecords [list $directoryExplorer $tag]
  set tag [$directoryExplorer AddObserver AnyEvent "$this processEvent $o(browser)"]
  lappend _observerRecords [list $directoryExplorer $tag]

  #
  # the current Path
  #
  set o(path) [vtkNew vtkKWEntryWithLabel]
  $o(path) SetParent $o(topFrame)
  $o(path) SetLabelText "Path: "
  $o(path) Create
  set tag [[$o(path) GetWidget] AddObserver DeleteEvent "itcl::delete $this"]
  lappend _observerRecords [list [$o(path) GetWidget] $tag]
  set tag [[$o(path) GetWidget] AddObserver AnyEvent "$this processEvent $o(path)"]
  lappend _observerRecords [list [$o(path) GetWidget] $tag]


  #
  # the options frame
  #
  set o(options) [vtkNew vtkKWFrameWithLabel]
  $o(options) SetParent $o(topFrame)
  $o(options) SetLabelText "Volume Options"
  $o(options) Create

  pack \
    [$o(browser) GetWidgetName] \
    [$o(path) GetWidgetName] \
    [$o(options) GetWidgetName] \
    -side top -anchor e -padx 2 -pady 2 -expand true -fill both

  #
  # the options contents
  #
  set o(centered) [vtkNew vtkKWCheckButton]
  $o(centered) SetParent [$o(options) GetFrame]
  $o(centered) SetText "Centered"
  $o(centered) Create

  set o(orient) [vtkNew vtkKWCheckButton]
  $o(orient) SetParent [$o(options) GetFrame]
  $o(orient) SetText "Orientation From File"
  $o(orient) Create

  set o(label) [vtkNew vtkKWCheckButton]
  $o(label) SetParent [$o(options) GetFrame]
  $o(label) SetText "Label Map"
  $o(label) Create

  set o(singlefile) [vtkNew vtkKWCheckButton]
  $o(singlefile) SetParent [$o(options) GetFrame]
  $o(singlefile) SetText "Single File"
  $o(singlefile) Create

  set o(name) [vtkNew vtkKWEntryWithLabel]
  $o(name) SetParent [$o(options) GetFrame]
  $o(name) SetLabelText "Name: "
  $o(name) Create

  pack [$o(centered) GetWidgetName] \
    -side left -anchor e -padx 2 -pady 2 -expand true -fill both

  pack [$o(orient) GetWidgetName] \
    -side left -anchor e -padx 2 -pady 2 -expand true -fill both

  pack [$o(label) GetWidgetName] \
    -side left -anchor e -padx 2 -pady 2 -expand true -fill both
  pack [$o(singlefile) GetWidgetName] \
    -side left -anchor e -padx 2 -pady 2 -expand true -fill both

  pack [$o(name) GetWidgetName] \
    -side top -anchor e -padx 2 -pady 2 -expand true -fill both

  #
  # a status label
  #
  set o(status) [vtkNew vtkKWLabel]
  $o(status) SetParent $o(toplevel)
  $o(status) Create
  $o(status) SetText ""
  $o(status) SetBalloonHelpString "Current status of the load process"
  pack [$o(status) GetWidgetName] -side top -anchor e -padx 2 -pady 2 -expand false -fill x


  #
  # the apply and cancel buttons
  #

  set o(buttonFrame) [vtkNew vtkKWFrame]
  $o(buttonFrame) SetParent $o(toplevel)
  $o(buttonFrame) Create
  pack [$o(buttonFrame) GetWidgetName] -side top -anchor nw -fill x

  set o(apply) [vtkNew vtkKWPushButton]
  $o(apply) SetParent $o(buttonFrame)
  $o(apply) Create
  $o(apply) SetText "Apply"
  $o(apply) SetBalloonHelpString "Load the listed files into slicer"
  set tag [$o(apply) AddObserver ModifiedEvent "$this processEvent $o(apply)"]
  lappend _observerRecords [list $o(apply) $tag]
  $o(apply) SetCommand $o(apply) Modified

  set o(cancel) [vtkNew vtkKWPushButton]
  $o(cancel) SetParent $o(buttonFrame)
  $o(cancel) Create
  $o(cancel) SetText "Cancel"
  $o(cancel) SetBalloonHelpString "Close window without loading files"
  set tag [$o(cancel) AddObserver ModifiedEvent "$this processEvent $o(cancel)"]
  lappend _observerRecords [list $o(cancel) $tag]
  $o(cancel) SetCommand $o(cancel) Modified

  pack \
    [$o(cancel) GetWidgetName] \
    [$o(apply) GetWidgetName] \
    -side right -anchor w -padx 4 -pady 2

  #
  # the dicom frame
  #
  set o(dicom) [vtkNew vtkKWFrameWithLabel]
  $o(dicom) SetParent $o(topFrame)
  $o(dicom) SetLabelText "DICOM Information"
  $o(dicom) Create

  #
  # parse dicom button - initially disabled, but enabled when
  # dicom file is selected
  #
  set o(dicomParse) [vtkNew vtkKWPushButton]
  $o(dicomParse) SetParent [$o(dicom) GetFrame]
  $o(dicomParse) Create
  $o(dicomParse) SetText "Parse Directory"
  $o(dicomParse) SetBalloonHelpString "Parse the current directory to select series to load as volume"
  set tag [$o(dicomParse) AddObserver ModifiedEvent "$this processEvent $o(dicomParse)"]
  lappend _observerRecords [list $o(dicomParse) $tag]
  $o(dicomParse) SetCommand $o(dicomParse) Modified
  $o(dicomParse) SetStateToDisabled

  #
  # the listbox of dicom info
  #

  set o(dicomList) [vtkNew vtkKWMultiColumnListWithScrollbars]
  $o(dicomList) SetParent [$o(dicom) GetFrame]
  $o(dicomList) Create
  $o(dicomList) SetHeight 4
  set w [$o(dicomList) GetWidget]
  $w SetSelectionTypeToCell
  $w MovableRowsOff
  $w MovableColumnsOn 
  $w SetPotentialCellColorsChangedCommand $w "ScheduleRefreshColorsOfAllCellsWithWindowCommand"
  $w SetColumnSortedCommand $w "ScheduleRefreshColorsOfAllCellsWithWindowCommand"

  set columns {Group/Element Description Value}
  foreach column $columns {
    set _dicomColumn($column) [$w AddColumn $column]
  }

  # configure the entries
  set widths {15 45 50}
  foreach column $columns width $widths {
    $w SetColumnEditWindowToCheckButton $_dicomColumn($column)
    $w ColumnEditableOn $_dicomColumn($column)
    $w SetColumnWidth $_dicomColumn($column) $width
  }


  #
  # SeriesSelection Tree
  #
  set o(dicomTree) [vtkNew vtkKWTreeWithScrollbars]
  $o(dicomTree) SetParent [$o(dicom) GetFrame]
  [$o(dicomTree) GetWidget] SetSelectionModeToSingle
  $o(dicomTree) Create

  set t [$o(dicomTree) GetWidget]
  set tag [$t AddObserver DeleteEvent "itcl::delete $this"]
  lappend _observerRecords [list $t $tag]
  set tag [$t AddObserver AnyEvent "$this processEvent $t"]
  lappend _observerRecords [list $t $tag]

  pack [$o(dicomParse) GetWidgetName] -side top -anchor e -padx 2 -pady 2 -expand false -fill none
  pack [$o(dicomTree) GetWidgetName] -side right -anchor e -padx 2 -pady 2 -expand true -fill both
  pack [$o(dicomList) GetWidgetName] -side left -anchor e -padx 2 -pady 2 -expand true -fill both
  pack [$o(dicom) GetWidgetName] -side bottom -anchor e -padx 2 -pady 2 -expand false -fill x

  #
  # pop up the dialog
  #
  $o(toplevel) Display
}


itcl::body LoadVolume::destructor {} {

  foreach record $_observerRecords {
    foreach {obj tag} $record {
      if { [info command $obj] != "" } {
        $obj RemoveObserver $tag
      }
    }
  }

  $this vtkDelete

}


#
# load the items from the list box
# into slicer return 0 on success, non-zero otherwise
#
itcl::body LoadVolume::apply { } {

  set centered [$o(centered) GetSelectedState]
  set oriented [$o(orient) GetSelectedState]
  set labelMap [$o(label) GetSelectedState]
  set singleFile [$o(singlefile) GetSelectedState]

  set loadingOptions [expr $labelMap * 1 + $centered * 2 + $singleFile * 4 + $oriented * 16]

  set fileTable [$o(browser) GetFileListTable]
  set fileName [$fileTable GetNthSelectedFileName 0]
  if { $fileName == "" } {
    $this errorDialog "No file selected"
    return 1
  }
  set name [[$o(name) GetWidget] GetValue]
  if { $name == "" } {
    set name [file root [file tail $fileName]]
  }

  puts "loading with archetype $fileName as $name"

  set volumeLogic [$::slicer3::VolumesGUI GetLogic]
  set ret [catch [list $volumeLogic AddArchetypeVolume "$fileName" $name $loadingOptions] node]
  if { $ret } {
    $this errorDialog "Could not load $fileName as a volume\n\nError is:\n$node"
    return 1
  }
  set selNode [$::slicer3::ApplicationLogic GetSelectionNode]
  if { $node == "" } {
    $this errorDialog "Could not load $fileName as a volume"
    return 1
  } else {
    if { $labelMap } {
      $selNode SetReferenceActiveLabelVolumeID [$node GetID]
    } else {
      $selNode SetReferenceActiveVolumeID [$node GetID]
    }
    $::slicer3::ApplicationLogic PropagateVolumeSelection

    $::slicer3::Application SetRegistry "OpenPath" [file dirname $fileName]
  }
  return 0
}


#
# handle gui events
# -basically just map button events onto methods
#
itcl::body LoadVolume::processEvent { {caller ""} {event ""} } {

  if { $_processingEvents } {
    return
  }

  if { $caller == $o(apply) } {
    if { [$this apply] == 0 } {
      after idle "itcl::delete object $this"
    }
    return
  }

  if { $caller == $o(cancel) } {
    after idle "itcl::delete object $this"
    return
  }

  if { $caller == $o(browser) } {
    set fileTable [$o(browser) GetFileListTable]
    set fileName [$fileTable GetNthSelectedFileName 0]
    if { [file isdirectory $fileName] } {
      set fileName [lindex [glob $fileName/*] 0]
    }
    set name [file root [file tail $fileName]]
    $this selectArchetype $fileName $name
    return
  }

  if { $caller == $o(path) } {
    set path [[$o(path) GetWidget] GetValue]
    set name [file tail $path]
    $this selectArchetype $path $name
    return
  }

  if { $caller == $o(dicomParse) } {
    set path [[$o(path) GetWidget] GetValue]
    if { [file isdirectory $path] } {
      set dir $path
    } else {
      set dir [file dirname $path]
    }
    $this populateDICOMTree $dir _dicomTree
    return
  }

  set t [$o(dicomTree) GetWidget]
  if { $caller == $t } {
    set selection [$t GetSelection]
    set name [$t GetNodeText $selection]
    switch -glob $selection {
      patient* {
        set patient $name
        set study [lindex $_dicomTree($patient,studies) 0]
        set series [lindex $_dicomTree($patient,$study,series) 0]
        set seriesNode series-[$this safeNodeName $series]
      }
      study* {
        set patientNode [$t GetNodeParent $selection]
        set patient [$t GetNodeText $patientNode]
        set study $name
        set series [lindex $_dicomTree($patient,$study,series) 0]
        set seriesNode series-[$this safeNodeName $series]
      }
      series* {
        set seriesNode $selection
      }
      file* {
        set seriesNode [$t GetNodeParent $selection]
      }
      default {
        errorDialog "can't parse selection \"$selection\""
      }
    }

    $t SelectSingleNode $seriesNode
    $t SeeNode $seriesNode

    set archetypeNode [lindex [$t GetNodeChildren $seriesNode] 0]
    set archetype [$t GetNodeText $archetypeNode]
    $this selectArchetype $archetype [$t GetNodeText $seriesNode]

    return
  }
  
  puts "unknown event from $caller"
}

itcl::body LoadVolume::selectArchetype { path name } {

  set fileTable [$o(browser) GetFileListTable]
  set directoryExplorer [$o(browser) GetDirectoryExplorer]

  if { [file isdirectory $path] } {
    set dir $path
  } else {
    set dir [file dirname $path]
  }

  set _processingEvents 1
  $directoryExplorer SelectDirectory $dir
  $fileTable ClearSelection
  $fileTable SelectFileName $path
  $fileTable ScrollToFile [file tail $path]

  [$o(path) GetWidget] SetValue $path
  [$o(name) GetWidget] SetValue $name
  $this populateDICOMTable $path
  set _processingEvents 0
}


itcl::body LoadVolume::errorDialog { errorText } {
  set dialog [vtkKWMessageDialog New]
  $dialog SetParent $o(toplevel)
  $dialog SetMasterWindow $o(toplevel)
  $dialog SetStyleToMessage
  $dialog SetText $errorText
  $dialog Create
  $dialog Invoke
  $dialog Delete
}

itcl::body LoadVolume::status { message } {
  $o(status) SetText $message
}

itcl::body LoadVolume::loadDICOMDictionary {} {
  if { [info exists _DICOM(loaded)] } {
    return
  }

  set dicomDict ""
  set dicomDictCandidates [list \
    $::env(ITK_BIN_DIR)/../Utilities/gdcm/Dicts/gdcm.dic \
    $::env(ITK_BIN_DIR)/../../Utilities/gdcm/Dicts/gdcm.dic]
  foreach dictFile $dicomDictCandidates {
    if { [file exists $dictFile] } {
      set dicomDict $dictFile
    }
  }

  if { $dicomDict == "" } {
    $this errorDialog "Cannot find dicom dictionary to load"
  set _DICOM(loaded) "Fail"
    return
  }

  set fp [open $dicomDict]
  while { ![eof $fp] } {
    gets $fp line
    set group [lindex $line 0]
    set element [lindex $line 1]
    set type [lindex $line 2]
    set rep [lindex $line 3]
    set description [lrange $line 4 end]
    set _DICOM($group|$element) $description
  }
  close $fp

  set _DICOM(loaded) "Success"
}

itcl::body LoadVolume::populateDICOMTable {fileName} {

  set scrollState [[lindex [[[$o(dicomList) GetWidget] GetWidgetName] cget -yscrollcommand] 0] get]
  [$o(dicomList) GetWidget] DeleteAllRows

  $this parseDICOMHeader $fileName header
  if { $fileName == "" || ![file exists $fileName] || !$header(isDICOM) } {
    $o(dicomParse) SetStateToDisabled
    return
  }
  $o(dicomParse) SetStateToNormal

  set w [$o(dicomList) GetWidget] 
  for {set n 0} {$n < $header(numberOfKeys)} {incr n} {
    set key $header($n,key)
    set description $header($key,description)
    set value $header($key,value)
    $w InsertCellText $n $_dicomColumn(Group/Element) $key
    $w InsertCellText $n $_dicomColumn(Description) $description
    $w InsertCellText $n $_dicomColumn(Value) $value
  }
  [[$o(dicomList) GetWidget] GetWidgetName] yview moveto [lindex $scrollState 0]
}

itcl::body LoadVolume::parseDICOMHeader {fileName arrayName} {

  upvar $arrayName header
  set header(fileName $fileName)
  set header(isDICOM) 0

  if { $fileName == "" || ![file exists $fileName] || [file isdirectory $fileName] } {
    return
  }

  $this loadDICOMDictionary

  set reader [vtkITKArchetypeImageSeriesReader New]
  $reader SetArchetype $fileName
  $reader SetSingleFile 1
  $reader UpdateInformation

  set isDICOM 0
  set header(numberOfKeys) [$reader GetNumberOfItemsInDictionary]
  for {set n 0} {$n < $header(numberOfKeys)} {incr n} {
    set key [$reader GetNthKey $n]
    set value [$reader GetTagValue $key]

    if { [info exists _DICOM($key)] } {
      set description $_DICOM($key)
      set isDICOM 1
    } else {
      set description "Unknown key"
    }
    set header($n,key) $key
    set header($key,description) $description
    set header($key,value) $value
  }
  set header(isDICOM) $isDICOM

  $reader Delete
}

itcl::body LoadVolume::safeNodeName {name} {
  set newname ""
  set len [string length $name]
  for {set n 0} {$n < $len} {incr n} {
    set c [string index $name $n]
    if { ![string is alnum $c] } {
      scan $c "%c" value
      set c [format %%%02x $value]
    }
    set newname $newname$c
  }
  return $newname
}

itcl::body LoadVolume::populateDICOMTree {directoryName arrayName} {

  upvar $arrayName tree
  $this parseDICOMDirectory $directoryName tree
  set t [$o(dicomTree) GetWidget]
  $t DeleteAllNodes

  foreach patient $tree(patients) {
    set patientNode patient-[$this safeNodeName $patient]
    $t AddNode "" $patientNode $patient
    $t OpenNode $patientNode
    foreach study $tree($patient,studies) {
      set studyNode study-[$this safeNodeName $study]
      $t AddNode $patientNode $studyNode $study
      $t OpenNode $studyNode
      foreach series $tree($patient,$study,series) {
        set seriesNode series-[$this safeNodeName $series]
        $t AddNode $studyNode $seriesNode $series
        foreach file $tree($patient,$study,$series,files) {
          set fileNode file-[$this safeNodeName $file]
          $t AddNode $seriesNode $fileNode $file
        }
      }
    }
  }
}

itcl::body LoadVolume::parseDICOMDirectory {directoryName arrayName} {

  upvar $arrayName tree

  set progressDialog [vtkKWProgressDialog New]
  $progressDialog SetParent $o(toplevel)
  $progressDialog SetMasterWindow $o(toplevel)
  $progressDialog SetTitle "Parsing DICOM Files in $directoryName..."
  $progressDialog SetMessageText "Starting..."
  $progressDialog SetDisplayPositionToMasterWindowCenter
  $progressDialog Create
  $progressDialog Display

  set PATIENT "0010|0010"
  set STUDY "0008|1030"
  set SERIES "0008|103e"

  set files [glob $directoryName/*]
  set totalFiles [llength $files]
  set fileCount 0

  set tree(patients) ""
  foreach f $files {

    set ff [file tail $f]
    incr fileCount
    set progress [expr pow((1. * $fileCount) / $totalFiles,2)]
    $progressDialog SetMessageText "Exaining $ff..."
    $progressDialog UpdateProgress $progress
    update

    $this parseDICOMHeader $f header

    foreach key "patient study series" {
      set tag [set [string toupper $key]]
      if { ![info exists header($tag,value)] } {
        set $key "Unknown"
      } else {
        set $key $header($tag,value)
      }
    }

    if { [lsearch $tree(patients) $patient] == -1 } {
      lappend tree(patients) $patient
      set tree($patient,studies) ""
    }
    if { [lsearch $tree($patient,studies) $study] == -1 } {
      lappend tree($patient,studies) $study
      set tree($patient,$study,series) ""
    }
    if { [lsearch $tree($patient,$study,series) $series] == -1 } {
      lappend tree($patient,$study,series) $series
    }
    lappend tree($patient,$study,$series,files) $f
  }

  $progressDialog SetParent ""
  $progressDialog SetMasterWindow ""
  $progressDialog Delete
}
