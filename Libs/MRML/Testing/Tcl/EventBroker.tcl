

#
# utilities for testing event broker
#

proc EventBrokerTmpDir {} {
  return $::env(SLICER_HOME)/Testing/Temporary
}

proc EventBrokerPrint {} {
  set broker [vtkEventBroker New]
  puts [$broker Print]
  $broker Delete
}

proc EventBrokerGraph { {fileName broker.dot} } {
  set broker [vtkEventBroker New]
  $broker GenerateGraphFile [EventBrokerTmpDir]/$fileName
  $broker Delete
}

proc EventBrokerUpdate {} {
  set broker [vtkEventBroker New]
  $broker ProcessEventQueue
  $broker Delete
}

proc EventBrokerAsync { {fileName broker.log} } {

  set broker [vtkEventBroker New]

  $broker SetLogFileName [EventBrokerTmpDir]/$fileName
  $broker EventLoggingOn

  $broker SetEventModeToAsynchronous

  $broker Delete
}

proc EventBrokerLogCommand { cmd {fileName brokercmd.log} } {

  set broker [vtkEventBroker New]

  $broker SetLogFileName [EventBrokerTmpDir]/$fileName
  $broker EventLoggingOn
  $broker OpenLogFile

  eval $cmd
  set timer [time "eval $cmd"]

  puts $timer

  $broker CloseLogFile
  $broker EventLoggingOff

  $broker Delete
}

proc EventBrokerLoadSampleScene { {sceneFileName ""} } {
  if { $sceneFileName == "" } {
    set sceneFileName $::env(SLICER_HOME)/../Slicer3/Libs/MRML/Testing/vol_and_cube_camera.mrml
  }

  $::slicer3::MRMLScene SetURL $sceneFileName
  $::slicer3::MRMLScene Connect
  update

}

proc EventBrokerTests {} {

  set broker [vtkEventBroker New]
  EventBrokerLogCommand "$::slicer3::MRMLScene Modified" scenemod.dot
  EventBrokerLogCommand "EventBrokerLoadSampleScene" sceneload.dot
  puts $broker
}
