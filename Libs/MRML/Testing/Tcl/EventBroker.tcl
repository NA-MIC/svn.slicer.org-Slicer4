
proc EventBrokerPrint {} {
  set broker [vtkEventBroker New]
  puts [$broker Print]
  $broker Delete
}

proc EventBrokerGraph { {fileName c:/tmp/broker.dot} } {
  set broker [vtkEventBroker New]
  $broker GenerateGraphFile $fileName
  $broker Delete
}

proc EventBrokerUpdate {} {
  set broker [vtkEventBroker New]
  $broker ProcessEventQueue
  $broker Delete
}

proc EventBrokerAsync { } {

  set broker [vtkEventBroker New]

  $broker SetLogFileName c:/tmp/broker.log
  $broker EventLoggingOn

  $broker SetEventModeToAsynchronous

  $broker Delete
}

proc EventBrokerLogCommand { cmd } {

  set broker [vtkEventBroker New]

  $broker SetLogFileName c:/tmp/brokercmd.log
  $broker EventLoggingOn
  $broker OpenLogFile

  eval $cmd

  $broker CloseLogFile

  $broker Delete
}
