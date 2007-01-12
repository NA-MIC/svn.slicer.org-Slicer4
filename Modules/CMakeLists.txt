PROJECT( Modules )

SUBDIRS(
  CommandLineModule
  ScriptedModule
  GradientAnisotropicDiffusionFilter
  EMSegment
  SlicerDaemon
  Editor
  QueryAtlas
  Tractography
  #Neurosurgery
  RealTimeImaging
  MRAblation
  #Opentracker
)


IF (Slicer3_BUILD_TESTING)
    SUBDIRS( Testing )
ENDIF (Slicer3_BUILD_TESTING)
