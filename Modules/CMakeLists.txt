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
  RealTimeImaging
  MRAblation
  NeuroNav 
  Neurosurgery
)


IF (Slicer3_BUILD_TESTING)
    SUBDIRS( Testing )
ENDIF (Slicer3_BUILD_TESTING)
