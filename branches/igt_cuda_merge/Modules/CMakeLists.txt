PROJECT( Modules )

SUBDIRS(
  NeuroNav
  RealTimeImaging
  WFEngineModule
  CommandLineModule
  ScriptedModule
  GradientAnisotropicDiffusionFilter
  EMSegment
  SlicerDaemon
  OpenIGTLinkDaemon
  Editor
  Python
  PythonGADScriptedModule
  Meshing
  QdecModule
  QueryAtlas
  Tractography
  VolumeRendering
  VolumeRenderingCuda
  Statistics
  RegistrationCuda
  OpenIGTLink
)


IF (Slicer3_BUILD_TESTING)
    SUBDIRS( Testing )
ENDIF (Slicer3_BUILD_TESTING)

IF ( USE_PYTHON AND PYTHON_INCLUDE_PATH )
  INCLUDE_DIRECTORIES (${PYTHON_INCLUDE_PATH})
ENDIF ( USE_PYTHON AND PYTHON_INCLUDE_PATH )
