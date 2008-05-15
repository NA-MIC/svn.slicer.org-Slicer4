PROJECT( MimxMeshing )
 
SUBDIRS(
  IO
  Filter
  #CommandLine
  Common
  Adapter
  Mesh
  BuildingBlock
  Widgets
  MeshingWorkflow
)


#IF (Slicer3_BUILD_TESTING)
#    SUBDIRS( Testing )
#ENDIF (Slicer3_BUILD_TESTING)
