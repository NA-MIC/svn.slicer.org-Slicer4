PROJECT( MimxMeshing )
 
SUBDIRS(
  #IO
  Common
  Filter
  Widgets
  
  ## Methods:
  #CommandLine
  BoundingBox
  MeshingWorkflow
)


#IF (Slicer3_BUILD_TESTING)
#    SUBDIRS( Testing )
#ENDIF (Slicer3_BUILD_TESTING)
