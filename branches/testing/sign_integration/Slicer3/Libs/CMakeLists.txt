PROJECT( SlicerLibs )

# Note: Order of of SUBDIRS is important if your projects depend
# on code in other projects (put dependent project later).
IF( TEEM_FOUND )
  SUBDIRS(
      vtkTeem
  )
ENDIF( TEEM_FOUND )

FIND_FILE(SIGN_DIR SIGN
${CMAKE_CURRENT_SOURCE_DIR}
 )

IF(SIGN_DIR)
SUBDIRS(
SIGN
)
ENDIF(SIGN_DIR)


SUBDIRS(
      vtkITK
      FreeSurfer
      MRML
      tclap
      ModuleDescriptionParser
      GenerateCLP
      SlicerIO
      ITKCommandIO
      MRMLIDImageIO
      MGHImageIO
      IGT
)
