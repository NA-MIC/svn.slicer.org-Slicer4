PROJECT(qCTKWidgetsPlugin)

#
# See qCTKWidgets/CMake/qCTKDesignerPluginMacros.cmake for details
#
  
# Source files
SET(designer_plugin_SRCS
  qCTKWidgetsPlugin.cxx  
  qCTKWidgetsAbstractPlugin.cxx
  )

SET(designer_plugin_MOC_SRCS
  qCTKWidgetsPlugin.h
  )

SET(designer_plugin_TARGET_LIBRARIES
  qCTKWidgets
  )

qctk_build_designer_plugin( 
  NAME ${PROJECT_NAME}
  SRCS ${designer_plugin_SRCS}
  MOC_SRCS ${designer_plugin_MOC_SRCS}
  RESOURCES Resources/qCTKWidgetsPlugins.qrc
  TARGET_LIBRARIES ${designer_plugin_TARGET_LIBRARIES}
  )

# TODO Move to a Slicer CMakeLists.txt
qctk_install_designer_plugin(
  NAME ${PROJECT_NAME}
  INSTALL_BIN_DIR ${Slicer3_INSTALL_BIN_DIR}
  INSTALL_LIB_DIR ${Slicer3_INSTALL_LIB_DIR}/designer
  INSTALL_HEADERS_DIR ${Slicer3_INSTALL_INCLUDE_DIR}
  )
