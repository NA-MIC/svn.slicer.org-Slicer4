
ADD_EXECUTABLE(VolumeRenderingCudaExample
  VolumeRenderingCudaExample.cxx
)

TARGET_LINK_LIBRARIES(VolumeRenderingCudaExample
  VolumeRenderingCudaModule
)

#ADD_TEST(VolumeRenderingCudaTest
#  ${EXECUTABLE_OUTPUT_PATH}/VolumeRenderingCudaTest ${CMAKE_CURRENT_SOURCE_DIR}/heart256.raw)

INCLUDE_DIRECTORIES(
 ${VolumeRenderingModule_SOURCE_DIR}
 ${VolumeRenderingModule_BINARY_DIR}
  ${CudaSupport_SOURCE_DIR}
  ${CudaSupport_BINARY_DIR}
  )
