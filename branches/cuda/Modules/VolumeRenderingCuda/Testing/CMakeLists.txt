
ADD_EXECUTABLE(VolumeRenderingCudaTest
  main.cxx
  ntkTimer.cxx
)

TARGET_LINK_LIBRARIES(VolumeRenderingCudaTest
  VolumeRenderingCudaModule_CudaLibrary
)

ADD_TEST(VolumeRenderingCudaTest
  ${EXECUTABLE_OUTPUT_PATH}/VolumeRenderingCudaTest ${CMAKE_CURRENT_SOURCE_DIR}/heart256.raw)
