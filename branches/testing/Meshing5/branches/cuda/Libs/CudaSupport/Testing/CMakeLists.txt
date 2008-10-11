PROJECT(CudaSupportTesting)

ADD_EXECUTABLE(MemoryCopyTest
        MemoryCopyTest.cxx
)

ADD_TEST(MemoryCopyTest
  ${EXECUTABLE_OUTPUT_PATH}/MemoryCopyTest
)

TARGET_LINK_LIBRARIES(MemoryCopyTest
  CudaSupport
)
