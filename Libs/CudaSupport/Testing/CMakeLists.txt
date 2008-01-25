PROJECT(CudaSupportTesting)

ADD_EXECUTABLE(MemoryCopyTest
        MemoryCopyTest.cxx
)

ADD_TEST(MemoryCopyTest
  MemoryCopyTest
)

TARGET_LINK_LIBRARIES(MemoryCopyTest
  CudaSupport
)
