project(libCLAHE CXX)

if (NOT TARGET libCLAHE)
    FIND_PACKAGE(OpenCV REQUIRED)
    add_library(libCLAHE clahe.cpp)
    target_link_libraries(libCLAHE ${OpenCV_LIBS})

endif()
