cmake_minimum_required (VERSION 2.6)
project (roll)

set (roll_VERSION_MAJOR 0)
set (roll_VERSION_MINOR 1)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-int-in-bool-context")

option(HELLOWORLD "Salute toggle" ON)

find_package(Eigen3)

FIND_PACKAGE(Boost 1.40 COMPONENTS serialization iostreams REQUIRED)

find_path(LBFGS_INCLUDE_DIR
    NAME lbfgs.h
    HINTS "${PROJECT_SOURCE_DIR}/deps/liblbfgs/include"
 )

if(NOT LBFGS_INCLUDE_DIR)
  message(FATAL_ERROR "lbfgs not found\n")
  return()
endif()

find_library(LBFGS_LIBRARY
    NAME liblbfgs.so
    HINTS "${PROJECT_SOURCE_DIR}/deps/liblbfgs/lib/.libs"
)

if(NOT LBFGS_LIBRARY)
  message(FATAL_ERROR "lbfgs static library not found\n")
  return()
endif()

find_path(SPDLOG_INCLUDE_DIR
    NAME spdlog
    HINTS "${PROJECT_SOURCE_DIR}/deps/spdlog/include"
)

if(NOT SPDLOG_INCLUDE_DIR)
  message(FATAL_ERROR "spdlog not found\n")
  return()
endif()

find_path(CLIPP_INCLUDE_DIR
    NAME clipp.h
    HINTS "${PROJECT_SOURCE_DIR}/deps/clipp/include"
)

if(NOT CLIPP_INCLUDE_DIR)
  message(FATAL_ERROR "clipp.h not found\n")
  return()
endif()

find_path(MURMURHASH3_INCLUDE_DIR
    NAME murmurhash3.hpp
    HINTS "${PROJECT_SOURCE_DIR}/deps/boost-bloom-filters/boost/bloom_filter/hash"
)

if(NOT MURMURHASH3_INCLUDE_DIR)
  message(FATAL_ERROR "murmurhash3.hpp not found\n")
  return()
endif()


configure_file (
  "${PROJECT_SOURCE_DIR}/include/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

include_directories("${CLIPP_INCLUDE_DIR}")
include_directories("${MURMURHASH3_INCLUDE_DIR}")
include_directories("${SPDLOG_INCLUDE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
INCLUDE_DIRECTORIES("${LBFGS_INCLUDE_DIR}")

set (EXTRA_LIBS ${EXTRA_LIBS} pthread ${LBFGS_LIBRARY} ${Boost_LIBRARIES})

add_executable (roll
    src/main.cc
    src/objective.cc
    src/stage1.cc
    src/stage2.cc
    src/stage3.cc
    src/roc.cc
    src/parse.cc
)

target_link_libraries (roll ${EXTRA_LIBS})

install (TARGETS roll DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/config.h"
         DESTINATION include)

