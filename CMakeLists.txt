cmake_minimum_required (VERSION 3.1)
project (roll)

set (roll_VERSION_MAJOR 0)
set (roll_VERSION_MINOR 1)

set_target_properties(PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-int-in-bool-context")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(LibLBFGS REQUIRED)
find_package(Boost COMPONENTS serialization iostreams REQUIRED)

find_path(TENSORBOARD_LOGGER_INCLUDE_DIR
    NAME tensorboard_logger.h
    HINTS "${PROJECT_SOURCE_DIR}/deps/tensorboard_logger/include"
 )

if(NOT TENSORBOARD_LOGGER_INCLUDE_DIR)
  message(FATAL_ERROR "tensorboard_logger not found\n")
  return()
endif()

find_library(TENSORBOARD_LOGGER_LIBRARY
    NAME libtensorboard_logger.a
    HINTS "${PROJECT_SOURCE_DIR}/deps/tensorboard_logger"
)

if(NOT TENSORBOARD_LOGGER_LIBRARY)
  message(FATAL_ERROR "tensorboard_logger static library not found\n")
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
    NAME murmur3.h
    HINTS "${PROJECT_SOURCE_DIR}/deps/murmur3"
)

if(NOT MURMURHASH3_INCLUDE_DIR)
  message(FATAL_ERROR "murmur3.h not found\n")
  return()
endif()


configure_file (
  "${PROJECT_SOURCE_DIR}/include/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
)

add_executable (${PROJECT_NAME}
        ${MURMURHASH3_INCLUDE_DIR}/murmur3.c
        src/main.cc
        src/objective.cc
        src/stage1.cc
        src/stage2.cc
        src/stage3.cc
        src/aroc.cc
        src/nll.cc
        src/parse.cc
        )

target_include_directories(${PROJECT_NAME} PRIVATE ${CLIPP_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${MURMURHASH3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${SPDLOG_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${LIBLBFGS_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${TENSORBOARD_LOGGER_INCLUDE_DIR})

set (EXTRA_LIBS ${EXTRA_LIBS} pthread protobuf ${LIBLBFGS_LIBRARY} ${Boost_LIBRARIES} ${TENSORBOARD_LOGGER_LIBRARY})

target_link_libraries (${PROJECT_NAME} ${EXTRA_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "cpproll")

install (TARGETS roll DESTINATION bin)
install (FILES "${PROJECT_BINARY_DIR}/config.h"
         DESTINATION include)

