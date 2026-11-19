cmake_minimum_required(VERSION 3.16)
project(plane_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # CGAL is very slow in Debug
endif()

find_package(CGAL 5.5 REQUIRED)

# Eigen: used by CGAL for PCA / least-squares plane fitting.
# No version in the request: Eigen 5's config rejects any "3.x" request,
# so asking for none accepts both Eigen 3.4 and 5.x.
find_package(Eigen3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

# TBB (optional): parallel normal estimation
find_package(TBB QUIET)
include(CGAL_TBB_support)

add_executable(plane_detect plane_detect.cpp)
# Same effect as CGAL::Eigen3_support, without its version-checked lookup
target_link_libraries(plane_detect PRIVATE CGAL::CGAL Eigen3::Eigen)
target_compile_definitions(plane_detect PRIVATE CGAL_EIGEN3_ENABLED)
if(TARGET CGAL::TBB_support)
  target_link_libraries(plane_detect PRIVATE CGAL::TBB_support)
  message(STATUS "TBB found: parallel normal estimation enabled")
else()
  message(STATUS "TBB not found: running sequentially")
endif()