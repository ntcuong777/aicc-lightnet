# Install script for directory: /home/cuong/AICC-darknet/aicc-lightnet

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/cuong/AICC-darknet/aicc-lightnet")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cuong/AICC-darknet/aicc-lightnet" TYPE SHARED_LIBRARY FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/libdarknet.so")
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so"
         OLD_RPATH "/usr/local/cuda/lib64:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/libdarknet.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/darknet" TYPE FILE FILES
    "/home/cuong/AICC-darknet/aicc-lightnet/include/darknet.h"
    "/home/cuong/AICC-darknet/aicc-lightnet/include/yolo_v2_class.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cuong/AICC-darknet/aicc-lightnet/uselib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cuong/AICC-darknet/aicc-lightnet" TYPE EXECUTABLE FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/uselib")
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib"
         OLD_RPATH "/usr/local/cuda/lib64:/home/cuong/AICC-darknet/aicc-lightnet/build_release:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cuong/AICC-darknet/aicc-lightnet/darknet")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cuong/AICC-darknet/aicc-lightnet" TYPE EXECUTABLE FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/darknet")
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet"
         OLD_RPATH "/usr/local/cuda/lib64:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/darknet")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/cuong/AICC-darknet/aicc-lightnet/uselib_track")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/cuong/AICC-darknet/aicc-lightnet" TYPE EXECUTABLE FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/uselib_track")
  if(EXISTS "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track"
         OLD_RPATH "/usr/local/cuda/lib64:/home/cuong/AICC-darknet/aicc-lightnet/build_release:/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/cuong/AICC-darknet/aicc-lightnet/uselib_track")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/darknet/DarknetTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/darknet/DarknetTargets.cmake"
         "/home/cuong/AICC-darknet/aicc-lightnet/build_release/CMakeFiles/Export/share/darknet/DarknetTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/darknet/DarknetTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/darknet/DarknetTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet" TYPE FILE FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/CMakeFiles/Export/share/darknet/DarknetTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet" TYPE FILE FILES "/home/cuong/AICC-darknet/aicc-lightnet/build_release/CMakeFiles/Export/share/darknet/DarknetTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/darknet" TYPE FILE FILES
    "/home/cuong/AICC-darknet/aicc-lightnet/build_release/CMakeFiles/DarknetConfig.cmake"
    "/home/cuong/AICC-darknet/aicc-lightnet/build_release/DarknetConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/cuong/AICC-darknet/aicc-lightnet/build_release/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
