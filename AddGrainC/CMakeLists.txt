set(PluginName "AddGrainC")

if (NOT WIN32)
  string(TOLOWER "${PluginName}" PluginName)
endif()

set(ProjectName "${PluginName}")
project(${ProjectName} LANGUAGES CXX)

file(GLOB AddGrainCSources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" *.cpp *.h)
add_library(${PluginName} SHARED ${AddGrainCSources})

#require sse2, some other plugins may need to set sse4.1 for quick msvc->gcc porting
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINTEL_INTRINSICS -msse2")

if (MSVC_IDE)
  IF(CLANG_IN_VS STREQUAL "1")
      # special AVX option for source files with *_avx.cpp pattern
      file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
      set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS " -mavx ")

      # special AVX2 option for source files with *_avx2.cpp pattern
      file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
      set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS " -mavx2 -mfma ")

      # special AVX512 option for source files with *_avx512.cpp pattern
      file(GLOB_RECURSE SRCS_AVX512 "*_avx512.cpp")
      set_source_files_properties(${SRCS_AVX512} PROPERTIES COMPILE_FLAGS " -mavx512f -mavx512bw ")
  ELSE()
      # special AVX option for source files with *_avx.cpp pattern
      file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
      set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS " /arch:AVX ")

      # special AVX2 option for source files with *_avx2.cpp pattern
      file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
      set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS " /arch:AVX2 ")

      # special AVX512 option for source files with *_avx512.cpp pattern
      file(GLOB_RECURSE SRCS_AVX512 "*_avx512.cpp")
      set_source_files_properties(${SRCS_AVX512} PROPERTIES COMPILE_FLAGS " /arch:AVX512 ")
  ENDIF()
else()
  # special AVX option for source files with *_avx.cpp pattern
  file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
  set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS " -mavx ")

  # special AVX2 option for source files with *_avx2.cpp pattern
  file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
  set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS " -mavx2 -mfma ")

  # special AVX512 option for source files with *_avx512.cpp pattern
  file(GLOB_RECURSE SRCS_AVX512 "*_avx512.cpp")
  set_source_files_properties(${SRCS_AVX512} PROPERTIES COMPILE_FLAGS " -mavx512f -mavx512bw ")
endif()

target_link_libraries(${ProjectName})
target_include_directories(${ProjectName} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

include(GNUInstallDirs)

INSTALL(TARGETS ${ProjectName}
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")
