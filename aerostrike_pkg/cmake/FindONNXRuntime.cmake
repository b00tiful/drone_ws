set(_ONNXRUNTIME_HINTS)

if(DEFINED ONNXRUNTIME_ROOT)
  list(APPEND _ONNXRUNTIME_HINTS "${ONNXRUNTIME_ROOT}")
endif()

if(DEFINED ENV{ONNXRUNTIME_ROOT})
  list(APPEND _ONNXRUNTIME_HINTS "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)

find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ONNXRuntime
  REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY
)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
  set_target_properties(
    ONNXRuntime::ONNXRuntime
    PROPERTIES
      IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
