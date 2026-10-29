# Regression tests for fork-only runtime options that must survive upstream merges.
# Run after the engine target is built:
#   cmake -DSTRATA_BIN=/absolute/path/to/strata -P tests/runtime_options_test.cmake
if(NOT DEFINED STRATA_BIN OR NOT EXISTS "${STRATA_BIN}")
  message(FATAL_ERROR "STRATA_BIN must name the built strata executable")
endif()

function(expect_accepted label)
  execute_process(
    COMMAND "${STRATA_BIN}" ${ARGN} --help
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    TIMEOUT 10)
  if(NOT "${rc}" STREQUAL "0")
    message(FATAL_ERROR "${label}: expected option acceptance/help exit 0, got ${rc}; stderr=${err}")
  endif()
  if("${err}" MATCHES "unknown argument")
    message(FATAL_ERROR "${label}: parser rejected option: ${err}")
  endif()
endfunction()

function(expect_rejected_value label expected)
  execute_process(
    COMMAND "${STRATA_BIN}" ${ARGN}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
    TIMEOUT 10)
  if("${rc}" STREQUAL "0" OR NOT "${err}" MATCHES "${expected}")
    message(FATAL_ERROR "${label}: expected validation diagnostic /${expected}/, got rc=${rc}; stderr=${err}")
  endif()
  if("${err}" MATCHES "unknown argument")
    message(FATAL_ERROR "${label}: parser rejected option instead of validating its value: ${err}")
  endif()
endfunction()

expect_accepted("shared KV pool option" --kv-pool-tokens 262144)
expect_rejected_value("negative shared KV pool size" "--kv-pool-tokens must be >= 0" --kv-pool-tokens -1)
expect_accepted("disk conversation-cache budget" --conversation-cache-disk-mib 402429)
expect_rejected_value("negative disk conversation-cache budget" "--conversation-cache-disk-mib needs a nonnegative integer" --conversation-cache-disk-mib -1)
expect_accepted("disk conversation-cache directory" --conversation-cache-dir /tmp/strata-cache-test)
expect_accepted("disk conversation-cache retention" --conversation-cache-days 7)
expect_rejected_value("negative disk conversation-cache retention" "--conversation-cache-days needs a nonnegative integer" --conversation-cache-days -1)
message(STATUS "fork runtime option regression tests passed")
