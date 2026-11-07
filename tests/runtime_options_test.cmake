# CLI validation test for --conversation-cache-disk-mib / --conversation-cache-dir / --conversation-cache-days
if(NOT DEFINED STRATA_BIN OR NOT EXISTS "${STRATA_BIN}")
  message(FATAL_ERROR "STRATA_BIN not set or does not exist: ${STRATA_BIN}")
endif()

function(expect_cli_Behavior label expect_substr)
  execute_process(
    COMMAND "${STRATA_BIN}" ${ARGN}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
  )
  set(combined "${out}\n${err}")
  if(rc EQUAL 0)
    message(FATAL_ERROR "${label}: expected non-zero exit code, got 0. Output:\n${combined}")
  endif()
  string(FIND "${combined}" "${expect_substr}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "${label}: expected '${expect_substr}' in output, got:\n${combined}")
  endif()
endfunction()

# 1. Both --conversation-cache-mib and --conversation-cache-disk-mib together are accepted (two-tier L1 RAM + L2 Disk)
#    (Without --tokens it fails on missing tokens, not on mutually exclusive cache flags.)
expect_cli_Behavior("two-tier L1+L2 accepted" "--tokens is required"
  --conversation-cache-mib 1024 --conversation-cache-disk-mib 4096)

# 2. Negative --conversation-cache-disk-mib rejected
expect_cli_Behavior("negative disk-mib" "--conversation-cache-disk-mib needs a nonnegative integer within range"
  --conversation-cache-disk-mib -1)

# 3. Negative --conversation-cache-days rejected
expect_cli_Behavior("negative cache-days" "--conversation-cache-days needs a nonnegative integer within range"
  --conversation-cache-days -1)

# 4. Missing argument to --conversation-cache-dir rejected
expect_cli_Behavior("missing cache-dir arg" "--conversation-cache-dir needs a value"
  --conversation-cache-dir)

message(STATUS "runtime_options_test: all checks passed")
