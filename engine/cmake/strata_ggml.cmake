# cmake/strata_ggml.cmake - where the engine's ggml comes from: the tree vendored in third_party/strata-ggml (default, no download), or -DSTRATA_GGML_DIR=<llama.cpp>.
#
# Every block below is a macro: it runs in the scope of the caller (CMakeLists.txt calls it where the block used to be), so variables, targets and
# their order are those of the single-file CMakeLists.txt it was cut from (cleanup 5.4; the build graph is identical).

# which ggml tree the engine builds from: the vendored third_party/strata-ggml, or the -DSTRATA_GGML_DIR override (repo_root: the checkout)
macro(strata_ggml_resolve repo_root)
if(DEFINED CACHE{GGML_CCACHE})
  option(GGML_CCACHE "ggml: use ccache if available" ON)
endif()
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from instead of the vendored third_party/strata-ggml (empty: the vendored tree)")
if(STRATA_GGML_DIR)
  set(_strata_ggml_root "${STRATA_GGML_DIR}")
else()
  get_filename_component(_strata_ggml_root "${repo_root}/third_party/strata-ggml" ABSOLUTE)   # (normalised: sycl/ and metal/ pass <repo>/sycl/..)
endif()
endmacro()

# the root project's ggml-base / ggml-cpu: the forced GGML_* options per build kind (portable, ISA floor, native) and the subproject
macro(strata_ggml_configure_root)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE AND NOT STRATA_CPU_X86)
    # arm64 (DGX Spark: Cortex-X925 + A725): ggml-cpu for any Armv8.2 CPU with dot product, FP16 and int8 matmul instead of the build
    # host's, so the engine built here does not depend on which core the compiler ran on.  Without it GGML_NATIVE OFF would leave
    # ggml-cpu at plain armv8-a, and a native pack's CPU experts would lose the NEON dot products.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(STRATA_ARM64_ARCH "armv8.2-a+dotprod+fp16+i8mm" CACHE STRING "arm64: ggml-cpu's -march for a STRATA_PORTABLE build")
    set(GGML_CPU_ARM_ARCH "${STRATA_ARM64_ARCH}" CACHE STRING "" FORCE)
    if(NOT MSVC)
      set_source_files_properties(src/kernels/cpu/portable.cpp PROPERTIES COMPILE_OPTIONS "-march=${STRATA_ARM64_ARCH}")
    endif()
  elseif(NOT STRATA_CPU_X86 AND NOT STRATA_ARM64_ISA STREQUAL "")
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    if(STRATA_ARM64_ISA STREQUAL "apple-m1")
      set(_arm_arch "armv8.4-a+dotprod+fp16")
    elseif(STRATA_ARM64_ISA MATCHES "^apple-m[234]$")
      set(_arm_arch "armv8.6-a+dotprod+fp16+i8mm+bf16")
    elseif(STRATA_ARM64_ISA STREQUAL "oryon-x1")
      set(_arm_arch "armv8.7-a+dotprod+fp16+i8mm+bf16")
    elseif(STRATA_ARM64_ISA STREQUAL "oryon-x2")
      set(_arm_arch "armv9.2-a+dotprod+fp16+i8mm+bf16")
    elseif(STRATA_ARM64_ISA STREQUAL "grace")
      set(_arm_arch "armv9-a+dotprod+fp16+i8mm+bf16+sve2")
    elseif(STRATA_ARM64_ISA STREQUAL "gb10")
      set(_arm_arch "armv9.2-a+dotprod+fp16+i8mm+bf16+sve2")
    else()
      message(FATAL_ERROR "Unknown STRATA_ARM64_ISA='${STRATA_ARM64_ISA}'")
    endif()
    set(GGML_CPU_ARM_ARCH "${_arm_arch}" CACHE STRING "" FORCE)
    if(NOT MSVC)
      set_source_files_properties(src/kernels/cpu/portable.cpp PROPERTIES COMPILE_OPTIONS "-march=${_arm_arch}")
    endif()
    message(STATUS "strata: Arm64 CPU kernels built for ${STRATA_ARM64_ISA} (-march=${_arm_arch})")
  elseif(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_ISA_FLOOR)
    # the experimental older-CPU build (STRATA_ISA_FLOOR above): one ggml-cpu for the floor, ggml's "sandybridge"
    # feature set (SSE4.2 + AVX) or SSE4.2 alone - not the build host's, which may be newer than the PC it runs on.
    # From the Strata_Dirigo fork (rwkeyes).  ggml's defaults turn the rest on when GGML_NATIVE is off: all off here.
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)
    if(STRATA_ISA_FLOOR STREQUAL "avx")
      set(GGML_AVX ON CACHE BOOL "" FORCE)
    else()
      set(GGML_AVX OFF CACHE BOOL "" FORCE)
    endif()
    foreach(_f GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2 GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
               GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
  elseif(STRATA_X86_ISA STREQUAL "zen5")
    # ggml-cpu for the Zen 5 ISA: MSVC's native probe cannot see VBMI/VNNI/BF16/AVX-VNNI (FindSIMD tests
    # /arch:AVX, /arch:AVX2, /arch:AVX512 only), so the extensions are named here and ggml's MSVC branch turns
    # them into the macros the kernels are #ifdef'd on (__AVX512VBMI__, __AVX512VNNI__, __AVX512BF16__,
    # __AVXVNNI__, __BMI2__) plus /arch:AVX512.  Native on: ggml-cpu's codegen targets this host; the binary
    # needs AVX-512 hardware (Zen 4+ / SPR+).  ggml_openmp stays off (the pool owns the threads).
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_SSE42 GGML_F16C GGML_FMA GGML_BMI2 GGML_AVX GGML_AVX2 GGML_AVX_VNNI
               GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
    foreach(_f GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
      set(${_f} OFF CACHE BOOL "" FORCE)
    endforeach()
    message(STATUS "strata: ggml-cpu built for the zen5 target (AVX-512 F/BW/VL/VBMI/VNNI/BF16 + AVX-VNNI + BMI2); requires AVX-512 hardware")
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  # ggml-cpu's AVX2 IQ1_S / IQ1_M dot products (ggml-cpu/arch/x86/quants.c) build their grid indices with BMI2 `pdep`
  # where __BMI2__ is defined.  AMD Zen 1 / Zen+ / Zen 2 (family 17h, and Hygon's 18h) run pdep in microcode (on the
  # order of a hundred cycles or more, against a few on Intel and Zen 3+), and there those two dots are 10-19x slower than
  # their #else branch, which builds the same indices with shifts and masks: the same integer sums, bit for bit.  They
  # are ggml's only __BMI2__ code (besides ggml_cpu_has_bmi2(), a report).  STRATA_GGML_PDEP=AUTO (the default)
  # compiles ggml-cpu without __BMI2__ in a native build whose build machine is such a CPU; OFF does it in every build,
  # ON in none.  The portable (release) build keeps pdep under AUTO: it also runs on the CPUs where pdep is fast.
  # __BMI2__ comes from GGML_BMI2 (MSVC, and -mbmi2 in a non-native build) or from -march=native (GCC / Clang).
  set(STRATA_GGML_PDEP "AUTO" CACHE STRING "ggml-cpu's IQ1_S/IQ1_M dots with BMI2 pdep: AUTO (not on a Zen 1/2 build machine), ON, OFF")
  set_property(CACHE STRATA_GGML_PDEP PROPERTY STRINGS AUTO ON OFF)
  set(_strata_no_pdep OFF)
  if(STRATA_GGML_PDEP STREQUAL "OFF")
    set(_strata_no_pdep ON)
  elseif(STRATA_GGML_PDEP STREQUAL "AUTO" AND NOT STRATA_PORTABLE AND NOT STRATA_ISA_FLOOR)
    set(_strata_cpu_vendor "")
    set(_strata_cpu_family "")
    if(EXISTS "/proc/cpuinfo")
      file(STRINGS "/proc/cpuinfo" _strata_cpuinfo LIMIT_COUNT 32)
      string(REGEX MATCH "vendor_id[ \t]*: ([A-Za-z]+)" _m "${_strata_cpuinfo}")
      set(_strata_cpu_vendor "${CMAKE_MATCH_1}")
      string(REGEX MATCH "cpu family[ \t]*: ([0-9]+)" _m "${_strata_cpuinfo}")
      set(_strata_cpu_family "${CMAKE_MATCH_1}")
    elseif(WIN32 AND DEFINED ENV{PROCESSOR_IDENTIFIER})   # "AMD64 Family 23 Model 49 Stepping 0, AuthenticAMD"
      string(REGEX MATCH "Family ([0-9]+) .*, ([A-Za-z]+)$" _m "$ENV{PROCESSOR_IDENTIFIER}")
      set(_strata_cpu_family "${CMAKE_MATCH_1}")
      set(_strata_cpu_vendor "${CMAKE_MATCH_2}")
    endif()
    if((_strata_cpu_vendor STREQUAL "AuthenticAMD" AND _strata_cpu_family STREQUAL "23") OR
       (_strata_cpu_vendor STREQUAL "HygonGenuine" AND _strata_cpu_family STREQUAL "24"))
      set(_strata_no_pdep ON)
    endif()
  endif()
  if(_strata_no_pdep)
    set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
    message(STATUS "Strata: ggml-cpu without BMI2 pdep in its IQ1_S/IQ1_M dots (STRATA_GGML_PDEP=${STRATA_GGML_PDEP})")
  endif()
  add_subdirectory(${_strata_ggml_root}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  if(_strata_no_pdep AND NOT MSVC)
    target_compile_options(ggml-cpu PRIVATE -U__BMI2__)   # -march=native defines it whatever GGML_BMI2 says
  endif()
endmacro()

