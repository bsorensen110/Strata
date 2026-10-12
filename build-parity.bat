@echo off
rem Reconfigures the existing build dir with the tests on and builds+runs kv_window_parity (the FP16 sliding
rem window + attention sinks, include/strata/kernels/kv_window.hpp).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\blsor\Strata\.venv\Scripts\ninja.exe -S C:\Users\blsor\Strata -B C:\Users\blsor\Strata\build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.EXE" -DSTRATA_GGML_DIR=C:\Users\blsor\Strata\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\blsor\Strata\build --target kv_window_parity -j 16 || exit /b 1
"C:\Users\blsor\Strata\build\kv_window_parity.exe"
