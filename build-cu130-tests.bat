@echo off
rem Reconfigures the CUDA 13.0 build dir with tests ON and builds every KV parity test, so the KV evidence for the
rem PR comes from the release's own toolchain. Run the tests afterwards with: ctest (or each .exe) in build-cu130.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA130=C:\Users\blsor\cuda130\nvidia\cu13
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\blsor\Strata\.venv\Scripts\ninja.exe -S C:\Users\blsor\Strata -B C:\Users\blsor\Strata\build-cu130 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=%CUDA130%\bin\nvcc.EXE" "-DCUDAToolkit_ROOT=%CUDA130%" -DSTRATA_GGML_DIR=C:\Users\blsor\Strata\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\blsor\Strata\build-cu130 --target kv_q4_parity kv_q8_parity kv_stream_parity kv_hybrid_parity kv_window_parity -j 16
