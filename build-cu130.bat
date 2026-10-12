@echo off
rem Builds the engine with the CUDA 13.0 toolkit (the release's toolchain): pip's cuda-toolkit components, laid
rem out under C:\Users\blsor\cuda130\nvidia\cu13. CUDA 13.2 produces a numerically broken engine for this model
rem on this box (a 5K recall test scores 0/8 where the release scores 8/8), so quality runs need this one.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA130=C:\Users\blsor\cuda130\nvidia\cu13
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\blsor\Strata\.venv\Scripts\ninja.exe -S C:\Users\blsor\Strata -B C:\Users\blsor\Strata\build-cu130 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=%CUDA130%\bin\nvcc.EXE" "-DCUDAToolkit_ROOT=%CUDA130%" -DSTRATA_GGML_DIR=C:\Users\blsor\Strata\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\blsor\Strata\build-cu130 --target strata -j 16
