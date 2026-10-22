@echo off
rem Quality build on CUDA 13.0 (pip components at C:\Users\blsor\cuda130\nvidia\cu13).
rem The system CUDA 13.2 toolkit builds numerically broken kernels for IQ3_XXS - do not use it here.
rem STRATA_Q6K_EXPERTS=ON is required: an old cache pinned it OFF.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\blsor\Strata\.venv\Scripts\ninja.exe -S C:\Users\blsor\Strata -B C:\Users\blsor\Strata\build-cu130 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=C:\Users\blsor\cuda130\nvidia\cu13\bin\nvcc.EXE" -DSTRATA_GGML_DIR=C:\Users\blsor\Strata\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\blsor\Strata\build-cu130 --target strata -j 8 && exit /b 0
echo   (the build stopped - trying it once more)
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\blsor\Strata\build-cu130 --target strata -j 8 || exit /b 1
