@echo off
REM ==================================================================
REM  HairFastGAN environment setup  (school PC)
REM
REM  This PC wipes drive C on reboot, so CUDA / BuildTools / env vars
REM  must be restored each session. This file lives on D: so it stays.
REM
REM  Usage:   D:\hairmatch\setup_env.bat
REM  Run it in a NEW cmd window (vars apply to that window only).
REM ==================================================================

echo.
echo === HairFastGAN environment setup ===
echo.

REM ---------- 1. CUDA Toolkit ----------
REM  MSVC 14.4x STL requires CUDA 12.4 or newer (error STL1002),
REM  so newer 12.x toolkits are searched first. torch is built with
REM  cu121 but any 12.x works (same major -> warning only).
set "CUDA_HOME="
for %%D in (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    "D:\CUDA\v12.6"
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
) do (
    if not defined CUDA_HOME if exist "%%~D\bin\nvcc.exe" set "CUDA_HOME=%%~D"
)

if not defined CUDA_HOME (
    echo [FAIL] CUDA Toolkit not found. Install it, or add the path above.
) else (
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    echo [ OK ] CUDA_HOME = %CUDA_HOME%
)

REM ---------- 2. MSVC compiler ----------
set "VCVARS="
for %%V in (
    "D:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if not defined VCVARS if exist %%V set "VCVARS=%%V"
)

if not defined VCVARS (
    echo [FAIL] vcvars64.bat not found. Install VS Build Tools.
) else (
    echo [ OK ] MSVC : %VCVARS%
    call %VCVARS% >nul
)

REM ---------- 2b. nvcc / build flags ----------
REM  CUDA 12.1 rejects MSVC 14.4x as "unsupported Visual Studio version".
REM  This flag overrides that host-compiler version check.
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
REM  8.6 = RTX 3070 (sm_86). Building only this arch cuts compile time a lot.
set "TORCH_CUDA_ARCH_LIST=8.6"
echo [ OK ] NVCC_PREPEND_FLAGS / TORCH_CUDA_ARCH_LIST set

REM ---------- 3. torch JIT cache on D: ----------
REM  Default is C:\Users\<user>\AppData\Local\torch_extensions which is
REM  wiped on reboot. Keeping it on D: preserves the compiled kernels.
set "TORCH_EXTENSIONS_DIR=D:\torch_ext"
if not exist "D:\torch_ext" mkdir "D:\torch_ext"
echo [ OK ] TORCH_EXTENSIONS_DIR = D:\torch_ext

REM ---------- 4. venv + workdir ----------
if exist "D:\hairmatch\venv_hf\Scripts\activate.bat" (
    call "D:\hairmatch\venv_hf\Scripts\activate.bat"
    echo [ OK ] venv_hf activated
) else (
    echo [FAIL] venv_hf not found
)

cd /d D:\hairmatch\HairFastGAN

REM ---------- 5. verify ----------
echo.
echo === verify ===
where nvcc >nul 2>&1 && (where nvcc) || echo [FAIL] nvcc NOT on PATH
where cl   >nul 2>&1 && (where cl)   || echo [FAIL] cl.exe NOT on PATH
echo.
python -c "import torch;print('torch',torch.__version__);print('cuda ok:',torch.cuda.is_available());print('gpu:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo.
echo Ready. Test with:
echo   python test_hairfast.py --faces FACE.png --shapes REF.png --keep-color
echo.