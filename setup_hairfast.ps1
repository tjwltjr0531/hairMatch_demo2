# HairFastGAN local setup (Windows + NVIDIA GPU)
#
# Prerequisites:
#   1. NVIDIA GPU + driver           -> check with nvidia-smi
#   2. Visual Studio Build Tools     -> required to compile StyleGAN2 CUDA kernels
#      https://visualstudio.microsoft.com/visual-cpp-build-tools/
#      Check the "Desktop development with C++" workload
#   3. Python 3.11, Git, Git LFS
#
# Run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_hairfast.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== 0. Preflight ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ($LASTEXITCODE -ne 0) {
    Write-Host "nvidia-smi failed - no NVIDIA GPU or driver." -ForegroundColor Red
    exit 1
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vs) { Write-Host "MSVC found: $vs" -ForegroundColor Green }
    else { Write-Host "WARNING: MSVC C++ tools missing - install Build Tools" -ForegroundColor Yellow }
} else {
    Write-Host "WARNING: VS Installer not found - install Build Tools" -ForegroundColor Yellow
}

Write-Host "`n=== 1. Virtual environment ===" -ForegroundColor Cyan
if (-not (Test-Path "venv_hf")) { py -3.11 -m venv venv_hf }
& .\venv_hf\Scripts\Activate.ps1
python -c "import sys; print('Python', sys.version.split()[0])"

Write-Host "`n=== 2. PyTorch (CUDA 12.1) ===" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

Write-Host "`n=== 3. Dependencies ===" -ForegroundColor Cyan
pip install "numpy<2" ninja addict dill face_alignment fpie Pillow opencv-python scikit-image gdown
pip install git+https://github.com/openai/CLIP.git

Write-Host "`n=== 4. Source ===" -ForegroundColor Cyan
if (-not (Test-Path "HairFastGAN")) { git clone https://github.com/AIRI-Institute/HairFastGAN.git }
Set-Location HairFastGAN

Write-Host "`n=== 5. Pretrained weights ===" -ForegroundColor Cyan
git lfs version
if ($LASTEXITCODE -ne 0) {
    Write-Host "git-lfs missing: winget install GitHub.GitLFS" -ForegroundColor Red
    exit 1
}
git lfs install

if (-not (Test-Path "pretrained_models")) {
    if (Test-Path "HairFastGAN") { Remove-Item -Recurse -Force HairFastGAN }
    git clone https://huggingface.co/AIRI-Institute/HairFastGAN
    Push-Location HairFastGAN; git lfs pull; Pop-Location
    Move-Item HairFastGAN\pretrained_models pretrained_models
    Move-Item HairFastGAN\input input
    Remove-Item -Recurse -Force HairFastGAN
}

foreach ($d in @('StyleGAN','PostProcess','Blending','Rotate','Alignment','ArcFace',
                 'BiSeNet','encoder4editing','FeatureStyleEncoder','CtrlHair',
                 'SEAN','STAR','ShapeAdaptor')) {
    New-Item -ItemType Directory -Force -Path "pretrained_models\$d" | Out-Null
}

Write-Host "`nLargest weight files (KB-sized means git-lfs failed):" -ForegroundColor Cyan
Get-ChildItem pretrained_models -Recurse -File |
    Sort-Object Length -Descending | Select-Object -First 8 |
    ForEach-Object { "{0,10:N1} MB  {1}" -f ($_.Length/1MB), $_.Name }

Write-Host "`n=== 6. Sanity run ===" -ForegroundColor Cyan
Write-Host "First run compiles CUDA kernels - takes several minutes. Do not interrupt." -ForegroundColor Yellow
python main.py --face_path=6.png --shape_path=7.png --color_path=6.png --input_dir=input --result_path=output/sanity.png

if (Test-Path "output\sanity.png") {
    Write-Host "`nSUCCESS: output\sanity.png" -ForegroundColor Green
    Write-Host "Next: copy ..\test_hairfast.py . ; python test_hairfast.py --keep-color" -ForegroundColor Green
} else {
    Write-Host "`nFAILED. Check the error above." -ForegroundColor Red
    Write-Host "  cl.exe error     -> install MSVC Build Tools" -ForegroundColor Yellow
    Write-Host "  ninja error      -> pip install ninja" -ForegroundColor Yellow
    Write-Host "  CUDA mismatch    -> install CUDA Toolkit 12.1" -ForegroundColor Yellow
    Write-Host "  still stuck      -> use WSL2 instead" -ForegroundColor Yellow
}
