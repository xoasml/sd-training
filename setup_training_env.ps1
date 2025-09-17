# setup_training_env.ps1
# LoRA 학습용 환경 세팅 (RTX 5080, CUDA 12.8 nightly)

# 1. 기존 venv 제거
if (Test-Path ".\venv") {
    Write-Host "Removing existing venv..."
    Remove-Item -Recurse -Force ".\venv"
}

# 2. 새 venv 생성
Write-Host "Creating new venv..."
python -m venv venv

# 3. venv 활성화
Write-Host "Activating venv..."
& .\venv\Scripts\Activate.ps1

# 4. pip 최신화
Write-Host "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# 5. torch / torchvision (로컬 GPU whl, 반드시 --no-deps)
Write-Host "Installing torch & torchvision (local GPU builds)..."
pip install --no-deps ./local_pakage/torch-2.9.0.dev20250714+cu128-cp39-cp39-win_amd64.whl
pip install --no-deps ./local_pakage/torchvision-0.24.0.dev20250714+cu128-cp39-cp39-win_amd64.whl

# 6. 필수 의존성 설치
Write-Host "Installing extra dependencies..."
pip install filelock fsspec jinja2 networkx sympy pillow numpy

# 7. xformers 설치 (GPU 빌드, no-deps)
Write-Host "Installing xformers (GPU build)..."
pip install --no-deps xformers==0.0.32.post2

# 8. HuggingFace 관련 필수 패키지
Write-Host "Installing HuggingFace ecosystem..."
pip install diffusers==0.30.2 transformers==4.44.2 peft==0.13.0 safetensors accelerate datasets scipy ftfy

# 9. 설치 확인
Write-Host "`n✅ Installation complete. Checking versions..."
python -c "import torch, diffusers, transformers, peft, xformers, safetensors, accelerate, datasets, scipy, ftfy; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); \
print('diffusers', diffusers.__version__); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__); \
print('xformers', xformers.__version__); \
print('safetensors', safetensors.__version__); \
print('accelerate', accelerate.__version__); \
print('datasets', datasets.__version__); \
import scipy; print('scipy', scipy.__version__); \
import ftfy; print('ftfy', ftfy.__version__)"
