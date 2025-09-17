#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#.\setup_training_env.ps1

# PowerShell 스크립트: setup_training_env.ps1

# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
. .\venv\Scripts\Activate.ps1

# 3. pip 최신화
python -m pip install --upgrade pip

# 4. PyTorch 2.7.1 + cu126 설치 (공식 인덱스)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5. xformers 설치
pip install xformers

# 6. transformers, accelerate, safetensors 설치
pip install transformers accelerate safetensors

# 7. 설치 확인
python -c "import torch, torchvision, torchaudio, xformers, transformers, accelerate, safetensors; print('✅ 모든 패키지 import 성공')"

# 8. LoRA + ControlNet 관련 설치
pip install diffusers peft

# 9. LoRA + ControlNet 관련 설치
pip install opencv-python omegaconf einops