
https://foxydog.tistory.com/219 - webUI 참고

# 구성 순서

- 환경 세팅 → Python + PyTorch + 필수 패키지 설치

- 모델 다운로드 → Stable Diffusion 1.5

- 데이터셋 준비 → 이미지 + 캡션

- LoRA 학습 → 엘리베이터 전용 가중치 생성

- ControlNet 다운로드 → canny/depth/openpose 등 선택

- 테스트 → WebUI에서 LoRA + ControlNet 함께 적용


# 환경 세팅 → Python + PyTorch + 필수 패키지 설치

✅ Python 3.9.x 필수

✅ Visual Studio 설치 필요

✅ C++ 개발 도구

✅ "Desktop development with C++" 선택

✅ MSVC v143 - VS 2022 C++ x64/x86 빌드 도구

✅ Windows 10 SDK (10.0.x 이상)


https://download.pytorch.org/whl/nightly/cu128/ 접속
torch : torch-2.9.0.dev20250714+cu128-cp39-cp39-win_amd64.whl
torchvision : torchvision-0.24.0.dev20250714+cu128-cp39-cp39-win_amd64.whl


``` powershell
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


#pip install --no-deps C:\Users\xoasm\Documents\sd-training\local_pakage/torch-2.10.0.dev20250910+cu128-cp310-cp310-win_amd64.whl
#pip install --no-deps C:\Users\xoasm\Documents\sd-training\local_pakage/torchvision-0.24.0.dev20250721+cu128-cp310-cp310-win_amd64.whl

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
```

# 모델 다운로드 → Stable Diffusion 1.5
 
 ### 1. Hugging Face 계정 생성
 1. https://huggingface.co/ 접속
 2. 오른쪽 상단 Sign up 클릭 → 계정 생성
    (이메일, GitHub, Google 로그인 가능)
 
 ### 2. Access Token 발급

1. 로그인 후 오른쪽 상단 프로필 아이콘 → Settings 클릭
2. 왼쪽 메뉴에서 Access Tokens 선택
3. New Token 클릭
4. Name: 예) sd15-download
5. Role: Read 선택
6. 발급된 토큰 문자열 복사 (예: hf_xxxxx...)

### 3. Git LFS 설치 확인
```powershell
git lfs install
```
(만약 git이 없으면 Git for Windows 설치)

### 4. Hugging Face CLI 로그인
```powershell
huggingface-cli login
```
→ 방금 발급한 토큰(hf_xxxxx...) 붙여넣기 → 엔터

### 5. Stable Diffusion 1.5 모델 다운로드
```powershell
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```
용량이 크기 때문에 콘솔의 내용 변화 없이 오래 작업함 88GB임.
→ 멈춘건지 작업 중인건지 의심 된다면 다운로드 받고 있는 폴더의 용량이 올라가는지 확인


# 데이터셋 준비 → 이미지 + 캡션
- pinterest에서 구했음
- jpg 이미지 (20~100장, 테스트면 30장 정도도 OK)
- txt 캡션 (영어 문장, 이미지와 동일한 파일명), elevx 트리거 토큰을 캡션 앞에 추가 하기
```arduino
elevator1.jpg
elevator1.txt → "elevx luxury elevator interior with marble floor and steel walls"
```

# LoRA 학습 → 엘리베이터 전용 가중치 생성


### 학습 방법 설정
파워쉘, bash등에서 실행
```powershell

$pip install datasets
...
$pip install accelerate
...
$pip show accelerate
...
$pip install peft trl
$pip install torchvision

pip install diffusers[torch] transformers accelerate safetensors xformers datasets


# 아래와 같이 설정 진행 
$ accelerate config
-----------------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-----------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:No
Do you wish to optimize your script with torch dynamo?[yes/NO]:no
Do you want to use DeepSpeed? [yes/NO]: no
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]: 그냥 엔터 치면 됨
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: no
-----------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
fp16
accelerate configuration saved at C:\Users\xoasm/.cache\huggingface\accelerate\default_config.yaml
```
### 학습 실행 명령어
```powershell
# 가상환경 활성화 후 실행
.\venv\Scripts\activate


# 빠르게 품질 확인
accelerate launch C:/Users/xoasm/Documents/sd-training/training/train_text_to_image_lora.py `
  --pretrained_model_name_or_path="C:/Users/xoasm/Documents/stable-diffusion-v1-5" `
  --train_file="C:/Users/xoasm/Documents/sd-training/training/dataset/dataset.csv" `
  --image_column="image" `
  --caption_column="text" `
  --output_dir="C:/Users/xoasm/Documents/sd-training/training/lora_outputs" `
  --resolution=512 `
  --train_batch_size=2 `
  --learning_rate=5e-5 `
  --max_train_steps=15000 `
  --mixed_precision=fp16 `
  --lora_r=8 `
  --lora_alpha=32 `
  --lora_dropout=0.05

# 최대 품질 지향
  accelerate launch C:/Users/xoasm/Documents/sd-training/training/train_text_to_image_lora.py `
  --pretrained_model_name_or_path="C:/Users/xoasm/Documents/stable-diffusion-v1-5" `
  --train_file="C:/Users/xoasm/Documents/sd-training/training/dataset/dataset.csv" `
  --image_column="image" `
  --caption_column="text" `
  --output_dir="C:/Users/xoasm/Documents/sd-training/training/lora_outputs" `
  --resolution=768 `
  --train_batch_size=1 `
  --learning_rate=5e-5 `
  --max_train_steps=18000 `
  --mixed_precision=fp16 `
  --lora_r=8 `
  --lora_alpha=32 `
  --lora_dropout=0.05

```
`pretrained_model_name_or_path` = 모델 경로 (SD 1.5 폴더)
`instance_data_dir` = 데이터셋 폴더 (jpg+txt 캡션 들어있는 곳)
`output_dir` = LoRA 결과 저장할 경로
