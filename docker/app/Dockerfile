# Use the official Python 3.11 slim image as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Add environment variables to prevent CUDA usage
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_CUDA_ARCH_LIST=""

# Install FFmpeg and TTS related dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    mecab \
    mecab-ipadic \
    libmecab-dev \
    swig \
    git \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch (CPU version) and torchaudio explicitly
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation (CPU version) during build
RUN python -c "import torch; print(f'PyTorch {torch.__version__} CPU version installed successfully.'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create a placeholder for the requirements file
# This ensures the COPY command doesn't fail if the file doesn't exist yet locally
RUN touch /app/requirements.txt

# Copy the requirements file into the container at /app
COPY ./app/requirements.txt /app/requirements.txt

# Install TTS dependencies first
RUN pip install --no-cache-dir pyyaml tqdm tensorboard numpy scipy soundfile librosa unidecode inflect pydub

# 명시적으로 phonemizer 설치 (일본어 포함 다국어 TTS에 필요)
RUN pip install --no-cache-dir phonemizer==3.2.1

# 일본어 및 다국어 지원에 필요한 추가 패키지
RUN pip install --no-cache-dir \
    fugashi \
    ipadic \
    unidic-lite \
    jaconv \
    Cython \
    mecab-python3

# Install TTS and language-specific dependencies together
RUN pip install --no-cache-dir TTS && \
    pip install --no-cache-dir 'TTS[ja]' && \
    pip install --no-cache-dir 'TTS[zh]' && \
    pip install --no-cache-dir 'TTS[fr]' && \
    pip install --no-cache-dir 'TTS[de]' && \
    pip install --no-cache-dir 'TTS[es]' && \
    pip install --no-cache-dir 'TTS[it]'

# Verify phonemizer and MeCab installation
RUN python -c "import phonemizer; print(f'Phonemizer version: {phonemizer.__version__}')" && \
    python -c "import MeCab; print('MeCab successfully installed')"

# Install remaining packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from the host to /app in the container
# This will copy the contents of the 'app' directory in the host
COPY ./app /app

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Define the command to run the app using streamlit
# Changed app.py to streamlit_app.py
CMD ["streamlit", "run", "streamlit_app.py"] 