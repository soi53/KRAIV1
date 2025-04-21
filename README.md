# Video Translator V1

이 프로젝트는 비디오 파일을 번역하는 도구입니다. Docker를 사용하여 환경을 설정하고, Python 스크립트를 통해 비디오 파일의 자막을 번역합니다.

## 설치 방법

1. **Docker 설치**: 이 프로젝트는 Docker를 사용하여 실행됩니다. Docker가 설치되어 있지 않다면 [Docker 공식 웹사이트](https://www.docker.com/)에서 설치하세요.

2. **프로젝트 클론**:
   ```bash
   git clone https://github.com/soi53/KRAIV1.git
   cd KRAIV1/video_translator_v1
   ```

3. **Docker 컨테이너 실행**:
   ```bash
   docker-compose up --build
   ```

## 사용법

1. `app/requirements.txt` 파일에 필요한 Python 패키지가 명시되어 있습니다. Docker 컨테이너가 실행되면 자동으로 설치됩니다.

2. 비디오 파일을 `/data/` 디렉토리에 추가하세요. 지원되는 파일 형식은 `.mp4`, `.avi`, `.wav`, `.srt`입니다.

3. 번역된 자막 파일은 `/data/` 디렉토리에 생성됩니다.

## 기여 방법

1. 이슈를 생성하여 버그를 보고하거나 기능을 제안하세요.
2. 포크를 생성하고, 기능을 추가한 후 풀 리퀘스트를 보내주세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.
