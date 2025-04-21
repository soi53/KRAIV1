import os
import sys
import torch
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import numpy as np
import soundfile as sf
from pathlib import Path
import traceback
import subprocess
import importlib.metadata
from TTS.api import TTS  # TTS 클래스를 가져오는 구문 추가

# TTS 버전 확인
try:
    tts_version = importlib.metadata.version('TTS')
    logger.info(f"TTS 버전: {tts_version}")
except:
    tts_version = "unknown"
    logger.warning("TTS 버전을 확인할 수 없습니다.")

# TTS가 CUDA를 강제로 찾지 않도록 환경 변수 설정
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from schemas import SubtitleSegment, SubtitleFile

# Global variable to cache TTS models
_tts_models = {}

# TTS 사용 가능 여부
TTS_AVAILABLE = True

# 언어별 의존성 상태 추적 (영어만 유지)
_language_dependencies_status = {
    "en": True  # 영어는 기본적으로 동작
}

# 기본 모델 설정 (영어만 유지)
DEFAULT_MODELS = {
    "en": {"model_name": "tts_models/en/ljspeech/tacotron2-DDC", "speaker": None, "language": "English"}  # ljspeech는 단일 화자 모델
}

# XTTS_v2 모델을 위한 화자 ID 매핑
XTTS_SPEAKERS = {
    "female": ""  # 빈 문자열로 기본 화자 사용
}

# XTTS_v2 모델에서 지원하는 언어 매핑
XTTS_LANGUAGES = {
    "en": "en"
}

# TTS 인스턴스 캐시
_tts_instances = {}

# 언어별 의존성 상태 저장
language_dependency_status = {}

def check_language_dependencies():
    """
    각 언어에 필요한 의존성이 설치되어 있는지 확인하고 상태를 업데이트합니다.
    """
    global language_dependency_status
    language_dependency_status = {}
    
    # 영어는 기본적으로 지원됨
    language_dependency_status["en"] = True
    logger.info(f"언어 의존성 상태: {language_dependency_status}")
    return language_dependency_status

def create_tts_instance(model_name, use_cuda=False):
    """
    TTS 버전에 따라 적절한 매개변수로 인스턴스를 생성합니다.
    
    Args:
        model_name (str): TTS 모델 이름
        use_cuda (bool): CUDA 사용 여부
        
    Returns:
        TTS: TTS 모델 인스턴스 또는 None (실패 시)
    """
    try:
        # PyTorch 2.6+ 이상에서 weights_only 문제를 해결하기 위한 코드
        # torch.load에 weights_only=False 옵션을 추가하여 기존 동작 유지
        import torch
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # torch.load 함수를 패치된 버전으로 대체
        torch.load = patched_torch_load
        
        # XTTS_v2 모델 사용 시 자동으로 라이센스에 동의 (비상업적 용도로 가정)
        if "xtts_v2" in model_name:
            # TTS 환경 변수 설정하여 라이센스 동의 자동화
            os.environ["COQUI_TOS_AGREED"] = "1"
            logger.info("XTTS_v2 모델에 대한 비상업적 라이센스 자동 동의 설정됨")
        
        # TTS 버전에 따라 다른 초기화 매개변수 사용
        if tts_version.startswith(('0.4', '0.5', '0.6', '0.7', '0.8')):
            # 구버전 TTS
            logger.info(f"구버전 TTS({tts_version}) 인스턴스 생성: {model_name}")
            return TTS(model_name=model_name, use_cuda=use_cuda)
        else:
            # 신버전 TTS
            logger.info(f"신버전 TTS({tts_version}) 인스턴스 생성: {model_name}")
            return TTS(model_name=model_name, gpu=use_cuda)
            
    except Exception as e:
        logger.error(f"TTS 인스턴스 생성 실패: {str(e)}")
        # 다른 방식도 시도해 봅니다
        try:
            # 모든 매개변수 제거하고 기본값으로 시도
            logger.info(f"대체 방법으로 TTS 인스턴스 생성 시도: {model_name}")
            return TTS(model_name)
        except Exception as alt_e:
            logger.error(f"대체 방법도 실패: {str(alt_e)}")
            return None

def get_tts_instance(lang_code: str = "en", model_name: str = None, use_cuda: bool = False):
    """
    언어 코드와 모델 이름에 맞는 TTS 인스턴스를 반환합니다.
    언어 의존성이 없는 경우 영어 TTS로 폴백합니다.
    
    Args:
        lang_code (str): 언어 코드 (예: 'en', 'zh')
        model_name (str, optional): 사용할 TTS 모델 이름. None이면 언어 기본값 사용
        use_cuda (bool, optional): CUDA 사용 여부
        
    Returns:
        TTS: TTS 모델 인스턴스 또는 모델 로드 실패 시 None
    """
    if not lang_code:
        lang_code = "en"  # 기본값은 영어
    
    # 언어 의존성 상태가 없으면 초기화
    global language_dependency_status
    if not language_dependency_status:
        check_language_dependencies()
    
    # 요청된 언어의 의존성이 설치되어 있는지 확인
    if lang_code != "en" and (lang_code not in language_dependency_status or not language_dependency_status[lang_code]):
        logger.warning(f"{lang_code} 언어에 필요한 의존성이 설치되어 있지 않습니다. 영어 TTS로 폴백합니다.")
        # 영어로 폴백
        lang_code = "en"
        model_name = None  # 기본 영어 모델 사용
    
    # 지정된 모델이 없으면 언어별 기본 모델 사용
    if model_name is None:
        if lang_code in DEFAULT_MODELS:
            model_name = DEFAULT_MODELS[lang_code]["model_name"]
            speaker = DEFAULT_MODELS[lang_code]["speaker"]
            logger.info(f"{lang_code} 언어에 기본 모델 사용: {model_name}, 화자: {speaker}")
        else:
            logger.warning(f"{lang_code} 언어에 대한 기본 모델이 없습니다. 영어 모델로 폴백합니다.")
            lang_code = "en"
            model_name = DEFAULT_MODELS[lang_code]["model_name"]
            speaker = DEFAULT_MODELS[lang_code]["speaker"]
    
    # 전역 캐시에서 이미 로드된 모델 확인
    cache_key = f"{lang_code}_{model_name}_{use_cuda}"
    if cache_key in _tts_instances:
        logger.info(f"캐시에서 TTS 인스턴스 로드: {cache_key}")
        return _tts_instances[cache_key]
    
    # 새 TTS 인스턴스 로드 시도
    try:
        logger.info(f"새 TTS 인스턴스 로드 시작: {lang_code}, 모델: {model_name}, CUDA: {use_cuda}")
        # 버전 호환성을 위한 함수 사용
        tts = create_tts_instance(model_name, use_cuda)
        
        if tts is not None:
            _tts_instances[cache_key] = tts
            logger.info(f"TTS 인스턴스 로드 성공: {cache_key}")
            return tts
        else:
            raise Exception("TTS 인스턴스를 생성할 수 없습니다.")
            
    except Exception as e:
        logger.error(f"TTS 인스턴스 로드 실패: {str(e)}")
        # 실패한 경우 영어 모델로 폴백 시도
        if lang_code != "en":
            logger.info("영어 TTS 모델로 폴백 시도")
            return get_tts_instance(lang_code="en", model_name=None, use_cuda=use_cuda)
        else:
            logger.error("영어 TTS 모델도 로드에 실패했습니다.")
        return None


def get_tts_model_for_language(lang_code, gender="female"):
    """
    특정 언어와 성별에 대한, 사용 가능한 최적의 TTS 모델 이름을 반환합니다.
    단순화: 영어만 지원합니다.

    Args:
        lang_code (str): 언어 코드 ('en')
        gender (str): 항상 'female'로 설정되며 호환성을 위해 유지됨

    Returns:
        str: TTS 모델 이름
    """
    logger.info(f"TTS 모델 검색: 언어={lang_code}")
    
    # 단순화된 언어 지원 - 영어만 지원
    if lang_code != "en":
        logger.warning(f"지원되지 않는 언어 코드: {lang_code}, 영어로 폴백합니다.")
        lang_code = "en"
    
    # 항상 여성 목소리 사용 (파라미터는 호환성을 위해 유지)
    gender = "female"
    
    # 영어(번역 대상) TTS 모델 - 단일 화자 모델 사용
    return "tts_models/en/ljspeech/tacotron2-DDC"  # 영어 단일 화자 모델


def generate_tts_audio(subtitle_segments: List[SubtitleSegment], 
                       language_code: str,
                       output_dir: str,
                       gender: str = "female",
                       file_id: str = None) -> Dict[str, str]:
    """
    Generate TTS audio for each subtitle segment.
    
    Args:
        subtitle_segments (List[SubtitleSegment]): List of subtitle segments with translated text
        language_code (str): ISO language code for the target language (only 'en' supported)
        output_dir (str): Directory to save audio files
        gender (str): Voice gender preference (only 'female' supported)
        file_id (str): Optional file ID for naming
        
    Returns:
        Dict[str, str]: Dictionary mapping segment indices to audio file paths
    """
    try:
        global TTS_AVAILABLE
        
        # 폴백 사용 여부 추적을 위한 변수
        used_fallback = False
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # TTS를 사용할 수 없는 경우 빈 딕셔너리 반환
        if not TTS_AVAILABLE:
            logger.warning("TTS 기능을 사용할 수 없습니다. 시스템 요구 사항을 확인하세요.")
            return {}
        
        # Generate a timestamp if file_id is not provided
        if not file_id:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_id = f"tts_{timestamp}"
        
        # 영어로 강제 설정
        language_code = "en"
        gender = "female"
        
        # Get the appropriate TTS model for the language (always English)
        model_name = get_tts_model_for_language(language_code, gender)
        
        # Load the TTS model
        tts_instance = get_tts_instance(lang_code=language_code, model_name=model_name, use_cuda=False)
        
        if tts_instance is None:
            logger.error("TTS 모델을 로드할 수 없습니다. 오디오를 생성할 수 없습니다.")
            return {}
        
        # Dictionary to store segment index to audio file path mappings
        audio_files = {}
        
        # Generate audio for each segment
        logger.info(f"Generating TTS audio for {len(subtitle_segments)} segments in {language_code}")
        
        # TTS 모델 유형 확인 (새 API 또는 레거시)
        is_new_api = hasattr(tts_instance, 'tts_to_file')
        
        if is_new_api:
            logger.info("최신 TTS API(0.22.0+)를 사용하여 오디오 생성")
        else:
            logger.info("레거시 TTS API를 사용하여 오디오 생성")
        
        # 멀티링구얼 모델인지 확인하고, 필요한 경우 speaker 설정
        is_multilingual = "multilingual" in model_name or "xtts_v2" in model_name
        
        # 멀티스피커 모델에 사용할 기본 화자
        default_speaker = None
        if is_multilingual:
            if gender.lower() == "male":
                default_speaker = XTTS_SPEAKERS["male"]  # XTTS 남성 화자
            else:
                default_speaker = XTTS_SPEAKERS["female"]  # XTTS 여성 화자
            logger.info(f"멀티링구얼 모델 감지됨: {model_name}, 기본 화자: {default_speaker if default_speaker else '기본값'}")
        elif language_code in DEFAULT_MODELS:
            default_speaker = DEFAULT_MODELS[language_code]["speaker"]
            logger.info(f"{language_code} 언어 모델 사용: {model_name}, 기본 화자: {default_speaker if default_speaker else '기본값'}")
            
        # 멀티링구얼 모델에 빈 문자열 화자 ID 사용 시 설정 보장
        if is_multilingual and (default_speaker is None):
            default_speaker = ""  # 빈 문자열 사용 (None이 아님을 보장)
            logger.info(f"멀티링구얼 모델을 위한 기본 화자 ID를 빈 문자열로 설정")
        
        for segment in subtitle_segments:
            # Skip segments without translated text
            if not hasattr(segment, 'translated_text') or not segment.translated_text:
                logger.warning(f"Skipping segment {segment.index}: No translated text")
                continue
            
            # Get text for TTS
            text = segment.translated_text.strip()
            
            # Generate audio file name based on segment index
            audio_filename = f"{file_id}_segment_{segment.index:04d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            # Generate speech
            logger.info(f"Generating TTS for segment {segment.index}: '{text[:30]}...' if len(text) > 30 else text")
            
            try:
                logger.info(f"Generating audio for text: {text[:50]}...")
                
                if is_new_api:
                    # 최신 TTS API 사용 (0.22.0+)
                    try:
                        # 멀티링구얼/멀티스피커 모델인지 확인
                        is_multispeaker = is_multilingual or "vctk" in model_name
                        
                        # 단일 화자 모델인 경우
                        if "ljspeech" in model_name or not is_multispeaker:
                            logger.info(f"단일 화자 모델 사용: {model_name}, speaker 파라미터 사용 안함")
                            tts_instance.tts_to_file(text=text, file_path=audio_path)
                        # 멀티링구얼 모델인 경우 language와 speaker 파라미터 추가
                        elif is_multilingual:
                            # 멀티링구얼 모델은 language 파라미터도 필요
                            xtts_lang = XTTS_LANGUAGES.get(language_code, "en")
                            # speaker가 빈 문자열이라도 반드시 제공
                            if default_speaker is None:
                                default_speaker = ""
                            logger.info(f"멀티링구얼 모델로 TTS 생성: 언어={xtts_lang}, 화자={default_speaker if default_speaker else '기본값'}")
                            tts_instance.tts_to_file(text=text, file_path=audio_path, speaker=default_speaker, language=xtts_lang)
                        # 일반 멀티스피커 모델인 경우 speaker 파라미터만 추가
                        else:
                            # speaker 제공
                            if default_speaker:
                                logger.info(f"멀티스피커 모델로 TTS 생성: 화자={default_speaker}")
                                tts_instance.tts_to_file(text=text, file_path=audio_path, speaker=default_speaker)
                            else:
                                # speaker가 없으면 시도할 가능성이 있는 speaker 목록
                                try_speakers = ["baker", "ljspeech", None]
                                success = False
                                for try_speaker in try_speakers:
                                    try:
                                        if try_speaker:
                                            tts_instance.tts_to_file(text=text, file_path=audio_path, speaker=try_speaker)
                                        else:
                                            tts_instance.tts_to_file(text=text, file_path=audio_path)
                                        success = True
                                        logger.info(f"화자 '{try_speaker}'로 TTS 생성 성공")
                                        break  # 성공하면 루프 종료
                                    except Exception as speaker_e:
                                        logger.warning(f"화자 '{try_speaker}'로 TTS 생성 시도 실패: {str(speaker_e)}")
                                
                                if not success:
                                    raise ValueError("모든 화자 시도가 실패했습니다")
                                
                        logger.info(f"Audio generated and saved at {audio_path}")
                        audio_files[segment.index] = audio_path
                    except Exception as e:
                        logger.error(f"TTS 생성 실패 (tts_to_file): {str(e)}")
                        # 보조 시도: 대체 방법으로 다시 시도
                        logger.warning(f"TTS 직접 파일 저장 실패, 대체 방법 시도: {str(e)}")
                        try:
                            # 멀티링구얼/멀티스피커 모델인지 확인
                            is_multispeaker = is_multilingual or "vctk" in model_name
                            
                            # 단일 화자 모델인 경우
                            if "ljspeech" in model_name or not is_multispeaker:
                                logger.info(f"대체 방법으로 단일 화자 TTS 시도: {model_name}")
                                wav = tts_instance.tts(text=text)
                            # 멀티링구얼 모델인 경우 speaker와 language 파라미터 필수
                            elif is_multilingual:
                                xtts_lang = XTTS_LANGUAGES.get(language_code, "en")
                                # speaker가 빈 문자열이라도 반드시 제공
                                if default_speaker is None:
                                    default_speaker = ""
                                logger.info(f"대체 방법으로 멀티링구얼 TTS 시도: 언어={xtts_lang}, 화자={default_speaker if default_speaker else '기본값'}")
                                wav = tts_instance.tts(text=text, speaker=default_speaker, language=xtts_lang)
                            # 일반 멀티스피커 모델인 경우
                            else:
                                # speaker 제공
                                if default_speaker:
                                    logger.info(f"대체 방법으로 멀티스피커 TTS 시도: 화자={default_speaker}")
                                    wav = tts_instance.tts(text=text, speaker=default_speaker)
                                else:
                                    # speaker가 없으면 시도할 가능성이 있는 speaker 목록
                                    try_speakers = ["baker", "ljspeech", None]
                                    success = False
                                    for try_speaker in try_speakers:
                                        try:
                                            if try_speaker:
                                                wav = tts_instance.tts(text=text, speaker=try_speaker)
                                            else:
                                                wav = tts_instance.tts(text=text)
                                            success = True
                                            logger.info(f"화자 '{try_speaker}'로 TTS 생성 성공")
                                            break  # 성공하면 루프 종료
                                        except Exception as speaker_e:
                                            logger.warning(f"화자 '{try_speaker}'로 TTS 생성 시도 실패: {str(speaker_e)}")
                                    
                                    if not success:
                                        raise ValueError("모든 화자 시도가 실패했습니다")
                                    
                            sf.write(audio_path, wav, 22050)  # 기본 샘플레이트
                            logger.info(f"대체 방법으로 오디오 생성 성공: {audio_path}")
                            audio_files[segment.index] = audio_path
                        except Exception as alt_e:
                            logger.error(f"대체 TTS 생성 방법도 실패: {str(alt_e)}")
                            continue
                else:
                    # 레거시 TTS API 사용
                    try:
                        # Synthesize audio
                        tts_result = tts_instance.tts(text)
                        
                        # 반환 값 처리 (다양한 형식 지원)
                        wav = None
                        sample_rate = 22050  # 기본 샘플레이트
                        
                        if isinstance(tts_result, tuple):
                            # 튜플 형태 (wav, sample_rate) 또는 (wav, sample_rate, ...)
                            wav = tts_result[0]
                            if len(tts_result) > 1:
                                sample_rate = tts_result[1]
                        elif isinstance(tts_result, dict):
                            # 딕셔너리 형태 {'wav': array, 'sample_rate': rate, ...}
                            if 'wav' in tts_result:
                                wav = tts_result['wav']
                            if 'sample_rate' in tts_result:
                                sample_rate = tts_result['sample_rate']
                        elif isinstance(tts_result, np.ndarray):
                            # 직접 wav 배열 반환
                            wav = tts_result
                        else:
                            # 기타 형식
                            wav = tts_result
                            logger.warning(f"예상치 못한 TTS 반환 타입: {type(tts_result)}")
                        
                        if wav is None:
                            raise ValueError("TTS 모델이 오디오 데이터를 생성하지 않았습니다")
                        
                        # Save as WAV file
                        sf.write(audio_path, wav, sample_rate)
                        
                        logger.info(f"Audio generated and saved at {audio_path}")
                        audio_files[segment.index] = audio_path
                        
                    except Exception as inner_e:
                        logger.error(f"TTS 생성 오류, 다른 방식 시도: {str(inner_e)}")
                        # 대체 방법 시도 (일부 레거시 버전)
                        try:
                            wav, sample_rate = tts_instance.tts(text, return_wav=True)
                            sf.write(audio_path, wav, sample_rate)
                            logger.info(f"대체 방법으로 오디오 생성 성공: {audio_path}")
                            audio_files[segment.index] = audio_path
                        except Exception as alt_e:
                            logger.error(f"대체 TTS 생성 방법도 실패: {str(alt_e)}")
                            continue
                
            except Exception as e:
                logger.error(f"Failed to generate TTS for segment {segment.index}: {str(e)}")
                continue
        
        logger.success(f"Generated TTS audio for {len(audio_files)} segments")
        return audio_files
        
    except Exception as e:
        logger.error(f"Error in generate_tts_audio: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def merge_subtitle_audio_files(audio_files: Dict[int, str], 
                              segments: List[SubtitleSegment],
                              output_dir: str,
                              file_id: str) -> Optional[str]:
    """
    Merge all segment audio files into a single audio file with proper timings.
    
    Args:
        audio_files (Dict[int, str]): Dictionary mapping segment indices to audio file paths
        segments (List[SubtitleSegment]): Subtitle segments with timing information
        output_dir (str): Directory to save merged audio file
        file_id (str): File ID for naming
        
    Returns:
        Optional[str]: Path to merged audio file or None if failed
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        merged_audio_path = os.path.join(output_dir, f"{file_id}_tts_merged.wav")
        
        # Calculate total duration based on last segment end time
        if segments:
            last_segment = max(segments, key=lambda x: x.end_time_ms)
            total_duration_ms = last_segment.end_time_ms
        else:
            logger.error("No segments provided for merging audio")
            return None
        
        # Create a silent audio array (44.1kHz sample rate)
        sample_rate = 22050  # Default sample rate for TTS
        total_samples = int((total_duration_ms / 1000) * sample_rate)
        merged_audio = np.zeros(total_samples)
        
        # Place each segment audio at the correct time position
        for segment in segments:
            segment_idx = segment.index
            
            # Skip if segment audio not available
            if segment_idx not in audio_files:
                logger.warning(f"No audio file for segment {segment_idx}, skipping")
                continue
            
            # Load segment audio
            segment_audio_path = audio_files[segment_idx]
            segment_audio, seg_sample_rate = sf.read(segment_audio_path)
            
            # Resample if needed
            if seg_sample_rate != sample_rate:
                # Simple resampling (for more complex cases, consider using librosa)
                from scipy import signal
                segment_audio = signal.resample(segment_audio, 
                                              int(len(segment_audio) * sample_rate / seg_sample_rate))
            
            # Calculate start position in samples
            start_sample = int((segment.start_time_ms / 1000) * sample_rate)
            
            # Calculate end position in samples
            end_sample = min(start_sample + len(segment_audio), total_samples)
            
            # Place the audio at the correct position
            merged_audio[start_sample:end_sample] = segment_audio[:end_sample-start_sample]
        
        # Save merged audio
        sf.write(merged_audio_path, merged_audio, sample_rate)
        logger.success(f"Merged audio saved to: {merged_audio_path}")
        
        return merged_audio_path
        
    except Exception as e:
        logger.error(f"Error merging audio files: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 

def generate_tts_for_subtitle(
    subtitle_text: str, 
    target_path: str, 
    lang_code: str = "en", 
    gender: str = "female",
    use_cuda: bool = False
) -> Tuple[bool, str]:
    """
    자막 텍스트에 대한 TTS를 생성합니다.
    
    Args:
        subtitle_text (str): TTS로 변환할 자막 텍스트
        target_path (str): 오디오 파일 저장 경로
        lang_code (str): 언어 코드 (현재는 영어(en)만 지원)
        gender (str): 목소리 성별 ('female'만 지원)
        use_cuda (bool): CUDA 사용 여부
        
    Returns:
        Tuple[bool, str]: (성공 여부, 결과 파일 경로 또는 오류 메시지)
    """
    try:
        # 항상 여성 목소리 사용
        gender = "female"
        
        # 영어 외의 언어는 지원하지 않음
        if lang_code != "en":
            logger.warning(f"{lang_code} 언어는 지원되지 않습니다. 영어로 폴백합니다.")
            lang_code = "en"
        
        # TTS 인스턴스 가져오기 전에 모델 이름 얻기
        model_name = get_tts_model_for_language(lang_code, gender)
        
        # TTS 인스턴스 가져오기
        tts = get_tts_instance(lang_code=lang_code, model_name=model_name, use_cuda=use_cuda)
        
        if tts is None:
            logger.error(f"TTS 인스턴스를 생성할 수 없습니다.")
            return False, "TTS 인스턴스 생성 실패"
        
        # 텍스트가 비어 있는지 확인
        if not subtitle_text or subtitle_text.strip() == "":
            logger.warning("TTS 생성을 위한 텍스트가 비어 있습니다.")
            return False, "TTS 생성을 위한 텍스트가 비어 있습니다."
        
        try:
            # 저장 디렉토리 확인
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            # 텍스트 전처리 (특수문자, 온점 등 처리)
            processed_text = preprocess_text_for_tts(subtitle_text, lang_code)
            
            # TTS 생성 실행
            logger.info(f"TTS 생성 중: {lang_code}, 화자: {gender}")
            
            # 모델 이름 확인
            is_multilingual = "multilingual" in model_name
            
            # 멀티링구얼 모델인 경우 적절한 화자 설정
            if is_multilingual:
                xtts_lang = XTTS_LANGUAGES.get(lang_code, "en")
                tts.tts_to_file(text=processed_text, file_path=target_path, speaker=gender, language=xtts_lang)
            else:
                if gender:
                    tts.tts_to_file(text=processed_text, file_path=target_path, speaker=gender)
                else:
                    # 여러 화자 시도
                    try_speakers = ["baker", "ljspeech", None]
                    success = False
                    
                    for try_speaker in try_speakers:
                        try:
                            if try_speaker:
                                tts.tts_to_file(text=processed_text, file_path=target_path, speaker=try_speaker)
                            else:
                                tts.tts_to_file(text=processed_text, file_path=target_path)
                            success = True
                            logger.info(f"화자 '{try_speaker}'로 TTS 생성 성공")
                            break
                        except Exception as speaker_e:
                            logger.warning(f"화자 '{try_speaker}'로 TTS 생성 시도 실패: {str(speaker_e)}")
                    
                    if not success:
                        raise ValueError("모든 화자 시도가 실패했습니다.")
        except Exception as e:
            logger.error(f"TTS 파일 생성 실패: {str(e)}")
            logger.info("대체 방식으로 TTS 생성 시도")
            
            try:
                # 대체 생성 방식 시도
                if is_multilingual:
                    xtts_lang = XTTS_LANGUAGES.get(lang_code, "en")
                    wav = tts.tts(text=processed_text, speaker=gender, language=xtts_lang)
                else:
                    if gender:
                        wav = tts.tts(text=processed_text, speaker=gender)
                    else:
                        wav = tts.tts(text=processed_text)
                
                sf.write(target_path, wav, 22050)  # 기본 샘플레이트
                logger.info(f"대체 방법으로 TTS 생성 성공")
            except Exception as alt_e:
                logger.error(f"대체 TTS 생성 방법도 실패: {str(alt_e)}")
                return False, f"TTS 생성 실패: {str(alt_e)}"
        
        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            logger.error("TTS 파일이 생성되지 않았거나 크기가 0입니다.")
            return False, "오디오 파일 생성에 실패했습니다."
        
        logger.info(f"TTS 생성 성공: {target_path}")
        return True, target_path
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"TTS 생성 중 오류 발생: {str(e)}")
        logger.debug(f"오류 상세 스택: {error_traceback}")
        return False, f"TTS 생성 중 오류 발생: {str(e)}"

def preprocess_text_for_tts(text: str, lang_code: str) -> str:
    """
    TTS 생성을 위해 텍스트를 전처리합니다.
    
    Args:
        text (str): 원본 텍스트
        lang_code (str): 언어 코드
        
    Returns:
        str: 전처리된 텍스트
    """
    # 기본 전처리: 공백 정리
    text = text.strip()
    
    # 중국어 특수 처리
    if lang_code == "zh" or lang_code == "zh-CN":
        # 중국어 문장 끝 마침표 확인
        if not text.endswith(("。", ".", "!", "?", "！", "？")):
            text += "。"  # 중국어 마침표 추가
    
    # 영어 특수 처리
    elif lang_code == "en":
        # 영어 문장 끝 마침표 확인
        if not text.endswith((".", "!", "?")):
            text += "."  # 마침표 추가
            
    # 필요한 경우 다른 언어에 대한 전처리 추가
    
    return text 