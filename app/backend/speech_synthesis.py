import streamlit as st
import os
import hashlib
import logging
import traceback
from tts_handler import get_tts_instance
from pydub import AudioSegment

def adjust_audio_speed(input_path, output_path, speed_factor):
    """
    오디오 파일의 재생 속도를 조절합니다.
    
    Args:
        input_path (str): 입력 오디오 파일 경로
        output_path (str): 출력 오디오 파일 경로
        speed_factor (float): 속도 조절 계수 (1.0 = 원래 속도)
    """
    try:
        # 오디오 파일 로드
        sound = AudioSegment.from_file(input_path)
        
        # 속도 변경 (frame_rate 조절)
        # frame_rate를 높이면 속도가 빨라지고, 낮추면 느려짐
        # speed_factor가 2.0이면 2배 빠르게, 0.5면 절반 속도로 재생
        new_frame_rate = int(sound.frame_rate * speed_factor)
        
        # 새 프레임 레이트로 오디오 내보내기
        sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": new_frame_rate
        })
        
        # 원래 프레임 레이트로 변환 (피치 유지)
        sound = sound.set_frame_rate(44100)
        
        # 파일로 저장
        sound.export(output_path, format="wav")
        return True
    except Exception as e:
        logging.error(f"오디오 속도 조절 실패: {e}")
        logging.debug(traceback.format_exc())
        return False

def text_to_speech(text, target_language, tts_voice="default", speed=1.0):
    """
    텍스트를 음성으로 변환
    
    Args:
        text (str): 읽을 텍스트
        target_language (str): 언어 코드 (예: 'en', 'ja', 'ko', 등)
        tts_voice (str): 사용할 TTS 음성 (현재는 무시됨)
        speed (float): 음성 재생 속도 조절 (1.0 = 정상)
        
    Returns:
        str: 생성된 오디오 파일 경로 또는 None (실패 시)
    """
    if not text or not text.strip():
        logging.warning("음성 합성을 위한 텍스트가 비어 있습니다.")
        return None
    
    # 캐시 디렉토리 확인
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache", "tts")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 캐시 파일명 생성 (텍스트, 언어, 속도의 해시)
    cache_key = f"{text}_{target_language}_{speed}"
    cache_filename = hashlib.md5(cache_key.encode()).hexdigest() + ".wav"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # 캐시 확인
    if os.path.exists(cache_path):
        logging.info(f"캐시된 TTS 오디오 사용: {cache_path}")
        return cache_path
    
    # TTS 인스턴스 가져오기
    try:
        tts = get_tts_instance(target_language)
        
        # TTS가 초기화되지 않았으면 None 반환
        if tts is None:
            logging.error("TTS 인스턴스를 초기화할 수 없습니다.")
            return None
        
        # 오디오 생성
        logging.info(f"TTS 오디오 생성 중: '{text[:30]}...' ({target_language})")
        
        # TTS 변환 수행
        tts.tts_to_file(
            text=text,
            file_path=cache_path
        )
        
        # 속도 조절이 필요한 경우
        if speed != 1.0:
            try:
                adjust_audio_speed(cache_path, cache_path, speed)
                logging.info(f"TTS 오디오 속도 조절 완료: {speed}x")
            except Exception as e:
                logging.error(f"오디오 속도 조절 실패: {e}")
        
        logging.info(f"TTS 오디오 생성 완료: {cache_path}")
        
        # 폴백 사용 여부에 따른 메시지 처리
        if hasattr(st, "session_state") and st.session_state.get("used_fallback_tts", False):
            fallback_reason = st.session_state.get("fallback_reason", "알 수 없는 이유로")
            logging.warning(f"TTS 폴백 사용됨: {fallback_reason} 영어 TTS 모델이 사용되었습니다.")
            # 폴백 사용 상태는 읽고 나서 초기화하지 않음 (streamlit_app.py에서 초기화)
        
        return cache_path
    
    except Exception as e:
        logging.error(f"TTS 변환 실패: {e}")
        logging.debug(traceback.format_exc())
        return None 