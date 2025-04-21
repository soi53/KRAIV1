import os
import sys
import json
from loguru import logger
from typing import List, Dict, Any, Optional

# Ensure the current directory is in sys.path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from schemas import SubtitleSegment, SubtitleFile
from config import settings

def translate_subtitle_segments(segments: List[SubtitleSegment], source_lang: str, target_lang: str) -> List[SubtitleSegment]:
    """
    번역 API를 사용하여 자막 세그먼트를 번역합니다.
    
    Args:
        segments: 번역할 자막 세그먼트 목록
        source_lang: 원본 언어 코드 (예: 'ko', 'en')
        target_lang: 대상 언어 코드 (예: 'en', 'ko')
        
    Returns:
        번역된 텍스트가 포함된 자막 세그먼트 목록
    """
    try:
        import openai
        from openai import OpenAI
        
        # API 키 설정
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            return segments
        
        # API 키 확인 로그 (디버깅 용도)
        logger.info(f"OpenAI API 키 확인: {api_key[:5]}...{api_key[-5:]}")
            
        # OpenAI 클라이언트 초기화
        logger.info("OpenAI 클라이언트 초기화 중...")
        client = OpenAI(api_key=api_key)
        
        # 번역 요청 준비
        batch_size = 10  # 한 번에 처리할 세그먼트 수
        batched_segments = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
        
        translated_segments = []
        total_batches = len(batched_segments)
        
        logger.info(f"총 {len(segments)}개 세그먼트를 {total_batches}개 배치로 번역 시작")
        
        for batch_idx, batch in enumerate(batched_segments):
            # 진행 상황 로그
            logger.info(f"번역 중: 배치 {batch_idx+1}/{total_batches}")
            
            # 번역할 텍스트 준비
            texts = [segment.text for segment in batch]
            indices = [segment.index for segment in batch]
            
            # 시스템 프롬프트 준비
            system_prompt = f"""당신은 전문 번역가입니다. 다음 지침을 따라 자막을 {source_lang}에서 {target_lang}으로 번역하세요:
1. 자막의 원래 의미와 뉘앙스를 정확하게 유지하세요.
2. 문화적 맥락을 고려하여 자연스러운 번역을 제공하세요.
3. 전문 용어나 특수 표현은 해당 언어권의 일반적인 표현으로 번역하세요.
4. 번역된 텍스트는 간결하게 유지하세요.
5. 화면에 자막으로 표시될 것이므로 문장이 너무 길어지지 않도록 주의하세요.
6. 결과는 원본 텍스트의 각 줄에 대한 번역만 포함해야 합니다. 다른 설명은 포함하지 마세요."""
            
            # 사용자 메시지 준비
            user_message = f"다음 {source_lang} 자막을 {target_lang}으로 번역해주세요:\n\n"
            for idx, text in zip(indices, texts):
                user_message += f"{idx}. {text}\n"
            
            logger.info(f"OpenAI API 호출 시작: gpt-4o 모델, 배치 {batch_idx+1}")
            
            try:
                # API 호출
                response = client.chat.completions.create(
                    model="gpt-4o",  # gpt-3.5-turbo에서 gpt-4o로 변경
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                )
                
                # 응답 처리
                translation_text = response.choices[0].message.content.strip()
                logger.info(f"API 응답 성공: {len(translation_text)} 글자 받음")
                
                # 번역 결과 파싱
                translation_lines = translation_text.split("\n")
                
                # 세그먼트에 번역 결과 적용
                for segment in batch:
                    # 현재 세그먼트의 인덱스와 일치하는 번역 라인 찾기
                    for line in translation_lines:
                        if line.startswith(f"{segment.index}."):
                            # "인덱스. " 부분 제거하고 번역 텍스트 추출
                            translated_text = line[line.find(" ") + 1:].strip()
                            segment.translated_text = translated_text
                            break
                    
                    translated_segments.append(segment)
            
            except Exception as batch_error:
                logger.error(f"배치 {batch_idx+1} 번역 중 오류: {str(batch_error)}")
                # 실패한 배치의 세그먼트를 그대로 추가
                translated_segments.extend(batch)
                continue
        
        # 모든 세그먼트가 번역되었는지 확인
        untranslated_count = sum(1 for segment in translated_segments if not hasattr(segment, 'translated_text') or not segment.translated_text)
        if untranslated_count > 0:
            logger.warning(f"{untranslated_count}개의 세그먼트가 번역되지 않았습니다.")
        
        logger.success(f"{len(segments) - untranslated_count}개의 세그먼트가 성공적으로 번역되었습니다.")
        return translated_segments
        
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return segments

def batch_translate_with_retry(segments: List[SubtitleSegment], source_lang: str, target_lang: str, max_retries: int = 3) -> List[SubtitleSegment]:
    """
    재시도 로직이 포함된 배치 번역 함수
    
    Args:
        segments: 번역할 자막 세그먼트 목록
        source_lang: 원본 언어 코드
        target_lang: 대상 언어 코드
        max_retries: 최대 재시도 횟수
        
    Returns:
        번역된 텍스트가 포함된 자막 세그먼트 목록
    """
    retry_count = 0
    result_segments = segments.copy()
    
    while retry_count < max_retries:
        # 번역이 필요한 세그먼트만 필터링
        untranslated = [s for s in result_segments if not hasattr(s, 'translated_text') or not s.translated_text]
        
        if not untranslated:
            break  # 모든 세그먼트가 번역됨
            
        logger.info(f"번역 시도 {retry_count+1}/{max_retries}: {len(untranslated)}개의 세그먼트")
        
        # 번역 시도
        translated = translate_subtitle_segments(untranslated, source_lang, target_lang)
        
        # 결과 업데이트
        translated_indices = {s.index for s in translated if hasattr(s, 'translated_text') and s.translated_text}
        
        for i, segment in enumerate(result_segments):
            if segment.index in translated_indices:
                # 번역된 세그먼트 찾기
                for t_segment in translated:
                    if t_segment.index == segment.index:
                        result_segments[i].translated_text = t_segment.translated_text
                        break
        
        # 아직 번역이 필요한 세그먼트 확인
        still_untranslated = sum(1 for s in result_segments if not hasattr(s, 'translated_text') or not s.translated_text)
        
        if still_untranslated == 0:
            logger.success("모든 세그먼트가 성공적으로 번역되었습니다.")
            break
            
        logger.warning(f"{still_untranslated}개의 세그먼트가 여전히 번역되지 않았습니다. 재시도 중...")
        retry_count += 1
    
    return result_segments

def get_language_name(lang_code: str) -> str:
    """
    언어 코드에 해당하는 언어 이름을 반환합니다.
    
    Args:
        lang_code: 언어 코드 (예: 'ko', 'en')
        
    Returns:
        언어 이름 (예: '한국어', '영어')
    """
    language_map = {
        'ko': '한국어',
        'en': '영어',
        'ja': '일본어',
        'zh': '중국어',
        'es': '스페인어',
        'fr': '프랑스어',
        'de': '독일어',
        'ru': '러시아어',
        'pt': '포르투갈어',
        'it': '이탈리아어',
        'nl': '네덜란드어',
        'ar': '아랍어',
        'hi': '힌디어',
        'th': '태국어',
        'vi': '베트남어'
    }
    return language_map.get(lang_code, lang_code) 