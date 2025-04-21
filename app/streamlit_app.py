import sys
import os

# 현재 파일 경로를 기준으로 상위 디렉토리(app 디렉토리)를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
from utils.logger_config import logger
from config import settings
from utils.file_handler import validate_video_file, save_uploaded_file
from backend.video_processor import get_video_duration
from schemas import VideoInfo

def get_language_name(lang_code: str) -> str:
    """
    언어 코드에 해당하는 언어 이름을 반환합니다.
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Video Translator V1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize session state ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'upload'  # 'upload', 'trim', 'subtitle', 'translate', 'tts', 'result'
if 'video_info' not in st.session_state:
    st.session_state.video_info = None

# --- Initialization ---
logger.info("Streamlit application started.")
logger.info(f"Log level set to: {settings.LOG_LEVEL}")
logger.info(f"OpenAI API Key Loaded: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
logger.info(f"Whisper API URL Set: {settings.WHISPER_API_URL if settings.WHISPER_API_URL else 'Not Set'}")

# --- Main Application UI ---
st.title("🎬 Video Translator and Dubbing Tool V1")
st.write("Welcome! This tool helps you translate and dub your videos.")
st.markdown("---")

# --- Progress Bar ---
steps = ['Upload', 'Trim', 'Subtitle', 'Translation', 'TTS', 'Result']
step_map = {'upload': 'Upload', 'trim': 'Trim', 'subtitle': 'Subtitle', 'translation': 'Translation', 'tts': 'TTS', 'result': 'Result'}
current_step_idx = steps.index(step_map.get(st.session_state.current_step, 'Upload')) if st.session_state.current_step else 0
st.progress(current_step_idx / (len(steps) - 1))

# --- Step indicators ---
cols = st.columns(len(steps))
for i, step in enumerate(steps):
    with cols[i]:
        if i < current_step_idx:
            st.markdown(f"✅ {step}")
        elif i == current_step_idx:
            st.markdown(f"🔄 **{step}**")
        else:
            st.markdown(f"⏸️ {step}")

# --- Upload step ---
if st.session_state.current_step == 'upload':
    st.header("Step 1: Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi"],
        help="Upload a video file in MP4 or AVI format. Maximum size: 2GB."
    )
    
    if uploaded_file is not None:
        # Validate uploaded file
        is_valid, message = validate_video_file(uploaded_file)
        
        if not is_valid:
            st.error(message)
        else:
            # Process the uploaded file
            with st.spinner("Uploading and processing video..."):
                # Save the file
                uploads_dir = "/data/uploads"
                file_info = save_uploaded_file(uploaded_file, uploads_dir)
                
                # Get video duration
                video_path = file_info["path"]
                duration_ms = get_video_duration(video_path)
                
                if duration_ms is None:
                    st.error("Failed to extract video duration. Please check if the video file is valid.")
                else:
                    # Create VideoInfo object
                    video_info = VideoInfo(
                        id=file_info["id"],
                        original_name=file_info["original_name"],
                        original_path=file_info["path"],
                        size=file_info["size"],
                        duration_ms=duration_ms,
                        extension=file_info["extension"]
                    )
                    
                    # Save to session state
                    st.session_state.video_info = video_info
                    st.session_state.current_step = 'trim'
                    
                    # Display success and redirect
                    st.success(f"Video '{video_info.original_name}' uploaded successfully!")
                    
                    # Display video information
                    st.subheader("Video Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Filename:** {video_info.original_name}")
                        st.write(f"**Size:** {video_info.size / (1024 * 1024):.2f} MB")
                    with col2:
                        minutes, seconds = divmod(video_info.duration_ms // 1000, 60)
                        st.write(f"**Duration:** {minutes} minutes, {seconds} seconds")
                        st.write(f"**Format:** {video_info.extension[1:].upper()}")
                    
                    # Add button to continue to next step
                    st.button("Continue to Trim Video", on_click=lambda: setattr(st.session_state, 'current_step', 'trim'))
                    
                    # Use st.cache_resource.clear() to avoid repeated file processing 
                    # if user manually refreshes the page but file is already uploaded
                    st.cache_resource.clear()

# --- trim step 부분 수정 ---
elif st.session_state.current_step == 'trim':
    st.header("Step 2: Trim Video (Optional)")
    
    # Check if video_info exists
    if st.session_state.video_info is None:
        st.error("No video information found. Please upload a video first.")
        if st.button("Go back to Upload"):
            st.session_state.current_step = 'upload'
    else:
        st.write("You can trim your video to focus on specific parts. This step is optional.")
        
        # Display video for reference
        video_info = st.session_state.video_info
        st.video(video_info.original_path)
        
        # Display video duration for reference
        total_duration_sec = video_info.duration_ms / 1000
        minutes, seconds = divmod(int(total_duration_sec), 60)
        st.info(f"Total video duration: {minutes} minutes and {seconds} seconds ({total_duration_sec:.2f} seconds)")
        
        # Trim settings
        st.subheader("Trim Settings")
        
        # Use time_converter module instead of inline functions
        from utils.time_converter import validate_time_format, time_to_ms
        
        # Get trim times from user
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.text_input(
                "Start time (MM:SS)",
                value="",
                help="Enter the start time in MM:SS format (e.g., 1:30 for 1 minute and 30 seconds)"
            )
        with col2:
            end_time = st.text_input(
                "End time (MM:SS)",
                value="",
                help="Enter the end time in MM:SS format (e.g., 2:45 for 2 minutes and 45 seconds)"
            )
        
        # Validate inputs
        start_ms = time_to_ms(start_time) if start_time else 0
        end_ms = time_to_ms(end_time) if end_time else video_info.duration_ms
        
        # Display buttons based on validation
        if start_time and not validate_time_format(start_time):
            st.error("Invalid start time format. Please use MM:SS format (e.g., 1:30).")
            can_trim = False
        elif end_time and not validate_time_format(end_time):
            st.error("Invalid end time format. Please use MM:SS format (e.g., 2:45).")
            can_trim = False
        elif start_ms is not None and end_ms is not None and start_ms >= end_ms:
            st.error("Start time must be before end time.")
            can_trim = False
        elif end_ms is not None and end_ms > video_info.duration_ms:
            st.error(f"End time exceeds video duration ({total_duration_sec:.2f} seconds).")
            can_trim = False
        else:
            can_trim = True
        
        # Only show trim button if inputs are valid
        trim_col, skip_col = st.columns(2)
        
        with trim_col:
            if can_trim and (start_ms > 0 or end_ms < video_info.duration_ms):
                trim_button = st.button("Trim Video")
                if trim_button:
                    # Implement actual trimming functionality
                    with st.spinner("Trimming video..."):
                        from backend.video_processor import trim_video
                        processed_dir = "/data/processed"
                        
                        # Call the trim_video function
                        trimmed_path = trim_video(
                            video_info.original_path, 
                            start_ms, 
                            end_ms, 
                            processed_dir
                        )
                        
                        if trimmed_path:
                            # Update session state with trimmed video info
                            video_info.trimmed = True
                            video_info.trimmed_path = trimmed_path
                            video_info.trim_start_ms = start_ms
                            video_info.trim_end_ms = end_ms
                            st.session_state.video_info = video_info
                            
                            st.success("Video trimmed successfully!")
                            # Show trimmed video
                            st.subheader("Trimmed Video Preview")
                            st.video(trimmed_path)
                            
                            # Add download button for trimmed video
                            with open(trimmed_path, "rb") as file:
                                file_data = file.read()  # 파일 데이터를 미리 읽음

                            # 다운로드 상태 추적
                            if 'download_clicked' not in st.session_state:
                                st.session_state.download_clicked = False

                            st.download_button(
                                label="Download Trimmed Video",
                                data=file_data,  # 미리 읽은 데이터 사용
                                file_name=os.path.basename(trimmed_path),
                                mime="video/mp4",
                                key="download_trim_btn",  # 고유 키 부여
                                on_click=lambda: setattr(st.session_state, 'download_clicked', True)  # 다운로드 상태 저장
                            )
                            
                            # Add button to continue to next step
                            st.button("Continue to Subtitle Extraction", 
                                     on_click=lambda: setattr(st.session_state, 'current_step', 'subtitle'))
                        else:
                            st.error("Failed to trim video. Please check the logs for details.")
            else:
                st.write("Enter valid times to trim the video")
        
        with skip_col:
            skip_button = st.button("Skip Trimming")
            if skip_button:
                st.session_state.current_step = 'subtitle'
                st.rerun()

# --- Subtitle extraction step ---
elif st.session_state.current_step == 'subtitle':
    st.header("Step 3: Subtitle Extraction")
    
    # Check if video_info exists
    if st.session_state.video_info is None:
        st.error("No video information found. Please upload a video first.")
        if st.button("Go back to Upload"):
            st.session_state.current_step = 'upload'
    else:
        video_info = st.session_state.video_info
        
        # Display active video
        st.subheader("Video Preview")
        st.video(video_info.active_video_path)
        
        # Initialize subtitle-related session state if needed
        if 'subtitle_file' not in st.session_state:
            st.session_state.subtitle_file = None
        
        # Two options: extract automatically or upload manually
        subtitle_tab1, subtitle_tab2 = st.tabs(["Extract Subtitles", "Upload Subtitles"])
        
        with subtitle_tab1:
            st.subheader("Extract Subtitles with Whisper API")
            
            # Extraction options
            col1, col2 = st.columns(2)
            with col1:
                model_size = st.selectbox(
                    "Whisper Model Size",
                    options=["tiny", "base", "small", "medium", "large"],
                    index=3,  # Default to "medium"
                    help="Larger models are more accurate but slower."
                )
                
                language = st.selectbox(
                    "Language",
                    options=["ko", "auto"],
                    index=0,  # Default to "ko" (Korean)
                    help="Choose 'ko' for Korean, or 'auto' for automatic detection."
                )
            
            with col2:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1,
                    help="Higher values will result in more random outputs."
                )
                
                prompt = st.text_area(
                    "Initial Prompt (Optional)",
                    value="",
                    help="Guide the transcription with an initial prompt."
                )
            
            # Extract button
            extract_button = st.button("Start Subtitle Extraction")
            if extract_button:
                with st.spinner("Extracting subtitles... This may take a while depending on video length."):
                    try:
                        from backend.subtitle_handler import extract_subtitles, parse_srt_file
                        processed_dir = "/data/processed"
                        
                        # Call the extraction function
                        srt_path = extract_subtitles(
                            video_path=video_info.active_video_path,
                            output_dir=processed_dir,
                            model_size=model_size,
                            language=language,
                            prompt=prompt,
                            temperature=temperature,
                            verbose=True
                        )
                        
                        if srt_path and os.path.exists(srt_path):
                            # Parse the SRT file
                            segments = parse_srt_file(srt_path)
                            
                            if segments:
                                # Create SubtitleFile object
                                from schemas import SubtitleFile
                                subtitle_file = SubtitleFile(
                                    file_id=video_info.id,
                                    segments=segments,
                                    language=language if language != "auto" else "detected",
                                    extracted_path=srt_path
                                )
                                
                                # Save to session state
                                st.session_state.subtitle_file = subtitle_file
                                
                                st.success(f"Successfully extracted {len(segments)} subtitle segments!")
                                
                                # Display a few segments as preview
                                st.subheader("Subtitle Preview")
                                for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                                    st.text(f"{i+1}. [{segment.format_start_time()} --> {segment.format_end_time()}] {segment.text}")
                                
                                if len(segments) > 5:
                                    st.text(f"... and {len(segments) - 5} more segments.")
                                
                                # Add download button for the SRT file
                                with open(srt_path, "rb") as file:
                                    file_data = file.read()
                                
                                st.download_button(
                                    label="Download SRT File",
                                    data=file_data,
                                    file_name=os.path.basename(srt_path),
                                    mime="text/plain",
                                    key="download_srt_btn"
                                )
                                
                                # Show continue button
                                st.button("Continue to Translation", 
                                         on_click=lambda: setattr(st.session_state, 'current_step', 'translation'))
                            else:
                                st.error("Failed to parse subtitles. No segments found.")
                        else:
                            st.error("Failed to extract subtitles. Please check the logs for details.")
                    
                    except Exception as e:
                        st.error(f"Error during subtitle extraction: {str(e)}")
                        logger.error(f"Subtitle extraction error: {str(e)}")
        
        with subtitle_tab2:
            st.subheader("Upload Subtitle File")
            st.write("If you already have a subtitle file, you can upload it here.")
            
            uploaded_subtitle = st.file_uploader(
                "Upload SRT or TXT file",
                type=["srt", "txt"],
                help="Upload a subtitle file in SRT or TXT format."
            )
            
            if uploaded_subtitle is not None:
                with st.spinner("Processing uploaded subtitle file..."):
                    try:
                        from backend.subtitle_handler import parse_uploaded_subtitle
                        processed_dir = "/data/processed"
                        
                        # Parse the uploaded subtitle
                        segments, saved_path = parse_uploaded_subtitle(uploaded_subtitle, processed_dir)
                        
                        if segments and saved_path:
                            # Create SubtitleFile object
                            from schemas import SubtitleFile
                            subtitle_file = SubtitleFile(
                                file_id=video_info.id,
                                segments=segments,
                                language="manual_upload",  # Placeholder language
                                extracted_path=saved_path
                            )
                            
                            # Save to session state
                            st.session_state.subtitle_file = subtitle_file
                            
                            st.success(f"Successfully processed {len(segments)} subtitle segments!")
                            
                            # Display a few segments as preview
                            st.subheader("Subtitle Preview")
                            for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                                st.text(f"{i+1}. [{segment.format_start_time()} --> {segment.format_end_time()}] {segment.text}")
                            
                            if len(segments) > 5:
                                st.text(f"... and {len(segments) - 5} more segments.")
                            
                            # Show continue button
                            st.button("Continue to Translation", 
                                     on_click=lambda: setattr(st.session_state, 'current_step', 'translation'))
                        else:
                            st.error("Failed to process the uploaded subtitle file.")
                    
                    except Exception as e:
                        st.error(f"Error processing uploaded subtitle: {str(e)}")
                        logger.error(f"Subtitle upload error: {str(e)}")
        
        # Display subtitle editor if subtitles exist
        if st.session_state.subtitle_file:
            st.subheader("Subtitle Editor")
            subtitle_file = st.session_state.subtitle_file
            
            # 수정된 자막 업로드 섹션 추가
            st.write("---")
            st.subheader("Upload Modified Subtitles")
            st.write("Download the subtitles, edit them locally, and upload the modified file here.")
            
            modified_subtitle = st.file_uploader(
                "Upload modified SRT file",
                type=["srt"],
                key="modified_subtitle_uploader",
                help="Upload your modified subtitle file in SRT format."
            )
            
            if modified_subtitle is not None:
                # 이미 처리된 파일인지 확인
                file_name = modified_subtitle.name
                if 'processed_files' not in st.session_state:
                    st.session_state.processed_files = set()
                
                # 이미 처리된 파일이 아닌 경우에만 처리
                if file_name not in st.session_state.processed_files:
                    with st.spinner("Processing modified subtitle file..."):
                        try:
                            from backend.subtitle_handler import parse_uploaded_subtitle
                            processed_dir = "/data/processed"
                            
                            # 업로드된 자막 파일 처리
                            segments, saved_path = parse_uploaded_subtitle(modified_subtitle, processed_dir)
                            
                            if segments and saved_path:
                                # 세션 상태의 자막 파일 업데이트
                                subtitle_file.segments = segments
                                subtitle_file.extracted_path = saved_path
                                st.session_state.subtitle_file = subtitle_file
                                
                                # 이 파일을 처리된 파일 목록에 추가
                                st.session_state.processed_files.add(file_name)
                                
                                st.success(f"Successfully imported {len(segments)} modified subtitle segments!")
                                
                                # 자막이 업데이트되었으면 번역 버튼 표시
                                if st.button("Continue to Translation", key="saved_to_translation_btn2", on_click=lambda: setattr(st.session_state, 'current_step', 'translation')):
                                    pass  # on_click 콜백에서 상태 변경됨
                            else:
                                st.error("Failed to process the uploaded subtitle file.")
                        
                        except Exception as e:
                            st.error(f"Error processing uploaded subtitle: {str(e)}")
                            logger.error(f"Modified subtitle upload error: {str(e)}")
                    
                    # 페이지 새로고침 대신 이미 처리된 상태로 표시
                    # st.rerun() 사용하지 않음
            
            st.write("---")
            
            # Convert subtitle segments to a DataFrame for easier editing
            import pandas as pd
                
            # Create DataFrame from segments
            df_data = []
            for segment in subtitle_file.segments:
                df_data.append({
                    "Index": segment.index,
                    "Start Time": segment.format_start_time(),
                    "End Time": segment.format_end_time(),
                    "Text": segment.text
                })
            
            df = pd.DataFrame(df_data)
            
            # Display the editor with a height limit
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="subtitle_editor",
                height=400
            )
            
            if 'subtitles_updated' not in st.session_state:
                st.session_state.subtitles_updated = False

            # Save Edited Subtitles 버튼
            if st.button("Save Edited Subtitles"):
                try:
                    # Convert edited DataFrame back to segments
                    from schemas import SubtitleSegment
                    from utils.time_converter import time_to_ms
                    
                    updated_segments = []
                    for _, row in edited_df.iterrows():
                        # Parse time strings back to milliseconds
                        # Assuming format_start_time returns "HH:MM:SS,mmm"
                        def parse_time_str(time_str):
                            # Extract hours, minutes, seconds, milliseconds
                            parts = time_str.replace(',', ':').split(':')
                            if len(parts) == 4:  # HH:MM:SS,mmm
                                h, m, s, ms = parts
                                return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
                            return 0  # Default if parsing fails
                        
                        start_ms = parse_time_str(row["Start Time"])
                        end_ms = parse_time_str(row["End Time"])
                        
                        segment = SubtitleSegment(
                            index=row["Index"],
                            start_time_ms=start_ms,
                            end_time_ms=end_ms,
                            text=row["Text"]
                        )
                        updated_segments.append(segment)
                    
                    # Update subtitle_file in session state
                    subtitle_file.segments = updated_segments
                    st.session_state.subtitle_file = subtitle_file
                    
                    # Save to file
                    from backend.subtitle_handler import save_subtitles_to_file
                    processed_dir = "/data/processed"
                    output_path = os.path.join(processed_dir, f"edited_subtitle_{video_info.id}.srt")
                    
                    saved_path = save_subtitles_to_file(
                        segments=updated_segments,
                        output_path=output_path,
                        file_format='srt',
                        translated=False
                    )
                    
                    if saved_path:
                        st.session_state.subtitles_updated = True
                        st.success("Subtitles updated and saved successfully!")
                        
                        # Update the path in subtitle_file
                        subtitle_file.extracted_path = saved_path
                        st.session_state.subtitle_file = subtitle_file
                        
                        # Add download button for the updated SRT file
                        with open(saved_path, "rb") as file:
                            file_data = file.read()
                        
                        st.download_button(
                            label="Download Updated SRT File",
                            data=file_data,
                            file_name=os.path.basename(saved_path),
                            mime="text/plain",
                            key="download_updated_srt_btn"
                        )

                        # 자막이 업데이트되었으면 번역 버튼 표시
                        if st.button("Continue to Translation", key="saved_to_translation_btn2", on_click=lambda: setattr(st.session_state, 'current_step', 'translation')):
                            pass  # on_click 콜백에서 상태 변경됨
                    else:
                        st.error("Failed to save updated subtitles.")
                
                except Exception as e:
                    st.error(f"Error saving edited subtitles: {str(e)}")
                    logger.error(f"Error saving edited subtitles: {str(e)}")

# --- Translation step ---
elif st.session_state.current_step == 'translation':
    st.header("Step 4: Subtitle Translation")
    
    # Check if subtitle_file exists
    if 'subtitle_file' not in st.session_state or st.session_state.subtitle_file is None:
        st.error("자막 파일이 없습니다. 먼저 자막을 추출하거나 업로드해주세요.")
        if st.button("자막 추출 단계로 돌아가기"):
            st.session_state.current_step = 'subtitle'
            st.rerun()
    else:
        subtitle_file = st.session_state.subtitle_file
        
        # 자막 미리보기 표시
        st.subheader("자막 미리보기")
        for i, segment in enumerate(subtitle_file.segments[:5]):  # 처음 5개 세그먼트 표시
            st.text(f"{i+1}. [{segment.format_start_time()} --> {segment.format_end_time()}] {segment.text}")
        
        if len(subtitle_file.segments) > 5:
            st.text(f"... 외 {len(subtitle_file.segments) - 5}개 세그먼트.")
        
        # 번역 옵션
        st.subheader("번역 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox(
                "원본 언어",
                options=["ko"],  # 한국어만 지원
                index=0,  # 기본값 "ko" (한국어)
                format_func=lambda x: f"{x} ({get_language_name(x)})",
                help="자막의 원본 언어를 선택하세요.",
                disabled=True  # 선택 불가능하게 설정
            )
            st.info("현재 버전에서는 한국어 원본만 지원합니다.")
        
        with col2:
            target_lang = st.selectbox(
                "번역 언어",
                options=["en"],
                index=0,  # 기본값 "en" (영어)
                format_func=lambda x: f"{x} ({get_language_name(x)})",
                help="번역할 대상 언어를 선택하세요."
            )
            st.info("현재 버전에서는 영어 번역만 지원합니다.")
            
            # 세션 상태에 번역 언어 저장
            st.session_state.target_language = target_lang
        
        # OpenAI API 키 상태 확인
        if not settings.OPENAI_API_KEY:
            st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        else:
            st.success(f"OpenAI API 키가 설정되어 있습니다: {settings.OPENAI_API_KEY[:5]}...{settings.OPENAI_API_KEY[-5:]}")
        
        # 번역 버튼
        translate_button = st.button("번역 시작", key="start_translation_btn")
        
        # 번역 상태 표시
        if 'is_translating' not in st.session_state:
            st.session_state.is_translating = False
            
        if 'translation_completed' not in st.session_state:
            st.session_state.translation_completed = False
            
        if translate_button:
            # st.rerun() 없이 바로 번역 작업 실행
            with st.spinner(f"{get_language_name(source_lang)}에서 {get_language_name(target_lang)}으로 번역 중... 잠시만 기다려주세요."):
                try:
                    from backend.translation_handler import batch_translate_with_retry
                    
                    # 번역 함수 직접 호출
                    translated_segments = batch_translate_with_retry(
                        segments=subtitle_file.segments,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        max_retries=3
                    )
                    
                    # 번역 결과 저장
                    subtitle_file.segments = translated_segments
                    st.session_state.subtitle_file = subtitle_file
                    st.session_state.translation_completed = True
                    
                    # 번역 완료 메시지
                    st.success(f"자막 번역이 완료되었습니다! {len(translated_segments)}개의 세그먼트가 {get_language_name(target_lang)}으로 번역되었습니다.")
                except Exception as e:
                    st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
                    logger.error(f"Translation error: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 번역 결과 표시 (번역이 완료된 경우)
        if st.session_state.translation_completed:
            st.subheader("번역 결과")
            
            # 작업 선택 탭 생성
            translate_tabs = st.tabs(["번역 결과 확인", "번역 자막 다운로드", "번역 자막 업로드"])
            
            with translate_tabs[0]:  # 번역 결과 확인 탭
                # 결과 표시를 위한 데이터프레임 생성
                import pandas as pd
                
                df_data = []
                for segment in subtitle_file.segments[:10]:  # 처음 10개만 표시
                    df_data.append({
                        "시간": f"{segment.format_start_time()} --> {segment.format_end_time()}",
                        "원본 텍스트": segment.text,
                        "번역 텍스트": segment.translated_text if hasattr(segment, 'translated_text') else ""
                    })
                
                # 데이터프레임으로 변환 및 표시
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                if len(subtitle_file.segments) > 10:
                    st.text(f"... 외 {len(subtitle_file.segments) - 10}개 세그먼트.")
            
            with translate_tabs[1]:  # 번역 자막 다운로드 탭
                # 번역된 자막 다운로드 버튼
                from backend.subtitle_handler import save_subtitles_to_file
                
                st.write("번역된 자막 파일을 다운로드하여 원하는 대로 편집할 수 있습니다.")
                
                try:
                    processed_dir = "/data/processed"
                    output_path = os.path.join(processed_dir, f"translated_{target_lang}_{subtitle_file.file_id}.srt")
                    
                    saved_path = save_subtitles_to_file(
                        segments=subtitle_file.segments,
                        output_path=output_path,
                        file_format='srt',
                        translated=True
                    )
                    
                    if saved_path:
                        with open(saved_path, "rb") as file:
                            file_data = file.read()
                        
                        st.download_button(
                            label=f"{get_language_name(target_lang)} 자막 다운로드",
                            data=file_data,
                            file_name=os.path.basename(saved_path),
                            mime="text/plain",
                            key="download_translated_srt_btn"
                        )
                    else:
                        st.error("번역된 자막 파일 저장에 실패했습니다.")
                
                except Exception as e:
                    st.error(f"번역된 자막 다운로드 중 오류: {str(e)}")
            
            with translate_tabs[2]:  # 번역 자막 업로드 탭
                st.write("자체적으로 편집한 번역 자막을 업로드할 수 있습니다.")
                st.write("참고: 업로드한 자막은 기존 번역을 대체합니다.")
                
                uploaded_translation = st.file_uploader(
                    "편집된 번역 자막 업로드 (SRT 형식)",
                    type=["srt"],
                    key="translated_subtitle_uploader",
                    help="외부에서 편집한 번역 자막 파일을 업로드하세요. 원본 자막과 동일한 세그먼트 구조를 유지해야 합니다."
                )
                
                if uploaded_translation is not None:
                    with st.spinner("번역 자막 파일 처리 중..."):
                        try:
                            from backend.subtitle_handler import parse_uploaded_subtitle
                            processed_dir = "/data/processed"
                            
                            # 업로드된 자막 파일 처리
                            uploaded_segments, saved_path = parse_uploaded_subtitle(uploaded_translation, processed_dir)
                            
                            if uploaded_segments and saved_path:
                                # 원본 세그먼트 수와 업로드된 세그먼트 수 비교
                                if len(uploaded_segments) != len(subtitle_file.segments):
                                    st.warning(f"업로드된 자막의 세그먼트 수({len(uploaded_segments)})가 원본 자막의 세그먼트 수({len(subtitle_file.segments)})와 다릅니다. 일부 자막이 누락되거나 추가되었을 수 있습니다.")
                                
                                # 각 세그먼트의 번역 텍스트를 업데이트
                                for i, segment in enumerate(subtitle_file.segments):
                                    if i < len(uploaded_segments):
                                        segment.translated_text = uploaded_segments[i].text
                                
                                # 세션 상태 업데이트
                                st.session_state.subtitle_file = subtitle_file
                                
                                st.success(f"번역 자막이 성공적으로 업로드되었습니다! {len(uploaded_segments)}개의 세그먼트가 업데이트되었습니다.")
                                
                                # 업데이트된 자막 미리보기
                                st.subheader("업데이트된 번역 자막 미리보기")
                                for i, segment in enumerate(subtitle_file.segments[:5]):
                                    st.text(f"{i+1}. [{segment.format_start_time()} --> {segment.format_end_time()}] {segment.translated_text if hasattr(segment, 'translated_text') else ''}")
                                
                                if len(subtitle_file.segments) > 5:
                                    st.text(f"... 외 {len(subtitle_file.segments) - 5}개 세그먼트.")
                            else:
                                st.error("번역 자막 파일 처리에 실패했습니다.")
                        
                        except Exception as e:
                            st.error(f"번역 자막 업로드 중 오류: {str(e)}")
                            logger.error(f"Translated subtitle upload error: {str(e)}")
            
            # 다음 단계 버튼 (향후 TTS 기능으로 연결)
            if st.button("TTS 단계로 계속", key="to_tts_btn"):
                st.session_state.current_step = 'tts'
                st.rerun()

# --- TTS step ---
elif st.session_state.current_step == 'tts':
    st.header("Step 5: TTS (Text-to-Speech)")
    
    # 필요한 상태 및 데이터 확인
    if st.session_state.get("subtitle_file") is None or not hasattr(st.session_state, "translation_completed"):
        st.error("번역된 자막이 준비되지 않았습니다. 번역 단계를 먼저 완료해주세요.")
        if st.button("번역 단계로 돌아가기"):
            st.session_state.current_step = 'translate'
            st.rerun()
    else:
        # 번역된 자막 정보 가져오기
        subtitle_file = st.session_state.subtitle_file
        target_lang = st.session_state.get("target_language", "en")  # 세션 상태에 없으면 기본값 "en" 사용
        
        st.write("번역된 자막을 음성으로 변환합니다. 선택한 언어에 적합한 TTS 모델이 자동으로 선택됩니다.")
        
        # TTS 언어 지원 상태 메시지 수정
        st.info("현재 TTS 기능은 영어 번역만 지원합니다.")
        
        with st.expander("🔊 TTS 설정", expanded=True):
            st.markdown("""
            ### 지원 언어
            - 영어(en): 안정적 지원
            """)

        # TTS 언어 재선택 옵션
        original_lang = st.session_state.get('original_tts_language', target_lang)
        
        # 변경된 언어 정보 표시
        st.info(f"현재 선택된 번역 언어: {target_lang} ({get_language_name(target_lang)})")
        
        # 언어 재선택 옵션 추가 - 영어만 선택 가능하므로 선택상자 대신 단순 정보 표시
        st.info("TTS는 영어 언어만 지원합니다.")
        target_lang = "en"
        st.session_state.target_language = target_lang
        
        # 초기 성별 값 설정 (위젯 생성 전에 설정해야 함)
        if "tts_gender_value" not in st.session_state:
            st.session_state.tts_gender_value = "female"
        
        # 음성 성별 선택 - 여성만 지원하므로 라디오 버튼 대신 정보 표시
        st.info("현재 버전에서는 여성 음성만 지원합니다.")
        selected_gender = "female"  # 항상 여성 음성으로 설정
        
        # TTS 모델 정보 표시
        from backend.tts_handler import get_tts_model_for_language
        
        model_name = get_tts_model_for_language(target_lang, selected_gender)
        st.info(f"선택된 TTS 모델: {model_name}")
        
        # 일부 자막 미리보기 표시
        st.subheader("번역된 자막 샘플")
        sample_size = min(5, len(subtitle_file.segments))
        for i, segment in enumerate(subtitle_file.segments[:sample_size]):
            st.text(f"{i+1}. [{segment.format_start_time()} --> {segment.format_end_time()}] {segment.translated_text if hasattr(segment, 'translated_text') and segment.translated_text else '(번역 없음)'}")
            
        if len(subtitle_file.segments) > sample_size:
            st.text(f"... 외 {len(subtitle_file.segments) - sample_size}개 세그먼트.")
        
        # TTS 처리 버튼
        generate_btn = st.button("TTS 생성", key="generate_tts_btn")
        
        if generate_btn:
            # 세션 상태에 이미 TTS 결과가 있는지 확인
            if st.session_state.get("tts_audio_files") is not None:
                st.warning("이미 TTS가 생성되었습니다. 새로 생성하면 기존 결과를 덮어씁니다.")
            
            # TTS 생성 진행
            with st.spinner("TTS 생성 중... 이 작업은 다소 시간이 걸릴 수 있습니다."):
                try:
                    from backend.tts_handler import generate_tts_audio, merge_subtitle_audio_files
                    
                    # 세그먼트에 번역된 텍스트가 있는지 확인
                    segments_with_translation = [
                        segment for segment in subtitle_file.segments 
                        if hasattr(segment, 'translated_text') and segment.translated_text
                    ]
                    
                    if not segments_with_translation:
                        st.error("번역된 텍스트가 없습니다. 번역 단계를 다시 확인해주세요.")
                    else:
                        # 출력 디렉토리 설정
                        output_dir = "/data/processed"
                        
                        # TTS 생성
                        st.text(f"{len(segments_with_translation)}개 세그먼트에 대해 TTS를 생성합니다...")
                        
                        # 번역된 세그먼트에 대해 TTS 생성
                        audio_files = generate_tts_audio(
                            subtitle_segments=segments_with_translation,
                            language_code=target_lang,
                            output_dir=output_dir,
                            gender=selected_gender,
                            file_id=subtitle_file.file_id
                        )
                        
                        if audio_files:
                            # 세션 상태에 저장
                            st.session_state.tts_audio_files = audio_files
                            st.session_state.tts_language = target_lang
                            st.session_state.tts_gender_value = selected_gender
                            st.session_state.tts_completed = True
                            
                            # 합쳐진 오디오 파일 생성
                            merged_audio_path = merge_subtitle_audio_files(
                                audio_files=audio_files,
                                segments=segments_with_translation,
                                output_dir=output_dir,
                                file_id=subtitle_file.file_id
                            )
                            
                            if merged_audio_path:
                                st.session_state.tts_merged_audio = merged_audio_path
                                st.success(f"TTS 생성 완료! {len(audio_files)}개 세그먼트에 대한 음성이 생성되었습니다.")
                                
                                # TTS 폴백 사용 여부 확인 및 알림
                                if st.session_state.get("used_fallback_tts", False):
                                    st.warning(f"선택한 {get_language_name(target_lang)} TTS 모델 대신 영어 TTS 모델을 사용했습니다.")
                                
                                # 생성된 오디오 표시
                                st.subheader("생성된 오디오 미리보기")
                                st.audio(merged_audio_path)
                                
                                # 다음 단계 버튼
                                if st.button("결과 단계로 계속", key="to_result_btn"):
                                    st.session_state.current_step = 'result'
                                    st.rerun()
                            else:
                                st.error("오디오 파일 합치기에 실패했습니다.")
                        else:
                            st.error("TTS 생성에 실패했습니다.")
                
                except Exception as e:
                    st.error(f"TTS 생성 중 오류가 발생했습니다: {str(e)}")
                    logger.error(f"TTS generation error: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 이미 TTS가 생성된 경우
        elif st.session_state.get("tts_completed") and st.session_state.get("tts_merged_audio"):
            st.success("TTS가 이미 생성되었습니다.")
            
            # 생성된 오디오 표시
            st.subheader("생성된 오디오 미리보기")
            st.audio(st.session_state.tts_merged_audio)
            
            # 다음 단계 버튼
            if st.button("결과 단계로 계속", key="to_result_btn2"):
                st.session_state.current_step = 'result'
                st.rerun()

elif st.session_state.current_step == 'result':
    st.header("Step 6: 최종 결과")
    st.write("이 단계에서는 번역된 자막과 생성된 음성이 합성된 최종 비디오를 생성합니다.")
    
    # 필요한 항목 확인
    video_info = st.session_state.get("video_info")
    subtitle_file = st.session_state.get("subtitle_file")
    translated_subtitles = st.session_state.get("translated_subtitles", {})
    tts_merged_audio = st.session_state.get("tts_merged_audio")
    target_lang = st.session_state.get("tts_language")
    
    if not video_info or not tts_merged_audio:
        st.error("비디오 정보나 TTS 오디오 파일이 없습니다. 이전 단계를 완료해주세요.")
    else:
        # 자막 선택 옵션
        with st.expander("자막 옵션", expanded=True):
            include_subtitles = st.checkbox("최종 비디오에 자막 포함하기", value=True)
            
            subtitle_path = None
            if include_subtitles and subtitle_file and subtitle_file.segments:
                subtitle_segments = [seg for seg in subtitle_file.segments if hasattr(seg, 'translated_text') and seg.translated_text]
                
                if subtitle_segments:
                    from backend.subtitle_handler import save_subtitles_to_file
                    
                    output_dir = "/data/processed"
                    os.makedirs(output_dir, exist_ok=True)
                    file_id = subtitle_file.file_id
                    subtitle_path = os.path.join(output_dir, f"final_subtitle_{file_id}_{target_lang}.srt")
                    
                    save_subtitles_to_file(
                        segments=subtitle_segments,
                        output_path=subtitle_path,
                        file_format="srt",
                        translated=True
                    )
                    
                    if os.path.exists(subtitle_path):
                        st.info(f"번역된 자막을 사용합니다: {os.path.basename(subtitle_path)}")
                    else:
                        st.warning("자막 파일 생성에 실패했습니다. 자막 없이 진행합니다.")
                        subtitle_path = None
                else:
                    st.warning("번역된 자막 정보가 없습니다. 자막 없이 진행합니다.")
                    subtitle_path = None
            elif include_subtitles:
                st.warning(f"선택된 언어({target_lang})의 번역된 자막이 없습니다. 자막 없이 진행합니다.")
        
        # 사용할 비디오 선택 (잘린 비디오가 있으면 그것을 사용, 아니면 원본)
        video_path = video_info.active_video_path
        
        # 최종 비디오 생성 섹션
        st.subheader("최종 비디오 생성")
        
        if not st.session_state.get("final_video_completed"):
            # 비디오 생성 버튼
            if st.button("최종 비디오 생성 시작", key="generate_final_video_btn"):
                with st.spinner("최종 비디오를 생성 중입니다... 잠시만 기다려주세요."):
                    try:
                        from backend.video_processor import combine_video
                        
                        # 최종 비디오 파일 경로를 저장할 output_dir
                        output_dir = "/data/results"
                        
                        # combine_video 함수 호출
                        final_video_path = combine_video(
                            video_path=video_path,
                            audio_path=tts_merged_audio,
                            subtitle_path=subtitle_path if include_subtitles else None,
                            output_dir=output_dir,
                            target_language=target_lang
                        )
                        
                        if final_video_path:
                            st.session_state.final_video_path = final_video_path
                            st.session_state.final_video_completed = True
                            st.success("최종 비디오 생성 완료!")
                            st.rerun()
                        else:
                            st.error("최종 비디오 생성에 실패했습니다.")
                    
                    except Exception as e:
                        st.error(f"비디오 합성 중 오류가 발생했습니다: {str(e)}")
                        logger.error(f"Video combination error: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
        
        # 이미 최종 비디오가 생성된 경우
        if st.session_state.get("final_video_completed") and st.session_state.get("final_video_path"):
            final_video_path = st.session_state.final_video_path
            
            # 최종 비디오 정보 표시
            st.success("최종 비디오가 성공적으로 생성되었습니다!")
            
            # 비디오 미리보기
            st.subheader("최종 비디오 미리보기")
            st.video(final_video_path)
            
            # 다운로드 버튼
            with open(final_video_path, "rb") as file:
                video_bytes = file.read()
                filename = os.path.basename(final_video_path)
                st.download_button(
                    label="최종 비디오 다운로드",
                    data=video_bytes,
                    file_name=filename,
                    mime="video/mp4"
                )
    
    # 이전 단계로 돌아가는 버튼
    if st.button("TTS 단계로 돌아가기"):
        st.session_state.current_step = 'tts'
        st.rerun()
    
    # 처음으로 돌아가는 버튼
    if st.button("처음으로 돌아가기"):
        st.session_state.current_step = 'upload'
        st.session_state.video_info = None
        # 모든 세션 상태 초기화
        for key in list(st.session_state.keys()):
            if key not in ['current_step']:
                del st.session_state[key]
        st.rerun()

else:
    st.header(f"Step {steps.index(step_map.get(st.session_state.current_step, 'Upload')) + 1}: {step_map.get(st.session_state.current_step, 'Upload')}")
    st.write("This feature will be implemented in subsequent phases.")
    
    if st.button("Go back to Upload"):
        st.session_state.current_step = 'upload'
        st.session_state.video_info = None
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Video Translator and Dubbing Tool V1 - Phase 2")

# --- Log app completion ---
logger.info("Streamlit application main script execution finished.")