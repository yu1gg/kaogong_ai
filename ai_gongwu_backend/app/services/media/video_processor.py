"""
视频与视觉特征处理模块。
集成 FFmpeg 音频剥离、OpenCV 原生面部特征分析与 Whisper 真实语音转录。
"""
import subprocess
import logging
import cv2
from pathlib import Path
from typing import Optional

# 核心导入：唤醒 Whisper 引擎的工厂函数
from app.services.media.audio_transcriber import get_transcriber

logger = logging.getLogger(__name__)

def extract_audio(video_path: str, output_audio_path: Optional[str] = None) -> str:
    """
    通过系统底层的 FFmpeg 将视频中的音频无损剥离为 WAV 格式。
    """
    if not output_audio_path:
        output_audio_path = str(Path(video_path).with_suffix('.wav'))
        
    logger.info(f"正在使用 FFmpeg 从 {video_path} 提取音频...")
    
    command = [
        'ffmpeg', '-y', '-i', video_path, 
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
        output_audio_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"音频提取成功: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 音频提取失败: {str(e)}")
        raise RuntimeError("音频提取失败，请检查视频文件是否损坏或格式不支持。")

def analyze_facial_behavior(video_path: str) -> str:
    """
    使用 OpenCV 原生 Haar 级联分类器进行面部行为分析。
    通过追踪面部边界框的中心点位移，计算考生的头部稳定性。
    """
    logger.info("正在启动 OpenCV 原生视觉特征检测引擎...")
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV 无法打开视频流: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: 
        fps = 25.0
        
    frame_step = max(1, int(fps / 2)) 
    
    frames_processed = 0
    faces_detected = 0
    head_movements = 0
    last_center_y = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_processed += 1
        
        if frames_processed % frame_step != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        if len(faces) > 0:
            faces_detected += 1
            x, y, w, h = faces[0]
            
            frame_height = frame.shape[0]
            center_y = (y + h / 2.0) / frame_height
            
            if last_center_y is not None:
                if abs(center_y - last_center_y) > 0.05: 
                    head_movements += 1
            last_center_y = center_y

    cap.release()
    
    if faces_detected == 0:
        return "【视觉检测警告】视频中未检测到清晰人脸，存在替考或光线极差风险。"
        
    movement_ratio = head_movements / faces_detected
    if movement_ratio > 0.3:
        behavior_desc = "考生头部晃动频繁，肢体语言表现较为紧张不安。"
    elif movement_ratio < 0.05:
        behavior_desc = "考生面部仪态端庄，姿态极佳，表现出极强的自信与专注力。"
    else:
        behavior_desc = "考生肢体语言自然，无明显异常小动作。"

    logger.info(f"视觉分析完成: {behavior_desc}")
    return f"【多模态视觉检测报告】: {behavior_desc}"


def process_video(video_path: str) -> str:
    """处理视频文件的主入口：视觉分析 + 真实 ASR 转录"""
    # 1. 真实剥离音频
    audio_file = extract_audio(video_path)
    
    # 2. 真实视觉分析 (OpenCV)
    visual_report = analyze_facial_behavior(video_path)
    
    # 3. 真实 ASR 转录 (Whisper)
    logger.info("正在唤醒 ASR 引擎转录音频内容...")
    transcriber = get_transcriber()
    asr_text = transcriber.transcribe(audio_file)
    
    # 如果考生一句话都没说，做个容错兜底
    if not asr_text.strip():
        asr_text = "（未能识别到任何有效的人声作答）"
    
    # 4. 合成多模态 Prompt 上下文
    multimodal_context = f"{visual_report}\n\n【考生答题语音转录】:\n{asr_text}"
    return multimodal_context


def process_audio(audio_path: str) -> str:
    """仅处理音频文件"""
    logger.info("仅音频模式：正在唤醒 ASR 引擎转录...")
    transcriber = get_transcriber()
    asr_text = transcriber.transcribe(audio_path)
    
    if not asr_text.strip():
        asr_text = "（未能识别到任何有效的人声作答）"
        
    return f"【多模态视觉检测报告】: 未提供视频流，无法评估肢体语言。\n\n【考生答题语音转录】:\n{asr_text}"