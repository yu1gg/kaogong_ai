"""Media processing helpers for audio/video interview submissions."""

import logging
from pathlib import Path
import subprocess
import tempfile

from app.core.config import settings
from app.models.schemas import MediaExtractionResult
from app.services.media.audio_transcriber import get_transcriber

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract a mono 16k WAV track from a video via ffmpeg."""

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return output_audio_path
    except FileNotFoundError as exc:
        raise RuntimeError("系统未安装 ffmpeg，无法处理视频文件。") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("音频提取失败，请检查视频文件格式或完整性。") from exc


def analyze_facial_behavior(video_path: str) -> str:
    """Run a lightweight face-stability analysis on the video."""

    if not settings.ENABLE_VISUAL_ANALYSIS:
        return "已关闭视觉分析。"

    try:
        import cv2
    except ImportError:
        logger.warning("未安装 OpenCV，跳过视觉分析。")
        return "视觉分析依赖未安装，已跳过。"

    logger.info("正在执行视觉行为分析: %s", video_path)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV 无法打开视频流: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 25.0

    frame_step = max(1, int(fps / 2))
    frames_processed = 0
    faces_detected = 0
    head_movements = 0
    last_center_y = None

    try:
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
                minSize=(30, 30),
            )

            if len(faces) == 0:
                continue

            faces_detected += 1
            x, y, w, h = faces[0]
            frame_height = frame.shape[0]
            center_y = (y + h / 2.0) / frame_height

            if last_center_y is not None and abs(center_y - last_center_y) > 0.05:
                head_movements += 1
            last_center_y = center_y
    finally:
        cap.release()

    if faces_detected == 0:
        return "视频中未检测到清晰人脸，无法稳定评估仪态。"

    movement_ratio = head_movements / faces_detected
    if movement_ratio > 0.3:
        return "考生头部晃动较多，可能存在紧张表现。"
    if movement_ratio < 0.05:
        return "考生仪态较稳定，整体表现较为从容。"
    return "考生肢体语言整体自然，无明显异常。"


def process_video(video_path: str) -> MediaExtractionResult:
    """Extract transcript and optional visual observation from a video file."""

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        extract_audio(video_path, temp_audio_path)
        transcriber = get_transcriber()
        transcript = transcriber.transcribe(temp_audio_path).strip()
        if not transcript:
            transcript = "（未能识别到任何有效的人声作答）"

        visual_observation = analyze_facial_behavior(video_path)
        return MediaExtractionResult(
            transcript=transcript,
            source="video",
            visual_observation=visual_observation,
        )
    finally:
        if temp_audio_path:
            temp_audio = Path(temp_audio_path)
            if temp_audio.exists():
                temp_audio.unlink(missing_ok=True)


def process_audio(audio_path: str) -> MediaExtractionResult:
    """Transcribe an audio-only submission."""

    transcriber = get_transcriber()
    transcript = transcriber.transcribe(audio_path).strip()
    if not transcript:
        transcript = "（未能识别到任何有效的人声作答）"

    return MediaExtractionResult(
        transcript=transcript,
        source="audio",
        visual_observation=None,
    )
