"""
video_utils.py

Summary:
- Reads video metadata using OpenCV.
- Interactively asks whether to change size / fps.
- Detects ffmpeg. If available, uses it for efficient scaling and streaming (rawvideo bgr24).
- Falls back to cv2 if ffmpeg is missing.
- Streams frames to user_callback without saving intermediate files.
- Uses ThreadPoolExecutor for parallel frame processing.
"""

import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Generator, Optional, Tuple, TypedDict

import cv2
import numpy as np

from .ffmpeg_utils import find_ffmpeg, verify_ffmpeg


class VideoMetadata(TypedDict):
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: Optional[float]


# Processing Callback Types
FrameData = cv2.typing.MatLike  # alias for clarity
FrameIndex = int
TimestampSec = float
FrameInfo = Tuple[FrameData, FrameIndex, TimestampSec]


# ---------------------------
# Helpers: metadata
# ---------------------------
def get_metadata_cv2(path: str) -> VideoMetadata:
    """Lazy metadata read via cv2."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = None
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    cap.release()
    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


def resolve_resolution(
    target_w: Optional[int], target_h: Optional[int], src_w: int, src_h: int
) -> Tuple[int, int]:
    """
    解析目標解析度，處理 -1 (自動比例) 的情況。
    確保回傳的是具體的整數 (w, h)。
    """
    # change None to -1 for easier handling
    target_w = target_w or -1
    target_h = target_h or -1

    # 如果兩個都是 -1，保持原樣
    if target_w <= 0 and target_h <= 0:
        return src_w, src_h

    final_w = target_w
    final_h = target_h

    # 情況 A: 指定寬度，高度自動 (-1)
    if target_w > 0 and target_h <= 0:
        ratio = src_h / src_w
        final_h = int(target_w * ratio)

    # 情況 B: 指定高度，寬度自動 (-1)
    elif target_h > 0 and target_w <= 0:
        ratio = src_w / src_h
        final_w = int(target_h * ratio)

    # 情況 C: 兩者都指定，直接使用 (可能變形)
    else:
        final_w = target_w
        final_h = target_h

    return int(final_w), int(final_h)


# ---------------------------
# Frame producers
# ---------------------------
def extract_with_ffmpeg_pipe(
    video_path: str,
    ffmpeg_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    target_fps: Optional[float] = None,
) -> Generator[FrameInfo, Any, None]:
    """
    Stream raw BGR24 frames via ffmpeg stdout pipe.
    Yields (frame_numpy_bgr, frame_index, timestamp_sec).
    """
    src_meta = get_metadata_cv2(video_path)
    src_w = src_meta["width"]
    src_h = src_meta["height"]
    src_fps = src_meta["fps"]

    tw = src_w  # target width
    th = src_h  # target height
    tf = src_fps  # target fps

    vf_args: list[str] = []
    if target_fps and target_fps != src_fps:
        vf_args.append(f"fps={target_fps}")
        tf = target_fps
    if target_size and target_size != (src_w, src_h):
        tw, th = target_size
        vf_args.append(f"scale={tw}:{th}")
    vf = ["-vf", ",".join(vf_args)] if vf_args else []

    # ffmpeg supports bgr24 pixel format which matches OpenCV's default.
    cmd = [
        # fmt: off
        ffmpeg_path,
        "-i", video_path,
        *vf,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-",
        # fmt: on
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Calculate frame size in bytes for reading from pipe
    frame_byte_size = tw * th * 3
    frame_idx = 0
    assert proc.stdout is not None, "ffmpeg process stdout is None"
    try:
        while True:
            in_bytes = proc.stdout.read(frame_byte_size)
            if not in_bytes or len(in_bytes) < frame_byte_size:
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape((th, tw, 3))
            timestamp = frame_idx / tf
            yield frame, frame_idx, timestamp
            frame_idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.kill()
        proc.wait(timeout=1)


def extract_with_cv2_time_based(
    video_path: str, target_size: Optional[Tuple[int, int]] = None, target_fps: Optional[float] = None
) -> Generator[FrameInfo, Any, None]:
    """
    Time-based frame extraction using cv2 to prevent drift.
    Yields (frame_bgr, index, timestamp)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open video for reading frames.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = None
    if src_fps > 0 and frame_count > 0:
        duration = frame_count / src_fps

    if target_fps is None:
        target_fps = src_fps

    if duration is None:
        print("Warning: source duration unknown; falling back to full scan sampling.")
        # Fallback: iterate all frames and sample by index accumulation
        frame_interval = src_fps / target_fps if src_fps > 0 and target_fps > 0 else 1.0
        idx = 0
        next_idx = 0.0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx >= next_idx:
                if target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                timestamp = saved / target_fps
                yield frame, saved, timestamp
                saved += 1
                next_idx += frame_interval
            idx += 1
        cap.release()
        return

    total_out_frames = int(math.floor(duration * target_fps + 1e-6))
    for i in range(total_out_frames):
        t = i / target_fps  # seconds
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            # Seeking to exact time might fail due to codec granularity; try next frame.
            ret2, frame2 = cap.read()
            if not ret2:
                break
            frame = frame2
        if target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        yield frame, i, t
    cap.release()


# ---------------------------
# Orchestrator
# ---------------------------
def process_frames_from_video(
    video_path: str,
    output_size: Optional[Tuple[int, int]],
    output_fps: Optional[float],
    callback: Callable[[FrameData, FrameIndex, TimestampSec], None],
    max_workers: int = 8,
    prefer_ffmpeg: bool = True,
    ffmpeg_exec_path: Optional[str] = None,
):
    """
    Streams frames (resized & resampled) and dispatches them to callback in threads.
    Does NOT store output video; frames are processed on the fly.
    """
    meta = get_metadata_cv2(video_path)
    src_w, src_h, src_fps = meta["width"], meta["height"], meta["fps"]
    frame_count = meta["frame_count"]
    duration = meta["duration"]
    print(
        f"[metadata] source size={src_w}x{src_h}, fps={src_fps}, frames={frame_count or 'unknown'}, duration={f'{duration}s' if duration else 'unknown'}"
    )

    target_size = output_size or (src_w, src_h)
    target_fps = output_fps or src_fps
    target_frame_count = None
    if duration is not None:
        target_frame_count = int(math.floor(duration * target_fps + 1e-6))
    print(
        f"[processing] target size={target_size[0]}x{target_size[1]}, fps={target_fps}, frames_approx={target_frame_count or 'unknown'}"
    )
    meta["width"] = target_size[0]
    meta["height"] = target_size[1]
    meta["fps"] = target_fps

    ffmpeg_available = False
    if prefer_ffmpeg:
        if not ffmpeg_exec_path:
            ffmpeg_exec_path = find_ffmpeg()
        if ffmpeg_exec_path:
            ffmpeg_available = verify_ffmpeg(ffmpeg_exec_path)
            ffmpeg_available = True

    if ffmpeg_exec_path and ffmpeg_available:
        print(f"[ffmpeg] using ffmpeg at: {ffmpeg_exec_path}")
        frames = extract_with_ffmpeg_pipe(video_path, ffmpeg_exec_path, output_size, output_fps)
        via = "ffmpeg -> pipe"
    else:
        if not ffmpeg_available:
            print("[ffmpeg] ffmpeg not found or invalid; fallback to cv2 pipeline.")
        else:
            print("[ffmpeg] not using ffmpeg; fallback to cv2 pipeline.")
        frames = extract_with_cv2_time_based(video_path, output_size, output_fps)
        via = "cv2 time-based"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        frame_count = 0
        for frame, idx, ts in frames:
            futures.append(executor.submit(callback, frame, idx, ts))
            frame_count += 1

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[worker] exception in callback: {e}", file=sys.stderr)
    print(f"[done] processed {frame_count} frames via {via}.")

    meta["frame_count"] = frame_count
    meta["duration"] = frame_count / target_fps
    return meta


# ---------------------------
# Example callback
# ---------------------------
def example_callback(frame: FrameData, index: FrameIndex, timestamp: TimestampSec):
    """
    Example callback: show minimal per-frame stats.
    Replace this with your pixel-level processing function.
    NOTE: frame is BGR numpy array (height, width, 3)
    """
    print(
        f"[frame {index}] timestamp={timestamp:.3f}s, shape={frame.shape}, mean_color={frame.mean(axis=(0, 1))}"
    )


# ---------------------------
# If executed as script
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_utils.py /path/to/video.mp4")
        sys.exit(1)
    video_file = sys.argv[1]
    if not os.path.exists("frame"):
        os.makedirs("frame")
    # call CLI with example callback; replace example_callback with your function
    meta = process_frames_from_video(
        video_file, output_size=None, output_fps=None, callback=example_callback, max_workers=4
    )
    print(f"[finish] processed video metadata: {meta}")
