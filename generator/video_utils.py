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

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Generator, Optional, Tuple, TypedDict

import cv2
import numpy as np


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
# Helpers: metadata & ffmpeg
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


def find_ffmpeg() -> Optional[str]:
    """Return path to ffmpeg if found on PATH."""
    path = shutil.which("ffmpeg")
    return path


def verify_ffmpeg(ffmpeg_exec: str) -> bool:
    """Check if path points to a valid ffmpeg executable."""
    try:
        proc = subprocess.run(
            [ffmpeg_exec, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )
        return proc.returncode == 0 and "ffmpeg version" in proc.stdout.lower()
    except Exception:
        return False


# ---------------------------
# Frame producers
# ---------------------------
def extract_with_ffmpeg_pipe(
    video_path: str, target_fps: float, target_size: Tuple[int, int]
) -> Generator[FrameInfo, Any, None]:
    """
    Stream raw BGR24 frames via ffmpeg stdout pipe.
    Yields (frame_numpy_bgr, frame_index, timestamp_sec).
    """
    tw, th = target_size or (0, 0)
    scale_str = f"scale={tw}:{th}" if (tw > 0 and th > 0) else "scale=iw:ih"
    vf = f"fps={target_fps},{scale_str}"

    # ffmpeg supports bgr24 pixel format which matches OpenCV's default.
    cmd = [
        # fmt: off
        "ffmpeg",
        "-i", video_path,
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-",
        # fmt: on
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Calculate frame size in bytes for reading from pipe
    frame_byte_size = target_size[0] * target_size[1] * 3
    frame_idx = 0
    assert proc.stdout is not None, "ffmpeg process stdout is None"
    try:
        while True:
            in_bytes = proc.stdout.read(frame_byte_size)
            if not in_bytes or len(in_bytes) < frame_byte_size:
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape((target_size[1], target_size[0], 3))
            timestamp = frame_idx / target_fps
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
    video_path: str, target_fps: float, target_size: Optional[Tuple[int, int]] = None
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
    output_fps: float,
    processing_callback: Callable[[FrameData, FrameIndex, TimestampSec], None],
    finish_callback: Optional[Callable[[VideoMetadata], None]] = None,
    max_workers: int = 8,
    prefer_ffmpeg: bool = True,
    ffmpeg_exec_path: Optional[str] = None,
):
    """
    Streams frames (resized & resampled) and dispatches them to user_callback in threads.
    Does NOT store output video; frames are processed on the fly.
    """
    meta = get_metadata_cv2(video_path)
    src_w, src_h, src_fps = meta["width"], meta["height"], meta["fps"]
    frame_count = meta["frame_count"]
    duration = meta["duration"]
    print(
        f"[metadata] source size={src_w}x{src_h}, fps={src_fps}, frames={frame_count or 'unknown'}, duration={f'{duration}s' if duration else 'unknown'}"
    )

    if output_size is None:
        target_size = (src_w, src_h)
    else:
        target_size = output_size

    target_frame_count = None
    if duration is not None:
        target_frame_count = int(math.floor(duration * output_fps + 1e-6))
    print(
        f"[processing] target size={target_size[0]}x{target_size[1]}, fps={output_fps}, frames={target_frame_count or 'unknown'}"
    )
    meta["width"] = target_size[0]
    meta["height"] = target_size[1]
    meta["fps"] = output_fps

    ffmpeg_available = False
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        ffmpeg_available = True
    if ffmpeg_exec_path:
        if verify_ffmpeg(ffmpeg_exec_path):
            ffmpeg_available = True
            ffmpeg_path = ffmpeg_exec_path
        else:
            print(f"[ffmpeg] provided path '{ffmpeg_exec_path}' is not a valid ffmpeg executable; ignoring.")

    use_ffmpeg = prefer_ffmpeg and ffmpeg_available
    if use_ffmpeg:
        print(f"[ffmpeg] using ffmpeg at: {ffmpeg_path}")
    else:
        print("[ffmpeg] not using ffmpeg; fallback to cv2 pipeline.")

    if use_ffmpeg:
        frames = extract_with_ffmpeg_pipe(
            video_path=video_path, target_fps=output_fps, target_size=target_size
        )
        via = "ffmpeg -> pipe"
    else:
        frames = extract_with_cv2_time_based(video_path, output_fps, output_size)
        via = "cv2 time-based"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        frame_count = 0
        for frame, idx, ts in frames:
            futures.append(executor.submit(processing_callback, frame, idx, ts))
            frame_count += 1

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[worker] exception in callback: {e}", file=sys.stderr)
    print(f"[done] processed {frame_count} frames via {via}.")

    if finish_callback:
        meta["frame_count"] = frame_count
        meta["duration"] = frame_count / output_fps
        finish_callback(meta)


# ---------------------------
# CLI wrapper
# ---------------------------
def _ask_size(default_w: int, default_h: int) -> Optional[Tuple[int, int]]:
    """
    Interactive size prompt.
    Accepts: empty (keep), WIDTHxHEIGHT, or WIDTH (auto-height).
    """
    prompt = f"Enter target size (e.g. 1280x720) or empty to keep original [{default_w}x{default_h}]: "
    v = input(prompt).strip()
    if v == "":
        return None
    elif "x" in v:
        parts = v.lower().split("x")
        try:
            w = int(parts[0])
            h = int(parts[1])
            return (w, h)
        except Exception:
            print("Invalid size format.")
            return _ask_size(default_w, default_h)
    else:
        try:
            w = int(v)
            h = int(round(default_h * (w / default_w)))
            return (w, h)
        except Exception:
            print("Invalid number.")
            return _ask_size(default_w, default_h)


def _ask_fps(default_fps: float) -> Optional[float]:
    prompt = f"Enter target fps (e.g. 24 or 30) or empty to keep original [{default_fps}]: "
    v = input(prompt).strip()
    if v == "":
        return None
    try:
        f = float(v)
        if f <= 0:
            print("FPS must be positive.")
            return _ask_fps(default_fps)
        return f
    except Exception:
        print("Invalid fps.")
        return _ask_fps(default_fps)


def process_video_cli(
    video_path: str,
    processing_callback: Callable[[FrameData, FrameIndex, TimestampSec], None],
    finish_callback: Optional[Callable[[VideoMetadata], None]] = None,
    max_workers: int = 8,
):
    """
    Interactive CLI entrypoint.
    Detects ffmpeg, asks for settings, and runs processing.
    """
    meta = get_metadata_cv2(video_path)
    print(json.dumps(meta, indent=2))
    target_size = _ask_size(meta["width"], meta["height"])
    target_fps = _ask_fps(meta["fps"])

    if target_fps is None:
        target_fps = meta["fps"] or 30.0

    ffmpeg_path = find_ffmpeg()
    ffmpeg_exec = None
    if ffmpeg_path:
        print(f"Found ffmpeg at: {ffmpeg_path}")
        ffmpeg_exec = ffmpeg_path
    else:
        resp = input(
            "ffmpeg not found on PATH. Enter path to ffmpeg executable to use it (or press Enter to fallback to cv2): "
        ).strip()
        if resp != "":
            if verify_ffmpeg(resp):
                ffmpeg_exec = resp
                print(f"Using ffmpeg at: {ffmpeg_exec}")
            else:
                print("Provided path is not valid ffmpeg. Falling back to cv2.")
        else:
            print("No ffmpeg provided; fallback to cv2.")

    process_frames_from_video(
        video_path=video_path,
        output_size=target_size,
        output_fps=target_fps,
        processing_callback=processing_callback,
        finish_callback=finish_callback,
        max_workers=max_workers,
        ffmpeg_exec_path=ffmpeg_exec,
    )
    return target_size, target_fps


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


def example_finish(meta: VideoMetadata):
    print(f"[finish] processed video metadata: {meta}")


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
    process_video_cli(
        video_file, processing_callback=example_callback, finish_callback=example_finish, max_workers=4
    )
