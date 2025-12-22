"""
video_utils.py

Summary:
- Reads video metadata (width, height, fps, frame_count, duration) using OpenCV.
- Interactively asks whether to change size / fps.
- Detects ffmpeg (shutil.which). If not found, asks the user for the ffmpeg executable path and verifies it.
- If ffmpeg is available, uses it to handle scale + fps and streams frames back to Python via pipe (rawvideo bgr24). Otherwise, falls back to cv2.
- Does not save an output video file (streams frames only).
- Provides each frame as a numpy.ndarray (BGR) to user_callback(frame, frame_index, timestamp_sec).
- Uses ThreadPoolExecutor to dispatch frame processing (worker count is configurable).
- CLI friendly: Can be used directly via process_video_cli("path/to/video.mp4").

Notes:
- Requires opencv-python.
- Performance is best if ffmpeg is present; if not, the cv2 fallback still works (but seeking might be slower).
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
    """Use cv2 to read basic metadata (lazy, no decoding of all frames)."""
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
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count, "duration": duration}


def find_ffmpeg() -> Optional[str]:
    """Try to locate ffmpeg executable via PATH. Return path or None."""
    path = shutil.which("ffmpeg")
    return path


def verify_ffmpeg(ffmpeg_exec: str) -> bool:
    """Validate that given path is a working ffmpeg executable."""
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
    Use ffmpeg to produce raw BGR24 frames via stdout pipe.
    Yields (frame_numpy_bgr, frame_index, timestamp_sec).
    """
    # ensure size is integers
    tw, th = target_size or (0, 0)
    # build vf filter: fps then scale
    scale_str = f"scale={tw}:{th}" if (tw > 0 and th > 0) else "scale=iw:ih"
    vf = f"fps={target_fps},{scale_str}"
    # ffmpeg pixel format: rgb24 or bgr24. OpenCV expects BGR; ffmpeg supports bgr24 as pix_fmt
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
    # If user supplied other ffmpeg path, user should set env or modify function to accept path.
    # We'll attempt to use 'ffmpeg' from PATH (the caller checked availability and set PATH accordingly if needed).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Determine expected frame size in bytes
    # Note: if scale uses -1 (preserve aspect), we must compute actual size: ffmpeg will output the scaled size used.
    # To avoid complexity, we only call this function with concrete integer target_size.
    frame_byte_size = target_size[0] * target_size[1] * 3
    frame_idx = 0
    try:
        while True:
            in_bytes = proc.stdout.read(frame_byte_size)  # type: ignore
            if not in_bytes or len(in_bytes) < frame_byte_size:
                break

            frame = np.frombuffer(in_bytes, np.uint8).reshape((target_size[1], target_size[0], 3))
            timestamp = frame_idx / target_fps
            yield frame, frame_idx, timestamp
            frame_idx += 1
    finally:
        try:
            proc.stdout.close()  # type: ignore
        except Exception:
            pass
        proc.kill()
        proc.wait(timeout=1)


def extract_with_cv2_time_based(
    video_path: str, target_fps: float, target_size: Optional[Tuple[int, int]] = None
) -> Generator[FrameInfo, Any, None]:
    """
    Use cv2 to extract frames time-based (set position by milliseconds) to avoid drift.
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
    # number of samples to produce:
    if duration is None:
        # unknown duration: iterate frames and sample based on indices using float interval
        # fallback to scanning all frames but using index accumulation method
        print("Warning: source duration unknown; falling back to full scan sampling.")
        # We'll approximate with reading frames and using index method:
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
            # sometimes seeking to exact time yields no frame (codec granularity); try read-next
            # read-next fallback:
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
    Main entrypoint to stream frames (resized & resampled) and process them via user_callback in threads.
    - user_callback(frame_bgr: np.ndarray, frame_index: int, timestamp_sec: float)
    - If prefer_ffmpeg and ffmpeg available + verified, uses ffmpeg pipe; otherwise use cv2 time-based extraction.
    - This function does NOT store output video; frames are processed on the fly.
    """
    # 1. read metadata via cv2 (as required)
    meta = get_metadata_cv2(video_path)
    src_w, src_h, src_fps = meta["width"], meta["height"], meta["fps"]
    frame_count = meta["frame_count"]
    duration = meta["duration"]
    print(
        f"[metadata] source size={src_w}x{src_h}, fps={src_fps}, frames={frame_count or 'unknown'}, duration={f'{duration}s' if duration else 'unknown'}"
    )

    # resolve target size
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

    # decide ffmpeg usability
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

    # start producer
    if use_ffmpeg:
        frames = extract_with_ffmpeg_pipe(
            video_path=video_path, target_fps=output_fps, target_size=target_size
        )
        via = "ffmpeg -> pipe"
    else:
        # else fallback to cv2 time-based extraction
        frames = extract_with_cv2_time_based(video_path, output_fps, output_size)
        via = "cv2 time-based"

    # Threaded processing: read frames sequentially, submit to pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        frame_count = 0
        for frame, idx, ts in frames:
            # submit callback
            futures.append(executor.submit(processing_callback, frame, idx, ts))
            frame_count += 1
        # wait results
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[worker] exception in callback: {e}", file=sys.stderr)
    print(f"[done] processed {frame_count} frames via {via}.")

    if finish_callback:
        meta["frame_count"] = frame_count
        finish_callback(meta)


# ---------------------------
# CLI wrapper
# ---------------------------
def _ask_size(default_w: int, default_h: int) -> Optional[Tuple[int, int]]:
    """
    Ask user to input target size. Accepts:
      - empty => keep original
      - WIDTHxHEIGHT e.g. 1280x720
      - WIDTH (single number) => WIDTH x auto-height (keep aspect)
    Returns None for keep original or tuple (w,h).
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
        # single width given: compute height to keep aspect
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
    Interactive CLI to:
    - read metadata via cv2
    - ask for size & fps changes
    - detect ffmpeg; if not found ask user to input ffmpeg path; verify
    - process frames and dispatch to user_callback using threads
    """
    meta = get_metadata_cv2(video_path)
    print(json.dumps(meta, indent=2))
    target_size = _ask_size(meta["width"], meta["height"])
    target_fps = _ask_fps(meta["fps"])

    # if user kept defaults, explicitly set to original
    if target_fps is None:
        target_fps = meta["fps"] or 30.0

    # ffmpeg detection
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

    # run processing
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
