import json
from typing import Optional, Tuple

from .ffmpeg_utils import find_ffmpeg, verify_ffmpeg
from .video_utils import get_metadata_cv2, resolve_resolution


def _ask_size(default_w: int, default_h: int) -> Optional[Tuple[int, int]]:
    """
    Interactive size prompt.
    Accepts: empty (keep), WIDTHxHEIGHT, or WIDTHx-1 (auto-height) or -1xHEIGHT (auto-width).
    """
    prompt = f"Enter target size (e.g. 640x480 or 1280x720), using -1 for auto dimension (e.g. 1280x-1), or empty to keep original [{default_w}x{default_h}]: "
    v = input(prompt).strip()
    if v == "":
        return None
    elif "x" in v:
        parts = v.lower().split("x")
        try:
            w = int(parts[0].strip())
            h = int(parts[1].strip())
            return resolve_resolution(target_w=w, target_h=h, src_w=default_w, src_h=default_h)
        except Exception:
            print("Invalid size format.")
            return _ask_size(default_w, default_h)
    else:
        print("Invalid input.")
        return _ask_size(default_w, default_h)


def _ask_fps(default_fps: float) -> Optional[float]:
    prompt = f"Enter target fps (e.g. 20, 24 or 30) or empty to keep original [{default_fps}]: "
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


def _ask_ffmpeg() -> Optional[str]:
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
                print("Provided path is not valid ffmpeg; fallback to cv2.")
        else:
            print("No ffmpeg provided; fallback to cv2.")

    return ffmpeg_exec


def ask_metadata(video_path: str):
    """
    Interactive CLI entrypoint.
    Detects ffmpeg, asks for settings, and runs processing.
    """
    meta = get_metadata_cv2(video_path)
    print("[INFO] Video original metadata: " + json.dumps(meta, indent=2, ensure_ascii=False))

    target_size = _ask_size(meta["width"], meta["height"])
    target_fps = _ask_fps(meta["fps"])
    ffmpeg_exec = _ask_ffmpeg()

    return target_size, target_fps, ffmpeg_exec
