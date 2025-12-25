import os
import subprocess
from functools import partial
from typing import Optional

from generator.ffmpeg_utils import find_ffmpeg, verify_ffmpeg


def segment_audio_with_ffmpeg(
    input_video: str, output_dir: str, segment_time: int = 10, ffmpeg_exec: str = "ffmpeg"
):
    """
    Splits audio into Ogg segments using ffmpeg.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_pattern = os.path.join(output_dir, "part_%d.ogg")
    cmd = [
        # fmt: off
        ffmpeg_exec, "-y",
        "-i", input_video,
        "-vn",
        "-c:a", "libvorbis",
        "-q:a", "3",
        "-f", "segment",
        "-segment_time", str(segment_time),
        output_pattern,
        # fmt: on
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    files = [f for f in os.listdir(output_dir) if f.startswith("part_") and f.endswith(".ogg")]
    files.sort(key=lambda x: int(x.removeprefix("part_").removesuffix(".ogg")))
    return files


def segment_audio_fallback(video_path: str, output_dir: str, segment_time: int = 10):
    """
    Split audio into Ogg segments without using ffmpeg (Fallback mode).
    """
    raise NotImplementedError("Fallback audio segmentation is not implemented yet.")


def segment_audio(
    input_video: str,
    output_dir: str,
    segment_time: int = 10,
    prefer_ffmpeg: bool = True,
    ffmpeg_exec_path: Optional[str] = None,
):
    """
    Splits audio into Ogg segments using ffmpeg.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ffmpeg_available = False
    if prefer_ffmpeg:
        if not ffmpeg_exec_path:
            ffmpeg_exec_path = find_ffmpeg()
        if ffmpeg_exec_path:
            ffmpeg_available = verify_ffmpeg(ffmpeg_exec_path)
            ffmpeg_available = True

    if ffmpeg_exec_path and ffmpeg_available:
        print(f"[Audio] using ffmpeg at: {ffmpeg_exec_path}")
        _segment_audio = partial(
            segment_audio_with_ffmpeg, input_video, output_dir, segment_time, ffmpeg_exec=ffmpeg_exec_path
        )
        via = "ffmpeg"
    else:
        print("[Audio] ffmpeg not available, using fallback method.")
        _segment_audio = partial(segment_audio_fallback, input_video, output_dir, segment_time)
        via = "fallback"

    files = _segment_audio()
    print(f"[Audio] Splitted audio into {len(files)} segments ({segment_time} seconds each) via {via}...")
    return files
