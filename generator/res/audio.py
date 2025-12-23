import os
import subprocess

from ..file_utils import PackGenerator


def segment_audio(input_video: str, output_dir: str, segment_time: int = 10):
    """
    Splits audio into Ogg segments using ffmpeg.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_pattern = os.path.join(output_dir, "part_%d.ogg")
    cmd = [
        # fmt: off
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-c:a", "libvorbis",
        "-q:a", "3",
        "-f", "segment",
        "-segment_time", str(segment_time),
        output_pattern,
        # fmt: on
    ]

    print(f"[Audio] Splitting audio into {segment_time} second segments...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    files = [f for f in os.listdir(output_dir) if f.startswith("part_") and f.endswith(".ogg")]
    return files


def generate_segmented_sounds_json(files: list[str], resourcepack: PackGenerator, namespace: str = "video"):
    sounds_data = {}
    for i in range(len(files)):
        sound_event = f"part_{i}"
        sounds_data[sound_event] = {
            "subtitle": f"Audio Part {i}",
            "sounds": [
                {
                    "name": f"{namespace}:part_{i}",
                    "stream": True,  # Streaming is safer for longer clips
                }
            ],
        }

    json_path = os.path.join("assets/video", "sounds.json")
    resourcepack.write_json(json_path, sounds_data)
