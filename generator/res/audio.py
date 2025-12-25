import os

from ..file_utils import PackGenerator


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
    print(f"[Done] Generated resourcepack sounds: {json_path} with {len(files)} entries")
