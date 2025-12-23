import math
import os
from tempfile import TemporaryDirectory

from ..file_utils import PackGenerator
from ..video_utils import FrameData, FrameIndex, TimestampSec, VideoMetadata
from .audio import generate_segmented_sounds_json, segment_audio

MAX_W = 256
MAX_H = 256
PIXEL_SCALE = 0.025  # 1 px = 0.025 blocks
ROW_HEIGHT_BLOCKS = MAX_H * PIXEL_SCALE  # 6.4


def processing_callback(
    frame: FrameData, index: FrameIndex, timestamp: TimestampSec, resourcepack: PackGenerator
):
    """
    Splits a single frame into multiple small tiles < 256x256 and saves them.
    Filename format: assets/video/textures/frame/{frame}_{row}_{col}.png
    """
    h, w, _ = frame.shape

    # Ceiling division
    cols = math.ceil(w / MAX_W)
    rows = math.ceil(h / MAX_H)

    for r in range(rows):
        for c in range(cols):
            x_start = c * MAX_W
            y_start = r * MAX_H
            x_end = min(x_start + MAX_W, w)
            y_end = min(y_start + MAX_H, h)

            # NumPy slicing is efficient
            tile = frame[y_start:y_end, x_start:x_end]

            filepath = os.path.join("assets/video/textures/frame", f"{index}_{r}_{c}.png")
            resourcepack.write_image(filepath, tile)


def finish_callback(meta: VideoMetadata, datapack: PackGenerator, resourcepack: PackGenerator):
    """
    Generates custom font json and text_display init mcfunction.
    Uses 1 px = 0.025 blocks measurement for automatic alignment.
    """
    target_w = meta["width"]
    target_h = meta["height"]
    fps = meta["fps"]
    total_frames = meta["frame_count"]

    cols = math.ceil(target_w / MAX_W)
    rows = math.ceil(target_h / MAX_H)

    # --- 1. Generate custom font json ---
    fonts: dict[str, list] = {}
    start_char = 0xE000

    print(f"[Info] Generating config... (Grid: {rows} rows x {cols} cols)")

    for i in range(total_frames):
        for r in range(rows):
            current_h = min((r + 1) * MAX_H, target_h) - (r * MAX_H)
            for c in range(cols):
                # Ascent=0 aligns the image top to the baseline.
                # The image renders downwards from the entity's Y.
                fonts.setdefault(f"frame_{r}_{c}", []).append(
                    {
                        "type": "bitmap",
                        "file": f"video:frame/{i}_{r}_{c}.png",
                        "ascent": 0,
                        "height": current_h,
                        "chars": [chr(start_char + i)],
                    }
                )

    for name, providers in fonts.items():
        json_path = os.path.join("assets/video/font", f"{name}.json")
        resourcepack.write_json(json_path, {"providers": providers})

    print("[Done] Generated resource pack config: assets/video/font")

    # --- 2. Output summon commands (Row-based) ---
    print("[Info] Generating summon commands...")

    # Ascent=0 means image grows downwards.
    # Place the first row at Y + total height so the bottom aligns with Y=0.
    total_height_blocks = target_h * PIXEL_SCALE
    start_y = total_height_blocks

    init_cmds: list[str] = []

    for r in range(rows):
        # Each row is 256px high (6.4 blocks).
        # Next row starts 6.4 blocks lower.
        trans_y = start_y - (r * ROW_HEIGHT_BLOCKS)

        # Placeholder text for initialization.
        # Unicode characters are swapped during playback.
        cmd = (
            "summon minecraft:text_display ~ ~ ~ "
            '{Tags:["video_player","frame"],'
            "transformation:{"
            "right_rotation:[0f,0f,0f,1f],"
            "left_rotation:[0f,0f,0f,1f],"
            f"translation:[0f,{trans_y}f,0f],"
            "scale:[1f,1f,1f]},"
            'text:["",'
            + ',{"text":"\\u200c","font":"video:nosplit"},'.join(
                '{"text":"\\uE000","font":"video:frame_r%d_c%d"}' % (r, c) for c in range(cols)
            )
            + "],line_width:2147483647,background:0}"
        )
        init_cmds.append(cmd)

    # --- 3. Generate frame Unicode mapping table ---
    frames_unicode = ",".join(f'"\\u{start_char + i:04x}"' for i in range(total_frames))
    init_cmds.append("data merge storage video_player:frame {frames:[%s]}" % frames_unicode)

    init_cmds.append("scoreboard players set frame video_player 0")
    init_cmds.append(f"scoreboard players set end_frame video_player {total_frames - 1}")

    # audio segment time (seconds)
    segment_time = 10
    init_cmds.append(f"scoreboard players set audio_segment video_player {int(segment_time * fps)}")

    mcfunction_dir = "data/video_player/function"

    init_path = os.path.join(mcfunction_dir, "init.mcfunction")
    datapack.write_text(init_path, "\n".join(init_cmds))
    print(f"[Done] Generated init commands: {init_path}")

    play_loop_path = os.path.join(mcfunction_dir, "play_frame.mcfunction")
    datapack.write_text(
        play_loop_path,
        "\n".join(
            f"$data modify entity @s text.extra[{c * 2}].text set from storage video_player:frame frames[$(frame)]"
            for c in range(cols)
        ),
    )
    print(f"[Done] Generated play frame commands: {play_loop_path}")

    with TemporaryDirectory() as tmpdir:
        audio_files = segment_audio(meta["path"], tmpdir, segment_time=segment_time)
        for filename in audio_files:
            source_path = os.path.join(tmpdir, filename)
            resourcepack.write_file_from_disk(f"assets/video/sounds/{filename}", source_path)
        generate_segmented_sounds_json(audio_files, resourcepack, namespace="video")
