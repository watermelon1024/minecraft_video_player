import math
import os
from typing import TypedDict

from ..file_utils import PackGenerator
from ..video_utils import FrameData, FrameIndex, TimestampSec, VideoMetadata


class FrameRelatedFunctions(TypedDict):
    init: str
    loop: str


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


def generate_frame_related(meta: VideoMetadata, resourcepack: PackGenerator) -> FrameRelatedFunctions:
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

    print(f"[Info] Generating frame related data... (Grid: {rows} rows x {cols} cols)")

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

    font_dir = "assets/video/font"
    for name, providers in fonts.items():
        json_path = os.path.join(font_dir, f"{name}.json")
        resourcepack.write_json(json_path, {"providers": providers})

    print(f"[Done] Generated resourcepack custom font: {font_dir} with {total_frames * rows * cols} entries")

    # --- 2. Output commands (Row-based) ---

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
                '{"text":"\\uE000","font":"video:frame_%d_%d"}' % (r, c) for c in range(cols)
            )
            + "],line_width:2147483647,background:0}"
        )
        init_cmds.append(cmd)

    # --- 3. Generate frame Unicode mapping table ---
    frames_unicode = ",".join(f'"\\u{start_char + i:04x}"' for i in range(total_frames))
    init_cmds.append("data merge storage video_player:frame {frames:[%s]}" % frames_unicode)

    init_cmds.append("scoreboard players set frame video_player 0")
    init_cmds.append(f"scoreboard players set end_frame video_player {total_frames - 1}")

    if fps != 20:
        init_cmds.append(
            'tellraw @p ["","Your video FPS is not 20; you need to execute ",{text:"/tick rate %s",bold:true,color:"gold",click_event:{action:"suggest_command",command:"/tick rate %s"}}," to fit the correct playback speed. (Or ",{text:"[Click Here]",color:"gold",click_event:{action:"run_command",command:"tick rate %s"}}," to execute.)"]'
            % ((f"{fps}",) * 3)
        )

    print(f"[Done] Generated frame init commands: {len(init_cmds)} lines")

    loop_cmds = [
        f"$data modify entity @s text.extra[{c * 2}].text set from storage video_player:frame frames[$(frame)]"
        for c in range(cols)
    ]
    print(f"[Done] Generated play frame commands: {len(loop_cmds)} lines")

    return {
        "init": "\n".join(init_cmds),
        "loop": "\n".join(loop_cmds),
    }
