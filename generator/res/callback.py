import json
import math
import os

import cv2


from ..video_utils import VideoMetadata, FrameData, FrameIndex, TimestampSec

MAX_W = 256
MAX_H = 256
PIXEL_SCALE = 0.025  # 1 px = 0.025 blocks
ROW_HEIGHT_BLOCKS = MAX_H * PIXEL_SCALE  # 6.4


def processing_callback(frame: FrameData, index: FrameIndex, timestamp: TimestampSec):
    """
    將單一畫面切割成多個 < 256x256 的小圖塊並存檔。
    檔名格式: assets/minecraft/textures/font/f{index}_r{row}_c{col}.png
    """

    # 取得原始尺寸
    h, w, _ = frame.shape

    # 計算需要切幾行幾列 (Ceiling division)
    cols = math.ceil(w / MAX_W)
    rows = math.ceil(h / MAX_H)

    # 確保輸出目錄存在
    output_dir = "frame"
    os.makedirs(output_dir, exist_ok=True)

    # --- 開始切割與存檔 ---
    for r in range(rows):
        for c in range(cols):
            # 計算裁切範圍
            x_start = c * MAX_W
            y_start = r * MAX_H
            x_end = min(x_start + MAX_W, w)
            y_end = min(y_start + MAX_H, h)

            # [核心] 利用 NumPy Slicing 切割 (速度極快)
            # frame[y:y, x:x]
            tile = frame[y_start:y_end, x_start:x_end]

            # 存檔
            # 檔名範例: f0_r0_c0.png
            filepath = os.path.join(output_dir, f"f{index}_r{r}_c{c}.png")

            # 使用 cv2 存檔 (預設壓縮參數即可)
            cv2.imwrite(filepath, tile)


def finish_callback(meta: VideoMetadata):
    """
    根據影片總幀數和尺寸，生成 custom font json 與 text_display init mcfunction。
    利用 1 px = 0.025 blocks 的測量數據進行自動對齊。
    """

    target_w = meta["width"]
    target_h = meta["height"]
    total_frames = meta["frame_count"]
    # 計算行列數
    cols = math.ceil(target_w / MAX_W)
    rows = math.ceil(target_h / MAX_H)

    # --- 1. 生成 custom font json ---
    fonts: dict[str, list] = {}
    start_char = 0xE000

    print(f"[Info] 正在生成設定檔... (Grid: {rows} rows x {cols} cols)")

    for i in range(total_frames):
        for r in range(rows):
            current_h = min((r + 1) * MAX_H, target_h) - (r * MAX_H)
            for c in range(cols):
                # Ascent 設為 0
                # 這代表基準線(Baseline)在圖片的最頂端。
                # 圖片會從實體的 Y 座標開始，向下渲染 current_h 的長度。
                fonts.setdefault(f"frame_r{r}_c{c}", []).append(
                    {
                        "type": "bitmap",
                        "file": f"video:frame/f{i}_r{r}_c{c}.png",
                        "ascent": 0,  # 頂部對齊
                        "height": current_h,  # 實際像素高度
                        "chars": [chr(start_char + i)],
                    }
                )

    font_dir = "res/font"
    os.makedirs(font_dir, exist_ok=True)
    for name, providers in fonts.items():
        json_path = os.path.join(font_dir, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"providers": providers}, f, separators=(",", ":"), ensure_ascii=False)

    print(f"[Done] 已生成資源包設定: {font_dir}")

    # --- 2. 輸出召喚指令 (Row-based) ---
    print("[Info] 生成召喚指令...")
    # 計算總寬度（用於置中）
    total_width_blocks = target_w * PIXEL_SCALE

    # 起始 X：往左移總寬的一半
    trans_x = -(total_width_blocks / 2)

    # 起始 Y：因為 ascent=0 代表圖片向下長，所以我們把第一排放在最高點
    # 假設腳下是 0，螢幕底部對齊眼睛高度(1.6)，或是直接浮空
    # 這裡設為：第一排的頂端在 Y + 總高度 (這樣螢幕底部大約在 Y=0)
    total_height_blocks = target_h * PIXEL_SCALE
    start_y = total_height_blocks

    cmd_list: list[str] = []

    # 只需要對每一「行」(Row) 生成一個實體
    for r in range(rows):
        # 1. 計算該行的 Y 座標
        # 因為每一行完整的高度都是 256px (除了最後一行，但最後一行起始點也是從上一行結束算起)
        # 且 ascent=0，所以下一行的起始點就是 Current_Y - 6.4
        trans_y = start_y - (r * ROW_HEIGHT_BLOCKS)

        # 3. 生成該行的初始化文字 (只用來佔位，實際播放時會替換)
        # 這裡我們不需要預先知道 Unicode，因為那是播放邏輯的事
        # 但為了測試，我們可以塞入第一幀的對應字符
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
        cmd_list.append(cmd)

    # --- 3. 生成幀 Unicode 對應表 ---
    frames_unicode = ",".join(f'"\\u{start_char + i:04x}"' for i in range(total_frames))
    cmd_list.insert(0, "data merge storage video_player:frame {frames:[%s]}" % frames_unicode)

    cmd_list.append("scoreboard players set frame video_player 0")
    cmd_list.append(f"scoreboard players set end_frame video_player {total_frames - 1}")

    mcfunction_dir = "dtp/function"
    os.makedirs(mcfunction_dir, exist_ok=True)
    init_path = os.path.join(mcfunction_dir, "init.mcfunction")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cmd_list))
    print(f"[Done] 已生成初始化指令: {init_path}")
