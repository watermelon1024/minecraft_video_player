"""
video_utils.py

功能摘要：
- 使用 OpenCV 讀取影片 metadata（width, height, fps, frame_count, duration）
- 互動式詢問是否要改變 size / fps
- 偵測 ffmpeg（shutil.which），若不存在詢問使用者輸入 ffmpeg 可執行檔路徑並驗證
- 若可用 ffmpeg，使用 ffmpeg 處理 scale + fps 並以 pipe (rawvideo bgr24) 傳回 Python，否則使用 cv2 完成相同工作
- 不保留輸出影片檔（僅 stream frames）
- 每一幀以 numpy.ndarray (BGR) 提供給 user_callback(frame, frame_index, timestamp_sec)
- 使用 ThreadPoolExecutor dispatch frame 處理（可設定 worker 數）
- CLI friendly：可直接用 process_video_cli("path/to/video.mp4")

注意：
- 需安裝 opencv-python
- ffmpeg 若存在則效能最好；若不存在，cv2 fallback 仍能工作（但跳時間定位可能較慢）
"""

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .agg import Block, agglomerative_merge
from .nbt import create_frame_structure


# ---------------------------
# Helpers: metadata & ffmpeg
# ---------------------------
def get_metadata_cv2(path: str) -> dict:
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
def extract_with_ffmpeg_pipe(video_path: str, target_fps: float, target_size: Tuple[int, int]):
    """
    Use ffmpeg to produce raw BGR24 frames via stdout pipe.
    Yields (frame_numpy_bgr, frame_index, timestamp_sec).
    """
    # ensure size is integers
    tw, th = target_size
    # build vf filter: fps then scale (scale accepts -1 to keep aspect)
    scale_str = f"scale={tw}:{th}" if (tw > 0 and th > 0) else "scale=iw:ih"
    vf = f"fps={target_fps},{scale_str}"
    # ffmpeg pixel format: rgb24 or bgr24. OpenCV expects BGR; ffmpeg supports bgr24 as pix_fmt
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-vcodec",
        "rawvideo",
        "-",
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


def extract_with_cv2_time_based(video_path: str, target_fps: float, target_size: Optional[Tuple[int, int]]):
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
    user_callback: Callable[[any, int, float], None],
    max_workers: int = 4,
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

    # For ffmpeg pipeline, ensure 'ffmpeg' command in PATH points to the verified path if provided.
    # If ffmpeg_path is non-default, we can set env or adjust command; for simplicity we'll require ffmpeg is in PATH
    # or that ffmpeg_exec_path was provided and valid. We'll try to use ffmpeg_exec_path if given.
    if use_ffmpeg and ffmpeg_exec_path:
        # temporarily use full path by setting shutil.which name is not used; modify environment PATH not necessary
        os.environ["FFMPEG_EXEC"] = ffmpeg_exec_path  # not used directly but kept for debugging

    # start producer
    if use_ffmpeg:
        # ensure we call ffmpeg from the supplied path if present
        # We'll temporarily set the command to the full path
        # NOTE: extract_with_ffmpeg_pipe uses "ffmpeg" literal; to use provided path, we can monkey-patch shutil.which or
        # simplest is to temporarily set PATH so that the desired ffmpeg is first. We'll invoke using full path by replacing 'ffmpeg' binary.
        # Simpler path: call subprocess with full path by setting env and using that name in cmd.
        # To avoid duplicating extract logic, we'll implement local ffmpeg extraction here using full path.
        ffmpeg_bin = ffmpeg_exec_path if ffmpeg_exec_path else ffmpeg_path

        # Build command with concrete integer target_size
        tw, th = target_size
        if tw <= 0 or th <= 0:
            tw, th = src_w, src_h

        scale_str = f"scale={tw}:{th}"
        vf = f"fps={output_fps},{scale_str}"

        cmd = [
            ffmpeg_bin,
            "-i",
            video_path,
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_byte_size = tw * th * 3

        with open("frame/frame.json", "w") as f:
            tar_h = int(target_size[1])
            tar_h_half = tar_h // 2
            json.dump(
                {
                    "providers": [
                        {
                            "type": "bitmap",
                            "file": f"frame/frame_{i}.png",
                            "height": tar_h,
                            "ascent": tar_h_half,
                            "chars": [chr(0xE000 + i)],
                        }
                        for i in range(target_frame_count or 10000)
                    ]
                },
                f,
                indent=2,
            )
        with open("frame/init.mcfunction", "w") as f:
            f.write(
                "data merge storage video_player:frame {frames:[%s]}\n"
                % ",".join(f'"\\u{0xE000 + i:04X}"' for i in range(target_frame_count or 10000))
            )

        # Threaded processing: read frames sequentially, submit to pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            frame_idx = 0
            try:
                while True:
                    chunk = proc.stdout.read(frame_byte_size)
                    if not chunk or len(chunk) < frame_byte_size:
                        break

                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape((th, tw, 3))
                    ts = frame_idx / float(output_fps)
                    # submit to worker
                    futures.append(executor.submit(user_callback, frame, frame_idx, ts))
                    frame_idx += 1
                # wait for futures to finish
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[worker] exception in callback: {e}", file=sys.stderr)
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                proc.kill()
                proc.wait(timeout=1)
        print(f"[done] processed {frame_idx} frames via ffmpeg -> pipe.")
        return

    # else fallback to cv2 time-based extraction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        frame_count = 0
        for frame, idx, ts in extract_with_cv2_time_based(video_path, output_fps, target_size):
            # submit callback
            futures.append(executor.submit(user_callback, frame, idx, ts))
            frame_count += 1
        # wait results
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[worker] exception in callback: {e}", file=sys.stderr)
    print(f"[done] processed {frame_count} frames via cv2 fallback.")


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
    if "x" in v:
        parts = v.lower().split("x")
        try:
            w = int(parts[0])
            h = int(parts[1])
            return (w, h)
        except:
            print("Invalid size format.")
            return _ask_size(default_w, default_h)
    elif v.endswith(("p", "P")):
        # height given with 'p' suffix
        try:
            h = int(v[:-1])
            w = int(round(default_w * (h / default_h)))
            return (w, h)
        except:
            print("Invalid size format.")
            return _ask_size(default_w, default_h)
    else:
        # single width given: compute height to keep aspect
        try:
            w = int(v)
            h = int(round(default_h * (w / default_w)))
            return (w, h)
        except:
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
    except:
        print("Invalid fps.")
        return _ask_fps(default_fps)


def process_video_cli(
    video_path: str, user_callback: Callable[[any, int, float], None], max_workers: int = 4
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
    if target_size is None:
        target_size = (meta["width"], meta["height"])
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
        user_callback=user_callback,
        max_workers=max_workers,
        prefer_ffmpeg=True,
        ffmpeg_exec_path=ffmpeg_exec,
    )

    generate_json_and_commands(int(meta["duration"] * target_fps), target_size[0], target_size[1])

    return target_size, target_fps


# ---------------------------
# Example user_callback (示範)
# ---------------------------


def rle_encode_row(row: np.ndarray):
    """
    對單一行像素執行 Run-Length Encoding。
    row: np.ndarray shape = (width, 3) (BGR)
    return: list of tuples [(color, count), ...]
    """
    if len(row) == 0:
        return []

    encoded = []
    current_color = tuple(row[0])
    count = 1

    for pixel in row[1:]:
        color = tuple(pixel)
        if color == current_color:
            count += 1
        else:
            encoded.append((current_color, count))
            current_color = color
            count = 1

    encoded.append((current_color, count))
    return encoded


# import matplotlib.pyplot as plt

# plt.switch_backend("Agg")


def color_gbr_to_hex(color: tuple[int, int, int]):
    return (
        np.uint32(color[0])
        | np.left_shift(color[1], 8, dtype=np.uint32)
        | np.left_shift(color[2], 16, dtype=np.uint32)
    )


def callback(frame, index, timestamp):
    """
    將單一畫面切割成多個 < 256x256 的小圖塊並存檔。
    檔名格式: assets/minecraft/textures/font/f{index}_r{row}_c{col}.png
    """

    # --- 設定切片參數 (可依需求調整，建議 256 以下) ---
    MAX_W = 256
    MAX_H = 256

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
            filename = f"f{index}_r{r}_c{c}.png"
            filepath = os.path.join(output_dir, filename)

            # 使用 cv2 存檔 (預設壓縮參數即可)
            cv2.imwrite(filepath, tile)


def generate_json_and_commands(total_frames, target_w, target_h):
    """
    根據影片總幀數和尺寸，生成 default.json 與 召喚實體的指令提示。
    利用 1 px = 0.025 blocks 的測量數據進行自動對齊。
    """
    MAX_W = 256
    MAX_H = 256
    PIXEL_SCALE = 0.025  # 1 px = 0.025 blocks
    ROW_HEIGHT_BLOCKS = MAX_H * PIXEL_SCALE  # 6.4

    # 計算行列數
    cols = math.ceil(target_w / MAX_W)
    rows = math.ceil(target_h / MAX_H)

    # --- 1. 生成 default.json ---
    fonts: dict[str, list] = {}
    start_char = 0xE000

    print(f"[Info] 正在生成設定檔... (Grid: {rows} rows x {cols} cols)")
    print("[Optimization] 使用橫向拼接與頂部對齊 (Ascent=0)")

    for i in range(total_frames):
        for r in range(rows):
            for c in range(cols):
                # 計算當前 Tile 的實際像素大小
                current_h = min((r + 1) * MAX_H, target_h) - (r * MAX_H)

                # 計算該字符的 Unicode

                # 檔名
                filename = f"video:frame/f{i}_r{r}_c{c}.png"

                # [優化] Ascent 設為 0
                # 這代表基準線(Baseline)在圖片的最頂端。
                # 圖片會從實體的 Y 座標開始，向下渲染 current_h 的長度。
                fonts.setdefault(f"frame_r{r}_c{c}", []).append(
                    {
                        "type": "bitmap",
                        "file": filename,
                        "ascent": 0,  # <--- 關鍵修改：頂部對齊
                        "height": current_h,  # 實際像素高度
                        "chars": [chr(start_char + i)],
                    }
                )

    for name, providers in fonts.items():
        json_path = f"./{name}.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"providers": providers}, f, ensure_ascii=False)

    print(f"[Done] 已生成資源包設定: {json_path}")

    # --- 2. 輸出召喚指令 (Row-based) ---
    print("\n" + "=" * 60)
    print(f"【優化版電視牆指令】 (1px = {PIXEL_SCALE} blocks)")
    print("-" * 60)

    # 計算總寬度（用於置中）
    total_width_blocks = target_w * PIXEL_SCALE

    # 起始 X：往左移總寬的一半
    start_x = -(total_width_blocks / 2)

    # 起始 Y：因為 ascent=0 代表圖片向下長，所以我們把第一排放在最高點
    # 假設腳下是 0，螢幕底部對齊眼睛高度(1.6)，或是直接浮空
    # 這裡設為：第一排的頂端在 Y + 總高度 (這樣螢幕底部大約在 Y=0)
    total_height_blocks = target_h * PIXEL_SCALE
    start_y = total_height_blocks

    cmd_list = []

    # 只需要對每一「行」(Row) 生成一個實體
    for r in range(rows):
        # 1. 計算該行的 Y 座標
        # 因為每一行完整的高度都是 256px (除了最後一行，但最後一行起始點也是從上一行結束算起)
        # 且 ascent=0，所以下一行的起始點就是 Current_Y - 6.4
        current_y = start_y - (r * ROW_HEIGHT_BLOCKS)

        # 2. 格式化座標
        trans_x = f"{start_x:.4f}"
        trans_y = f"{current_y:.4f}"

        tag_name = f"screen_row_{r}"

        # 3. 生成該行的初始化文字 (只用來佔位，實際播放時會替換)
        # 這裡我們不需要預先知道 Unicode，因為那是播放邏輯的事
        # 但為了測試，我們可以塞入第一幀的對應字符

        # 這裡生成第一幀該行的所有字符字串: "\uE000\uE001..."
        # 這裡僅作範例，實際 Datapack 播放時要動態組裝
        demo_text = ""
        # 這裡的邏輯僅適用於 frame 0 預覽
        first_frame_base = start_char
        row_base_char = first_frame_base + (r * cols)

        chars_in_row = []
        for c in range(cols):
            chars_in_row.append(f"\\u{row_base_char + c:04X}")

        # 拼成 JSON string: "text":"\uE000\uE001..."
        text_json = "".join(chars_in_row)
        cmd = (
            """summon minecraft:text_display ~ ~ ~ {Tags:["video_player","frame"],transformation:{right_rotation:[0f,0f,0f,1f],left_rotation:[0f,0f,0f,1f],translation:[%sf,%sf,0f],scale:[1f,1f,1f]},text:["",{"text":"SCREEN",font:"video:frame_r0_c0"},{"text":"\u200c",font:"video:nosplit"},{"text":"SCREEN",font:"video:frame_r0_c1"},{"text":"\u200c",font:"video:nosplit"},{"text":"SCREEN",font:"video:frame_r0_c2"},{"text":"\u200c",font:"video:nosplit"},{"text":"SCREEN",font:"video:frame_r0_c3"}],line_width:2147483647,background:0}"""
            % (trans_x, trans_y)
        )

        cmd_list.append(cmd)
        print(cmd)

    print("-" * 60)
    print(f"實體數量已減少為: {rows} 個 (原為 {rows*cols} 個)")
    print("提示：播放時，請確保更新該 Row 實體時，Text 內容包含該行所有的 Tile 字符。")
    print("=" * 60 + "\n")


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
    size, fps = process_video_cli(video_file, user_callback=callback, max_workers=4)
    print(size[0] * 9)
