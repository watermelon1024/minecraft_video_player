import pysubs2

from ..file_utils import PackGenerator


def generate_subtitle_init_mcfunction(subtitle_path: str, fps: float, datapack: PackGenerator) -> str:
    """
    根據影片路徑產生對應的字幕檔路徑
    例如: video/subtitles/{video_filename}_subtitles.srt
    """
    subs = pysubs2.load(subtitle_path)

    subtitles: dict[int, str] = {}

    for line in subs:
        # 將時間轉換為 frame index
        start_sec = int(line.start / 1000 * fps)
        end_sec = int(line.end / 1000 * fps)
        text = line.text.replace(r"\N", "\n")  # 處理換行

        subtitles[start_sec] = text
        subtitles[end_sec] = ""  # 清除字幕

    storage = ",".join(f'{k}:"{v}"' for k, v in subtitles.items())

    return "data merge storage video_player:subtitle {%s}" % storage
