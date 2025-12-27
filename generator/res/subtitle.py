import pysubs2


def generate_subtitle_init_mcfunction(subtitle: pysubs2.SSAFile, fps: float, subtitle_scale: float = 2.0):
    """
    根據影片路徑產生對應的字幕檔路徑
    例如: video/subtitles/{video_filename}_subtitles.srt
    """
    subtitles_dict: dict[int, str] = {}

    for line in subtitle:
        # 將時間轉換為 frame index
        start_sec = int(line.start / 1000 * fps)
        end_sec = int(line.end / 1000 * fps)
        text = line.text.replace(r"\N", "\n")  # 處理換行

        subtitles_dict[start_sec] = text
        subtitles_dict[end_sec] = ""  # 清除字幕

    storage = ",".join(f'{k}:"{v}"' for k, v in subtitles_dict.items())
    summon = (
        'summon minecraft:text_display ~ ~ ~0.5 {Tags:["video_player","subtitle"],'
        'text:"SUBTITLE",line_width:2147483647,background:0x60808080,'
        "transformation:{right_rotation:[0f,0f,0f,1f],left_rotation:[0f,0f,0f,1f],"
        f"scale:[{subtitle_scale}f,{subtitle_scale}f,{subtitle_scale}f],"
        "translation:[0f,0.5f,0f]}}"
    )
    return "\n".join([summon, "data modify storage video_player:subtitle subtitle set value {%s}" % storage])
