import argparse
import math
import os
from functools import partial
from tempfile import TemporaryDirectory

from .audio_utils import segment_audio
from .cli_utils import ask_metadata
from .ffmpeg_utils import verify_ffmpeg
from .file_utils import PackGenerator, PackMode
from .res.audio import generate_segmented_sounds_json
from .res.callback import generate_frame_related, processing_callback
from .res.subtitle import generate_subtitle_init_mcfunction
from .subtitle_utils import extract_and_parse_subtitles_from_video, load_subtitles_from_file
from .video_utils import process_frames_from_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Video Player Generator")
    parser.add_argument("video", help="Path to the video file")

    # Output filename option
    parser.add_argument(
        "-o",
        "--output",
        help="Base name for output files (e.g. 'myvideo' -> 'myvideo_datapack', 'myvideo_resourcepack')",
    )

    # Subtitle options
    parser.add_argument(
        "-s",
        "--subtitle",
        help="Path to the subtitle file (.srt, .ass, etc.)",
    )
    parser.add_argument(
        "-ss",
        "--subtitle-scale",
        type=float,
        default=2.0,
        help="Scale for subtitle text display (default: 2.0)",
    )
    parser.add_argument(
        "-ns",
        "--no-subtitle",
        action="store_true",
        help="Do not attempt to extract subtitles from video (Do not affect if subtitle file is provided)",
    )

    # Zip options
    parser.add_argument(
        "-nz",
        "--no-zip",
        action="store_true",
        help="Generate as folders instead of zip files",
    )

    # Resourcepack options
    parser.add_argument(
        "-nr",
        "--no-resourcepack",
        action="store_true",
        help="Do not generate resource pack",
    )

    # Workers option
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads to use for processing",
    )

    # FFmpeg options
    parser.add_argument(
        "--ffmpeg-exec",
        help="Path to FFmpeg executable (if not in PATH)",
    )
    parser.add_argument(
        "--ffprobe-exec",
        help="Path to FFprobe executable (if not in PATH)",
    )

    args = parser.parse_args()

    video_file = args.video
    subtitle_file = args.subtitle
    subtitle_scale = args.subtitle_scale
    no_subtitle = args.no_subtitle
    use_zip = not args.no_zip
    no_resourcepack = args.no_resourcepack
    output_base = args.output
    workers = args.workers
    ffmpeg_exec = args.ffmpeg_exec
    ffprobe_exec = args.ffprobe_exec

    if no_resourcepack:
        raise NotImplementedError("Datapack-only generation is not yet supported.")

    if workers < 1:
        workers = 1

    prefer_ffmpeg = True
    if ffmpeg_exec and not verify_ffmpeg(ffmpeg_exec):
        print("[Error] Provided FFmpeg executable is not valid, will fallback to pure-python methods.")
        ffmpeg_exec = None
        prefer_ffmpeg = False
    if ffprobe_exec and not verify_ffmpeg(ffprobe_exec):
        print("[Error] Provided FFprobe executable is not valid, will fallback to pure-python methods.")
        ffprobe_exec = None

    # target_size, target_fps, ffmpeg_exec = ask_metadata(video_file)
    target_size, target_fps = ask_metadata(video_file)

    mode = PackMode.ZIP if use_zip else PackMode.FOLDER

    if output_base:
        datapack_name = f"{output_base}_datapack"
        resourcepack_name = f"{output_base}_resourcepack"
    else:
        datapack_name = "datapack"
        resourcepack_name = "resourcepack"

    if use_zip:
        datapack_name += ".zip"
        resourcepack_name += ".zip"

    with PackGenerator(
        datapack_name, template_path="template/datapack", mode=mode
    ) as datapack, PackGenerator(
        resourcepack_name, template_path="template/resourcepack", mode=mode
    ) as resourcepack:
        # handle video frames
        meta = process_frames_from_video(
            video_path=video_file,
            output_size=target_size,
            output_fps=target_fps,
            callback=partial(processing_callback, resourcepack=resourcepack),
            max_workers=workers,
            prefer_ffmpeg=prefer_ffmpeg,
            ffmpeg_exec_path=ffmpeg_exec,
        )
        frame_func = generate_frame_related(meta, resourcepack)
        init_cmds = [frame_func["init"]]

        fps = meta["fps"]

        # handle audio
        segment_time = 10  # audio segment time (seconds)
        with TemporaryDirectory() as tmpdir:
            audio_files = segment_audio(
                meta["path"],
                tmpdir,
                segment_time=segment_time,
                prefer_ffmpeg=prefer_ffmpeg,
                ffmpeg_exec_path=ffmpeg_exec,
            )
            for filename in audio_files:
                source_path = os.path.join(tmpdir, filename)
                resourcepack.write_file_from_disk(f"assets/video/sounds/{filename}", source_path)
            generate_segmented_sounds_json(audio_files, resourcepack, namespace="video")

        init_cmds.append(
            f"scoreboard players set audio_segment video_player {int(math.floor(segment_time * fps + 1e-6))}"
        )

        # handle subtitle
        subs = None
        if subtitle_file:
            subs = load_subtitles_from_file(subtitle_file)

        if subs is None and not no_subtitle:
            subs = extract_and_parse_subtitles_from_video(
                video_file, prefer_ffmpeg=prefer_ffmpeg, ffmpeg_exec=ffmpeg_exec, ffprobe_exec=ffprobe_exec
            )

        if subs:
            init_cmds.append(generate_subtitle_init_mcfunction(subs, fps, scale=subtitle_scale))
            print(f"[Done] Generated subtitles with {len(subs)} entries.")
        else:
            # empty subtitles
            init_cmds.append("data modify storage video_player:subtitle subtitle set value {}")
            print("[Info] No subtitles found, skipping subtitle generation.")

        # save mcfunctions
        mcfunction_dir = "data/video_player/function"

        init_path = os.path.join(mcfunction_dir, "init.mcfunction")
        datapack.write_text(init_path, "\n".join(init_cmds))
        print(f"[Done] Generated datapack init: {init_path}")

        play_loop_path = os.path.join(mcfunction_dir, "play_frame.mcfunction")
        datapack.write_text(play_loop_path, frame_func["loop"])
        print(f"[Done] Generated datapack play loop: {play_loop_path}")
