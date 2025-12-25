import argparse
import math
import os
from functools import partial
from tempfile import TemporaryDirectory

from .audio_utils import segment_audio
from .cli_utils import ask_metadata
from .file_utils import PackGenerator, PackMode
from .res.audio import generate_segmented_sounds_json
from .res.callback import generate_frame_related, processing_callback
from .res.subtitle import generate_subtitle_init_mcfunction
from .video_utils import process_frames_from_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft Video Player Generator")
    parser.add_argument("video_file", help="Path to the video file")

    # Zip options (default: True)
    zip_group = parser.add_mutually_exclusive_group(required=False)
    zip_group.add_argument(
        "-z", "--zip", dest="zip", action="store_true", help="Generate as zip files (default)"
    )
    zip_group.add_argument(
        "--no-zip", dest="zip", action="store_false", help="Generate as folders instead of zip files"
    )
    parser.set_defaults(zip=True)

    # Resource pack options
    parser.add_argument("-nr", "--no-resourcepack", action="store_true", help="Do not generate resource pack")

    # Output filename option
    parser.add_argument(
        "-o",
        "--output",
        help="Base name for output files (e.g. 'myvideo' -> 'myvideo_datapack', 'myvideo_resourcepack')",
    )

    args = parser.parse_args()

    video_file = args.video_file
    is_zip = args.zip
    no_resourcepack = args.no_resourcepack
    output_base = args.output

    if no_resourcepack:
        raise NotImplementedError("Datapack-only generation is not yet supported.")

    target_size, target_fps, ffmpeg_exec = ask_metadata(video_file)

    mode = PackMode.ZIP if is_zip else PackMode.FOLDER

    datapack_name = "datapack"
    resourcepack_name = "resourcepack"
    if output_base:
        datapack_name = f"{output_base}_datapack"
        resourcepack_name = f"{output_base}_resourcepack"
    if is_zip:
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
            max_workers=16,
            prefer_ffmpeg=bool(ffmpeg_exec),
            ffmpeg_exec_path=ffmpeg_exec,
        )
        frame_func = generate_frame_related(meta, resourcepack)
        init_cmds = [frame_func["init"]]

        fps = meta["fps"]

        # handle audio
        segment_time = 10  # audio segment time (seconds)
        with TemporaryDirectory() as tmpdir:
            audio_files = segment_audio(meta["path"], tmpdir, segment_time=segment_time)
            for filename in audio_files:
                source_path = os.path.join(tmpdir, filename)
                resourcepack.write_file_from_disk(f"assets/video/sounds/{filename}", source_path)
            generate_segmented_sounds_json(audio_files, resourcepack, namespace="video")

        init_cmds.append(
            f"scoreboard players set audio_segment video_player {int(math.floor(segment_time * fps + 1e-6))}"
        )

        # handle subtitle
        subtitle_path = "subtitle.srt"
        if os.path.exists(subtitle_path):
            init_cmds.append(generate_subtitle_init_mcfunction(subtitle_path, fps, datapack))
        else:
            init_cmds.append("data merge storage video_player:subtitle {}")  # empty subtitles

        # save mcfunctions
        mcfunction_dir = "data/video_player/function"

        init_path = os.path.join(mcfunction_dir, "init.mcfunction")
        datapack.write_text(init_path, "\n".join(init_cmds))
        print(f"[Done] Generated datapack init: {init_path}")

        play_loop_path = os.path.join(mcfunction_dir, "play_frame.mcfunction")
        datapack.write_text(play_loop_path, frame_func["loop"])
        print(f"[Done] Generated datapack play loop: {play_loop_path}")
