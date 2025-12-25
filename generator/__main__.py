if __name__ == "__main__":
    import argparse
    from functools import partial

    from generator.file_utils import PackGenerator, PackMode

    from .cli_utils import ask_metadata
    from .res.callback import finish_callback, processing_callback
    from .video_utils import process_frames_from_video

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
        meta = process_frames_from_video(
            video_path=video_file,
            output_size=target_size,
            output_fps=target_fps,
            callback=partial(processing_callback, resourcepack=resourcepack),
            max_workers=16,
            prefer_ffmpeg=bool(ffmpeg_exec),
            ffmpeg_exec_path=ffmpeg_exec,
        )
        finish_callback(meta, datapack, resourcepack)
