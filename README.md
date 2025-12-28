# Minecraft Video Player

A powerful tool that converts video files into Minecraft Datapacks and Resourcepacks, allowing you to play high-quality videos directly in Minecraft Java Edition using `text_display` entities.

## Features

- **High-Quality Video Playback**: Uses `text_display` entities with custom bitmap fonts to render video frames with high performance.
- **Audio Support**: Automatically segments and synchronizes audio with the video.
- **Subtitle Support**: Can extract subtitles from the video file or load external `.srt`, `.ass`, etc... files.
- **Efficient Processing**: Multithreaded processing for fast generation.
- **Customizable**: Options for subtitle scaling, output naming, and more.
- **No Mods Required**: Works with vanilla Minecraft (Datapack + Resourcepack).

## Requirements

### Software

- **Python 3.8+**
- **FFmpeg** (Not strictly required but strongly recommended for best performance and audio/subtitle extraction)

### Minecraft

- **Minecraft Java Edition 1.20.2+** (Required for `text_display` and macro support)

## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/minecraft_video_player.git
    cd minecraft_video_player
    ```

1. **Install Python dependencies**

   - If you have ffmpeg installed:

       ```bash
       pip install -r requirements.txt
       ```

   - If you do not have ffmpeg installed:

       ```bash
       pip install -r requirements_audio.txt
       ```

## Usage

### Generating the Pack

Run the generator module with your video file:

```bash
python -m generator "path/to/video.mp4" [options]
```

#### Common Options

- `-o`, `--output <name>`: Base name for the output files (e.g., `-o myvideo` creates `myvideo_datapack.zip` and `myvideo_resourcepack.zip`).
- `-s`, `--subtitle <file>`: Path to an external subtitle file.
- `-ss`, `--subtitle-scale <float>`: Scale for subtitle text (default: 2.0).
- `-w`, `--workers <int>`: Number of worker threads (default: 16).
- `-nz`, `--no-zip`: Generate folders instead of zip files.

    Use `-h` or `--help` to see all available options.

**Example:**

```bash
python -m generator "bad_apple.mp4" -o bad_apple -s "bad_apple.srt"
```

### In-Game Instructions

1. **Install the Packs**:
   - Place the generated `*_datapack.zip` into your world's `datapacks` folder.
   - Place the generated `*_resourcepack.zip` into your `resourcepacks` folder and enable it in-game.

2. **Initialize the Player**:
   Run the initialization function to spawn the screen and set up scoreboards.

   ```mcfunction
   /function video_player:init
   ```

   *Note: The screen will spawn at your current location.*

3. **Control Playback**:
   - **Start**:

     ```mcfunction
     /function video_player:start
     ```

   - **Pause/Stop**:

     ```mcfunction
     /function video_player:stop
     ```

     *(This pauses playback; run `start` to resume.*

     *Note: Because audio are segmented with 10 seconds chunks, pausing and resuming may not have immediate effect.)*

   - **Remove/Reset**:

     ```mcfunction
     /function video_player:reset
     ```

     *(This removes the screen entities. Run `init` again to respawn.)*

4. **FPS Adjustment**:
   If your video is not 20 FPS, the generator will suggest a `/tick rate` command during initialization. Run it to ensure correct playback speed.

   ```mcfunction
   /tick rate <fps>
   ```

## How It Works

- **Video**: Frames are split into tiles (max 256x256) and saved as font textures. `text_display` entities cycle through these characters to display the video.
- **Audio**: Audio is segmented into small chunks and played using the `playsound` command, synchronized with the video frames.
- **Subtitles**: Subtitles are displayed using a separate `text_display` entity, updated based on the current frame.

## License

[Apache-2.0 License](LICENSE)
