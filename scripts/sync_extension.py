"""Copy core files from web directory to Chrome extension directory."""

import shutil
from pathlib import Path


def sync_extension_files():

    repo_dir = Path(__file__).resolve().parent.parent
    web_dir = repo_dir / "whisperlivekit" / "web"
    extension_dir = repo_dir / "chrome-extension"

    files_to_sync = [
        "live_transcription.html", "live_transcription.js", "live_transcription.css"
    ]

    svg_files = [
        "system_mode.svg",
        "light_mode.svg",
        "dark_mode.svg",
        "settings.svg"
    ]

    for file in files_to_sync:
        src_path = web_dir / file
        dest_path = extension_dir / file

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    for svg_file in svg_files:
        src_path = web_dir / "src" / svg_file
        dest_path = extension_dir / "web" / "src" / svg_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    # PCM mode loads these URLs from /web in both the page and extension.
    for filename in ("pcm_worklet.js", "recorder_worker.js"):
        dest_path = extension_dir / "web" / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(web_dir / filename, dest_path)


if __name__ == "__main__":

    sync_extension_files()
