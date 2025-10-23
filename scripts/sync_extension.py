"""Copy core files from web directory to Chrome extension directory."""

import os
import shutil
from pathlib import Path


def sync_extension_files():
    
    web_dir = Path("whisperlivekit/web")
    extension_dir = Path("chrome-extension")
    
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
    

if __name__ == "__main__":

    sync_extension_files()