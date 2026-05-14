"""
Build script to create a standalone .exe with PyInstaller.
Run: python build_exe.py
"""

import PyInstaller.__main__

PyInstaller.__main__.run([
    "photo_scan_gui.py",
    "--onefile",
    "--windowed",
    "--name", "PhotoScan",
    "--hidden-import", "extract_photos",
    "--add-data", "extract_photos.py:.",
])
