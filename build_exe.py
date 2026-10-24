"""
Build script: packages NEON Market Scanner into a single EXE
with the API key embedded. The EXE runs the Flask dashboard
and opens the browser automatically.

Usage:  python build_exe.py
Output: dist/NeonMarketScanner.exe
"""
import os
import re
import shutil
import subprocess
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
ANALYZER = os.path.join(PROJECT, "analyzer.py")
API_KEY = os.environ.get("ZAI_API_KEY", "SET_ZAI_API_KEY_ENV")


def embed_key():
    """Temporarily embed API key in analyzer.py for the build."""
    src = open(ANALYZER, encoding="utf-8").read()
    patched = re.sub(
        r'DEFAULT_API_KEY\s*=\s*.*',
        f'DEFAULT_API_KEY = os.environ.get("ZAI_API_KEY") or "{API_KEY}"',
        src,
    )
    open(ANALYZER, "w", encoding="utf-8").write(patched)
    print("[1/4] API key embedded in analyzer.py")


def build_exe():
    """Run PyInstaller to create the EXE."""
    print("[2/4] Running PyInstaller (this takes ~2 minutes)...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "NeonMarketScanner",
        "--add-data", f"{PROJECT}\\dashboard.html;.",
        "--add-data", f"{PROJECT}\\index.html;.",
        "--hidden-import", "flask",
        "--hidden-import", "zai",
        "--hidden-import", "faster_whisper",
        "--hidden-import", "yt_dlp",
        "--hidden-import", "numpy",
        "--exclude-module", "onnxruntime",
        "--exclude-module", "matplotlib",
        "--exclude-module", "scipy",
        "--exclude-module", "pandas",
        "--exclude-module", "tkinter",
        "--exclude-module", "PIL",
        "--exclude-module", "pytest",
        "--exclude-module", "IPython",
        "--clean",
        "--noconfirm",
        os.path.join(PROJECT, "server.py"),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT)
    print("[3/4] PyInstaller build done")


def make_launcher():
    """Create a friendly launcher .bat next to the EXE."""
    bat = os.path.join(PROJECT, "dist", "run.bat")
    with open(bat, "w") as f:
        f.write("@echo off\necho Starting NEON Market Scanner...\necho Dashboard will open at http://localhost:5000\nstart http://localhost:5000\nNeonMarketScanner.exe\npause\n")
    print(f"[4/4] Launcher created: {bat}")


def restore_key_removal():
    """Remove the key again so the source tree stays clean."""
    src = open(ANALYZER, encoding="utf-8").read()
    patched = re.sub(
        r'DEFAULT_API_KEY\s*=\s*.*',
        'DEFAULT_API_KEY = os.environ.get("ZAI_API_KEY", "")',
        src,
    )
    open(ANALYZER, "w", encoding="utf-8").write(patched)
    print("[cleanup] API key removed from source")


if __name__ == "__main__":
    embed_key()
    try:
        build_exe()
        make_launcher()
    finally:
        restore_key_removal()
    exe = os.path.join(PROJECT, "dist", "NeonMarketScanner.exe")
    if os.path.exists(exe):
        size_mb = os.path.getsize(exe) / 1024 / 1024
        print(f"\nDONE!  {exe}  ({size_mb:.0f} MB)")
        print("Send this EXE to your friend — it runs out of the box.")
    else:
        print("\nBUILD FAILED — check PyInstaller output above")
