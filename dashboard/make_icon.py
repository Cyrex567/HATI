"""Make hati.ico from the user's logo PNG (for the .exe icon). Needs Pillow."""
from pathlib import Path

A = Path(__file__).resolve().parent / "static" / "assets"
src = A / "hati_logo.png"
try:
    from PIL import Image
    if not src.exists():
        raise FileNotFoundError("drop your logo at dashboard/static/assets/hati_logo.png first")
    Image.open(src).convert("RGBA").save(A / "hati.ico",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"icon -> {A/'hati.ico'}")
except Exception as e:  # noqa: BLE001
    print(f"icon skipped ({e}); the exe builds fine without it -- remove --icon from build_exe.bat")
