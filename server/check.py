"""Post-install check: print the device this machine will use and fail if torch is broken."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import diffusers
    import gradio
    import torch
except Exception as error:  # noqa: BLE001
    sys.exit(f"Error: the Python environment is incomplete ({error}). Run Install again.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loader  # noqa: E402

LAUNCHER = Path(__file__).resolve().parent.parent
(LAUNCHER / "outputs").mkdir(exist_ok=True)

# Pinokio generates ENVIRONMENT per user and keeps it out of git, so the ISR_* settings ship
# in server/environment.defaults and are added here, once, for the Configure tab.
environment = LAUNCHER / "ENVIRONMENT"
current = environment.read_text(encoding="utf-8") if environment.exists() else ""
if "ISR_PROFILE" not in current:
    defaults = (LAUNCHER / "server" / "environment.defaults").read_text(encoding="utf-8")
    environment.write_text(current.rstrip("\n") + ("\n\n" if current.strip() else "") + defaults, encoding="utf-8")
    print("Added the ISR_* settings to ENVIRONMENT (Pinokio's Configure tab).")

device = loader.pick_device("auto")
print(f"torch {torch.__version__}, diffusers {diffusers.__version__}, gradio {gradio.__version__}")
print(f"Device: {loader.describe_device(device)}")
print(f"Auto memory profile: {loader.auto_profile(device)}")
if device.type == "cpu":
    print("No usable GPU found: the app will run on the CPU, which takes minutes per image.")
