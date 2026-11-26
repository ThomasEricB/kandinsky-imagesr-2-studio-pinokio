"""Install the PyTorch build that fits this machine into the active venv.

Pinokio's stock torch.js pins torch 2.7, but the Kandinsky ImageSR 2.0 Space
requires torch>=2.10,<2.14, and the right CUDA flavour depends on the GPU
generation and the driver, which only nvidia-smi reports reliably:

  * CUDA 13.0 wheels need driver >= 580 and dropped Maxwell/Pascal/Volta.
  * CUDA 12.8 wheels need driver >= 570 but stop at torch 2.11.
  * CUDA 12.6 wheels still carry sm_50..sm_90 (GTX 9xx/10xx, Titan V, V100)
    but nothing newer than Hopper, so Blackwell can't use them.

Run inside the app venv:  python install_torch.py --gpu <nvidia|amd|intel|apple|none>
Uses only the standard library, so it runs before torch exists.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys

TORCH = "2.13.0"
TORCH_CU128 = "2.11.0"  # newest release that still ships cu128 wheels
INDEX = "https://download.pytorch.org/whl/"


def nvidia_info() -> tuple[float, float] | None:
    """Return (driver major.minor, compute capability) of GPU 0, or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
        driver, cap = (field.strip() for field in out[0].split(","))
        major, minor = driver.split(".")[:2]
        return float(f"{major}.{minor}"), float(cap)
    except Exception as error:  # noqa: BLE001 - any failure means "unknown"
        print(f"nvidia-smi query failed: {error}")
        return None


def choose(gpu: str) -> tuple[str, str | None, str]:
    """Return (torch version, index URL or None for PyPI, reason)."""
    system, machine = sys.platform, platform.machine().lower()

    if system == "darwin":
        if machine != "arm64":
            sys.exit("Error: Intel Macs are not supported (PyTorch >= 2.10 has no x86_64 macOS wheels).")
        return TORCH, None, "Apple Silicon: PyPI wheel with MPS"

    if gpu == "nvidia":
        info = nvidia_info()
        if info is None:
            return TORCH_CU128, INDEX + "cu128", "NVIDIA, nvidia-smi unavailable: cu128 (driver >= 570)"
        driver, cap = info
        print(f"NVIDIA driver {driver}, compute capability {cap}")
        if cap < 5.0:
            return TORCH, INDEX + "cpu", "GPU too old for current CUDA wheels: CPU build"
        if cap < 7.5:
            return TORCH, INDEX + "cu126", "Maxwell/Pascal/Volta: cu126 (last build with sm_50-sm_70)"
        if driver >= 580:
            return TORCH, INDEX + "cu130", "Turing or newer, driver >= 580: cu130"
        if driver >= 570:
            return TORCH_CU128, INDEX + "cu128", "driver 570-579: cu128 (torch 2.11)"
        if cap >= 10.0:
            sys.exit(f"Error: Blackwell GPUs need NVIDIA driver 570 or newer (found {driver}). Update the driver.")
        return TORCH, INDEX + "cu126", f"old driver {driver}: cu126"

    if gpu == "amd":
        if system == "linux" and machine in ("x86_64", "amd64"):
            return TORCH, INDEX + "rocm7.1", "AMD on Linux: ROCm 7.1"
        print("AMD on Windows has no official PyTorch GPU wheel for this torch range: falling back to CPU.")
        return TORCH, INDEX + "cpu", "AMD on Windows: CPU build"

    if gpu == "intel" and machine in ("x86_64", "amd64"):
        return TORCH, INDEX + "xpu", "Intel GPU: XPU build (falls back to CPU at runtime if unusable)"

    if system == "win32" and machine in ("arm64", "aarch64"):
        return TORCH, None, "Windows on ARM: PyPI wheel (CPU)"
    return TORCH, INDEX + "cpu", "no supported GPU: CPU build"


def uv(*args: str) -> None:
    command = ["uv", "pip", "install", "--python", sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", default="none", help="Pinokio's {{gpu}} value")
    args = parser.parse_args()

    version, index, reason = choose(args.gpu.lower())
    print(f"PyTorch choice: torch=={version} from {index or 'PyPI'} ({reason})", flush=True)
    uv(f"torch=={version}", *(["--index-url", index] if index else []))
    try:  # only the "balanced" (int8) profile needs torchao; the others work without it
        uv("torchao")
    except subprocess.CalledProcessError:
        print("torchao could not be installed: the 'balanced' memory profile will be unavailable.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"Error: {' '.join(error.cmd)} failed with exit code {error.returncode}")
