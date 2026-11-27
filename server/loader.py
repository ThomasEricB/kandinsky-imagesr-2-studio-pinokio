"""Load Kandinsky ImageSR 2.0 on whatever this machine has, from big GPUs down to the CPU.

The official Space only knows one setup: the whole bf16 pipeline on one CUDA GPU. This
module adds device detection (CUDA, ROCm, XPU, MPS, CPU) and memory profiles, and keeps
everything else (pipeline, tiling, colour handling, PNG writer) from the Space's own
`kandinsky_image_sr` package, which must be importable (run from the `app/` folder).

Settings come from environment variables, normally set through Pinokio's ENVIRONMENT file
or the Start menu:

    ISR_PROFILE     auto | full | balanced | low | minimum | cpu
    ISR_DEVICE      auto | cuda | cuda:1 | xpu | mps | cpu
    ISR_QUANT       auto | none | int8                    (auto = what the profile says)
    ISR_VAE_TILING  auto | 0 | 1                          (auto = what the profile says)
    ISR_DTYPE       auto | bf16 | fp16 | fp32
    ISR_TILE_BATCH  1 | 2 | ...                           (tiles per transformer pass)
    ISR_ATTENTION   native | flash | sage                 (flash/sage need their package)
    ISR_BAKE_MODULATION  1 | 0                            (see bake_modulation)
    ISR_OUTPUT_DIR  folder for saved results (default: <launcher>/outputs)
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Two harmless startup warnings from torchao, which only the "balanced" (int8) profile uses:
# torchao 0.18 still registers enums with a pytree call that torch 2.13 deprecated, and
# diffusers 0.41 looks for old torchao classes needed only to load checkpoints saved already
# quantized by torchao, which this app never does. Only these two loggers are quietened.
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("diffusers.quantizers.torchao.torchao_quantizer").setLevel(logging.ERROR)

REPO_ID = "kandinskylab/Kandinsky-ImageSR-2.0"
# Optional cap on the result's longest side, chosen in the UI (0 = none, the default). The
# Space caps at 8192 px. Here the only real cost is memory: the pipeline blends its tiles into
# an fp32 image on the compute device, 16 bytes per output pixel (1 GiB at 8192 x 8192).
OUTPUT_CAPS = [("No limit", 0), ("4096 px", 4096), ("8192 px (the Space's limit)", 8192), ("16384 px", 16384)]
GB = 1024**3
LAUNCHER_DIR = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# What each profile does. Measured on an RTX 5090 (PCIe 5) upscaling 512 -> 2048 px:
#
#   profile   weights            peak VRAM   time    output vs full
#   full      bf16, on GPU       11.5 GB     4.2 s   reference
#   balanced  int8, on GPU        8.4 GB     4.6 s   46 dB PSNR (visually identical)
#   low       bf16, streamed      4.5 GB     4.7 s   bit-identical
#   minimum   streamed + VAE      1.8 GB     5.3 s   34.5 dB (VAE tile seams)
#             tiling
#
# "Streamed" keeps the transformer in system RAM and copies one block at a time to the GPU
# on a side CUDA stream, overlapping the copy with compute. It is lossless, but its speed
# depends on PCIe bandwidth; on slow links (laptops, PCIe 3 x8, eGPUs) "balanced" can be
# faster. "auto" never picks balanced for that reason: it prefers lossless.
PROFILES: dict[str, dict] = {
    "full": {"quant": "none", "offload": False, "vae_tiling": False},
    "balanced": {"quant": "int8", "offload": False, "vae_tiling": False},
    "low": {"quant": "none", "offload": True, "vae_tiling": False},
    "minimum": {"quant": "none", "offload": True, "vae_tiling": True},
    "cpu": {"quant": "none", "offload": False, "vae_tiling": False},
}
# Free memory (GiB) auto needs before it picks a profile (peak reserved + ~1 GiB margin).
PROFILE_MIN_FREE_GB = {"full": 12.5, "low": 5.5}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


@dataclass
class Settings:
    profile: str = field(default_factory=lambda: _env("ISR_PROFILE", "auto").lower())
    device: str = field(default_factory=lambda: _env("ISR_DEVICE", "auto").lower())
    quant: str = field(default_factory=lambda: _env("ISR_QUANT", "auto").lower())
    dtype: str = field(default_factory=lambda: _env("ISR_DTYPE", "auto").lower())
    # More tiles per pass was slower on an RTX 5090 (5.0 s vs 4.2 s for 4 tiles) and costs
    # ~2.5 GiB per extra tile, so the default stays at one.
    tile_batch: str = field(default_factory=lambda: _env("ISR_TILE_BATCH", "1").lower())
    attention: str = field(default_factory=lambda: _env("ISR_ATTENTION", "native").lower())
    bake_modulation: bool = field(default_factory=lambda: _env("ISR_BAKE_MODULATION", "1") != "0")
    output_dir: Path = field(
        default_factory=lambda: Path(_env("ISR_OUTPUT_DIR", str(LAUNCHER_DIR / "outputs"))).resolve()
    )


# --------------------------------------------------------------------------- devices


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def pick_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():  # also true for ROCm builds
        return torch.device("cuda")
    if _xpu_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_info(device: torch.device) -> tuple[float, float] | None:
    """(free, total) GiB on the device, or None where it can't be known (CPU, MPS)."""
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return free / GB, total / GB
        if device.type == "xpu":
            if hasattr(torch.xpu, "mem_get_info"):
                free, total = torch.xpu.mem_get_info(device)
                return free / GB, total / GB
            total = torch.xpu.get_device_properties(device).total_memory
            return total / GB, total / GB
        if device.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):
            total = torch.mps.recommended_max_memory() / GB
            return total, total
    except Exception:  # noqa: BLE001 - memory probing is best effort
        return None
    return None


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
        label = f"{name} ({backend})"
    elif device.type == "xpu":
        label = f"{torch.xpu.get_device_name(device)} (Intel XPU)"
    elif device.type == "mps":
        label = "Apple GPU (MPS)"
    else:
        label = f"CPU ({os.cpu_count()} threads)"
    memory = memory_info(device)
    if memory:
        label += f", {memory[0]:.1f} of {memory[1]:.1f} GiB free"
    return label


def auto_profile(device: torch.device) -> str:
    if device.type == "cpu":
        return "cpu"
    memory = memory_info(device)
    if memory is None:  # MPS without a memory API: unified memory, try the full model
        return "full"
    free = memory[0]
    for name in ("full", "low"):
        if free >= PROFILE_MIN_FREE_GB[name]:
            return name
    return "minimum"


def pick_dtype(device: torch.device, preference: str) -> torch.dtype:
    named = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if preference in named:
        return named[preference]
    if device.type == "cuda" and not torch.version.hip and torch.cuda.get_device_capability(device)[0] < 8:
        # Turing and older have no bf16 tensor cores. fp16 is fast there; the VAE stays fp32.
        return torch.float16
    return torch.bfloat16


# --------------------------------------------------------------------------- loading


@dataclass
class LoadReport:
    device: torch.device
    device_label: str
    profile: str
    quant: str
    dtype: torch.dtype
    offload: bool
    vae_tiling: bool
    tile_batch: int
    attention: str
    load_seconds: float
    notes: list[str]

    def lines(self) -> list[str]:
        out = [
            f"Device: {self.device_label}",
            f"Profile: {self.profile} (weights {self.quant}, compute {str(self.dtype).removeprefix('torch.')}"
            f"{', group offload' if self.offload else ''}{', VAE tiling' if self.vae_tiling else ''})",
            f"Tiles per pass: {self.tile_batch}, attention: {self.attention}",
            f"Model loaded in {self.load_seconds:.1f} s",
        ]
        return out + self.notes


def _from_pretrained(dtype: torch.dtype):
    from kandinsky_image_sr import KandinskyImageSRPipeline

    try:  # install.js already put the weights in the shared HF cache; don't touch the network
        return KandinskyImageSRPipeline.from_pretrained(REPO_ID, dtype=dtype, local_files_only=True)
    except OSError:
        print(f"{REPO_ID} is not in the Hugging Face cache yet: downloading (~10.4 GB).", flush=True)
        return KandinskyImageSRPipeline.from_pretrained(REPO_ID, dtype=dtype)


def bake_modulation(pipe, device: torch.device, compute_dtype: torch.dtype) -> float:
    """Replace each block's fp32 modulation MLP by a lookup table of its outputs.

    The release bakes text conditioning away, so the time embedding a block sees is always
    one row of `transformer.time_embed_table` (19 rows, one per supported timestep). The
    modulation MLP (512 -> 9 x 2560, kept in fp32) therefore only ever produces 19 different
    outputs. Precomputing them on the target device, under the same autocast the pipeline
    uses, gives bit-identical results and frees about 2.4 GB of weights.

    Returns the GiB of parameters freed.
    """
    transformer = pipe.transformer
    table = transformer.time_embed_table
    autocast = pipe._autocast(device, compute_dtype)
    freed = 0
    with torch.no_grad():
        for block in transformer.visual_transformer_blocks:
            modulation = block.visual_modulation
            freed += sum(p.numel() * p.element_size() for p in modulation.parameters())
            modulation.to(device)
            # One row at a time: at run time the GEMM sees one tile per row too, and a
            # different batch size can pick a different kernel and round differently.
            with autocast:
                outputs = torch.cat([modulation(row.to(device)) for row in table.split(1)])
            block.visual_modulation = ModulationLookup(table, outputs.detach().cpu())
            del modulation, outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return freed / GB


class ModulationLookup(torch.nn.Module):
    """Stands in for a Kandinsky5Modulation whose inputs come from a fixed table."""

    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("inputs", inputs.detach().clone().float(), persistent=False)
        self.register_buffer("outputs", outputs, persistent=False)

    def forward(self, time_embed: torch.Tensor) -> torch.Tensor:
        # Each row of time_embed is an exact copy of one row of the table.
        match = (time_embed.float()[:, None, :] == self.inputs[None]).all(dim=-1)
        if not bool(match.any(dim=1).all()):
            raise RuntimeError("time embedding is not one of the baked rows; set ISR_BAKE_MODULATION=0")
        return self.outputs.index_select(0, match.float().argmax(dim=1))


def _quantizable(module: torch.nn.Module, fqn: str) -> bool:
    # Only the big linears of the 50 transformer blocks. Embeddings, the output layer and the
    # (fp32) modulation stay as they are.
    return isinstance(module, torch.nn.Linear) and fqn.startswith("visual_transformer_blocks.") and (
        "modulation" not in fqn
    )


def quantize_transformer(transformer: torch.nn.Module, quant: str, device: torch.device) -> None:
    """int8 weight-only (torchao). fp8 and nf4 were tried and dropped: on an RTX 5090 they were
    no faster and lost more quality (fp8 41.5 dB, nf4 31.5 dB vs int8 46 dB PSNR)."""
    if quant != "int8":
        raise ValueError(f"unknown ISR_QUANT {quant!r}; use auto, none or int8")
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    transformer.to(device)
    quantize_(transformer, Int8WeightOnlyConfig(), filter_fn=_quantizable)


def load(settings: Settings | None = None) -> tuple[object, LoadReport]:
    settings = settings or Settings()
    started = time.perf_counter()
    notes: list[str] = []

    device = pick_device(settings.device)
    device_label = describe_device(device)
    profile = settings.profile if settings.profile != "auto" else auto_profile(device)
    if profile not in PROFILES:
        raise ValueError(f"unknown ISR_PROFILE {profile!r}; use one of auto, {', '.join(PROFILES)}")
    if device.type == "cpu":
        profile = "cpu"
    recipe = dict(PROFILES[profile])
    if settings.quant != "auto":
        recipe["quant"] = settings.quant
    if device.type not in ("cuda", "xpu"):
        recipe["offload"] = False  # nothing to offload to on unified memory or the CPU
    dtype = pick_dtype(device, settings.dtype)

    print(f"Loading {REPO_ID} for {device_label}, profile {profile}", flush=True)
    pipe = _from_pretrained(dtype)
    if dtype != torch.bfloat16 or device.type == "cpu":
        pipe.vae.to(torch.float32)  # fp16 overflows in the FLUX VAE; fp32 is also faster on CPU

    if settings.bake_modulation:
        try:
            freed = bake_modulation(pipe, device, dtype)
            notes.append(f"Modulation baked into lookup tables ({freed:.1f} GiB of weights freed).")
        except Exception as error:  # noqa: BLE001 - keep the original modules if this fails
            notes.append(f"Modulation baking skipped: {error}")

    if recipe["quant"] != "none":
        quantize_transformer(pipe.transformer, recipe["quant"], device)

    if recipe["offload"]:
        pipe.transformer.enable_group_offload(
            onload_device=device,
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        pipe.vae.to(device)
    else:
        pipe.to(device)

    tiling = _env("ISR_VAE_TILING", "auto")
    if tiling in ("0", "1"):
        recipe["vae_tiling"] = tiling == "1"
    if recipe["vae_tiling"]:
        _enable_vae_tiling(pipe.vae)

    attention = settings.attention
    if attention != "native":
        try:
            pipe.transformer.set_attention_backend(attention)
        except Exception as error:  # noqa: BLE001
            notes.append(f"Attention backend {attention!r} unavailable ({error}); using PyTorch SDPA.")
            attention = "native"

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tile_batch = max(1, int(settings.tile_batch)) if settings.tile_batch.isdigit() else 1

    report = LoadReport(
        device=device, device_label=describe_device(device), profile=profile, quant=recipe["quant"],
        dtype=dtype, offload=recipe["offload"], vae_tiling=recipe["vae_tiling"], tile_batch=tile_batch,
        attention=attention, load_seconds=time.perf_counter() - started, notes=notes,
    )
    for line in report.lines():
        print(line, flush=True)
    return pipe, report


def _enable_vae_tiling(vae) -> None:
    """Decode 1024 px tiles as 512 px VAE tiles (the default threshold is the full 1024)."""
    vae.enable_tiling()
    for name, value in (
        ("tile_sample_min_size", 512), ("tile_sample_min_height", 512), ("tile_sample_min_width", 512),
        ("tile_latent_min_size", 64), ("tile_latent_min_height", 64), ("tile_latent_min_width", 64),
    ):
        if hasattr(vae, name):
            setattr(vae, name, value)


# --------------------------------------------------------------------------- inference


def parse_cap(value) -> int:
    """The output cap from the UI or the API: a number of pixels, a "4096 px" string, or 0/None."""
    text = str(value or 0).lower().replace("px", "").strip()
    try:
        return max(0, int(float(text)))
    except ValueError:
        return 0


def fit_input(image, scale: int, max_output_side: int = 0):
    """Shrink inputs whose output would pass `max_output_side` (0 = no limit)."""
    from PIL import Image

    if not max_output_side:
        return image
    limit = max(1, max_output_side // scale)
    if max(image.size) <= limit:
        return image
    ratio = limit / max(image.size)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(path)


class OutOfMemory(RuntimeError):
    pass


def _oom_message(report: LoadReport) -> str:
    lower = {"full": "balanced", "balanced": "low", "low": "minimum", "minimum": "cpu"}.get(report.profile)
    hint = f" Start with the '{lower}' memory profile instead." if lower else ""
    return f"Ran out of {report.device.type.upper()} memory in the '{report.profile}' profile.{hint}"


_ERROR_LABEL = re.compile(r"(\w*(?:Error|Exception)):")


def log(message: str) -> None:
    """Print to the terminal Pinokio shows.

    Pinokio stops a shell whose output contains "Error:" (its default break pattern), which
    would take the whole server down over one failed image. So "SomeError:" is printed as
    "SomeError -" instead.
    """
    print(_ERROR_LABEL.sub(r"\1 -", message), flush=True)


def describe_error(error: BaseException) -> str:
    message = getattr(error, "message", None)
    if type(error).__name__ == "Error" and isinstance(message, str):
        return message  # gradio.Error: already a sentence meant for the user
    text = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return f"{type(error).__name__} - {text}" if text else type(error).__name__


def tile_count(pipe, width: int, height: int, scale: int) -> int:
    """How many 1024 px tiles the pipeline will run for this input (mirrors its own tiling)."""
    from kandinsky_image_sr.tiling import tile_positions

    lq_tile = int(pipe.config.tile_size) // scale
    stride = max(1, int(lq_tile * (1.0 - float(pipe.config.overlap))))
    rows = tile_positions(max(height, lq_tile), lq_tile, stride)
    cols = tile_positions(max(width, lq_tile), lq_tile, stride)
    return len(rows) * len(cols)


class SavingPipeline:
    """Wraps the pipeline for the Space's UI: adds tile batching, saves every result, reports
    progress, results and failures in the terminal, and turns out-of-memory errors into a
    readable message (the Space only keeps results for 1 h)."""

    def __init__(self, pipe, report: LoadReport, output_dir: Path,
                 error_factory: type[Exception] = OutOfMemory) -> None:
        self.pipe = pipe
        self.report = report
        self.output_dir = output_dir
        self.error_factory = error_factory
        self.last_saved: Path | None = None
        self._bar = None
        # The pipeline has no progress callback; count tiles as its tile loop calls _infer_tiles.
        infer_tiles = pipe._infer_tiles

        def counted(pixels, *args, **kwargs):
            out = infer_tiles(pixels, *args, **kwargs)
            if self._bar is not None:
                self._bar.update(int(pixels.shape[0]))
            return out

        pipe._infer_tiles = counted

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def run(self, image, upscale_factor: int, creativity: int):
        from tqdm.auto import tqdm

        width, height = image.size
        tiles = tile_count(self.pipe, width, height, upscale_factor)
        log(f"▶ Upscaling {width}×{height} ×{upscale_factor}, creativity {creativity} → "
            f"{width * upscale_factor}×{height * upscale_factor} ({tiles} tile{'s' if tiles != 1 else ''})")
        self._bar = tqdm(total=tiles, unit="tile", desc="  progress", file=sys.stdout,
                         dynamic_ncols=True, mininterval=0.5)
        try:
            return self.pipe(
                image, upscale_factor=upscale_factor, creativity=creativity,
                tile_batch_size=self.report.tile_batch,
            ).images[0]
        except torch.OutOfMemoryError as error:
            if self.report.device.type == "cuda":
                torch.cuda.empty_cache()
            message = _oom_message(self.report)
            log(f"✖ Failed - {message}")
            failure = self.error_factory(message)
            failure.isr_logged = True
            failure.print_exception = False  # gradio.Error: don't print "...Error: ..." again
            raise failure from error
        except Exception as error:
            log(f"✖ Failed - {describe_error(error)}")
            error.isr_logged = True
            raise
        finally:
            self._bar.close()
            self._bar = None

    def __call__(self, image, *, upscale_factor: int = 4, creativity: int | None = 4, **_ignored):
        from kandinsky_image_sr.png import save_png

        started = time.perf_counter()
        result = self.run(image, upscale_factor, creativity)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = _unique(self.output_dir / f"{stamp}_x{upscale_factor}_c{creativity}.png")
        save_png(result, path)
        self.last_saved = path
        log(f"✔ Saved {path} ({result.width}×{result.height}, {time.perf_counter() - started:.1f} s)")
        return _Output([result])


@dataclass
class _Output:
    images: list


# --------------------------------------------------------------------------- batch


def find_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Images directly in `folder`, or in it and all its subfolders when `recursive`."""
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


# Results this app writes: <name>_x<2|4>_c<0-8>.png, or <name>-<n>_x..._c....png on a name clash.
RESULT_NAME = re.compile(r"_x[24]_c[0-8]$")


def is_result(path: Path) -> bool:
    return path.suffix.lower() == ".png" and bool(RESULT_NAME.search(path.stem))


def plan_batch(sources: Iterable[str | Path], output_root: Path, scale: int, creativity: int,
               recursive: bool = False) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Turn picked folders and files into (source image, result path) pairs.

    A folder's images go to `output_root/<folder name>/` (keeping the subfolder layout when
    `recursive`); single files go straight into `output_root`. Subfolders are only scanned
    when `recursive` is set. Returns (pairs, problems), e.g. missing paths.

    When scanning a folder, the output folder and earlier results of this app are left out:
    otherwise a folder that holds its own results (output folder inside the input folder)
    gets its upscales upscaled again on every run. Files picked one by one are always used.
    """
    pairs: list[tuple[Path, Path]] = []
    problems: list[str] = []
    taken: set[Path] = set()

    def add(source: Path, target: Path) -> None:
        n = 2
        while target in taken:  # two picked files with the same name
            target = target.with_name(f"{source.stem}-{n}_x{scale}_c{creativity}.png")
            n += 1
        taken.add(target)
        pairs.append((source, target))

    for raw in sources:
        path = Path(str(raw).strip()).expanduser()
        if not str(raw).strip():
            continue
        if path.is_dir():
            out = output_root.resolve()
            found = find_images(path, recursive)
            images = [p for p in found if not is_result(p) and not p.resolve().is_relative_to(out)]
            if len(images) < len(found):
                problems.append(f"skipped {len(found) - len(images)} earlier result(s) of this app in {path}")
            if not images:
                problems.append(f"no images to upscale in {path}")
            for image in images:
                relative = image.relative_to(path)
                add(image, output_root / path.name / relative.parent / f"{relative.stem}_x{scale}_c{creativity}.png")
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            add(path, output_root / f"{path.stem}_x{scale}_c{creativity}.png")
        elif path.is_file():
            problems.append(f"not an image: {path}")
        else:
            problems.append(f"not found: {path}")
    return pairs, problems


def upscale_files(
    runner: SavingPipeline,
    items: Iterable[tuple[Path, Path]],
    scale: int,
    creativity: int,
    overwrite: bool = False,
    on_progress: Callable[[int, int, Path, str], None] | None = None,
    start: int = 1,
    total: int | None = None,
    max_output_side: int = 0,
) -> tuple[list[Path], list[str]]:
    """Upscale (source, target) pairs, logging each one to the terminal.

    `start`/`total` number the images when a caller feeds a longer batch in pieces.
    Returns (written files, error messages).
    """
    from kandinsky_image_sr.image_processor import to_srgb
    from kandinsky_image_sr.png import save_png
    from PIL import Image

    items = list(items)
    written: list[Path] = []
    errors: list[str] = []
    total = total or len(items)
    for index, (source, target) in enumerate(items, start=start):
        log(f"[{index}/{total}] {source}")
        if target.exists() and not overwrite:
            status = "skipped (exists)"
            log(f"↷ Skipped, result exists: {target}")
        else:
            try:
                started = time.perf_counter()
                with Image.open(source) as opened:
                    image = fit_input(to_srgb(opened), scale, max_output_side)
                result = runner.run(image, scale, creativity)
                target.parent.mkdir(parents=True, exist_ok=True)
                save_png(result, target)
                written.append(target)
                status = f"{image.width}x{image.height} -> {result.width}x{result.height} " \
                         f"in {time.perf_counter() - started:.1f} s"
                log(f"✔ Saved {target} ({result.width}×{result.height}, {time.perf_counter() - started:.1f} s)")
            except runner.error_factory:
                raise  # out of memory: every following image would fail the same way
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the batch
                status = f"failed: {describe_error(error)}"
                errors.append(f"{source.name}: {describe_error(error)}")
                if not getattr(error, "isr_logged", False):  # run() already logged pipeline failures
                    log(f"✖ Failed - {source}: {describe_error(error)}")
        if on_progress:
            on_progress(index, total, source, status)
    return written, errors
