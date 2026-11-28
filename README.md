<p align="center"><img src="icon.png" width="160" alt="Kandinsky ImageSR 2.0 Studio"></p>

# Kandinsky ImageSR 2.0 Studio

**A new image upscaler made to run on almost any PC: a CPU-only mode, plus memory profiles
for GPUs with 16+ GB, 10 GB, 6 GB or just 3 GB of VRAM.**

A 1-click [Pinokio](https://pinokio.co) launcher for
[Kandinsky ImageSR 2.0](https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0) by KandinskyLab,
a one-step diffusion super-resolution model. It upscales images ×2 or ×4, with a
**creativity** level from 0 (faithful) to 8 (invents more detail).

It runs the official [demo Space](https://huggingface.co/spaces/kandinskylab/Kandinsky-ImageSR-2.0-Demo)
UI unchanged, and adds:

- **Memory profiles** so it runs on far more than the H100 the model card benchmarks:
  16+ GB, 10 GB, 6 GB and 3 GB GPUs, Apple Silicon, Intel Arc, AMD (ROCm) and plain CPU.
- **Saved results.** Every upscale is written to `outputs/`. The Space itself deletes results after an hour.
- **Batch mode** in the web UI, for whole folders or many files.
- **About 2.2 GB less VRAM at identical output.** The model's fp32 modulation layers only ever
  see 19 fixed inputs, so their outputs are precomputed into small tables at load time.

## Requirements

| | |
|---|---|
| Disk | ~10.4 GB model (in Pinokio's shared Hugging Face cache) + ~5 GB Python environment |
| System RAM | 16 GB recommended (the `low`/`minimum` profiles keep the model in RAM) |
| NVIDIA | GTX 900 series and newer. RTX 20 and newer get CUDA 13 (driver ≥ 580) or CUDA 12.8 (driver ≥ 570); GTX 9xx/10xx and Volta get CUDA 12.6 automatically |
| AMD | Linux with ROCm. Windows falls back to CPU |
| Apple | Apple Silicon, 16 GB+ unified memory (24 GB+ comfortable). Intel Macs are not supported |
| Intel | Arc and Core Ultra GPUs (XPU) |
| CPU only | Works, ~40 s per 1024 px output tile on a 16-core desktop CPU |

The installer picks the right PyTorch build for your GPU and driver by itself
(`server/install_torch.py`).

**Tested so far:**

- Linux + RTX 5090 (driver 620, CUDA 13.0), with every profile, VRAM caps from 4 to 10 GB,
  CPU-only mode, and fp16 compute (the pre-Ampere path; 38 dB vs bf16, no NaNs).
- Windows with an NVIDIA GPU: works.
- macOS (Apple Silicon): works.

AMD and Intel GPUs follow the same code paths but haven't been run yet. Reports are welcome.

## How to use

1. In Pinokio, choose **Download from URL** and paste
   `https://github.com/ThomasEricB/kandinsky-imagesr-2-studio-pinokio`.
   Then click **Install**, then **Start**. The web UI opens once the model has loaded.
2. Drop an image, pick ×2 or ×4 and a creativity level, and press **Upscale**. Drag the
   slider to compare with bicubic, or use **Inspect details** for a side-by-side crop.
3. Results are saved in `outputs/` (menu: **Open outputs**) as `<original name>_x4_c4.png` (scale and creativity).

There is no limit on the output size. The Space caps it at 8192 px; here **Limit output
size** (on the Upscale and Batch tabs) can set a cap of 4096, 8192 or 16384 px, or any size
you type, and bigger inputs are then shrunk first. Without a cap, the only cost is memory:
the pipeline blends its tiles into a full-size image on the GPU, about 1 GB per
8192×8192 px of output (4.3 GB at 16384×16384) on top of the profile's usual peak. The
model works on 1024 px output tiles, so time grows with the output area.

### Batch mode

Use the **Batch** tab. **📁 Add folder…** and **🖼️ Add files…** open your system's own file
dialog (several files at once, and as many folders as you like). You can also paste paths
one per line, or drop images into the upload box.

| System | Dialog |
|---|---|
| Linux | the desktop portal, which shows your desktop's own dialog (KDE on Plasma, GNOME on Ubuntu/Fedora, GTK on Xfce, Cinnamon and MATE); falls back to `kdialog` or `zenity` |
| Windows | the standard Explorer dialog, also for folders |
| macOS | Finder's dialog |

Only the images directly inside an added folder are used. Tick **Include subfolders of
added folders** to also scan the folders inside it. It only affects added folders: single
files and dropped images are always upscaled. When scanning, the output folder and earlier results of
this app (`…_x4_c4.png` and the like) are skipped, so they are never upscaled twice.

Results go to `outputs/batch/` by default. A folder's images go to
`outputs/batch/<folder name>/` (keeping the subfolder layout when subfolders are included),
and single files land directly in `outputs/batch/`. Name clashes get a `-2` suffix.

Files whose result already exists are skipped unless you tick *Overwrite*, so an
interrupted batch can simply be run again.

### Memory profiles

`auto` (the default) picks one from free VRAM. To force one, use **Start with memory profile**
in the menu, or set `ISR_PROFILE` in Pinokio's **Configure** tab.

Measured on an RTX 5090 (PCIe 5), upscaling 512 → 2048 px (9 tiles):

| Profile | For GPUs with | What it does | Peak VRAM | Time | Output vs `full` |
|---|---|---|---|---|---|
| `full` | 16+ GB | whole model on the GPU in bf16 | 11.5 GB | 4.2 s | reference |
| `balanced` | 10 GB+ | int8 weights on the GPU | 8.4 GB | 4.6 s | 46 dB PSNR, visually identical |
| `low` | 6 GB+ | bf16 weights streamed from system RAM block by block | 4.5 GB | 4.7 s | **bit-identical** |
| `minimum` | 3 GB+ | streamed + VAE tiling | 1.8 GB | 5.3 s | 34.5 dB, faint tile seams possible |
| `cpu` | no GPU | runs on the processor | — | ~40 s per tile | bf16 on CPU |

"For GPUs with" leaves room for the driver, the desktop and the CUDA context on top of the
measured peak.

`auto` picks `full` with ≥ 12.5 GB free, `low` with ≥ 5.5 GB free, else `minimum`. It never
picks `balanced`. Streaming speed depends on PCIe bandwidth, so on laptops, eGPUs and
PCIe 3 x8 slots `balanced` can be noticeably faster than `low` on a 10–12 GB card.
Try both.

fp8 and nf4 weights were tested too and dropped. They were no faster than int8 and lost
more quality (41.5 dB and 31.5 dB). Processing several tiles per pass was also slower on
the 5090 (5.0 s vs 4.2 s), so `ISR_TILE_BATCH` defaults to 1.

### All settings (Configure tab / `ENVIRONMENT`)

| Variable | Values | Default |
|---|---|---|
| `ISR_PROFILE` | `auto` `full` `balanced` `low` `minimum` `cpu` | `auto` |
| `ISR_DEVICE` | `auto` `cuda` `cuda:1` `xpu` `mps` `cpu` | `auto` |
| `ISR_QUANT` | `auto` `none` `int8` | `auto` (profile) |
| `ISR_VAE_TILING` | `auto` `0` `1` | `auto` (profile) |
| `ISR_TILE_BATCH` | `1`, `2`, … | `1` |
| `ISR_DTYPE` | `auto` `bf16` `fp16` `fp32` | `auto`: bf16, fp16 on pre-Ampere NVIDIA |
| `ISR_ATTENTION` | `native` `flash` `sage` | `native` (PyTorch SDPA) |
| `ISR_BAKE_MODULATION` | `1` `0` | `1` |
| `ISR_OUTPUT_DIR` | a folder | `outputs/` |

The **System** tab in the web UI shows which device, profile and settings are in use.

## API

The app is a Gradio server. When it's running, Pinokio shows its URL (normally
`http://127.0.0.1:7860`). There are two endpoints:

| Endpoint | Inputs | Returns |
|---|---|---|
| `/enhance` | `image`, `scale` (2 or 4), `creativity` (0–8), `max_output_side` (default `0`) | `[[bicubic_png, upscaled_png], summary]` |
| `/batch` | `files` (uploads, list), `paths` (folders/files on the app's machine, one per line), `scale`, `creativity`, `output_folder` (`""` = default), `overwrite`, `subfolders` (default `False`), `max_output_side` (default `0`) | `[log_markdown, gallery]` |

`max_output_side` caps the result's longest side in pixels; `0` (the default) means no cap.

Every `/enhance` call is also saved to `outputs/`.

### Python

```python
# pip install gradio_client
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
(bicubic, upscaled), summary = client.predict(
    handle_file("photo.jpg"), 4, 4, api_name="/enhance"
)
print(summary, upscaled)  # upscaled = local path of the downloaded PNG

# folders and/or files on the machine running the app, one per line
log, gallery = client.predict([], "/path/to/photos\n/path/to/one.jpg", 4, 4, "", False, False, api_name="/batch")
print(log)
```

### JavaScript

```javascript
// npm install @gradio/client
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const image = await (await fetch("https://example.com/photo.jpg")).blob();
const result = await client.predict("/enhance", [handle_file(image), 4, 4]);
const [[bicubic, upscaled], summary] = result.data;
console.log(summary, upscaled.url);
```

### curl

```bash
URL=http://127.0.0.1:7860
# 1. upload the image
FILE=$(curl -s -F "files=@photo.jpg" $URL/gradio_api/upload | python -c 'import json,sys; print(json.load(sys.stdin)[0])')
# 2. start the job
ID=$(curl -s -X POST $URL/gradio_api/call/enhance -H 'Content-Type: application/json' \
  -d "{\"data\": [{\"path\": \"$FILE\", \"meta\": {\"_type\": \"gradio.FileData\"}}, 4, 4]}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["event_id"])')
# 3. read the result (server-sent events). data[0][1].url is the upscaled PNG
curl -sN $URL/gradio_api/call/enhance/$ID
```

## Troubleshooting

- **"Ran out of CUDA memory"**: start with the next lower profile (the message says which).
  Other GPU apps running at the same time count against the free memory `auto` sees at startup.
  For very large outputs, a **Limit output size** cap also helps (see *How to use*).
- **Black or noisy output on GTX 10/16 or RTX 20 series**: set `ISR_DTYPE=fp32` (needs more
  memory, so combine with `low`).
- **`low` or `minimum` is slow**: your PCIe link is the bottleneck; try `balanced`.
- **Logs**: `logs/api/start.js/latest` (Pinokio's terminal shows the same).
- **Free the disk space of the model**: after **Reset**, run `hf cache delete` in Pinokio's terminal and
  select `kandinskylab/Kandinsky-ImageSR-2.0`. Reset alone keeps it, since other apps can share the cache.

## How it works

`install.js` clones the official Space into `app/`, creates a Python 3.12 venv, installs
the PyTorch build that `server/install_torch.py` chooses, then the Space's requirements,
then downloads the weights. `start.js` runs `server/serve.py`. That script imports the Space's
`app.py` unchanged, loads the pipeline through `server/loader.py` (device, profile,
modulation tables, int8, block streaming), and serves the Space's UI next to the Batch and
System tabs. No file of the Space is modified, so **Update** can simply `git pull` it.

## Credits and licenses

- Model, pipeline and demo: [KandinskyLab](https://kandinskylab.ai/),
  [kandinsky-image-sr](https://github.com/kandinskylab/kandinsky-image-sr), MIT.
- VAE: from `black-forest-labs/FLUX.1-schnell`, Apache-2.0.
- Icon: KandinskyLab's Kandinsky SR 2.0 logo, rearranged on a square.
- This launcher: MIT, see [LICENSE](LICENSE).
