"""Pinokio entry point: the official Space UI, plus memory profiles, saved results and batch mode.

start.js runs this from the cloned Space folder (`app/`), so the Space's own `app.py`,
`assets/` and `kandinsky_image_sr` package are imported as they are, without edits. The
Space's `__main__` only knows one CUDA GPU with the full model; here `loader.load()` picks
the device and memory profile, and the Space's `build_demo()` gets a wrapped pipeline.
"""

from __future__ import annotations

import contextlib
import functools
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

APP_DIR = Path.cwd()
sys.path.insert(0, str(APP_DIR))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

import loader  # noqa: E402


_last_folder: Path | None = None
DIALOG = Path(__file__).resolve().parent / "dialog.py"


def _start_folder() -> Path:
    if _last_folder and _last_folder.is_dir():
        return _last_folder
    pictures = Path.home() / "Pictures"
    return pictures if pictures.is_dir() else Path.home()


def native_pick(kind: str) -> list[str]:
    """Show the operating system's own file dialog (see dialog.py) and return the picked paths.

    It runs in a subprocess so a dialog problem can't take the server down. The dialog
    appears on the machine running the app, which is the same machine as the browser in
    normal Pinokio use.
    """
    global _last_folder
    title = "Choose a folder of images" if kind == "folder" else "Choose images"
    try:
        result = subprocess.run([sys.executable, str(DIALOG), kind, title, str(_start_folder())],
                                capture_output=True, text=True, timeout=1900)
    except (OSError, subprocess.SubprocessError) as error:
        loader.log(f"✖ File dialog failed - {loader.describe_error(error)}")
        raise gr.Error(f"The file dialog failed: {error}", print_exception=False) from error
    for line in result.stderr.strip().splitlines()[-5:]:
        loader.log(f"  dialog: {line}")
    if result.returncode == 3:
        loader.log("✖ No native file dialog available on this system")
        raise gr.Error("No file dialog is available on this system. Type or paste the paths instead.",
                       print_exception=False)
    if result.returncode != 0:
        raise gr.Error("The file dialog failed; see the terminal. You can type or paste paths instead.",
                       print_exception=False)
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if paths:
        first = Path(paths[0])
        _last_folder = first.parent if kind == "folder" or first.is_file() else first
        loader.log(f"Picked {len(paths)} {'folder' if kind == 'folder' else 'file'}{'s' if len(paths) != 1 else ''}: "
                   + ", ".join(paths[:3]) + (" …" if len(paths) > 3 else ""))
    return paths


# --------------------------------------------------------------------------- output size cap


def cap_control() -> gr.Dropdown:
    return gr.Dropdown(
        loader.OUTPUT_CAPS, value=0, label="Limit output size", allow_custom_value=True,
        info="Longest side of the result; bigger inputs are shrunk first. Without a limit the GPU "
             "needs about 1 GB extra per 8192×8192 px of output. Pick a size or type one.")


@contextlib.contextmanager
def space_cap(space, max_output_side: int):
    """Run the Space's enhance() with this cap: it reads its MAX_OUTPUT_SIDE on every call."""
    saved = space.MAX_OUTPUT_SIDE
    space.MAX_OUTPUT_SIDE = max_output_side or 10**9
    try:
        yield
    finally:
        space.MAX_OUTPUT_SIDE = saved


def size_warning(image, scale, cap) -> str:
    """The Space's "image too large" note, for the chosen cap instead of its fixed 8192 px."""
    cap = loader.parse_cap(cap)
    if image is None or not cap or max(image.size) * int(scale) <= cap:
        return ""
    return (f"**Image too large for ×{scale} within the {cap} px limit.** It will be resized before "
            f"upscaling so the output's longest side stays at {cap} px. Choose *No limit* to keep its size.")


@contextlib.contextmanager
def cap_in_space_ui(found: list):
    """While the Space builds its UI, add cap_control() right after its Creativity slider.

    The Space's app.py stays unmodified (Update git-pulls it), so its gr.Slider call for
    Creativity is intercepted instead.
    """
    original = gr.Slider

    def slider(*args, **kwargs):
        made = original(*args, **kwargs)
        if kwargs.get("label") == "Creativity" and not found:
            found.append(cap_control())
        return made

    gr.Slider = slider
    try:
        yield
    finally:
        gr.Slider = original


def wire_cap(space, blocks: gr.Blocks, cap: gr.Dropdown) -> bool:
    """Feed the cap to the Space's /enhance event and to its size warning.

    Returns False, leaving the Space's 8192 px cap in place, if a Space update changed them.
    """
    enhance = next((f for f in blocks.fns.values() if f.api_name == "enhance"), None)
    warning = next((f for f in blocks.fns.values() if f.fn is space._size_warning), None)
    if enhance is None or len(enhance.inputs) != 3 or warning is None or len(warning.inputs) != 2:
        return False
    original = enhance.fn

    # Parameter names are the API's: /enhance(image, scale, creativity, max_output_side).
    def enhance_capped(image, scale, creativity, max_output_side=0):
        token = loader.SOURCE_NAME.set(loader.source_stem(image))
        try:
            with space_cap(space, loader.parse_cap(max_output_side)):
                return original(image, scale, creativity)
        finally:
            loader.SOURCE_NAME.reset(token)

    enhance.fn = enhance_capped
    enhance.inputs.append(cap)
    warning.fn = size_warning
    warning.inputs.append(cap)
    source, scale = warning.inputs[:2]
    with blocks:
        cap.change(size_warning, [source, scale, cap], warning.outputs, queue=False, api_visibility="private")
    return True


def batch_tab(runner: loader.SavingPipeline, settings: loader.Settings) -> None:
    default_out = settings.output_dir / "batch"
    gr.Markdown(
        "Upscale many images with the model that is already loaded. Pick **folders** and/or **files** "
        "on this computer, or drop images in the upload box. Only the images directly in a folder are "
        "used unless you tick *Include subfolders of added folders*. Results are saved as PNG."
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="card"):
            paths = gr.Textbox(
                label="Folders and files on this computer", lines=4, max_lines=12,
                placeholder="Use the buttons below, or paste paths here, one per line.",
            )
            with gr.Row():
                add_folder = gr.Button("📁 Add folder…", size="sm")
                add_files = gr.Button("🖼️ Add files…", size="sm")
                clear = gr.Button("Clear", size="sm")
            files = gr.File(label="…or drop images here", file_count="multiple", file_types=["image"])
            with gr.Group():
                subfolders = gr.Checkbox(
                    label="Include subfolders of added folders", value=False,
                    info="Only affects folders (📁 Add folder… or a pasted folder path): also upscale the "
                         "images in the folders inside them, keeping that layout in the output folder. "
                         "Single files and dropped images are always upscaled.")
            scale = gr.Radio([("×2", 2), ("×4", 4)], value=4, label="Upscale")
            creativity = gr.Slider(0, 8, value=4, step=1, label="Creativity",
                                   info="0 · Faithful   4 · Default   8 · Creative")
            cap = cap_control()
            output = gr.Textbox(label="Output folder", value=str(default_out))
            overwrite = gr.Checkbox(label="Overwrite results that already exist", value=False)
            with gr.Row():
                run = gr.Button("✨ Upscale all", variant="primary")
                stop = gr.Button("Stop", variant="stop")
        with gr.Column(scale=7, elem_classes="card"):
            log = gr.Markdown()
            gallery = gr.Gallery(label="Results", columns=4, height="auto", format="png")

    def append(kind: str, current: str) -> str:
        listed = [line for line in (current or "").splitlines() if line.strip()]
        listed += [p for p in native_pick(kind) if p not in listed]
        return "\n".join(listed)

    add_folder.click(lambda current: append("folder", current), paths, paths,
                     queue=False, api_visibility="private")
    add_files.click(lambda current: append("files", current), paths, paths,
                    queue=False, api_visibility="private")
    clear.click(lambda: ("", None), None, [paths, files], queue=False, api_visibility="private")

    def go(uploaded, paths_text, scale_value, creativity_value, output_text, overwrite_value,
           subfolders_value=False, max_output_side=0):
        scale_value, creativity_value = int(scale_value), int(creativity_value)
        max_output_side = loader.parse_cap(max_output_side)
        out_root = Path(output_text.strip() or default_out).expanduser()
        sources = (paths_text or "").splitlines()
        sources += [item if isinstance(item, str) else item.name for item in uploaded or []]
        pairs, problems = loader.plan_batch(sources, out_root, scale_value, creativity_value,
                                            recursive=bool(subfolders_value))
        for problem in problems:
            loader.log(f"⚠ Batch: {problem}")
        if not pairs:
            message = "Add a folder or some images first." + (f" ({'; '.join(problems)})" if problems else "")
            loader.log(f"✖ Batch not started - {message}")
            raise gr.Error(message, print_exception=False)
        loader.log(f"■ Batch: {len(pairs)} image(s), ×{scale_value}, creativity {creativity_value} → {out_root}")

        lines = [f"**{len(pairs)} image(s)** → `{out_root}`", ""]
        lines += [f"- ⚠️ {problem}" for problem in problems]
        # Gradio only serves files from allowed_paths, so previews work inside the outputs folder.
        previewable = out_root.resolve().is_relative_to(settings.output_dir)
        if not previewable:
            lines.insert(1, "_Previews are shown only for output folders inside the app's outputs folder._")
        done: list[str] = []
        started = time.perf_counter()
        yield "\n".join(lines), done
        counts = {"written": 0, "failed": 0}
        for number, (source, target) in enumerate(pairs, start=1):
            written, errors = loader.upscale_files(
                runner, [(source, target)], scale_value, creativity_value, overwrite_value,
                on_progress=lambda i, n, src, status: lines.append(f"- `{src.name}`: {status}"),
                start=number, total=len(pairs), max_output_side=max_output_side,
            )
            counts["written"] += len(written)
            counts["failed"] += len(errors)
            done += [str(p) for p in written] if previewable else []
            yield "\n".join(lines), done[-40:]  # the gallery shows the latest 40
        seconds = time.perf_counter() - started
        skipped = len(pairs) - counts["written"] - counts["failed"]
        lines += ["", f"Finished in {seconds:.0f} s."]
        loader.log(f"■ Batch done: {counts['written']} written, {skipped} skipped, {counts['failed']} failed, "
                   f"in {seconds:.0f} s → {out_root}")
        yield "\n".join(lines), done[-40:]

    event = run.click(
        go, [files, paths, scale, creativity, output, overwrite, subfolders, cap], [log, gallery],
        api_name="batch", concurrency_id="imagesr",
    )
    stop.click(None, cancels=[event], queue=False, api_visibility="private")


def log_failures(blocks: gr.Blocks, api_name: str) -> None:
    """Print failures of one of the Space's events (e.g. "Upload an image first.") to the terminal.

    Gradio would print its own traceback, ending in "gradio.exceptions.Error: ...", which is
    the text Pinokio stops a shell on. So each failure is logged once, cleanly (pipeline
    failures already by SavingPipeline.run), and Gradio's printout is switched off.
    Unexpected exceptions keep their traceback, reworded by loader.log.
    """
    for block_fn in blocks.fns.values():
        if block_fn.api_name != api_name:
            continue
        original = block_fn.fn

        @functools.wraps(original)
        def wrapped(*args, original=original, **kwargs):
            try:
                return original(*args, **kwargs)
            except gr.Error as error:
                if not getattr(error, "isr_logged", False):
                    loader.log(f"✖ Failed - {loader.describe_error(error)}")
                error.print_exception = False
                raise
            except Exception as error:
                if not getattr(error, "isr_logged", False):
                    loader.log(f"✖ Failed - {loader.describe_error(error)}")
                loader.log(traceback.format_exc().rstrip())
                raise gr.Error(loader.describe_error(error), print_exception=False) from error

        block_fn.fn = wrapped


def system_md(report: loader.LoadReport, settings: loader.Settings) -> str:
    rows = "\n".join(f"- {line}" for line in report.lines())
    return f"""### This machine
{rows}
- Saved results: `{settings.output_dir}`

### Changing the setup
Pick another profile from **Start with memory profile** in Pinokio's menu, or edit the
`ISR_*` settings in Pinokio's **Configure** tab (the launcher's `ENVIRONMENT` file), then
restart. Profiles: `full` (whole model on the GPU), `balanced` (int8 weights on the GPU),
`low` (lossless, blocks streamed from system RAM), `minimum` (streamed + VAE tiling), `cpu`.
"""


def main() -> None:
    settings = loader.Settings()
    pipe, report = loader.load(settings)
    runner = loader.SavingPipeline(pipe, report, settings.output_dir, error_factory=gr.Error)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    # Pillow refuses images above ~179 megapixels ("decompression bomb"), a guard for servers
    # that open strangers' uploads. Here the images are the user's own, and large ones are the point.
    Image.MAX_IMAGE_PIXELS = None

    import app as space  # the Space's app.py, unchanged

    found: list = []
    with cap_in_space_ui(found):
        upscale = space.build_demo(runner)
    if not (found and wire_cap(space, upscale, found[0])):
        loader.log("⚠ The Space's UI changed: its 8192 px output limit stays on the Upscale tab")
    log_failures(upscale, "enhance")
    with gr.Blocks(title="Kandinsky ImageSR 2.0 Studio", fill_width=True, delete_cache=(3600, 3600)) as root:
        with gr.Tabs():
            with gr.Tab("Upscale"):
                upscale.render()
            with gr.Tab("Batch"):
                batch_tab(runner, settings)
            with gr.Tab("System"):
                gr.Markdown(system_md(report, settings))

    root.queue(default_concurrency_limit=1).launch(
        theme=space.theme(),
        css=(space.DEMO_DIR / "theme.css").read_text(encoding="utf-8"),
        allowed_paths=[str(space.ASSETS_DIR), str(settings.output_dir)],
    )


if __name__ == "__main__":
    main()
