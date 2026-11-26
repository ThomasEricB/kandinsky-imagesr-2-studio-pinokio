module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Intel Macs have no PyTorch >= 2.10 wheels. Fail early with a readable message.
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        message: "printf \"Error%s\\n\" \": Intel Macs are not supported (PyTorch 2.10+ has no x86_64 macOS build).\""
      }
    },
    // The official demo Space: Gradio UI + the kandinsky_image_sr pipeline package.
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone https://huggingface.co/spaces/kandinskylab/Kandinsky-ImageSR-2.0-Demo app"
        ]
      }
    },
    // Create the venv on Python 3.12, the version the Space runs on.
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        path: "app",
        message: "python --version"
      }
    },
    // PyTorch first, so the requirements below keep this build instead of pulling PyPI's.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Gradio is not in the Space's requirements.txt (Spaces set it through sdk_version).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt -r ../server/requirements.txt gradio==6.27.0"
        ]
      }
    },
    // Weights (~10.4 GB) go to Pinokio's shared Hugging Face cache. The promo video is skipped.
    {
      method: "hf.download",
      params: {
        _: ["kandinskylab/Kandinsky-ImageSR-2.0"],
        exclude: "assets/*"
      }
    },
    // Report what this machine will run on, and fail loudly if torch can't import.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python ../server/check.py"
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    }
  ]
}
