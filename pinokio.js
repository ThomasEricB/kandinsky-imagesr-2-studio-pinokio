const profiles = [
  { profile: "full", text: "Full GPU (16 GB+)", icon: "fa-solid fa-bolt" },
  { profile: "balanced", text: "Balanced (10 GB+, int8, best for laptops)", icon: "fa-solid fa-scale-balanced" },
  { profile: "low", text: "Low VRAM (6 GB+, streamed, lossless)", icon: "fa-solid fa-feather" },
  { profile: "minimum", text: "Minimum (3 GB+, streamed + VAE tiling)", icon: "fa-solid fa-battery-quarter" },
  { profile: "cpu", text: "CPU only (slow)", icon: "fa-solid fa-microchip" }
]

module.exports = {
  version: "8.0.0",
  title: "Kandinsky ImageSR 2.0 Studio",
  description: "New image upscaler that was made to run on almost any PC: cpu-only mode + memory profiles for GPUs with 16+ GB, 10 GB, 6 GB or just 3 GB of VRAM. One-step diffusion image super-resolution (×2 / ×4) by KandinskyLab, with batch mode. https://huggingface.co/kandinskylab/Kandinsky-ImageSR-2.0",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }, {
            icon: "fa-solid fa-folder-open",
            text: "Open outputs",
            href: "outputs?fs=true",
          }]
        } else {
          return [{
            default: true,
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }, {
          icon: "fa-solid fa-sliders",
          text: "Start with memory profile",
          menu: profiles.map((p) => ({
            icon: p.icon,
            text: p.text,
            href: "start.js",
            params: { profile: p.profile }
          }))
        }, {
          icon: "fa-solid fa-folder-open",
          text: "Open outputs",
          href: "outputs?fs=true",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Reset",
          href: "reset.js",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
