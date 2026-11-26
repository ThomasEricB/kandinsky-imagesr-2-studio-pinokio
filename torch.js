// Installs the PyTorch build that matches this machine (GPU vendor, generation and driver).
// The stock Pinokio torch.js pins torch 2.7, which is older than the torch>=2.10,<2.14 the
// Kandinsky ImageSR Space needs, so the choice lives in server/install_torch.py instead.
module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "{{args && args.venv ? args.venv : 'env'}}",
      venv_python: "3.12",
      path: "{{args && args.path ? args.path : 'app'}}",
      message: "python ../server/install_torch.py --gpu {{gpu}}"
    }
  }]
}
