module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "git pull"
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app"
      }
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install -r requirements.txt -r ../server/requirements.txt gradio==6.27.0"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "python ../server/check.py"
    }
  }]
}
