module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          GRADIO_SERVER_NAME: "127.0.0.1",
          PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True",
          PYTORCH_ENABLE_MPS_FALLBACK: "1",
          // A profile picked from the Start menu wins over the one in ENVIRONMENT.
          ISR_PROFILE: "{{args && args.profile ? args.profile : (env.ISR_PROFILE || 'auto')}}"
        },
        message: [
          "python ../server/serve.py"
        ],
        on: [{
          // Gradio prints the URL only after the model has loaded, so the UI opens when it is ready.
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
