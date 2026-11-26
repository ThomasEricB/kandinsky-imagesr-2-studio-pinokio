// Removes the app and its venv. The model stays in Pinokio's shared Hugging Face cache
// and your results stay in outputs/.
module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "app"
    }
  }]
}
