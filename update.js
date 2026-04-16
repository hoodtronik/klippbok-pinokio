// CLAUDE-NOTE: Update mirrors install steps 2-4: pull code, re-sync deps,
// re-apply GPU torch (in case torch version bumped), re-dump --help (catches
// CLI surface drift from the Klippbok upstream).
module.exports = {
  run: [
    {
      method: "shell.run",
      params: { message: "git pull" }
    },
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    {
      method: "script.start",
      params: { uri: "torch.js", params: { venv: "../env", path: "app" } }
    },
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app",
        message: "python dump_help.py"
      }
    }
  ]
}
