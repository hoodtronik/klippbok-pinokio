// CLAUDE-NOTE: `git pull` first, then re-sync deps in case requirements.txt
// changed. The torch.js call is commented out here for the same reason as in
// install.js — step 3 will re-enable it alongside the klippbok[all] migration.
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    // {
    //   method: "script.start",
    //   params: { uri: "torch.js", params: { venv: "../env", path: "app" } }
    // },
  ]
}
