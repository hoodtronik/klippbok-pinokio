// CLAUDE-NOTE: Step-2 minimal install — creates the 3.11 venv and installs
// only `gradio` so we can verify the Pinokio Install -> Start -> Open Web UI
// loop end-to-end before pulling in Klippbok's heavy deps (torch, transformers,
// opencv, scenedetect). Step 3 of the plan expands this to:
//   - Add klippbok[all] to app/requirements.txt
//   - Re-enable the script.start torch.js invocation below
//   - Append a shell.run block that dumps --help for every subcommand into
//     docs/cli_help.txt so the Gradio form fields match real argparse output.
module.exports = {
  run: [
    // Create the 3.11 venv at repo root. `uv venv --python 3.11` auto-downloads
    // a managed CPython 3.11 if the user's system Python doesn't match.
    {
      method: "shell.run",
      params: {
        path: ".",
        message: "uv venv --python 3.11 env"
      }
    },
    // Install Gradio into the new venv.
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    // CLAUDE-NOTE: torch.js call intentionally disabled until step 3 adds
    // klippbok[all] to requirements.txt. Triage needs CUDA/ROCm torch; the
    // default `uv pip install klippbok[all]` would pull CPU torch on Windows.
    // {
    //   method: "script.start",
    //   params: { uri: "torch.js", params: { venv: "../env", path: "app" } }
    // },
  ]
}
