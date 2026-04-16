// CLAUDE-NOTE: Install pipeline — (1) create 3.11 venv, (2) install Klippbok
// + Gradio + Pillow, (3) overwrite torch with a GPU-appropriate build via
// torch.js, (4) dump every subcommand's --help into docs/cli_help.txt so
// the Gradio forms can be verified against authoritative argparse output.
module.exports = {
  run: [
    // 1. Create the 3.11 venv at repo root. `uv venv --python 3.11`
    //    auto-downloads a managed CPython 3.11 if the user's system Python
    //    doesn't match — no preinstall required.
    {
      method: "shell.run",
      params: {
        path: ".",
        message: "uv venv --python 3.11 env"
      }
    },
    // 2. Install Klippbok[all], Gradio, and Pillow.
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app",
        message: "uv pip install -r requirements.txt"
      }
    },
    // 3. Swap in a GPU-aware torch build (NVIDIA CUDA / AMD ROCm or DirectML /
    //    macOS MPS / CPU fallback). torch.js's `when` branches route by
    //    {{gpu}} + {{platform}} + {{arch}}.
    {
      method: "script.start",
      params: { uri: "torch.js", params: { venv: "../env", path: "app" } }
    },
    // 4. Dump argparse --help output for every subcommand into
    //    docs/cli_help.txt. This file is the ground truth for the Gradio
    //    form fields in app.py — regenerate when upgrading Klippbok.
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
