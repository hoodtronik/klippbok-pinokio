// CLAUDE-NOTE: Platform/GPU-aware PyTorch install, adapted from
// pinokiofactory/z-image-turbo. NOT yet invoked by install.js — step 3 of the
// build plan enables it alongside adding klippbok[all] to requirements.txt.
//
// Branches (in `when` order):
//   1. NVIDIA + Windows -> CUDA 12.4 wheel
//   2. NVIDIA + Linux   -> CUDA 12.4 wheel
//   3. AMD    + Windows -> DirectML
//   4. AMD    + Linux   -> ROCm 6.1 wheel
//   5. macOS arm64      -> CPU wheel (uses MPS at runtime)
//   6. macOS intel      -> older CPU wheel (last Intel-compatible torch)
//   7. Fallback         -> CPU wheel
//
// Versions are pinned so triage results are reproducible. Upgrade deliberately.
// CLAUDE-NOTE: 2026-04-18 — bumped torch 2.4.1 -> 2.6.0 (torchvision 0.19.1 ->
// 0.21.0) to close CVE-2025-32434 in torch.load weights-only path. Intel Mac
// branch stays at 2.2.2 because 2.6 has no Intel Mac wheels.
module.exports = {
  run: [
    // nvidia windows
    {
      when: "{{gpu === 'nvidia' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps"
      },
      next: null
    },
    // nvidia linux
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall"
      },
      next: null
    },
    // amd windows (DirectML)
    {
      when: "{{gpu === 'amd' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch torch-directml torchvision numpy==1.26.4 --force-reinstall"
      },
      next: null
    },
    // amd linux (rocm)
    {
      when: "{{gpu === 'amd' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.1 --force-reinstall --no-deps"
      },
      next: null
    },
    // apple silicon mac
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
      },
      next: null
    },
    // intel mac
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
      },
      next: null
    },
    // cpu fallback
    {
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: "uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
      }
    }
  ]
}
