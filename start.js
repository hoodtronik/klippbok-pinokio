// CLAUDE-NOTE: URL capture regex /(http:\/\/[0-9.:]+)/ is the standard
// Pinokio pattern (shared by z-image-turbo, ace-step, and most Gradio launchers).
// Gradio prints "Running on local URL:  http://127.0.0.1:<port>" to stdout on
// boot — the event:done:true stops matching after the first hit, then
// local.set({url:...}) makes it available to pinokio.js's menu function.
//
// {{port}} is a Pinokio template var that injects a free port. app.py accepts
// --port so we never hardcode 7860 and avoid collisions with other Pinokio apps.
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "../env",
        // CLAUDE-NOTE: PYTHONUNBUFFERED=1 + `python -u` is mandatory — without
        // both, Python buffers stdout when Pinokio captures it (non-TTY), so
        // the "Running on local URL" line arrives in a flush at shutdown and
        // the `on` regex below never matches during normal operation. Caught
        // during local smoke test: Monitor hit 30s timeout with 0 stdout lines;
        // same command with `python -u` showed output instantly.
        env: { PYTHONUNBUFFERED: "1" },
        path: "app",
        message: "python -u app.py --host 127.0.0.1 --port {{port}}",
        on: [{
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
