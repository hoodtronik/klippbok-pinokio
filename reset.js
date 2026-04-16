// CLAUDE-NOTE: Only wipes the venv. Keeps `app/` source, any saved reviewer
// output, the user's .env, and their cache/ thumbnails. Pinokio's Reset button
// is for recovering from a broken install, not starting over — a fresh clone
// covers that case.
module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "env"
    }
  }]
}
