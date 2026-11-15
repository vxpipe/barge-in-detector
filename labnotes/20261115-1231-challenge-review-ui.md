# Challenge review UI

- User needs a browser interface while working over SSH, then requested direct
  Tailscale access with a `0.0.0.0` binding.
- Added stdlib Python server and buildless HTML UI. No application dependencies.
  One utterance at a time, blind labels, keyboard decisions, notes, previous,
  skip, search, filters, progress and downloadable annotations.
- Frozen source CSV remains unchanged. Separate annotation store uses source
  SHA256, revision checks, file locking and atomic replacement with fsync.
- Tests first: absent review module failed collection; implemented store and
  server; five focused tests pass, covering persistence, source mismatch,
  stale writes, blind responses, origin protection and HTTP export.
- Running in local tmux `review-ui`, binding `0.0.0.0:8765`; Tailscale endpoint
  returns all 98 cases. Allowed hosts include the machine Tailscale IP and DNS
  name. Temporary check server stopped without touching real annotations.
- Playwright and Chromium were downloaded to transient tool caches, not project
  dependencies. User prohibited Playwright before any browser was launched.
  Switched to installed agent-browser as requested. Its launch failed because
  Chromium has no usable sandbox on this host. Did not disable the sandbox.
  Browser interaction and screenshot verification remain unperformed.
- Mechanical UI detector ran once, no findings, but degraded HTML parser mode
  cannot verify computed contrast or layout. HTTP and Python checks are the
  verification evidence. User-facing operation documented in docs/review-ui.md.
- Completion checks: Ruff format and lint passed; pytest 116 passed,
  one external integration test deselected (9.96 seconds). Frozen source
  SHA256 still matches the original challenge manifest.
