# Challenge review UI

Start from the repository root:

```shell
uv run python -m barge_in_detector.review
```

Default: `http://127.0.0.1:8765`. For SSH forwarding, run on your own computer:

```shell
ssh -N -L 8765:127.0.0.1:8765 user@server
```

For direct Tailscale access, bind all interfaces and allow the machine address:

```shell
uv run python -m barge_in_detector.review --host 0.0.0.0 --port 8765 \
  --allow-host YOUR_TAILSCALE_IP --allow-host YOUR_TAILSCALE_DNS_NAME
```

Open `http://YOUR_TAILSCALE_IP:8765`. The server has no sign-in; use the
trusted network intended for this workflow. It rejects unconfigured Host
headers and cross-origin saves. Repeat `--allow-host` for other hostnames.

The running session on this development machine is named `review-ui`:
`tmux attach -t review-ui`. The service uses Python's standard library,
without GPU or additional application dependencies.

## Review

Assume the caller is speaking while the agent is talking. Proposed labels,
categories and model predictions are hidden. Choose:

- **1 — Let the agent continue:** backchannel.
- **2 — Stop the agent:** barge-in.

Every case must have one of these two labels using its text alone. There is no
context-dependent or ambiguous decision option.

Each decision saves to disk and advances. Optional notes save with the chosen
decision. Previous returns to the last example for corrections. Search and
filters let you revisit saved cases. Shortcuts are disabled while
typing or interacting with form controls; click the page heading to resume.

Review annotations live in `data/evaluation/reviews/english-challenge-v1.json`.
They resume on reload and across server restarts. The frozen input CSV is never
modified. Reviews record a source hash, UTC timestamp and per-case revision.
Concurrent stale saves are rejected rather than overwriting another tab.
Download reviews exports decisions and notes as CSV. The `.lock` file beside
annotations coordinates writes and is not review data.

`--source` and `--reviews` select alternate files. Use a new review destination
when the source dataset changes. These annotations are owner review, not a
representative independent evaluation set.
