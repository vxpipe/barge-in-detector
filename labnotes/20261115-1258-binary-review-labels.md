# Binary review labels

- Owner clarified that context cannot be supplied: every review must choose
  backchannel or barge-in, with no needs-context option.
- Relabelled en-031 (Yes yes yes yes yes) as backchannel through ReviewStore,
  preserving previous rationale and appending the owner clarification.
  Repeated acknowledgments contain no textual floor claim.
- All 98 reviewed cases now have binary decisions: 50 backchannel, 48 barge-in.
- Removed the third action, keyboard shortcut, filter and flag count from the
  UI; store rejects needs_context writes. Updated review instructions.
- Test first: focused invalid-decision test failed because needs_context was
  accepted; passes after restricting save validation.
- Ruff formatting and lint passed; 116 tests passed, one integration test
  deselected (9.28 s). Node JavaScript syntax check passed.
- Restarted local review-ui tmux server on the existing Tailscale endpoint;
  API confirms binary counts. No browser automation used for this change.
