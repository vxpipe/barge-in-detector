# Combined-feature improvement

## Scope and decisions

The owner clarified that the objective is improving the classifier, not merely
running the frozen-E5 experiment. Continued locally after downloading/verifying
the GPU artifacts and destroying the single authorized RTX 5090 instance. No
further paid compute or generation; all 14 languages and original split hashes
remain unchanged. The 98-case challenge is development/regression evidence.

Hypothesis: the character model's short-request clues and longer-utterance
features can complement each other. Tested two controlled combinations, each
with a grid fixed before inspecting that candidate's challenge predictions:
C=1/10/100 and complementary-feature scale=0.25/1/4. Fit TF-IDF/weights on train
only, select on validation macro F1, tie by first grid entry, fixed threshold 0.5.
No challenge fitting, relabeling, synthetic data insertion or threshold tuning.

## Results

| Candidate | Selected scale / C | Synthetic test | Reviewed English | Short requests | False stops | Prefix rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Character + frozen E5 embeddings | 0.25 / 10 | 326/331 | 90/98 | 25/30 | 3/50 | 24/24 |
| Character + word unigrams/bigrams | 0.25 / 10 | 320/331 | 90/98 | 26/30 | 4/50 | 24/24 |

The E5 hybrid still misses “stop” and retains the encoder computation. The
lexical candidate correctly predicts it (p=0.53264) and corrects two original
character-baseline errors with no new development mistakes: “that is interesting
yes i see” and “yeah yeah no that will not work for me”. Reviewed unseen subset
85/93. Remaining misses: enough; go back; you misunderstood; how much. Four false
stops are longer acknowledgments. This does not establish independent population
accuracy; we have repeatedly inspected these development cases.

Synthetic multilingual regressions remain: lexical Hindi/Italian 16/18 each,
Spanish 33/36, versus original E5's perfect synthetic test. Retain all results,
including unsuccessful candidates. Configuration trials and per-language counts
are under `reports/hybrid-run1/`, `reports/lexical-run1/` and
`reports/model-comparison.json`.

Validation-only threshold diagnostic: maximize validation macro F1 over observed
probabilities plus 0/0.5/1, ties closest to 0.5 then higher threshold. It retains
0.5 for both combined-feature candidates. Frozen E5 selects 0.517328 with no
challenge decision changes. No runtime threshold was deployed.

## Portable candidate and CPU evidence

Exported the selected lexical model to `models/lexical-run1/portable/` as JSON
vocabularies/IDF/weights plus tokenizer, with no pickle needed for inference.
The private intermediate joblib file is produced/loaded only by the local
experiment. Inference truncates via the shared tokenizer's offsets, preserving
the first 64 tokens including special tokens; no token IDs are stored.

All validation/test/challenge predictions reproduce exactly after portable export
and budget enforcement (maximum difference 0, zero changed labels).
Added tokenizers explicitly to optional baseline dependencies and retained all
pre-existing pyproject/lock changes. Inference command:

`uv run --group baseline python -m barge_in_detector.training.lexical "stop"`

Owner-designated local AMD EPYC-Genoa CPU, one thread, warmed batch one, 100 warmups
and 1,000 measurements per scenario, includes tokenizer through probability:

| Scenario | Model p50 | Model p95 | Model p99 | Whole-path p50/p95/p99 |
| --- | ---: | ---: | ---: | --- |
| Multilingual test | 0.90 ms | 1.27 ms | 1.42 ms | Unmeasured |
| Full 64-token input | 2.16 ms | 4.43 ms | 5.35 ms | Unmeasured |

Passes the <=20 ms model gate on this machine. Vxpipe boundary is absent, so the
<=50 ms whole-path gate stays open. Artifact size 22,898,881 bytes; verification
process peak RSS 469.96 MiB includes extra export/comparison objects. It is not
isolated serving RSS. Report: `reports/lexical-run1/cpu-benchmark.json`.

## Verification and next checkpoint

- Test first: lexical tests failed because the module did not exist; implementation
  then passed round-trip, class probability, 64-token and native-script preservation
  checks. Experimental model comparisons are verified with saved metrics rather
  than tests that mirror fitting.
- Model files remain ignored. No published model changed, and no commits created.
- `docs/model-improvement-results.md` records the improved candidate, remaining
  errors and next concrete work: new evaluation coverage and deterministic
  generation coverage across all 14 languages, followed by calibration and
  Vxpipe whole-path measurement. Exact new criteria/counts and spend still need
  review before paid generation under the existing milestone.
- Final completion commands passed: `uv run ruff format --check` (103 files),
  `uv run ruff check`, `uv run pytest` (**149 passed, 1 integration deselected**,
  11.45 seconds), and `git diff --check`. Reviewed final status/diffs; preserved
  unrelated release/submodule work. Final Vast.ai query confirms zero instances.
