# Project README onboarding

## Request and direction

Owner requested marketing-oriented getting-started READMEs on GitHub and both
HF repos, with shared badges and reciprocal links. All three now share the
project title, introduction, and four-badge navigation: GitHub source, HF model,
HF dataset, and MIT licence. Each has a repository-specific quick start and a
common table explaining source/model/data ownership.

## Changes

- GitHub: explain the overlapping-speech task, show intended runtime responses,
  provide a CPU CLI quick start and review entry point, and link generation,
  training, reports, contribution milestones and checks.
- Model card: CPU environment setup and executable Transformers example, input/
  output contract, evaluation limits, CPU timing and release provenance.
- Dataset card: environment/setup and datasets.load_dataset example, split counts,
  useful columns, label definitions, languages, file layout and review limitations.
- Keep the experimental status and 25/30 short-interruption failures visible;
  do not turn the synthetic 100% score into a production/generalization claim.
- Preserve HF YAML metadata/configuration, artifact/data contents and revision pins.
  GitHub remains the development home and HF trees remain release-only.

## Verification

- All four badge images returned SVG and all four destinations returned HTTP 200.
  The badge rows are byte-identical across all three READMEs.
- Model README Python example executed on CPU: p_barge_in=0.98690003, barge_in,
  for please stop for a moment.
- Dataset README Python example loads train/validation/test plus challenge;
  train=2640 and challenge=98 checked explicitly.
- Hugging Face ModelCard and DatasetCard schemas validate.
- Ruff formatting and Git whitespace checks pass for all three README files.
- Documentation-only change: no retraining, generation, GPU usage, new unit tests,
  or repeated full-suite run. Each README is published in its own repo commit;
  this checkpoint note accompanies the GitHub change.
