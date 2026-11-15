# Original plus phrase supplement v1

Derived comparison dataset: all **6,741 accepted samples** across 14 languages,
split into **4,918 train / 915 validation / 908 test**. These files concatenate
the original splits with the separately split phrase supplement; no original
sample changes its assignment. The original 3,304-sample dataset and the
3,437-sample supplement remain independently available.

Build with `uv run python -m barge_in_detector.supplements`. Compare original and
supplement test results separately; their distributions differ. See
[results](../../../docs/phrase-coverage-results.md) and the
[split audit](../../../reports/phrase-coverage-v1/splits.json).
