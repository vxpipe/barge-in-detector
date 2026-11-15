# After the 98-case owner review

The review is complete. Compare model approaches using the entire existing
accepted dataset across all 14 languages and identical train/validation/test
splits before changing the data (owner clarification, 2026-11-15). Both completed
experiments already use this scope. The current reviewed set is a development
regression set; it cannot establish production accuracy.

Follow-up experiments and a runnable improved candidate are now recorded in
[Model improvement results](model-improvement-results.md). The frozen-E5 comparison
is complete; character plus word features reach 90/98 on the development review
and model-path p99 1.42 ms (5.35 ms for a full 64-token input) on the owner's CPU.
This is not a release acceptance claim; see the multilingual and false-stop limits.

## Evidence, 2026-11-15

The audit reads the saved owner decisions, checks the challenge source hash,
joins frozen probabilities by case ID and checks text equality. All 98 decisions
are binary and agree with the original labels. Five rows overlap the historical
splits (three train, two validation); the other 93 are also scored separately.

| Measure | Existing e5-small encoder | Character TF-IDF + logistic regression |
| --- | ---: | ---: |
| Reviewed challenge correct | 72/98 (73.5%) | 88/98 (89.8%) |
| Correct on 93 unseen challenge rows | 67/93 (72.0%) | 83/93 (89.2%) |
| Short interruptions caught | 5/30 | 26/30 |
| All interruptions caught | 22/48 | 43/48 |
| Backchannels wrongly interrupted | 0/50 | 5/50 |
| Longer backchannels correct | 24/24 | 19/24 |
| Prefix rows correct | 23/24 | 23/24 |
| Synthetic test correct | 331/331 (100%) | 320/331 (96.7%) |

The baseline fits character 2–5-gram TF-IDF and balanced logistic regression on
the existing 2,640 training rows. It chooses C from 0.1, 1 and 10 using macro F1
on the 333 validation rows, with ties favoring the smaller C. C=10 won. Threshold
is fixed at 0.5. It never fits on the challenge or its annotations. Split hashes
are checked against the original encoder's source manifest. Its lower synthetic
test score and higher reviewed challenge score show why synthetic validation
alone is insufficient for choosing the next model.

This baseline is an experiment, not a replacement serving artifact: probabilities
are uncalibrated, it uses full cleaned strings rather than the encoder's 64-token
truncation, and no export or runtime integration has been built. No reference
CPU or whole-path latency result is claimed. The 98-case set is small,
deliberately challenging, English-only and contains related prefixes; 89.8% is
not a population accuracy estimate.

Machine-readable evidence:

- [Coverage, owner labels and errors](../reports/reviewed-error-audit-v1.json).
- [Baseline configuration, selection trials and comparison](../reports/character-baseline-v1/report.json).
- [Baseline per-case probabilities](../reports/character-baseline-v1/challenge-predictions.csv).

## The coverage gap

Lengths below are whitespace words, meaningful within English/Hindi rather than
comparable measures across all scripts. Counts include each language's mixed
forms as recorded in the dataset.

| Training slice | Total | 1–2 words | 3–5 words | 6+ words |
| --- | ---: | ---: | ---: | ---: |
| English interruptions | 432 | **0** | 8 | 424 |
| Hindi interruptions | 96 | **0** | 2 | 94 |
| English backchannels | 211 | 22 | 156 | 33 |
| Hindi backchannels | 48 | 14 | 31 | 3 |

All 54 English and all 12 Hindi interruptions in each validation/test split have
at least six words. Thus the synthetic holdouts never tested the shortest cases
that failed owner review. This supports prioritizing data coverage; it does not
prove the encoder relies exclusively on length. The baseline's results show that
the learning method also matters.

All 3,304 accepted export rows match the splits exactly. No cleaned text is
duplicated within or across splits. Same-language character-trigram Jaccard
similarity >=0.8 finds four near-matching validation rows and one test row against
train; examples are acknowledgments/continuers. This heuristic is not a semantic
paraphrase audit and cannot rule out shared synthetic templates.

Lowering the encoder threshold is insufficient as the sole intervention. At 0.5
it catches 5/30 short interruptions with 0/50 false interruptions; at 0.0133 it
catches 29/30 but falsely interrupts 25/50 backchannels. These are diagnostic
sensitivity measurements, not a selected threshold.

Illustrative matched-length targets for the new generation criteria (these are
design examples, not additions to the dataset or replacements for owner review):

| Language | Words | Backchannel | Barge-in |
| --- | ---: | --- | --- |
| English | 2 | yes indeed | pause please |
| Hindi | 2 | जी बिल्कुल | ज़रा रुकिए |
| English | 5 | okay yes i am following | okay please change the date |
| English | 9 | yes i understand and i am following along | yes i understand but please change the appointment time |

## Next comparison checkpoint

Keep all 3,304 accepted samples, with the existing deterministic split of 2,640
train / 333 validation / 331 test. All 14 languages occur in each split. Reuse
these exact files and hashes for every approach, rather than reshuffling between
runs or restricting training to particular languages. Rejected and retired
generation attempts are not training examples.

The existing fully fine-tuned E5 model, the character classifier and the
frozen-E5-embeddings linear classifier have now been compared on this fixed
dataset, followed by two combined-feature experiments. Learn parameters on train only, select
settings and thresholds on validation, and report test and owner-reviewed
challenge metrics separately. Include aggregate and per-language synthetic
results, false/missed interruptions, and comparisons at matched false-interruption
rates using validation-selected thresholds. The same numeric threshold on two
uncalibrated models does not establish an equal operating tradeoff.

The 98 English cases remain a shared development/regression check. Human review
availability limits which languages have verified evidence, not which languages
participate in training or synthetic testing. Fresh human evaluation and serving
checks remain required before release. Additional generation is deferred until
the fixed-dataset comparison is assessed.

## Subsequent data and release work (deferred)

1. **Make length a deterministic generation criterion.** Add an optional
   `length_band` factor with a legacy `default` and language-appropriate short,
   medium and longer bands. English/Hindi whitespace bands of 1–2, 3–5 and 6–12
   words are examples, not universal measures for every script. Keep default
   criterion UUIDs unchanged; new bands get new UUIDs. Version the schema and
   prompt, persist the factor through storage/export/splits, and reject samples
   outside their specified band after cleaning. The generator still writes
   natural punctuated text. Judge the new criterion alongside intent/script.
   Do not pad utterances to fill a band; catalogue constraints must exclude
   unnatural combinations.
2. **Build a coverage plan before buying generation.** Preserve all 14 languages
   and their native-script/code-switching rules. Cover short stop/wait, rejection,
   repeat/clarification and skip requests, plus natural same-length backchannels.
   Include longer acknowledgments and utterances that begin as acknowledgments
   and then claim the floor. English mixing stays permitted where the script and
   length constraints are feasible. Retain equal required counts per criterion;
   start with five, consistent with the current plan policy. Produce the exact
   feasible criteria/counts and budget by dry run for review before paid calls.
   Increasing the existing plan's count alone does not enforce these gaps.
3. **Protect the evaluation boundary.** Exclude exact cleaned-text matches to the
   98 challenge cases from new training data. Preserve historical overlaps in
   the report. Fix split growth stability before filling new criteria: the
   current splitter re-ranks held-out rows and alternates sorted criterion IDs,
   so adding rows/criteria can move existing examples. Add a regression test and
   persist existing assignments. Keep related prefix families together and audit
   near duplicates. New generated rows must go through storage and CSV export.
4. **Retrain both methods on the revised splits.** Preserve run1. Compare the
   same encoder architecture/training settings and the character baseline to
   measure the data change first. Select using validation, report false and
   missed interruptions separately, and inspect both short requests and long
   acknowledgments. Do not tune thresholds against these 98 cases.
5. **Collect fresh English/Hindi review after the candidate is fixed.** Prepare
   a separate blind review batch with short requests, longer backchannels and
   prefix transitions; exclude texts/families used for training or selection.
   Use separate reviewed validation data for threshold/calibration decisions.
   Report raw counts per language and class, and confidence intervals where
   independence permits. Agree false/missed-interruption floors before tuning;
   the development score alone cannot set these product tradeoffs.
6. **Then verify serving.** Calibrate on validation reflecting runtime priors,
   measure reference-CPU p50/p95/p99 and memory with the 64-token contract, and
   measure the Vxpipe boundary. The <=20 ms model / <=50 ms whole-path p99 gates
   remain open; the reference CPU is still unspecified in existing reports.

An immediate model swap is not justified: the baseline trades better interruption
recall for false interruptions, and Hindi has no fresh owner-reviewed evidence.
Training on the 98 reviewed rows would remove their value as a regression test.
A universal stop/ignore word list would violate the classifier contract and
would not handle continuers that develop into requests.

## Reproduce this checkpoint

```shell
uv run python -m experiments.audit_reviewed_errors
uv run --group baseline python -m experiments.compare_character_baseline
```

Both commands are local. The baseline group is optional and pinned by `uv.lock`;
it does not pull in torch. The baseline report records file hashes, source base
commit and dirty-tree status because this checkpoint has not been committed.
