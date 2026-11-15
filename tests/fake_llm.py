"""A fake LlmClient: writes valid utterances from the rendered request prompt
and answers every judge question yes, unless told otherwise.

It reads the factor lines the real prompt contains, so the runner, prompts and
validation are exercised together. It never calls the network.
"""

import itertools
import json
import re
import threading

from barge_in_detector.generation.llm import DecisionResult, LlmError, LlmResult

LENGTH = {"w1": 1, "w2_3": 2, "w4_8": 4, "w9_plus": 9}
_IDS = itertools.count()


def _factor(lines, name):
    for line in lines:
        m = re.match(rf"- {re.escape(name)}: `([^`]+)`", line)
        if m:
            return m.group(1)
    return None


def _label(lines):
    for line in lines:
        m = re.search(r"\(intended label: (\w+)\)", line)
        if m:
            return m.group(1)
    return None


class FakeLlm:
    def __init__(
        self,
        *,
        judge_fails=False,
        generator_fails=False,
        label_override=None,
        fixed_text=None,
        judge_no=(),
    ):
        self.judge_fails = judge_fails
        self.generator_fails = generator_fails
        # Report this label instead of the intended one, or write this text for
        # every sample (to exercise deduplication).
        self.label_override = label_override
        self.fixed_text = fixed_text
        # Answer no (P(yes) 0.1) to judge questions whose key starts with one of
        # these prefixes, yes (0.95) to every other question.
        self.judge_no = tuple(judge_no)
        self.calls = {"generate": 0, "judge": 0}
        self.system_prompts = set()
        self._lock = threading.Lock()

    def generate(self, *, model, system_prompt, user_prompt, schema, thinking_level=None):
        with self._lock:
            self.calls["generate"] += 1
            self.system_prompts.add(system_prompt)
        if self.generator_fails:
            raise LlmError("503: unavailable")
        lines = user_prompt.splitlines()
        count = int(re.search(r"Write (\d+) utterances?\b", user_prompt).group(1))
        length = LENGTH[_factor(lines, "Length")]
        development = _factor(lines, "Development")
        label = self.label_override or _label(lines)
        samples = []
        for _ in range(count):
            uid = next(_IDS)
            word = None
            if label == "barge_in":
                word = 0 if development == "stable" else 1
            text = " ".join([self.fixed_text or f"u{uid}", *["x"] * (length - 1)])
            samples.append(
                {"text": text, "label": label, "barge_in_from_word": word, "rationale": "fake"}
            )
        text = json.dumps({"samples": samples})
        usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 200}
        return LlmResult(text=text, response={"text": text, "usage": usage}, usage=usage)

    def decide(self, *, model, input, questions):
        with self._lock:
            self.calls["judge"] += 1
        if self.judge_fails:
            raise LlmError("503: unavailable")
        assert set(json.loads(input)) == {"text"}
        answers = {}
        for key in questions:
            p = 0.1 if self.judge_no and key.startswith(self.judge_no) else 0.95
            answers[key] = {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}
        usage = {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 0}
        return DecisionResult(response={"answers": answers, "usage": usage}, usage=usage)

    def close(self):
        pass
