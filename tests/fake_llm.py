"""A fake LlmClient that writes valid samples from the rendered request prompt.

It reads the factor lines the real prompt contains, so the runner, prompts and
validation are exercised together. The judge agrees with the generator unless
told otherwise.
"""

import itertools
import json
import re
import threading

from barge_in_detector.generation.llm import DecisionResult, LlmError, LlmResult

LENGTH = {"w1": 1, "w2_3": 2, "w4_8": 4, "w9_plus": 9}

# Shared by every FakeLlm, so a later client can judge text an earlier one wrote.
_LABELS: dict[tuple[str, str], tuple[str, int | None]] = {}
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
    def __init__(self, *, judge_disagrees=False, judge_fails=False, generator_fails=False):
        self.judge_disagrees = judge_disagrees
        self.judge_fails = judge_fails
        self.generator_fails = generator_fails
        self.calls = {"generate": 0, "judge": 0}
        self.system_prompts = set()
        self._lock = threading.Lock()

    def generate(self, *, model, system_prompt, user_prompt, schema, thinking_level=None):
        self.system_prompts.add(system_prompt)
        return self._generate(user_prompt)

    def close(self):
        pass

    def _result(self, payload):
        text = json.dumps(payload)
        usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 200}
        return LlmResult(text=text, response={"text": text, "usage": usage}, usage=usage)

    def _generate(self, prompt):
        with self._lock:
            self.calls["generate"] += 1
        if self.generator_fails:
            raise LlmError("503: unavailable")
        lines = prompt.splitlines()
        sets = re.search(r"Write (\d+) sets", prompt)
        if sets:
            shared_end = next(i for i, line in enumerate(lines) if line.startswith("Member with"))
            shared = lines[:shared_end]
            members = []
            current = None
            for line in lines[shared_end:]:
                if line.startswith("Member with"):
                    current = []
                    members.append(current)
                elif current is not None and line.startswith("- "):
                    current.append(line)
            share_agent = "context_turn, speaker," in prompt
            count = int(sets.group(1))
            member_lines = [shared + m for m in members]
        else:
            count = int(re.search(r"Write (\d+) samples", prompt).group(1))
            member_lines = [lines]
            share_agent = False
        samples = []
        for set_index in range(count):
            uid = next(_IDS)
            for member_index, ml in enumerate(member_lines):
                agent_uid = uid if share_agent else f"{uid}m{member_index}"
                samples.append(self._sample(ml, uid, agent_uid, set_index, member_index))
        return self._result({"samples": samples})

    def _sample(self, lines, uid, agent_uid, set_index, member_index):
        cut = _factor(lines, "Cut position")
        pattern = _factor(lines, "Speaker pattern")
        length = LENGTH[_factor(lines, "Length of listener_text")]
        development = _factor(lines, "Development")
        label = _label(lines)
        full = f"Manager says item {agent_uid} and more words here. Then another sentence."
        said = {"start": 2, "early": 3, "middle": 5, "near_end": 9}.get(cut)
        spoken = full if said is None else " ".join(full.split()[:said])
        first = f"c{uid}m{member_index}" if agent_uid == uid else f"c{uid}"
        candidate = " ".join([first, *["x"] * (length - 1)])
        word = None
        if label == "barge_in":
            word = 0 if development == "stable" else 1
        with self._lock:
            _LABELS[(spoken, candidate)] = (label, word)
        return {
            "set_index": set_index,
            "member_index": member_index,
            "context_turn": {
                "speaker": "Guest" if pattern == "returning_speaker" else "Kim",
                "text": "I need help.",
            },
            "speaker": "Hotel Manager",
            "speaker_full_text": full,
            "speaker_spoken_text": spoken,
            "listener": "Guest",
            "listener_text": candidate,
            "label": label,
            "barge_in_from_word": word,
            "rationale": "fake",
        }

    def decide(self, *, model, input, questions):
        with self._lock:
            self.calls["judge"] += 1
        if self.judge_fails:
            raise LlmError("503: unavailable")
        conversation = json.loads(input)["conversation"]
        spoken, candidate = conversation[-2]["transcript"], conversation[-1]["transcript"]
        label, word = _LABELS[(spoken, candidate)]
        if self.judge_disagrees:
            label = "backchannel" if label == "barge_in" else "barge_in"
            word = 0 if label == "barge_in" else None
        p = {
            "is_backchannel": 0.9 if label == "backchannel" else 0.1,
            "is_barge_in": 0.9 if label == "barge_in" else 0.1,
        }
        for key in questions:
            if key.startswith("prefix_"):
                k = int(key.removeprefix("prefix_"))  # the first k words
                p[key] = 0.9 if word is not None and k > word else 0.1
        for key in questions:
            p.setdefault(key, 0.95)  # conformance and factor_* checks pass
        assert set(p) == set(questions)
        answers = {
            k: {"type": "noul", "noul": v, "confidence": max(v, 1 - v)} for k, v in p.items()
        }
        usage = {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 0}
        return DecisionResult(response={"answers": answers, "usage": usage}, usage=usage)
