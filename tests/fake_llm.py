"""A fake LlmClient that writes valid samples from the rendered request prompt.

It reads the factor lines the real prompt contains, so the runner, prompts and
validation are exercised together. The judge agrees with the generator unless
told otherwise.
"""

import itertools
import json
import re
import threading

from barge_in_detector.generation.llm import LlmError, LlmResult

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
        if "items" in schema["properties"]:
            return self._judge(user_prompt)
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
            share_agent = "context_turn, agent_speaker" in prompt
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
        length = LENGTH[_factor(lines, "Length of candidate_text")]
        development = _factor(lines, "Development")
        label = _label(lines)
        full = f"Agent says item {agent_uid} and more words here. Then another sentence."
        spoken = full if cut == "just_finished" else f"Agent says item {agent_uid} and more"
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
                "speaker": "Sam" if pattern == "returning_speaker" else "Kim",
                "role": "caller",
                "text": "I need help.",
            },
            "agent_speaker": "Agent",
            "agent_full_text": full,
            "agent_spoken_text": spoken,
            "candidate_speaker": "Sam",
            "candidate_role": "caller",
            "candidate_text": candidate,
            "label": label,
            "barge_in_from_word": word,
            "rationale": "fake",
        }

    def _judge(self, prompt):
        with self._lock:
            self.calls["judge"] += 1
        if self.judge_fails:
            raise LlmError("503: unavailable")
        items = []
        blocks = prompt.split("\nItem ")[1:]
        for block in blocks:
            lines = block.splitlines()
            index = int(lines[0])
            agent_line = next(line for line in lines if line.startswith("Agent: "))
            spoken = agent_line.removeprefix("Agent: ").rsplit(" [", 1)[0]
            candidate = lines[-1].split(": ", 1)[1]
            label, word = _LABELS[(spoken, candidate)]
            if self.judge_disagrees:
                label = "backchannel" if label == "barge_in" else "barge_in"
                word = 0 if label == "barge_in" else None
            items.append(
                {
                    "index": index,
                    "label": label,
                    "barge_in_from_word": word,
                    "language_ok": True,
                    "script_ok": True,
                    "cut_ok": True,
                    "natural_ok": True,
                    "notes": "",
                }
            )
        return self._result({"items": items})
