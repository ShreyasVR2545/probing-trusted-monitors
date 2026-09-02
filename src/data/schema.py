"""Trajectory schema and the monitor-side rendering of a trajectory.

Two things happen here that materially shape every downstream number, so both
are configurable and both are logged:

1. What the monitor is allowed to see. The mrt agent system prompt states the
   secondary task verbatim, and the agent's <model_thinking> scratchpad
   routinely announces the attack in plain language. Neither is visible to a
   monitor watching a deployed agent, so both are stripped by default.
2. How long trajectories are compressed. Individual tool results run past
   30,000 characters. Rather than truncating the trajectory tail (which would
   systematically delete the end of the attack), long tool results are elided
   in the middle with an explicit marker.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

THINKING_RE = re.compile(r"<model_thinking>.*?</model_thinking>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINKING_RE = re.compile(r"<model_thinking>.*\Z", re.DOTALL | re.IGNORECASE)


@dataclass
class Trajectory:
    """One agent trajectory plus the metadata needed to split and label it."""

    traj_id: str
    task: str
    task_suite: str            # "shade_arena" | "computer_use"
    family_dir: str            # raw mrt directory name
    strategy: str              # red-team strategy family (the RQ2 split axis)
    hint_condition: str        # no_hint | hint | human_hint | feedback | monitor_prompt
    agent_model: str
    label: int                 # 1 = attack (hidden secondary task), 0 = benign
    side_task_success: bool | None
    original_monitor_verdict: int | None
    n_messages: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""
    source_path: str = ""

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("messages")
        return d


def content_hash_of(messages: list[dict[str, Any]]) -> str:
    """Hash the agent-side content only.

    The same agent trajectory is re-monitored under many monitor configurations
    and therefore appears under several `transcripts_*` directories. Without
    this, near-duplicates would straddle the train/test boundary and inflate
    every number in the project.
    """
    payload = json.dumps(
        [[m.get("role"), str(m.get("content"))[:4000]] for m in messages],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:16]


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove hidden scratchpad blocks. Returns (text, did_strip)."""
    new, n = THINKING_RE.subn("", text)
    new, n2 = UNCLOSED_THINKING_RE.subn("", new)
    return new.strip(), bool(n or n2)


def _elide(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return (
        f"{text[:head]}\n\n[... {omitted} characters elided ...]\n\n{text[-tail:]}",
        True,
    )


def _stringify_tool_calls(calls: Any) -> str:
    if not calls:
        return ""
    parts = []
    for c in calls if isinstance(calls, list) else [calls]:
        if isinstance(c, dict):
            name = c.get("function") or c.get("name") or "tool"
            if isinstance(name, dict):
                name = name.get("name", "tool")
            args = c.get("args", c.get("arguments", ""))
            parts.append(f"{name}({json.dumps(args, default=str)[:1500]})")
        else:
            parts.append(str(c)[:1500])
    return "TOOL CALLS: " + "; ".join(parts)


def render_for_monitor(
    traj: Trajectory,
    strip_agent_thinking: bool = True,
    strip_agent_system_prompt: bool = True,
    max_tool_result_chars: int = 2000,
    max_message_chars: int = 6000,
) -> tuple[str, dict[str, Any]]:
    """Render a trajectory as the text a monitor reads.

    Returns (text, stats) where stats records what was removed, so the writeup
    can report elision rates per condition -- a difference across conditions
    would be a confound.
    """
    lines: list[str] = []
    stats = {
        "system_prompts_dropped": 0,
        "thinking_blocks_stripped": 0,
        "tool_results_elided": 0,
        "messages_elided": 0,
        "n_messages_rendered": 0,
    }
    for msg in traj.messages:
        role = (msg.get("role") or "").lower()
        content = msg.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, default=str)

        if role == "system":
            if strip_agent_system_prompt:
                stats["system_prompts_dropped"] += 1
                continue
            label = "SYSTEM"
        elif role == "user":
            label = "USER"
        elif role == "tool":
            label = "TOOL RESULT"
        else:
            label = "AGENT"

        if role == "assistant" and strip_agent_thinking:
            content, did = strip_thinking(content)
            stats["thinking_blocks_stripped"] += int(did)

        if role == "assistant":
            tc = _stringify_tool_calls(msg.get("tool_calls"))
            if tc:
                content = (content + "\n" + tc).strip()

        limit = max_tool_result_chars if role == "tool" else max_message_chars
        content, elided = _elide(content, limit)
        if elided:
            stats["tool_results_elided" if role == "tool" else "messages_elided"] += 1

        if not content.strip():
            continue
        lines.append(f"[{label}]\n{content}")
        stats["n_messages_rendered"] += 1

    text = "\n\n".join(lines)
    stats["rendered_chars"] = len(text)
    return text, stats
