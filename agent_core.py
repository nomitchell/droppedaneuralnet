import difflib
import hashlib
import json
import random
import re
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from config import (
    ATTEMPT_SUMMARY_TSV_PATH,
    BEST_RESULT_PATH,
    CONTEXT_EXPERIMENT_LIMIT,
    EMPTY_MESSAGE_CONTENT_RETRIES,
    EXPERIMENTS_JSONL_PATH,
    KNOWN_FACTS_PATH,
    MAX_MEMORY_ENTRY_CHARS,
    MAX_MEMORY_LIST_ITEMS,
    MEMORY_EXTRACT_MAX_PER_LIST,
    MEMORY_SHORT_LEN,
    MEMORY_SIMILARITY_DEDUPE,
    MEMORY_UPDATE_MAX_TOKENS,
    MODEL,
    MODEL_MAX_OUTPUT_TOKENS,
    NIM_API_KEY,
    NIM_BASE_URL,
    PRIOR_BEST_SOLVER_SCRIPT_CHARS,
    PRIOR_INVESTIGATE_SCRIPT_CHARS,
    PRIOR_SOLVE_ATTEMPT_SCRIPT_CHARS,
    PROBLEM_PATH,
    RESEARCH_LOG_PATH,
    RESEARCH_SYNTHESIS_PATH,
    RESULTS_DIR,
    RESULTS_TSV_PATH,
    SCRIPT_COMPILE_REPAIR_ATTEMPTS,
    SYNTHESIS_CONTEXT_TAIL,
)

_MAX_PLAN_CHARS_JSONL = 900
_MAX_PLAN_CHARS_CONTEXT = 220
ALLOWED_ACTIONS = {"THINK", "INVESTIGATE", "SOLUTION", "SYNTHESIS"}

client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY)
PROBLEM_DESCRIPTION = PROBLEM_PATH.read_text(encoding="utf-8")

ACTION_POLICY_MARKDOWN = """
## Action policy (use SOLUTION as a checkpoint, not only a “final submit”)

- **INVESTIGATE** — Read data, print diagnostics, probe shapes/stats. Output is logged to a text file **only**; runs **do not** update `results/best_result.json`, **do not** append official MSE for a candidate permutation to the experiment ratchet, and are easy to lose in long logs. Use for quick prints, not for “try permutation X and see MSE” as your only record.
- **SOLUTION** — Any step where you run a **solver script** that writes `results/solution_output.txt` (comma-separated 97 indices) so the harness can score it. Use SOLUTION for **baselines, heuristics, partial searches, and checkpoints** — not reserved for a perfect final answer. Each SOLUTION run is archived (`solve_attempt_iter*.py`), logged to `experiments.jsonl` + `attempt_summary.tsv` with MSE/status/fingerprints, and **improving** runs update **Current Best** in this context (`best_result.json`, `best_solver.py`, `best_permutation.txt`).
- **THINK** — Reasoning and memory only.
- **SYNTHESIS** — Meta checkpoint: read the situational context and **append** a concise new section to `results/research_synthesis.md` that continues where the last summary left off (state of the search, what worked/failed, open questions, suggested next moves). Use after long stretches, confusion, or before a strategy pivot—not every step.

**Rule of thumb:** If your next code would **assign a full permutation and measure official MSE** (or optimize toward it), choose **SOLUTION** so metrics and best-so-far are stored. Use INVESTIGATE for analysis that does not need that audit trail.

Use the **Attempt register** JSON for prior `plan_excerpt` / `perm_sha16` / `artifacts` (same `perm_sha16` ⇒ same permutation).

Inside solver scripts, add a short top-of-file **module docstring** (algorithm, search space, random seed) so `best_solver.py` remains self-explanatory. You may also write extra summaries under `results/` (e.g. `results/solver_run_notes.txt`) if useful.
""".strip()


def log(entry: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with open(RESEARCH_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n---\n**[{ts}]**\n{entry}\n")
    try:
        print(entry)
    except UnicodeEncodeError:
        print(entry.encode("utf-8", errors="replace").decode("utf-8"))


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _truncate_plan_one_line(plan: str | None, max_len: int) -> str:
    if not plan:
        return ""
    s = " ".join(str(plan).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _perm_sha16(perm_str: str | None) -> str | None:
    if not perm_str or not str(perm_str).strip():
        return None
    h = hashlib.sha256(str(perm_str).strip().encode("utf-8")).hexdigest()
    return h[:16]


def _collect_artifact_paths(row: dict) -> list[str]:
    paths: list[str] = []
    it = row.get("iteration")
    action = row.get("action")
    if action == "SOLUTION" and it is not None:
        paths.append(f"results/solve_attempt_iter{it}.py")
    if action == "INVESTIGATE":
        paths.append("results/investigate.py")
    if action == "SYNTHESIS":
        paths.append("results/research_synthesis.md")
    for k in ("output_file", "compile_error_file", "stderr_file", "stdout_file", "permutation_file"):
        v = row.get(k)
        if v:
            try:
                rel = Path(v).as_posix()
                if rel not in paths:
                    paths.append(rel)
            except Exception:
                continue
    return paths[:10]


def _tsv_cell(s: str | None) -> str:
    t = "" if s is None else str(s)
    return " ".join(t.split()).replace("\t", " ")[:500]


def log_experiment(row: dict) -> None:
    row = dict(row)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    if row.get("plan") is not None:
        row["plan"] = _truncate_plan_one_line(row.get("plan"), _MAX_PLAN_CHARS_JSONL) or None
    row["artifact_paths"] = _collect_artifact_paths(row)
    append_jsonl(EXPERIMENTS_JSONL_PATH, row)

    if not ATTEMPT_SUMMARY_TSV_PATH.exists() or ATTEMPT_SUMMARY_TSV_PATH.stat().st_size == 0:
        ATTEMPT_SUMMARY_TSV_PATH.write_text(
            "ts\titeration\taction\tstatus\tmse\tperm_sha16\teval_error\tartifacts\tplan_excerpt\n",
            encoding="utf-8",
        )
    arts = ";".join(row.get("artifact_paths") or [])
    mse = row.get("mse")
    line = "\t".join(
        [
            _tsv_cell(row.get("ts")),
            _tsv_cell(row.get("iteration")),
            _tsv_cell(row.get("action")),
            _tsv_cell(row.get("status")),
            _tsv_cell("" if mse is None else str(mse)),
            _tsv_cell(row.get("perm_sha16")),
            _tsv_cell(row.get("eval_error")),
            _tsv_cell(arts),
            _tsv_cell(row.get("plan")),
        ]
    )
    with open(ATTEMPT_SUMMARY_TSV_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_last_experiments(limit: int = 8) -> list[dict]:
    if not EXPERIMENTS_JSONL_PATH.exists():
        return []
    out = []
    for line in EXPERIMENTS_JSONL_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def compact_experiments_for_context(limit: int = CONTEXT_EXPERIMENT_LIMIT) -> list[dict]:
    compact = []
    for r in read_last_experiments(limit=limit):
        item = {
            "iteration": r.get("iteration"),
            "ts": r.get("ts"),
            "action": r.get("action"),
            "status": r.get("status"),
            "mse": r.get("mse"),
            "improved": r.get("improved"),
            "solved": r.get("solved"),
            "plan_excerpt": _truncate_plan_one_line(r.get("plan") or "", _MAX_PLAN_CHARS_CONTEXT) or None,
            "perm_sha16": r.get("perm_sha16"),
            "eval_error": (r.get("eval_error") or "")[:200] or None,
            "artifacts": r.get("artifact_paths") or _collect_artifact_paths(r),
        }
        compact.append({k: v for k, v in item.items() if v not in (None, "", [])})
    return compact


def _parse_json_object_loose(raw: str):
    s = (raw or "").strip()
    if not s:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_action_label(action_val) -> str | None:
    if action_val is None:
        return None
    u = str(action_val).strip().strip('"').strip("'").upper().replace(" ", "_")
    aliases = {
        "THINK": "THINK",
        "INVESTIGATE": "INVESTIGATE",
        "INVESTIGATION": "INVESTIGATE",
        "SOLVE": "SOLUTION",
        "SOLUTION": "SOLUTION",
        "SYNTHESIS": "SYNTHESIS",
        "SYNTHESIZE": "SYNTHESIS",
        "SUMMARY": "SYNTHESIS",
    }
    out = aliases.get(u, u)
    return out if out in ALLOWED_ACTIONS else None


def _json_for_prompt(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _extract_last_failure(log_text: str) -> str:
    markers = ["No output file written", "Action parse failed", "TIMED OUT", "Traceback", "STDERR:", "\"error\":", "**Metrics:**"]
    latest_idx = max((log_text.rfind(m) for m in markers), default=-1)
    if latest_idx == -1:
        return "(none)"
    return log_text[max(0, latest_idx - 700): min(len(log_text), latest_idx + 1200)].strip()


def _read_synthesis_tail(max_chars: int) -> str:
    if not RESEARCH_SYNTHESIS_PATH.exists():
        return "(No synthesis file yet.)"
    data = RESEARCH_SYNTHESIS_PATH.read_text(encoding="utf-8", errors="replace").rstrip()
    if not data:
        return "(Synthesis file is empty — write an opening checkpoint.)"
    if len(data) <= max_chars:
        return data
    return "... [earlier synthesis omitted] ...\n\n" + data[-max_chars:].strip()


def build_context(for_synthesis_call: bool = False) -> str:
    log_text = RESEARCH_LOG_PATH.read_text(encoding="utf-8") if RESEARCH_LOG_PATH.exists() else "(empty)"
    best = load_json(BEST_RESULT_PATH, {})
    known = load_json(KNOWN_FACTS_PATH, {})
    last_failure = known.get("last_failure_summary", "(none)") or _extract_last_failure(log_text)
    compact_known = {
        "confirmed_facts": known.get("confirmed_facts", [])[-10:],
        "rejected_hypotheses": known.get("rejected_hypotheses", [])[-10:],
        "last_failure": last_failure,
    }
    synthesis_section = ""
    if not for_synthesis_call:
        synthesis_section = f"\n## Latest synthesis (tail)\n{_read_synthesis_tail(SYNTHESIS_CONTEXT_TAIL)}\n"
    return f"""
## Problem
{PROBLEM_DESCRIPTION}

## Environment Contract (must follow)
- Input files: data/pieces/piece_*.pth and data/historical_data.csv
- Output artifacts: results/
- Solver must write: results/solution_output.txt
- Valid action labels only: THINK, INVESTIGATE, SOLUTION, SYNTHESIS
- Rolling narrative: `results/research_synthesis.md` (append-only; latest tail below)
- Attempt log: `results/experiments.jsonl` (model context below); same rows mirrored in `results/attempt_summary.tsv` for external tools (not duplicated here).

{ACTION_POLICY_MARKDOWN}

## Known Facts (durable memory)
{_json_for_prompt(compact_known)}

## Current Best
{_json_for_prompt(best)}

## Attempt register (recent)
{_json_for_prompt(compact_experiments_for_context(limit=CONTEXT_EXPERIMENT_LIMIT))}
{synthesis_section}
## Research Log (recent tail)
{log_text[-4200:]}
"""


def _normalize_one_entry(e) -> str | None:
    if not isinstance(e, str):
        return None
    s = " ".join(e.strip().split())
    if not s:
        return None
    return s[:MAX_MEMORY_ENTRY_CHARS].rstrip()


def _memory_entry_similarity(a: str, b: str) -> float:
    la, lb = a.lower(), b.lower()
    if la == lb:
        return 1.0
    if min(len(la), len(lb)) >= 14 and (la in lb or lb in la):
        shorter, longer = (la, lb) if len(la) <= len(lb) else (lb, la)
        if len(shorter) >= 0.75 * len(longer):
            return 0.92
        if len(shorter) >= 0.5 * len(longer):
            return 0.88
    return difflib.SequenceMatcher(None, la, lb).ratio()


def _is_near_duplicate_memory(new: str, existing: str) -> bool:
    if len(new) < MEMORY_SHORT_LEN or len(existing) < MEMORY_SHORT_LEN:
        return new.lower() == existing.lower()
    return _memory_entry_similarity(new, existing) >= MEMORY_SIMILARITY_DEDUPE


_META_NOISE_SUBSTRINGS = ("output only python", "write only python", "avoid extra reconstruction", "no inline comments", "embedded natural language")


def _is_low_value_memory_noise(s: str) -> bool:
    t = s.lower()
    return any(p in t for p in _META_NOISE_SUBSTRINGS)


def _dedupe_memory_list(items: list[str]) -> list[str]:
    out: list[str] = []
    for raw in items:
        s = _normalize_one_entry(raw)
        if not s or _is_low_value_memory_noise(s):
            continue
        if any(_is_near_duplicate_memory(s, o) for o in out):
            continue
        out.append(s)
    return out


def _normalize_short_entries(entries: list[str]) -> list[str]:
    return _dedupe_memory_list(entries)


def _merge_memory_into(existing: list[str], incoming: list[str]) -> list[str]:
    base = _dedupe_memory_list(existing)
    for s in _dedupe_memory_list(incoming):
        if any(_is_near_duplicate_memory(s, o) for o in base):
            continue
        base.append(s)
    return base


def update_known_facts(last_failure_summary: str | None = None, best_mse: float | None = None, solved: bool | None = None):
    known = load_json(KNOWN_FACTS_PATH, {})
    if last_failure_summary is not None:
        known["last_failure_summary"] = last_failure_summary
    if best_mse is not None:
        known["best_mse"] = best_mse
    if solved is not None:
        known["solved"] = solved
    save_json(KNOWN_FACTS_PATH, known)


def update_known_fact_entries(facts: list[str] | None = None, rejections: list[str] | None = None):
    known = load_json(KNOWN_FACTS_PATH, {})
    known["confirmed_facts"] = _merge_memory_into(known.get("confirmed_facts", []), facts or [])[-MAX_MEMORY_LIST_ITEMS:]
    known["rejected_hypotheses"] = _merge_memory_into(known.get("rejected_hypotheses", []), rejections or [])[-MAX_MEMORY_LIST_ITEMS:]
    save_json(KNOWN_FACTS_PATH, known)


def compact_known_facts_store():
    if not KNOWN_FACTS_PATH.exists():
        return
    known = load_json(KNOWN_FACTS_PATH, {})
    known["confirmed_facts"] = _dedupe_memory_list(known.get("confirmed_facts", []))[-MAX_MEMORY_LIST_ITEMS:]
    known["rejected_hypotheses"] = _dedupe_memory_list(known.get("rejected_hypotheses", []))[-MAX_MEMORY_LIST_ITEMS:]
    save_json(KNOWN_FACTS_PATH, known)


def init_results_store():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESEARCH_LOG_PATH.exists():
        RESEARCH_LOG_PATH.write_text(f"# Research Log\nStarted: {datetime.now()}\n", encoding="utf-8")
    if not BEST_RESULT_PATH.exists():
        save_json(BEST_RESULT_PATH, {"mse": float("inf"), "solved": False})
    if not KNOWN_FACTS_PATH.exists():
        save_json(
            KNOWN_FACTS_PATH,
            {
                "environment_contract": {"data_inputs": ["data/pieces/piece_*.pth", "data/historical_data.csv"], "solver_output": "results/solution_output.txt", "results_dir": "results/"},
                "confirmed_facts": [],
                "rejected_hypotheses": [],
                "last_failure_summary": "(none)",
                "best_mse": float("inf"),
                "solved": False,
            },
        )
    compact_known_facts_store()
    if not EXPERIMENTS_JSONL_PATH.exists():
        EXPERIMENTS_JSONL_PATH.write_text("", encoding="utf-8")
    if not ATTEMPT_SUMMARY_TSV_PATH.exists():
        ATTEMPT_SUMMARY_TSV_PATH.write_text("ts\titeration\taction\tstatus\tmse\tperm_sha16\teval_error\tartifacts\tplan_excerpt\n", encoding="utf-8")
    if not RESULTS_TSV_PATH.exists():
        RESULTS_TSV_PATH.write_text("iteration\taction\tstatus\tmse\tsolved\timproved\tplan_family\tnotes\n", encoding="utf-8")
    if not RESEARCH_SYNTHESIS_PATH.exists():
        RESEARCH_SYNTHESIS_PATH.write_text(
            "# Research synthesis\n\nAppend-only checkpoints. Each **SYNTHESIS** action adds a dated section below. Later syntheses should **continue** this narrative.\n\n",
            encoding="utf-8",
        )


def _chat_completion_debug_fields(resp) -> dict:
    out: dict = {"response_id": getattr(resp, "id", None), "model": getattr(resp, "model", None)}
    choices = getattr(resp, "choices", None) or []
    if not choices:
        out["note"] = "response.choices is empty"
        return out
    ch = choices[0]
    msg = ch.message
    out["finish_reason"] = getattr(ch, "finish_reason", None)
    content = msg.content
    out["content_python_type"] = type(content).__name__
    if isinstance(content, str):
        out["content_len"] = len(content)
    elif content is None:
        out["content_len"] = 0
    else:
        out["content_repr_preview"] = repr(content)[:500]
    ref = getattr(msg, "refusal", None)
    if ref:
        out["refusal"] = ref[:2000] if isinstance(ref, str) else repr(ref)[:500]
    tc = getattr(msg, "tool_calls", None)
    if tc:
        out["tool_calls_count"] = len(tc)
        out["tool_calls_names"] = [getattr(getattr(t, "function", None), "name", str(t)) for t in tc[:8]]
    return out


def call_model(messages: list, max_tokens=MODEL_MAX_OUTPUT_TOKENS, temperature: float = 0.6, *, response_debug: dict | None = None) -> str:
    max_attempts = 6
    base_delay = 3.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
            empty_try = 0
            while True:
                choice = resp.choices[0] if resp.choices else None
                raw = choice.message.content if choice and choice.message else ""
                raw = raw or ""
                if raw.strip():
                    if response_debug is not None:
                        response_debug.clear()
                    return raw
                if empty_try >= EMPTY_MESSAGE_CONTENT_RETRIES:
                    if response_debug is not None:
                        response_debug.clear()
                        response_debug.update(_chat_completion_debug_fields(resp))
                    return raw
                empty_try += 1
                time.sleep(0.8 + random.uniform(0.0, 0.6))
                resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
        except (RateLimitError, APITimeoutError, APIError):
            if attempt == max_attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 1.5))
    return ""


def propose_memory_updates(source: str, snippet: str | None) -> tuple[list[str], list[str]]:
    snippet = (snippet or "")[:6000]
    known = load_json(KNOWN_FACTS_PATH, {})
    cur_f = known.get("confirmed_facts", [])[-25:]
    cur_r = known.get("rejected_hypotheses", [])[-25:]
    resp = call_model(
        [
            {"role": "system", "content": textwrap.dedent(f"""Extract optional durable memory updates from {source}.
Return ONLY valid JSON with schema: {{"facts": [], "rejections": []}}
Keep each entry <= {MAX_MEMORY_ENTRY_CHARS} chars. Up to {MEMORY_EXTRACT_MAX_PER_LIST} per list.""")},
            {"role": "user", "content": f"Existing facts:\n{json.dumps(cur_f)}\nExisting rejections:\n{json.dumps(cur_r)}\n--- Source ---\n{snippet}"},
        ],
        max_tokens=MEMORY_UPDATE_MAX_TOKENS,
        temperature=0.15,
    )
    parsed = _parse_json_object_loose(resp)
    if not isinstance(parsed, dict):
        return [], []
    facts = _dedupe_memory_list(parsed.get("facts", []) if isinstance(parsed.get("facts"), list) else [])[:MEMORY_EXTRACT_MAX_PER_LIST]
    rejections = _dedupe_memory_list(parsed.get("rejections", []) if isinstance(parsed.get("rejections"), list) else [])[:MEMORY_EXTRACT_MAX_PER_LIST]
    return facts, rejections


def extract_inline_memory_updates(text: str | None) -> tuple[list[str], list[str]]:
    facts: list[str] = []
    rejections: list[str] = []
    for line in (text or "").splitlines():
        m_fact = re.match(r"^\s*[-*]?\s*FACT:\s*(.+)$", line, flags=re.IGNORECASE)
        m_rej = re.match(r"^\s*[-*]?\s*REJECT:\s*(.+)$", line, flags=re.IGNORECASE)
        if m_fact:
            facts.append(m_fact.group(1))
        if m_rej:
            rejections.append(m_rej.group(1))
    return _normalize_short_entries(facts), _normalize_short_entries(rejections)


def append_tsv_row(iteration: int, action: str, status: str, mse: float | str, solved: bool, improved: bool, plan_family: str, notes: str):
    line = f"{iteration}\t{action}\t{status}\t{mse}\t{solved}\t{improved}\t{plan_family}\t{notes}\n"
    with open(RESULTS_TSV_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def _read_prior_script_block(path: Path, *, max_chars: int) -> str:
    rel = path.as_posix()
    if not path.exists():
        return f"### Prior `{rel}`\n_(not present yet — first generation for this path.)_\n"
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return f"### Prior `{rel}`\n_(empty file.)_\n"
    if len(raw) <= max_chars:
        return f"### Prior `{rel}` ({len(raw)} chars)\n```python\n{raw.rstrip()}\n```\n"
    head = max_chars * 2 // 3
    tail = max(800, max_chars - head - 120)
    omitted = len(raw) - head - tail
    return f"### Prior `{rel}` (truncated; {len(raw)} chars total)\n```python\n{raw[:head].rstrip()}\n\n# ... [{omitted} chars omitted] ...\n\n{raw[-tail:].lstrip()}\n```\n"


def prior_investigate_script_for_prompt() -> str:
    return _read_prior_script_block(RESULTS_DIR / "investigate.py", max_chars=PRIOR_INVESTIGATE_SCRIPT_CHARS)


def prior_solver_scripts_for_prompt() -> str:
    attempt_path = RESULTS_DIR / "solve_attempt.py"
    best_path = RESULTS_DIR / "best_solver.py"
    parts = [_read_prior_script_block(attempt_path, max_chars=PRIOR_SOLVE_ATTEMPT_SCRIPT_CHARS)]
    if best_path.exists():
        best_txt = best_path.read_text(encoding="utf-8", errors="replace")
        att_txt = attempt_path.read_text(encoding="utf-8", errors="replace") if attempt_path.exists() else ""
        if best_txt.strip() and best_txt != att_txt:
            parts.append(_read_prior_script_block(best_path, max_chars=PRIOR_BEST_SOLVER_SCRIPT_CHARS))
    return "\n".join(parts)


def sanitize_model_python_script(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("\ufeff"):
        s = s[1:].lstrip()
    if "```" in s:
        m = re.search(r"```(?:python|py)?\s*\r?\n?(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
        else:
            lines = s.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
    s = "\n".join([ln for ln in s.splitlines() if ln.strip() != "```"]).strip()
    return s


def subprocess_text_kw() -> dict:
    return {"text": True, "encoding": "utf-8", "errors": "replace"}


def python_compile_check(code: str, logical_name: str) -> tuple[bool, str]:
    try:
        compile(code, logical_name, "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}\n  {(e.text or '').rstrip()}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def repair_script_until_compile(sanitized: str, logical_name: str, *, label: str) -> tuple[str, bool, str, int]:
    if not (sanitized or "").strip():
        return sanitized, False, "empty script", 0
    s = sanitized
    compiles, err = python_compile_check(s, logical_name)
    if compiles:
        return s, True, "", 0
    repairs_done = 0
    for _ in range(SCRIPT_COMPILE_REPAIR_ATTEMPTS):
        repairs_done += 1
        snippet = s if len(s) <= 14_000 else s[:10_000] + "\n# ... [truncated middle for repair prompt] ...\n" + s[-3500:]
        fix = call_model(
            [
                {"role": "system", "content": f"You fix Python so it passes compile(). Output ONLY Python. Target: {label}."},
                {"role": "user", "content": f"compile() failed for `{logical_name}`.\nError:\n{err}\n\n```python\n{snippet}\n```\nReturn full corrected script."},
            ],
            max_tokens=min(MODEL_MAX_OUTPUT_TOKENS, 12_000),
            temperature=0.12,
        )
        s = sanitize_model_python_script(fix)
        if not s.strip():
            err = "repair model returned empty script"
            continue
        compiles, err = python_compile_check(s, logical_name)
        if compiles:
            return s, True, "", repairs_done
    return s, False, err, repairs_done


def decide_action(iteration: int) -> tuple[str, str]:
    context = build_context()
    messages = [
        {
            "role": "system",
            "content": textwrap.dedent(
                """
            You are an AI researcher solving a neural network reconstruction puzzle.
            Respond ONLY with valid JSON:
            {"action":"THINK|INVESTIGATE|SOLUTION|SYNTHESIS","reasoning":"...","plan":"..."}
            """
            ),
        },
        {"role": "user", "content": f"Iteration {iteration}.\n\n{context}\n\nWhat action do you take?"},
    ]
    last_resp = ""
    for _ in range(4):
        resp = call_model(messages)
        last_resp = resp or ""
        parsed = _parse_json_object_loose(last_resp)
        if not isinstance(parsed, dict):
            messages.append({"role": "user", "content": "Reply with one valid JSON object only."})
            continue
        action = _normalize_action_label(parsed.get("action"))
        if action not in ALLOWED_ACTIONS:
            messages.append({"role": "user", "content": "Invalid action. Use THINK/INVESTIGATE/SOLUTION/SYNTHESIS."})
            continue
        reasoning = str(parsed.get("reasoning") or "").strip()
        plan = str(parsed.get("plan") or "").strip()
        body = f"{reasoning}\n{plan}".strip()
        if body:
            return action, body
    snippet = last_resp[:500].replace("\n", " ")
    raise ValueError(f"Unable to parse valid action JSON after retries. Last response: {snippet!r}...")
