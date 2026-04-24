import json
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from datetime import datetime, timezone

from config import (
    BEST_RESULT_PATH,
    MAX_ITERATIONS,
    RESEARCH_SYNTHESIS_PATH,
    RESULTS_DIR,
    SCRIPT_COMPILE_REPAIR_ATTEMPTS,
    SOLUTION_TIMEOUT_SECS,
    SYNTHESIS_MAX_OUTPUT_TOKENS,
    SYNTHESIS_PRIOR_TAIL,
)
from agent_core import (
    append_tsv_row,
    build_context,
    call_model,
    decide_action,
    extract_inline_memory_updates,
    init_results_store,
    load_json,
    log,
    log_experiment,
    prior_investigate_script_for_prompt,
    prior_solver_scripts_for_prompt,
    propose_memory_updates,
    repair_script_until_compile,
    sanitize_model_python_script,
    save_json,
    subprocess_text_kw,
    update_known_fact_entries,
    update_known_facts,
    _normalize_short_entries,
    _perm_sha16,
    _read_synthesis_tail,
)


def run_think(plan: str, iteration: int):
    context = build_context()
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a neural net reconstruction puzzle. "
                "Be concise and specific. Optional memory protocol: FACT:/REJECT: lines."
            ),
        },
        {"role": "user", "content": f"{context}\n\nYour plan: {plan}\n\nThink carefully about next moves."},
    ]
    thoughts = ""
    for _ in range(2):
        thoughts = (call_model(messages) or "").strip()
        if thoughts:
            break
        messages.append({"role": "user", "content": "Your reply was empty. Write concrete analysis."})
    log(f"### THINK (iter {iteration})\n{thoughts or '(no text in model response)'}")
    facts_inline, rej_inline = extract_inline_memory_updates(thoughts)
    facts_llm, rej_llm = propose_memory_updates("THINK output", thoughts)
    facts = _normalize_short_entries(facts_inline + facts_llm)[:10]
    rejections = _normalize_short_entries(rej_inline + rej_llm)[:10]
    if facts or rejections:
        update_known_fact_entries(facts=facts, rejections=rejections)
        log(
            "### MEMORY UPDATED (from THINK)\n"
            f"facts_added={json.dumps(facts)}\n"
            f"rejections_added={json.dumps(rejections)}"
        )


def run_synthesis(plan: str, iteration: int) -> str:
    prior = _read_synthesis_tail(SYNTHESIS_PRIOR_TAIL)
    ctx = build_context(for_synthesis_call=True)
    messages = [
        {
            "role": "system",
            "content": "Write a markdown-only research synthesis checkpoint. Continue prior narrative.",
        },
        {
            "role": "user",
            "content": (
                f"Iteration {iteration}.\n\nPlan:\n{plan}\n\nPrevious synthesis:\n```\n{prior}\n```\n\nContext:\n{ctx}"
            ),
        },
    ]
    body = ""
    for _ in range(2):
        body = (call_model(messages, max_tokens=SYNTHESIS_MAX_OUTPUT_TOKENS, temperature=0.35) or "").strip()
        if body:
            break
        messages.append({"role": "user", "content": "Reply was empty. Provide a substantive markdown synthesis."})

    block = (
        f"\n---\n\n## Checkpoint — iteration {iteration} — "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n{body or '(no text in model response)'}\n"
    )
    with open(RESEARCH_SYNTHESIS_PATH, "a", encoding="utf-8") as f:
        f.write(block)

    status = "ok" if body else "empty"
    log_experiment(
        {
            "iteration": iteration,
            "action": "SYNTHESIS",
            "status": status,
            "plan": plan,
            "output_file": str(RESEARCH_SYNTHESIS_PATH),
        }
    )
    log(f"### SYNTHESIS (iter {iteration}) — appended to `{RESEARCH_SYNTHESIS_PATH}`\n{(body or '')[:4500]}")
    if body:
        facts_inline, rej_inline = extract_inline_memory_updates(body)
        facts_llm, rej_llm = propose_memory_updates("SYNTHESIS output", body)
        facts = _normalize_short_entries(facts_inline + facts_llm)[:10]
        rejections = _normalize_short_entries(rej_inline + rej_llm)[:10]
        if facts or rejections:
            update_known_fact_entries(facts=facts, rejections=rejections)
            log(
                "### MEMORY UPDATED (from SYNTHESIS)\n"
                f"facts_added={json.dumps(facts)}\n"
                f"rejections_added={json.dumps(rejections)}"
            )
    return status


def run_investigate(plan: str, iteration: int) -> str:
    context = build_context()
    empty_debug: dict = {}
    script = call_model(
        [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """
                Write a Python diagnostic script.
                - Read data/pieces/ and data/historical_data.csv
                - Print diagnostics only
                - Output ONLY Python code
                """
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\nPlan: {plan}\n\n{prior_investigate_script_for_prompt()}\n"
                    "Write the investigation script as full replacement for results/investigate.py."
                ),
            },
        ],
        response_debug=empty_debug,
    )
    sanitized = sanitize_model_python_script(script)
    investigate_path = RESULTS_DIR / "investigate.py"
    if not sanitized.strip():
        investigate_path.write_text("", encoding="utf-8")
        output = "Harness: no investigation script text from the model."
        status = "empty_script"
    else:
        sanitized, compiles, compile_err, repairs = repair_script_until_compile(
            sanitized,
            "investigate.py",
            label="diagnostic investigation script",
        )
        if repairs > 0:
            log(f"### INVESTIGATE (iter {iteration}): compile repair invoked ({repairs}/{SCRIPT_COMPILE_REPAIR_ATTEMPTS}); compile_ok={compiles}")
        investigate_path.write_text(sanitized, encoding="utf-8")
        if not compiles:
            output = "Harness: script failed compile() before execution.\n" + compile_err
            status = "compile_error"
        else:
            try:
                result = subprocess.run([sys.executable, str(investigate_path)], capture_output=True, timeout=90, **subprocess_text_kw())
                output = (result.stdout or "") + (f"\nSTDERR:\n{result.stderr}" if (result.stderr or "").strip() else "")
                status = "ok" if result.returncode == 0 else "runtime_error"
                if result.returncode != 0:
                    output = f"(exit code {result.returncode})\n{output}".lstrip()
            except subprocess.TimeoutExpired:
                output = "TIMED OUT after 90 seconds"
                status = "timeout"
    out_path = RESULTS_DIR / f"investigate_iter{iteration}_output.txt"
    out_path.write_text(output, encoding="utf-8")
    log_experiment({"iteration": iteration, "action": "INVESTIGATE", "status": status, "plan": plan, "output_file": str(out_path)})
    facts, rejections = propose_memory_updates("INVESTIGATE output", f"Plan:\n{plan}\n\nOutput:\n{output}")
    if facts or rejections:
        update_known_fact_entries(facts=facts, rejections=rejections)
    if status != "ok":
        update_known_facts(last_failure_summary=f"INVESTIGATE iter {iteration}: {status} (see {out_path.name})")
    excerpt = output if len(output) <= 7000 else output[:3500] + "\n\n... [truncated for log] ...\n\n" + output[-3000:]
    log(
        f"### INVESTIGATE (iter {iteration}) — harness **{status}**\n"
        f"**Plan:** {plan}\n\n**Full output file:** `{out_path}`\n\n**Output excerpt:**\n```\n{excerpt}\n```"
    )
    return status


def run_solution(plan: str, iteration: int) -> bool:
    context = build_context()
    empty_debug: dict = {}
    script = call_model(
        [
            {
                "role": "system",
                "content": "Write a Python solver script. Output ONLY Python code.",
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\nPlan: {plan}\n\n{prior_solver_scripts_for_prompt()}\n"
                    "Write the solver as full replacement for results/solve_attempt.py."
                ),
            },
        ],
        response_debug=empty_debug,
    )
    sanitized = sanitize_model_python_script(script)
    solve_attempt_path = RESULTS_DIR / "solve_attempt.py"

    if not sanitized.strip():
        solve_attempt_path.write_text("", encoding="utf-8")
        shutil.copy(solve_attempt_path, RESULTS_DIR / f"solve_attempt_iter{iteration}.py")
        msg = f"SOLUTION iter {iteration}: model returned no assistant text in message.content"
        update_known_facts(last_failure_summary=msg)
        log_experiment({"iteration": iteration, "action": "SOLUTION", "status": "empty_script", "plan": plan, "mse": None, "solved": False, "improved": False})
        log(f"### {msg}")
        return False

    sanitized, compiles, compile_err, repairs = repair_script_until_compile(
        sanitized,
        "solve_attempt.py",
        label="solver script (writes results/solution_output.txt)",
    )
    if repairs > 0:
        log(f"### SOLUTION (iter {iteration}): compile repair invoked ({repairs}/{SCRIPT_COMPILE_REPAIR_ATTEMPTS}); compile_ok={compiles}")
    solve_attempt_path.write_text(sanitized, encoding="utf-8")
    shutil.copy(solve_attempt_path, RESULTS_DIR / f"solve_attempt_iter{iteration}.py")
    if not compiles:
        err_path = RESULTS_DIR / f"solve_attempt_iter{iteration}_compile_error.txt"
        err_path.write_text(compile_err, encoding="utf-8")
        msg = f"SOLUTION iter {iteration}: script compile failed after repair attempts (not executed)"
        update_known_facts(last_failure_summary=msg)
        log_experiment({"iteration": iteration, "action": "SOLUTION", "status": "compile_error", "plan": plan, "compile_error_file": str(err_path), "mse": None, "solved": False, "improved": False})
        log(f"### {msg}\nDetails saved to `{err_path}`\n{compile_err[:2000]}")
        return False
    try:
        proc = subprocess.run([sys.executable, str(solve_attempt_path)], timeout=SOLUTION_TIMEOUT_SECS, capture_output=True, **subprocess_text_kw())
    except subprocess.TimeoutExpired:
        msg = f"SOLUTION iter {iteration}: TIMED OUT"
        update_known_facts(last_failure_summary=msg)
        log_experiment({"iteration": iteration, "action": "SOLUTION", "status": "timeout", "plan": plan, "mse": None, "solved": False, "improved": False})
        log(f"### {msg}")
        return False

    solution_output_path = RESULTS_DIR / "solution_output.txt"
    if not solution_output_path.exists():
        stderr_path = RESULTS_DIR / f"solve_attempt_iter{iteration}_stderr.txt"
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        msg = f"SOLUTION iter {iteration}: No output file written"
        update_known_facts(last_failure_summary=msg)
        log_experiment({"iteration": iteration, "action": "SOLUTION", "status": "no_output", "plan": plan, "stderr_file": str(stderr_path), "mse": None, "solved": False, "improved": False})
        log(f"### {msg}\nSTDERR saved to `{stderr_path}`\nSTDERR preview:\n{(proc.stderr or '')[:1400]}")
        return False

    perm_str = solution_output_path.read_text(encoding="utf-8").strip()
    eval_result = subprocess.run([sys.executable, "evaluate.py", perm_str], capture_output=True, **subprocess_text_kw())
    try:
        metrics = json.loads(eval_result.stdout)
    except json.JSONDecodeError:
        msg = f"SOLUTION iter {iteration}: evaluator output parse failed"
        update_known_facts(last_failure_summary=msg)
        log_experiment({"iteration": iteration, "action": "SOLUTION", "status": "eval_parse_failed", "plan": plan, "mse": None, "solved": False, "improved": False, "perm_sha16": _perm_sha16(perm_str)})
        log(f"### {msg}")
        return False

    best = load_json(BEST_RESULT_PATH, {"mse": float("inf"), "solved": False})
    mse = metrics.get("mse", float("inf"))
    solved = bool(metrics.get("solved", False))
    improved = mse < best.get("mse", float("inf"))
    status = "accepted" if improved else "rejected"

    sol_row = {
        "iteration": iteration,
        "action": "SOLUTION",
        "status": status,
        "plan": plan,
        "mse": mse,
        "solved": solved,
        "improved": improved,
        "permutation_file": str(solution_output_path),
        "perm_sha16": _perm_sha16(perm_str),
    }
    if metrics.get("error"):
        sol_row["eval_error"] = str(metrics.get("error"))[:500]
    log_experiment(sol_row)
    append_tsv_row(iteration, "SOLUTION", status, mse, solved, improved, " ".join(plan.lower().split())[:140], "objective gate applied")

    if improved:
        save_json(BEST_RESULT_PATH, {**metrics, "iteration": iteration, "permutation": perm_str})
        shutil.copy(solve_attempt_path, RESULTS_DIR / "best_solver.py")
        (RESULTS_DIR / "best_permutation.txt").write_text(perm_str, encoding="utf-8")
        update_known_facts(best_mse=mse, solved=solved, last_failure_summary=f"latest solution status={status}, mse={mse}")
    else:
        update_known_facts(last_failure_summary=f"latest solution rejected (mse={mse}, best={best.get('mse')})")

    log(
        f"### SOLUTION iter {iteration}: {status.upper()}\n"
        f"**Metrics:** {json.dumps(metrics)}\n"
        f"**Objective gate:** {'improved' if improved else 'not improved'}"
    )
    if solved:
        (RESULTS_DIR / "SOLUTION.txt").write_text(perm_str, encoding="utf-8")
        return True
    return False


def main():
    init_results_store()
    recent_actions: list[str] = []
    recent_solution_improvement = True
    solved_flag = False

    for iteration in range(MAX_ITERATIONS):
        print(f"\n{'=' * 50}\nIteration {iteration}\n{'=' * 50}")
        try:
            action, plan = decide_action(iteration)
        except Exception as e:
            msg = f"Action parse failed iter {iteration}: {e}"
            update_known_facts(last_failure_summary=msg)
            log(f"### {msg}")
            continue

        log(f"### ACTION CHOSEN: {action} (iter {iteration})\n{plan}")
        recent_actions.append(action)
        recent_actions = recent_actions[-6:]

        try:
            if action == "THINK":
                run_think(plan, iteration)
                append_tsv_row(iteration, "THINK", "ok", "n/a", False, False, "think", "analysis step")
            elif action == "SYNTHESIS":
                syn_status = run_synthesis(plan, iteration)
                append_tsv_row(iteration, "SYNTHESIS", syn_status, "n/a", False, False, "synthesis", "checkpoint")
            elif action == "INVESTIGATE":
                inv_status = run_investigate(plan, iteration)
                append_tsv_row(iteration, "INVESTIGATE", inv_status, "n/a", False, False, "investigate", "diagnostic step")
            elif action == "SOLUTION":
                best_before = load_json(BEST_RESULT_PATH, {"mse": float("inf")}).get("mse", float("inf"))
                solved = run_solution(plan, iteration)
                best_after = load_json(BEST_RESULT_PATH, {"mse": float("inf")}).get("mse", float("inf"))
                recent_solution_improvement = bool(best_after < best_before)
                if solved:
                    solved_flag = True
                    break
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = f"Harness exception on {action} iter {iteration}: {type(e).__name__}: {e}"
            update_known_facts(last_failure_summary=msg[:900])
            log(f"### {msg}\n```\n{traceback.format_exc()[:5000]}\n```")
            append_tsv_row(iteration, action, "harness_crash", "n/a", False, False, "error", str(e)[:120])
        time.sleep(1.5)

    if not solved_flag:
        print(
            f"\n{'=' * 50}\n"
            f"Agent stopped: max iterations ({MAX_ITERATIONS}) reached without solve.\n"
            f"{'=' * 50}\n"
        )
