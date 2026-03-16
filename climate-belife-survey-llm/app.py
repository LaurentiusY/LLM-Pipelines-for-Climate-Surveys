"""
Terminal UI for Climate Belief Survey LLM Pipeline.

Usage:
    python app.py
"""

import os
import json
import time
import threading
import traceback
from types import SimpleNamespace
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_lines(path: str) -> int:
    """Count lines in a file (for progress tracking)."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _scan_runs() -> list[str]:
    """Return list of run directories sorted newest-first."""
    runs_dir = "runs"
    if not os.path.isdir(runs_dir):
        return []
    dirs = [
        os.path.join(runs_dir, d)
        for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("run_")
    ]
    dirs.sort(key=lambda p: os.path.basename(p), reverse=True)
    return dirs


def _load_summary(run_dir: str) -> pd.DataFrame | None:
    """Load the Summary sheet from evaluation.xlsx."""
    xlsx = os.path.join(run_dir, "evaluation.xlsx")
    if not os.path.exists(xlsx):
        return None
    try:
        return pd.read_excel(xlsx, sheet_name="Summary")
    except Exception:
        return None


def _collect_excel_files(run_dir: str) -> list[tuple[str, str]]:
    """Return (display_name, full_path) for Excel output files."""
    names = [
        "evaluation.xlsx",
        "evaluation_raw.xlsx",
        "evaluation_by_group.xlsx",
        "evaluation_plans.xlsx",
    ]
    out = []
    for n in names:
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            out.append((n, p))
    return out


def _load_sheet(run_dir: str, sheet_name: str) -> "pd.DataFrame | None":
    """Load a specific sheet from evaluation.xlsx."""
    xlsx = os.path.join(run_dir, "evaluation.xlsx")
    if not os.path.exists(xlsx):
        return None
    try:
        return pd.read_excel(xlsx, sheet_name=sheet_name)
    except Exception:
        return None


def _display_results(run_dir: str):
    """Show metrics and file list for a finished run."""
    summary = _load_summary(run_dir)
    if summary is None or len(summary) == 0:
        print("  No evaluation results found in this run.")
        return

    print("\n  ── Continuous Metrics (MAE / Mean Error) ─────────────────")

    belief = summary[(summary["Task"] == "Belief") & (summary["Scale"] == "original")]
    policy = summary[(summary["Task"] == "Policy") & (summary["Scale"] == "original")]

    if len(belief) > 0:
        r = belief.iloc[0]
        print(f"  Belief          MAE={r['MAE']:.2f}  Mean_Error={r['Mean_Error']:.2f}  N={int(r['N'])}")
    else:
        print("  Belief          N/A")

    if len(policy) > 0:
        r = policy.iloc[0]
        print(f"  Policy          MAE={r['MAE']:.2f}  Mean_Error={r['Mean_Error']:.2f}  N={int(r['N'])}")
    else:
        print("  Policy          N/A")

    print("\n  ── Correlation (Pearson r / Spearman r) ──────────────────")
    for task_label, sheet in [("Belief", "Belief_Correlation"),
                               ("Policy", "Policy_Correlation")]:
        df = _load_sheet(run_dir, sheet)
        if df is not None and len(df) > 0:
            overall = df[df["question"] == "OVERALL"] if "question" in df.columns else df
            if len(overall) > 0:
                r = overall.iloc[0]
                pr = r.get("pearson_r",  float("nan"))
                sr = r.get("spearman_r", float("nan"))
                n  = r.get("N",          float("nan"))
                print(f"  {task_label:<16}  pearson_r={pr:.4f}  spearman_r={sr:.4f}  N={int(n)}")
        else:
            print(f"  {task_label:<16}  N/A")

    print("\n  ── Share (Social Media Sharing) ──────────────────────────")
    share = summary[summary["Task"] == "Share"]
    if len(share) > 0:
        r = share.iloc[0]
        print(f"  Share           Accuracy={r['Accuracy']:.4f}  F1={r['F1']:.4f}  N={int(r['N'])}")
    else:
        print("  Share           N/A")

    print("\n  ── Trees / WEPT (Pro-environmental Effort) ───────────────")
    trees_cont = summary[(summary["Task"] == "Trees") & (summary["Scale"] == "continuous")]
    trees_bin  = summary[(summary["Task"] == "Trees") & (summary["Scale"] == "binary")]
    trees_tern = summary[(summary["Task"] == "Trees") & (summary["Scale"] == "ternary")]

    if len(trees_cont) > 0:
        r = trees_cont.iloc[0]
        print(f"  Trees (cont.)   MAE={r['MAE']:.2f}  Mean_Error={r['Mean_Error']:.2f}  N={int(r['N'])}")
    if len(trees_bin) > 0:
        r = trees_bin.iloc[0]
        print(f"  Trees (binary)  Accuracy={r['Accuracy']:.4f}  F1={r['F1']:.4f}  N={int(r['N'])}")
    if len(trees_tern) > 0:
        r = trees_tern.iloc[0]
        print(f"  Trees (ternary) Accuracy={r['Accuracy']:.4f}  F1={r['F1']:.4f}  N={int(r['N'])}")
    if len(trees_cont) == 0 and len(trees_bin) == 0:
        print("  Trees           N/A")

    print()
    excel_files = _collect_excel_files(run_dir)
    if excel_files:
        print("  Output Files:")
        for name, path in excel_files:
            print(f"    - {path}")
    else:
        print("  No Excel output files found.")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    """Prompt user for input with an optional default."""
    if default:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {prompt}: ").strip()


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"    Invalid number, using default: {default}")
        return default


def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"    Invalid number, using default: {default}")
        return default


def _ask_bool(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    options = "/".join(choices)
    raw = input(f"  {prompt} ({options}) [{default}]: ").strip()
    if not raw:
        return default
    if raw in choices:
        return raw
    # case-insensitive match
    for c in choices:
        if c.lower() == raw.lower():
            return c
    print(f"    Invalid choice, using default: {default}")
    return default


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class PipelineState:
    """Shared mutable state between pipeline thread and progress display."""
    def __init__(self):
        self.running = False
        self.step = ""          # "persona" | "call" | "eval" | "done" | "error"
        self.run_dir = ""
        self.error_msg = ""
        self.total_tasks = 0


def _run_pipeline(method: str, cfg: SimpleNamespace,
                  profile_mode: str, target_condition: str,
                  run_dir: str, state: PipelineState):
    """Execute the full persona -> call -> eval pipeline."""
    try:
        # --- Step 1: Persona ---
        state.step = "persona"
        if method == "cot":
            import persona.run_cot as persona_mod
        else:
            import persona.run_vbn as persona_mod
        persona_mod.PROFILE_MODE = profile_mode
        persona_mod.TARGET_CONDITION = target_condition
        prompts_data = persona_mod.main(run_dir)

        # figure out total tasks for progress
        if prompts_data:
            n_samples = len(prompts_data.get("row_ids", []))
            if cfg.SELECTED_ROW_IDS:
                n_samples = min(n_samples, len(cfg.SELECTED_ROW_IDS))
            if cfg.USE_RANDOM_SAMPLING:
                n_samples = min(n_samples, cfg.RANDOM_SAMPLE_SIZE)
            state.total_tasks = n_samples * 4 * cfg.REPEAT_TIMES

        # --- Step 2: Call LLM ---
        state.step = "call"
        from call.llm_runner import main as call_main
        call_main(run_dir, cfg, prompts_data)

        # --- Step 3: Eval ---
        state.step = "eval"
        if method == "cot":
            from eval.run_cot import main as eval_main
        else:
            from eval.run_vbn import main as eval_main
        eval_main(run_dir)

        state.step = "done"

    except Exception as e:
        state.error_msg = f"{e}\n\n{traceback.format_exc()}"
        state.step = "error"
    finally:
        state.running = False


def _show_progress(state: PipelineState):
    """Poll and display progress until the pipeline finishes."""
    step_labels = {
        "persona": "Step 1/3 - Generating prompts...",
        "call":    "Step 2/3 - Calling LLM...",
        "eval":    "Step 3/3 - Evaluating results...",
    }
    last_msg = ""

    while state.running:
        step = state.step
        msg = step_labels.get(step, "Running...")

        if step == "call":
            temp_file = os.path.join(state.run_dir, "temp_results.jsonl")
            done = _count_lines(temp_file)
            total = state.total_tasks
            if total > 0:
                pct = min(done / total, 1.0)
                msg = f"Step 2/3 - Calling LLM... {done}/{total} tasks ({pct:.0%})"
            else:
                msg = f"Step 2/3 - Calling LLM... {done} tasks completed"

        if msg != last_msg:
            print(f"\r  {msg}          ", end="", flush=True)
            last_msg = msg

        time.sleep(0.5)

    # Final newline
    print()


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def action_run():
    """Interactive pipeline configuration and execution."""
    print("\n" + "=" * 60)
    print("  Configure Pipeline")
    print("=" * 60)

    # Method
    print("  Press Enter to accept the [default value] shown in brackets.\n")
    method = _ask_choice("Method  (cot=Chain-of-Thought / vbn=Value-Belief-Norm)", ["cot", "vbn"], "cot")

    # LLM settings
    print("\n  -- LLM Settings --")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        masked = env_key[:8] + "..." + env_key[-4:]
        api_key = _ask(f"API Key  (default: read from env OPENAI_API_KEY={masked})", env_key)
    else:
        api_key = _ask("API Key  (required, no default)")
    if not api_key:
        print("  ERROR: No API key provided. Aborting.")
        return

    base_url = _ask("Base URL  (API proxy or official endpoint)", "https://api.openai.com/v1")
    llm_model = _ask("Model  (recommended: gpt-4o-mini / gpt-4o)", "gpt-4o-mini")
    if not llm_model.strip():
        print("  ERROR: No model specified. Aborting.")
        return

    default_s1 = 500 if method == "vbn" else 400
    max_tokens_s1 = _ask_int("Max tokens Stage 1  (reasoning plan, more = more detail)", default_s1)
    max_tokens_s2 = _ask_int("Max tokens Stage 2  (final answer)", 120)
    temperature = _ask_float("Temperature  (0 = deterministic, >0 = random)", 0.0)

    # Sampling settings
    print("\n  -- Sampling Settings --")
    use_random = _ask_bool("Random sampling?  (y = sample N rows, N = use full dataset)", False)
    sample_size = 100
    sample_seed = 42
    if use_random:
        sample_size = _ask_int("  Sample size  (number of rows to randomly draw)", 100)
        sample_seed = _ask_int("  Sample seed  (for reproducibility)", 42)

    # Execution settings
    print("\n  -- Execution Settings --")
    max_workers = _ask_int("Concurrency workers  (parallel threads, recommended 4-8)", 6)
    llm_seed    = _ask_int("LLM random seed  (seed passed to the model)", 42)

    # Persona settings
    print("\n  -- Persona Settings --")
    profile_mode = _ask_choice(
        "Profile mode  (both=demographics+psychology / demographics / psychology)",
        ["both", "demographics", "psychology"], "both")
    target_condition = _ask_choice(
        "Target condition  (Both=all participants / Control / Treatment)",
        ["Both", "Control", "Treatment"], "Both")

    # Confirm
    print("\n" + "-" * 60)
    print(f"  Method:      {method.upper()}")
    print(f"  Model:       {llm_model}")
    print(f"  Temperature: {temperature}")
    print(f"  Workers:     {max_workers}")
    if use_random:
        print(f"  Sampling:    random ({sample_size} samples, seed={sample_seed})")
    else:
        print(f"  Sampling:    full dataset")
    print(f"  Profile:     {profile_mode}")
    print(f"  Condition:   {target_condition}")
    print("-" * 60)

    if not _ask_bool("Start pipeline?", True):
        print("  Cancelled.")
        return

    # Build config
    cfg = SimpleNamespace(
        API_KEY=api_key,
        BASE_URL=base_url,
        LLM_MODEL=llm_model,
        MAX_TOKENS_STAGE1=max_tokens_s1,
        MAX_TOKENS_STAGE2=max_tokens_s2,
        TEMPERATURE=temperature,
        MAX_WORKERS=max_workers,
        REPEAT_TIMES=1,
        LLM_RANDOM_SEED=llm_seed,
        USE_RANDOM_SAMPLING=use_random,
        RANDOM_SAMPLE_SIZE=sample_size,
        RANDOM_SAMPLE_SEED=sample_seed,
        SELECTED_ROW_IDS=[],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run_{timestamp}_{method}")
    os.makedirs(run_dir, exist_ok=True)

    state = PipelineState()
    state.running = True
    state.step = "persona"
    state.run_dir = run_dir

    print(f"\n  Run directory: {run_dir}")
    print()

    t = threading.Thread(
        target=_run_pipeline,
        args=(method, cfg, profile_mode, target_condition, run_dir, state),
        daemon=True,
    )
    t.start()

    _show_progress(state)

    if state.step == "done":
        print("\n  Pipeline complete!")
        _display_results(run_dir)
    elif state.step == "error":
        print("\n  Pipeline FAILED!")
        print(f"\n{state.error_msg}")


def action_history():
    """View past runs."""
    runs = _scan_runs()
    if not runs:
        print("\n  No past runs found.")
        return

    print("\n  Past Runs:")
    print("  " + "-" * 40)
    for i, r in enumerate(runs, 1):
        print(f"  {i}. {os.path.basename(r)}")

    raw = input(f"\n  Select a run (1-{len(runs)}) or press Enter to cancel: ").strip()
    if not raw:
        return
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(runs):
            selected = runs[idx]
            print(f"\n  Run: {selected}")
            _display_results(selected)
        else:
            print("  Invalid selection.")
    except ValueError:
        print("  Invalid input.")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  Climate Belief Survey - LLM Pipeline")
    print("=" * 60)

    while True:
        print("\n  1. Run Experiment")
        print("  2. View History")
        print("  3. Exit")

        choice = input("\n  Select [1/2/3]: ").strip()

        if choice == "1":
            action_run()
        elif choice == "2":
            action_history()
        elif choice == "3":
            print("  Bye!")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
