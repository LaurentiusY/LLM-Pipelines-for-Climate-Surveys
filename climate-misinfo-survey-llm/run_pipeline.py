"""
Climate Misinfo Survey LLM Pipeline

Usage:
    python run_pipeline.py cot              # full CoT pipeline
    python run_pipeline.py vbn              # full VBN pipeline
    python run_pipeline.py cot --test 5     # random sample of 5 for testing

Each run creates runs/run_{YYYYMMDD_HHMMSS}_{method}/ and writes
all three stages (persona -> call -> eval) into the same folder.
"""
import sys
import os
from datetime import datetime


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('cot', 'vbn'):
        print("Usage: python run_pipeline.py <cot|vbn> [--test N]")
        sys.exit(1)

    method = sys.argv[1]
    test_mode = '--test' in sys.argv
    test_size = 5  # default test sample count
    if test_mode and '--test' in sys.argv:
        idx = sys.argv.index('--test')
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            test_size = int(sys.argv[idx + 1])

    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f'run_{timestamp}_{method}')
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Climate Misinfo Survey LLM Pipeline")
    print(f"{'='*60}")
    print(f"Method:   {method.upper()}")
    print(f"Run dir:  {run_dir}")
    if test_mode:
        print(f"Mode:     TEST ({test_size} random samples)")
    print(f"{'='*60}\n")

    # ====== Step 1: Persona (generate prompts) ======
    print(f"[Step 1/3] Generating prompts...")
    if method == 'cot':
        from persona.run_cot import main as persona_main
    else:
        from persona.run_vbn import main as persona_main
    prompts_data = persona_main(run_dir)

    # ====== Step 2: Call (invoke LLM) ======
    print(f"\n[Step 2/3] Calling LLM...")
    if method == 'cot':
        import call.config_cot as call_cfg
    else:
        import call.config_vbn as call_cfg

    # test mode: enable random sampling temporarily
    if test_mode:
        call_cfg.USE_RANDOM_SAMPLING = True
        call_cfg.RANDOM_SAMPLE_SIZE = test_size
        call_cfg.RANDOM_SAMPLE_SEED = 42

    from call.llm_runner import main as call_main
    call_main(run_dir, call_cfg, prompts_data)

    # ====== Step 3: Eval (evaluate results) ======
    print(f"\n[Step 3/3] Evaluating results...")
    if method == 'cot':
        from eval.run_cot import main as eval_main
    else:
        from eval.run_vbn import main as eval_main
    eval_main(run_dir)

    # ====== Done ======
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"All outputs: {run_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
