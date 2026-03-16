import json
import time
import concurrent.futures
import threading
import os
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

lock = threading.Lock()


def parse_json_content(content):
    try:
        return json.loads(content)
    except:
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return content


def load_completed_tasks(temp_file):
    completed = set()
    if os.path.exists(temp_file):
        with open(temp_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if item.get('success') and item.get('stage2_success'):
                        key = (item['row_id'], item['repeat'], item['task_type'])
                        completed.add(key)
                except:
                    continue
    return completed


def save_result_realtime(result, temp_file):
    with lock:
        with open(temp_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def call_llm(system_prompt, user_prompt, max_tokens, temperature, seed, client, model):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed
    )
    content = resp.choices[0].message.content
    return content, resp.usage


def process_one_task_two_stage(task_args, temp_file, client, cfg):
    row_id, repeat_idx, task_type, task_obj, seed_base, plan_token = task_args

    max_retries = 6
    base_delay = 2

    for attempt in range(max_retries):
        try:
            # ---------- Stage 1 ----------
            s1_sys = task_obj["stage1"]["system_prompt"]
            s1_user = task_obj["stage1"]["user_prompt"]

            seed_stage1 = seed_base * 10 + 1
            s1_content, s1_usage = call_llm(
                s1_sys, s1_user,
                max_tokens=cfg.MAX_TOKENS_STAGE1,
                temperature=cfg.TEMPERATURE,
                seed=seed_stage1,
                client=client,
                model=cfg.LLM_MODEL
            )
            plan_parsed = parse_json_content(s1_content)

            if isinstance(plan_parsed, (dict, list)):
                plan_str = json.dumps(plan_parsed, ensure_ascii=False)
            else:
                plan_str = str(plan_parsed)

            # ---------- Stage 2 ----------
            s2_sys = task_obj["stage2"]["system_prompt"]
            s2_user_template = task_obj["stage2"]["user_prompt"]

            if plan_token not in s2_user_template:
                raise ValueError(f"Plan token '{plan_token}' not found in stage2 user prompt.")

            s2_user = s2_user_template.replace(plan_token, plan_str)

            seed_stage2 = seed_base * 10 + 2
            s2_content, s2_usage = call_llm(
                s2_sys, s2_user,
                max_tokens=cfg.MAX_TOKENS_STAGE2,
                temperature=cfg.TEMPERATURE,
                seed=seed_stage2,
                client=client,
                model=cfg.LLM_MODEL
            )
            final_parsed = parse_json_content(s2_content)

            result = {
                "success": True,
                "stage2_success": True,
                "row_id": row_id,
                "repeat": repeat_idx + 1,
                "task_type": task_type,
                "plan": plan_parsed,
                "data": final_parsed,
                "usage": {
                    "stage1": {
                        "prompt": s1_usage.prompt_tokens,
                        "completion": s1_usage.completion_tokens,
                        "total": s1_usage.total_tokens
                    },
                    "stage2": {
                        "prompt": s2_usage.prompt_tokens,
                        "completion": s2_usage.completion_tokens,
                        "total": s2_usage.total_tokens
                    },
                    "total": {
                        "prompt": s1_usage.prompt_tokens + s2_usage.prompt_tokens,
                        "completion": s1_usage.completion_tokens + s2_usage.completion_tokens,
                        "total": s1_usage.total_tokens + s2_usage.total_tokens
                    }
                },
                "timestamp": datetime.now().isoformat()
            }

            save_result_realtime(result, temp_file)
            return result

        except RateLimitError:
            sleep_time = base_delay * (2 ** attempt)
            time.sleep(sleep_time)

        except APIError as e:
            print(f"\nAPIError on Row {row_id}-{task_type}: {e}")
            sleep_time = base_delay * (2 ** attempt)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"\nError on Row {row_id}-{task_type}: {e}")
            time.sleep(base_delay)

    result = {
        "success": False,
        "stage2_success": False,
        "row_id": row_id,
        "repeat": repeat_idx + 1,
        "task_type": task_type,
        "error": "Max retries reached",
        "timestamp": datetime.now().isoformat()
    }
    save_result_realtime(result, temp_file)
    return result


def reconstruct_from_jsonl(temp_file):
    results_flat = []

    with open(temp_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                results_flat.append(item)
            except:
                continue

    organized_data = {}

    for item in results_flat:
        if not item.get('success') or not item.get('stage2_success'):
            continue

        key = (item['row_id'], item['repeat'])

        if key not in organized_data:
            organized_data[key] = {
                "row_id": item['row_id'],
                "repeat": item['repeat'],
                "affect_science": None,
                "affect_action": None,
                "ccb": None,
                "mist": None,
                "plans": {
                    "affect_science": None,
                    "affect_action": None,
                    "ccb": None,
                    "mist": None,
                },
                "total_usage": {"prompt": 0, "completion": 0, "total": 0}
            }

        target = organized_data[key]
        task_type = item['task_type']
        data = item.get('data')
        plan = item.get('plan')

        if task_type in ('affect_science', 'affect_action', 'ccb', 'mist'):
            target[task_type] = data
            target['plans'][task_type] = plan

        usage_total = item.get('usage', {}).get('total', {})
        target['total_usage']['prompt'] += usage_total.get('prompt', 0)
        target['total_usage']['completion'] += usage_total.get('completion', 0)
        target['total_usage']['total'] += usage_total.get('total', 0)

    final_list = list(organized_data.values())
    final_list.sort(key=lambda x: (x['row_id'], x['repeat']))
    return final_list


def filter_row_ids(row_ids, cfg):
    import random

    if cfg.SELECTED_ROW_IDS:
        selected_set = set(cfg.SELECTED_ROW_IDS)
        row_ids = [rid for rid in row_ids if rid in selected_set]
        print(f"Filtered by row_id: {len(row_ids)} samples selected")

    if cfg.USE_RANDOM_SAMPLING and len(row_ids) > cfg.RANDOM_SAMPLE_SIZE:
        random.seed(cfg.RANDOM_SAMPLE_SEED)
        row_ids = random.sample(row_ids, cfg.RANDOM_SAMPLE_SIZE)
        print(f"Random sampling: {len(row_ids)} samples selected (seed={cfg.RANDOM_SAMPLE_SEED})")

    return row_ids


SAMPLE_BUFFER_SIZE = 10


def main(run_dir, cfg, prompts_data=None):
    output_file = os.path.join(run_dir, 'results.json')
    temp_file = os.path.join(run_dir, 'temp_results.jsonl')

    print(f"--- Scenario 2-Stage Runner for {cfg.LLM_MODEL} ---")
    print(f"Configuration:")
    print(f"  - Run dir: {run_dir}")
    print(f"  - Repeat times per sample: {cfg.REPEAT_TIMES}")
    print(f"  - LLM random seed base: {cfg.LLM_RANDOM_SEED}")
    if cfg.SELECTED_ROW_IDS:
        print(f"  - Selected row_ids: {cfg.SELECTED_ROW_IDS}")
    if cfg.USE_RANDOM_SAMPLING:
        print(f"  - Random sampling: {cfg.RANDOM_SAMPLE_SIZE} samples (seed={cfg.RANDOM_SAMPLE_SEED})")
    print()

    client = OpenAI(
        api_key=cfg.API_KEY,
        base_url=cfg.BASE_URL
    )

    if prompts_data is None:
        input_file = os.path.join(run_dir, 'prompts.json')
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)

    metadata = prompts_data.get("metadata", {})
    plan_token = metadata.get("stage2_plan_token", "__PLAN_JSON__")

    get_persona = prompts_data.get("get_persona")
    build_prompts_fn = prompts_data.get("build_prompts")
    lazy_mode = get_persona is not None and build_prompts_fn is not None

    if lazy_mode:
        row_ids = filter_row_ids(prompts_data["row_ids"], cfg)
    else:
        samples = prompts_data["samples"]
        if cfg.SELECTED_ROW_IDS:
            selected_set = set(cfg.SELECTED_ROW_IDS)
            samples = [s for s in samples if s['row_id'] in selected_set]
        if cfg.USE_RANDOM_SAMPLING and len(samples) > cfg.RANDOM_SAMPLE_SIZE:
            import random
            random.seed(cfg.RANDOM_SAMPLE_SEED)
            samples = random.sample(samples, cfg.RANDOM_SAMPLE_SIZE)
        row_ids = [s['row_id'] for s in samples]

    if not row_ids:
        print("No samples to process after filtering!")
        return

    completed_tasks = load_completed_tasks(temp_file)
    if completed_tasks:
        print(f"Found {len(completed_tasks)} completed tasks (stage2 done), will skip them.")

    task_types = ["affect_science", "affect_action", "ccb", "mist"]
    total_count = 0
    for rid in row_ids:
        for rep in range(cfg.REPEAT_TIMES):
            for task_type in task_types:
                if (rid, rep + 1, task_type) not in completed_tasks:
                    total_count += 1

    print(f"Total samples after filtering: {len(row_ids)}")
    print(f"Remaining tasks (each includes stage1+stage2): {total_count}")
    print(f"Max Workers: {cfg.MAX_WORKERS}")
    if lazy_mode:
        print(f"Sample buffer: {SAMPLE_BUFFER_SIZE} (lazy mode)")

    if total_count == 0:
        print("All tasks already completed!")
    else:
        print("\nStarting execution (Ctrl+C to stop, progress is saved)...")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            with tqdm(total=total_count, unit="task") as pbar:

                for batch_start in range(0, len(row_ids), SAMPLE_BUFFER_SIZE):
                    batch_rids = row_ids[batch_start:batch_start + SAMPLE_BUFFER_SIZE]

                    batch_futures = []
                    for rid in batch_rids:
                        if lazy_mode:
                            persona_text = get_persona(rid)
                            full_prompts = build_prompts_fn(rid, persona_text)
                        else:
                            full_prompts = next(s for s in samples if s['row_id'] == rid)

                        task_key_map = {
                            "affect_science": "task_1",
                            "affect_action": "task_2",
                            "ccb": "task_3",
                            "mist": "task_4",
                        }

                        for rep in range(cfg.REPEAT_TIMES):
                            base_seed = cfg.LLM_RANDOM_SEED * 1000000 + rid * 1000 + rep * 10
                            for i, task_type in enumerate(task_types):
                                if (rid, rep + 1, task_type) in completed_tasks:
                                    continue
                                seed_base = base_seed + i
                                task_obj = full_prompts[task_key_map[task_type]]
                                task_args = (rid, rep, task_type, task_obj, seed_base, plan_token)
                                f = executor.submit(process_one_task_two_stage,
                                                    task_args, temp_file, client, cfg)
                                batch_futures.append(f)

                    for f in concurrent.futures.as_completed(batch_futures):
                        try:
                            f.result()
                        except Exception as e:
                            print(f"\nUnexpected error: {e}")
                        pbar.update(1)

        duration = time.time() - start_time
        print(f"\nExecution finished in {duration:.2f} seconds.")

    print("Reconstructing final output...")
    final_list = reconstruct_from_jsonl(temp_file)

    output_json = {
        "metadata": {
            "model": cfg.LLM_MODEL,
            "method": metadata.get("technique", "scenario_two_stage"),
            "repeat_times": cfg.REPEAT_TIMES,
            "llm_random_seed": cfg.LLM_RANDOM_SEED,
            "selected_row_ids": cfg.SELECTED_ROW_IDS if cfg.SELECTED_ROW_IDS else "all",
            "random_sampling": {
                "enabled": cfg.USE_RANDOM_SAMPLING,
                "size": cfg.RANDOM_SAMPLE_SIZE if cfg.USE_RANDOM_SAMPLING else None,
                "seed": cfg.RANDOM_SAMPLE_SEED if cfg.USE_RANDOM_SAMPLING else None
            },
            "processed_at": datetime.now().isoformat(),
            "total_samples": len(final_list),
            "stage2_plan_token": plan_token,
            "max_tokens": {
                "stage1": cfg.MAX_TOKENS_STAGE1,
                "stage2": cfg.MAX_TOKENS_STAGE2
            }
        },
        "results": final_list
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(final_list)} complete samples to {output_file}")
    print(f"Temp file kept at: {temp_file}")
