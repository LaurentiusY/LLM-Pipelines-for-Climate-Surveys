import os
import json
from shared.io_utils import read_csv_robust
from shared.profile import generate_profile
from persona.prompts_vbn import (
    SYSTEM_PROMPT_STAGE1, SYSTEM_PROMPT_STAGE2,
    build_user_prompt_stage1, build_user_prompt_stage2,
)
from persona.questions import (
    QUESTION_1, REQUIRED_FORMAT_1,
    QUESTION_2, REQUIRED_FORMAT_2,
    QUESTION_3, REQUIRED_FORMAT_3,
    QUESTION_4, REQUIRED_FORMAT_4,
)

# =========================================================================
# 【配置区】
# =========================================================================
TARGET_CONDITION = 'All'
INPUT_FILE = 'data/Showdown_short.csv'
PROFILE_MODE = 'both'
TECHNIQUE = 'vbn_two_stage_v1'

TASK_TYPE_MAP = {
    "task_1": "affect_science",
    "task_2": "affect_action",
    "task_3": "ccb",
    "task_4": "mist",
}


# =========================================================================
# Prompt generation per sample (two-stage, VBN)
# =========================================================================
def generate_prompts_for_sample(row_id, persona_text):
    def mk_task(task_key, task_type, description, question_text, required_format):
        return {
            "description": description,
            "stage1": {
                "system_prompt": SYSTEM_PROMPT_STAGE1,
                "user_prompt": build_user_prompt_stage1(task_type, persona_text, question_text, required_format),
                "output_format_note": 'JSON: {"evidence":[...],"VBN":{"AC":"...","AR":"...","PN":"..."},"synthesis":"..."}'
            },
            "stage2": {
                "system_prompt": SYSTEM_PROMPT_STAGE2,
                "user_prompt": build_user_prompt_stage2(task_type, persona_text, question_text),
                "plan_placeholder_token": "__PLAN_JSON__"
            }
        }

    return {
        "row_id": row_id,
        "persona": persona_text,
        "task_1": mk_task("task_1", "affect_science", "Affect Response to Science-Denial Tweets", QUESTION_1, REQUIRED_FORMAT_1),
        "task_2": mk_task("task_2", "affect_action", "Affect Response to Action-Opposition Tweets", QUESTION_2, REQUIRED_FORMAT_2),
        "task_3": mk_task("task_3", "ccb", "Climate Change Beliefs", QUESTION_3, REQUIRED_FORMAT_3),
        "task_4": mk_task("task_4", "mist", "Truth Discernment (MIST)", QUESTION_4, REQUIRED_FORMAT_4),
    }


# =========================================================================
# Main
# =========================================================================
def main(run_dir):
    print("=" * 60)
    print("Prompt Generation Script (VBN 2-Stage)")
    print("=" * 60)

    print(f"\n[1] Reading data from: {INPUT_FILE}")
    df_all = read_csv_robust(INPUT_FILE)
    df_all.insert(0, 'row_id', range(1, len(df_all) + 1))

    print(f"[2] Filtering for Inoculation_Group = '{TARGET_CONDITION}'")
    if TARGET_CONDITION == 'All':
        df_filtered = df_all.copy()
    else:
        if 'Inoculation_Group' not in df_all.columns:
            raise ValueError("Column 'Inoculation_Group' not found!")
        df_filtered = df_all[df_all['Inoculation_Group'] == TARGET_CONDITION].copy()
    print(f"    Found {len(df_filtered)} rows")

    if len(df_filtered) == 0:
        print("No data found. Exiting.")
        return None

    print(f"[3] Profile Mode set to: '{PROFILE_MODE}'")

    df_indexed = df_filtered.set_index('row_id')
    profile_mode = PROFILE_MODE

    def get_persona(row_id):
        row = df_indexed.loc[row_id]
        return generate_profile(row, mode=profile_mode)

    row_ids = df_filtered['row_id'].tolist()

    output_data = {
        "metadata": {
            "source_file": INPUT_FILE,
            "condition": TARGET_CONDITION,
            "total_samples": len(row_ids),
            "profile_mode": PROFILE_MODE,
            "technique": TECHNIQUE,
            "stage2_plan_token": "__PLAN_JSON__"
        },
        "row_ids": row_ids,
        "get_persona": get_persona,
        "build_prompts": generate_prompts_for_sample,
    }

    print(f"    Prepared {len(row_ids)} samples (lazy mode)")
    print("\n[DONE] Persona generation complete!")
    return output_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m persona.run_vbn <run_dir>")
        sys.exit(1)
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1])
