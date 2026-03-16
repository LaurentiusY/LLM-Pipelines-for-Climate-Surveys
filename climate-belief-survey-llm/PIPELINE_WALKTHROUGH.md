# Climate Belief Survey LLM — Complete Guide

> A step-by-step reference covering every detail of the pipeline: what each file does,
> what data flows in and out of each stage, exactly what is sent to the language model,
> and how to interpret every output metric.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [The Dataset](#4-the-dataset)
5. [Quick Start](#5-quick-start)
6. [Running via the Interactive Terminal (app.py)](#6-running-via-the-interactive-terminal-apppy)
7. [Running via the Command Line (run_pipeline.py)](#7-running-via-the-command-line-run_pipelinepy)
8. [Stage 1 — Persona & Prompt Generation](#8-stage-1--persona--prompt-generation)
9. [Stage 2 — LLM Calling](#9-stage-2--llm-calling)
10. [Stage 3 — Evaluation](#10-stage-3--evaluation)
11. [Output Files](#11-output-files)
12. [Evaluation Metrics Explained](#12-evaluation-metrics-explained)
13. [CoT vs. VBN — Choosing a Method](#13-cot-vs-vbn--choosing-a-method)
14. [Configuration Reference](#14-configuration-reference)
15. [Resuming Interrupted Runs](#15-resuming-interrupted-runs)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What This Project Does

This project investigates whether a large language model (LLM) can replicate the
climate-related attitudes of real survey respondents when given a detailed description
of each respondent's demographic and psychological profile.

The pipeline works as follows:

1. For each real respondent in a climate belief survey dataset, it builds a
   **natural-language persona description** from their survey profile (age, gender,
   country, political ideology, values, etc.).
2. It sends this persona to an LLM and asks it to answer four survey tasks
   **as if it were that person**.
3. It compares the LLM's answers to the respondent's actual answers using a rich
   set of statistical metrics.

The four survey tasks are:

| Task | Description | Scale |
|------|-------------|-------|
| **Belief** | Rate agreement with 4 statements about climate change (e.g., "Climate change is happening") | 0–100 slider |
| **Policy** | Rate support for 9 climate policy items | 0–100 slider |
| **Share** | Would this person share a climate video on social media? | Yes / No |
| **Trees (WEPT)** | How many pages of a pro-environmental newsletter would this person read? | 0–8 pages |

---

## 2. Prerequisites

### 2.1 Python

Python **3.10 or higher** is required (the project uses `list[str]` and `X | Y`
type hints that require Python 3.10+).

Check your version:

```bash
python3 --version
```

### 2.2 Dependencies

All dependencies are listed in `requirements.txt`:

```
pandas
numpy
scikit-learn
openai
tqdm
python-dotenv
openpyxl
scipy
```

Install them inside a virtual environment (strongly recommended):

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

> **Note on scipy:** `scipy` is required for Pearson and Spearman correlation
> computations. If it is not listed in your local `requirements.txt`, install it
> manually: `pip install scipy`.

### 2.3 OpenAI API Key

You need an API key for an OpenAI-compatible endpoint. Set it in one of two ways:

**Option A — `.env` file (recommended):**

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=sk-...your-key-here...
```

The project loads this automatically via `python-dotenv`.

**Option B — Shell environment variable:**

```bash
export OPENAI_API_KEY=sk-...your-key-here...
```

If neither is set, `app.py` will prompt you to enter the key interactively at runtime.

---

## 3. Project Structure

```
climate-belife-survey-llm/
│
├── app.py                        # Interactive terminal UI (recommended entry point)
├── run_pipeline.py               # Command-line entry point (for scripting / test mode)
├── config.py                     # Default configuration values (used by run_pipeline.py)
├── requirements.txt
├── .env                          # Your API key — create this file manually
│
├── data/
│   └── climate_belief_data_notimers.csv   # Survey dataset (~800 respondents)
│
├── persona/
│   ├── run_cot.py                # Stage 1: persona + prompt generation (CoT method)
│   ├── run_vbn.py                # Stage 1: persona + prompt generation (VBN method)
│   └── prompts/                  # Prompt template files used by both persona modules
│
├── call/
│   └── llm_runner.py             # Stage 2: calls the LLM API, saves results
│
├── eval/
│   ├── run_cot.py                # Stage 3: full evaluation for CoT runs
│   ├── run_vbn.py                # Stage 3: full evaluation for VBN runs
│   ├── metrics.py                # All metric computation functions
│   ├── data_loading.py           # Load and parse results.json + original CSV
│   ├── group_analysis.py         # Per-country group-level analysis
│   ├── excel_output.py           # Write all results to Excel files
│   └── utils.py                  # Scale discretisation helpers
│
└── runs/
    └── run_YYYYMMDD_HHMMSS_cot/  # Each run gets its own timestamped directory
        ├── results.json
        ├── temp_results.jsonl
        ├── evaluation.xlsx
        ├── evaluation_raw.xlsx
        ├── evaluation_plans.xlsx
        └── evaluation_by_group.xlsx
```

---

## 4. The Dataset

**File:** `data/climate_belief_data_notimers.csv`

This is a real multi-country climate belief survey dataset. Each row represents one
survey respondent. The pipeline defaults to the **Control group** (respondents who
received no experimental treatment), so that baseline attitudes are measured without
intervention effects.

### Key columns used to build each persona

| Column | Description | Profile category |
|--------|-------------|-----------------|
| `country` | Country of residence | Demographics |
| `age` | Age in years | Demographics |
| `gender` | 1=Male, 2=Female, 3=Other | Demographics |
| `education` | Highest educational level | Demographics |
| `income` | Subjective income level | Demographics |
| `political_orientation` | 1 (very left) – 10 (very right) | Demographics |
| `trust_scientists` | Trust in scientists, 1–5 | Psychology |
| `env_identity` | Environmental self-identity score | Psychology |
| `SDO` | Social Dominance Orientation | Psychology |
| `NEP_*` | New Ecological Paradigm subscales | Psychology |

### Key columns used only in evaluation (ground truth)

| Column | Task |
|--------|------|
| `Belief.in.CC_1`, `_2`, `_4`, `_5` | Belief task Q1–Q4 |
| `CC_policy_2`, `_7`, ... (9 total) | Policy task Q1–Q9 |
| `Share` | Share task (binary) |
| `WEPT1confirm` … `WEPT8confirm` | Trees task Q2–Q9 |

---

## 5. Quick Start

```bash
# 1. Navigate to the project directory
cd climate-belife-survey-llm

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Create your .env file with your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Run a quick test (10 random samples, CoT method)
python run_pipeline.py cot --test 10

# 6. OR launch the interactive terminal for full control
python app.py
```

---

## 6. Running via the Interactive Terminal (app.py)

`app.py` is the recommended way to run the pipeline. Launch it with:

```bash
python app.py
```

You will see the main menu:

```
============================================================
  Climate Belief Survey - LLM Pipeline
============================================================

  1. Run Experiment
  2. View History
  3. Exit

  Select [1/2/3]:
```

### Menu Option 1 — Run Experiment

Selecting `1` starts an interactive configuration wizard.
**Press Enter at any prompt to accept the default value shown in `[brackets]`.**

```
  Press Enter to accept the [default value] shown in brackets.

  Method  (cot=Chain-of-Thought / vbn=Value-Belief-Norm) (cot/vbn) [cot]:
```

#### LLM Settings

| Prompt | Default | Notes |
|--------|---------|-------|
| API Key | Read from `OPENAI_API_KEY` env | Required; shown masked if found in env |
| Base URL | `https://api.openai.com/v1` | Change for proxy or custom endpoints |
| Model | `gpt-4o-mini` | Any OpenAI-compatible model name |
| Max tokens Stage 1 | `400` (CoT) / `500` (VBN) | Controls length of the reasoning plan |
| Max tokens Stage 2 | `120` | Controls length of the final JSON answer |
| Temperature | `0.0` | 0 = fully deterministic output |

#### Sampling Settings

| Prompt | Default | Notes |
|--------|---------|-------|
| Random sampling? | `N` (use full dataset) | Enter `y` to draw a random subset |
| Sample size | `100` | Only asked if random sampling is `y` |
| Sample seed | `42` | Ensures identical samples across runs |

#### Execution Settings

| Prompt | Default | Notes |
|--------|---------|-------|
| Concurrency workers | `6` | Number of parallel API call threads |
| LLM random seed | `42` | Seed passed directly to the model API |

#### Persona Settings

| Prompt | Default | Options |
|--------|---------|---------|
| Profile mode | `both` | `both` = demographics + psychology; `demographics` = demographics only; `psychology` = psychology only |
| Target condition | `Both` | `Both` = all respondents; `Control` = control group only; `Treatment` = treatment group only |

After you confirm, the pipeline runs and displays live progress:

```
  Run directory: runs/run_20260315_223516_cot

  Step 1/3 - Generating prompts...
  Step 2/3 - Calling LLM... 47/120 tasks (39%)
  Step 3/3 - Evaluating results...

  Pipeline complete!

  ── Continuous Metrics (MAE / Mean Error) ─────────────────
  Belief          MAE=12.43  Mean_Error=-3.21  N=120
  Policy          MAE=11.87  Mean_Error=-1.98  N=120

  ── Correlation (Pearson r / Spearman r) ──────────────────
  Belief          pearson_r=0.3412  spearman_r=0.3198  N=120
  Policy          pearson_r=0.2876  spearman_r=0.2954  N=120

  ── Share (Social Media Sharing) ──────────────────────────
  Share           Accuracy=0.6250  F1=0.5943  N=120

  ── Trees / WEPT (Pro-environmental Effort) ───────────────
  Trees (cont.)   MAE=2.31  Mean_Error=-0.87  N=120
  Trees (binary)  Accuracy=0.6667  F1=0.6124  N=120

  Output Files:
    - runs/run_20260315_223516_cot/evaluation.xlsx
    - runs/run_20260315_223516_cot/evaluation_raw.xlsx
    - runs/run_20260315_223516_cot/evaluation_plans.xlsx
    - runs/run_20260315_223516_cot/evaluation_by_group.xlsx
```

### Menu Option 2 — View History

Displays a numbered list of all past runs, sorted newest first. Select a run number
to re-display its result summary.

---

## 7. Running via the Command Line (run_pipeline.py)

For scripting, batch jobs, or quick smoke tests, use `run_pipeline.py` directly.
All configuration is read from `config.py`.

```bash
# Full run — CoT method (all respondents in the configured condition)
python run_pipeline.py cot

# Full run — VBN method
python run_pipeline.py vbn

# Test mode: 10 randomly selected rows (seed=42, reproducible)
python run_pipeline.py cot --test 10

# Test mode: custom sample count
python run_pipeline.py vbn --test 50
```

Edit `config.py` to change the model, API key, worker count, or any other setting
before running.

---

## 8. Stage 1 — Persona & Prompt Generation

**Files:** `persona/run_cot.py` and `persona/run_vbn.py`

### What it does

Stage 1 reads the survey CSV and prepares a **lazy container** — it does not generate
any text upfront. Instead it stores callable functions that produce the persona
description and task prompts on demand as each respondent is processed. This avoids
holding large amounts of text in memory simultaneously.

### Input

- `data/climate_belief_data_notimers.csv`
- `TARGET_CONDITION` — which experimental group to include (`'Control'`, `'Treatment'`, or `'Both'`)
- `PROFILE_MODE` — which profile variables to include (`'demographics'`, `'psychology'`, or `'both'`)

### Output returned to the pipeline

A Python dictionary with the following keys:

```python
{
    "metadata": {
        "source_file":    "data/climate_belief_data_notimers.csv",
        "condition":      "Control",
        "total_samples":  812,
        "profile_mode":   "both",
        "technique":      "scenario_two_stage_v1",
        "task4_mode":     "one_shot"
    },
    "row_ids":      [1, 2, 3, ...],              # list of row IDs to process
    "get_persona":  callable(row_id) -> str,     # generates persona text on demand
    "build_prompts": callable(row_id, persona_text) -> dict  # builds all 4 task prompts
}
```

### The Persona Description

`get_persona(row_id)` looks up the respondent's row in the CSV and assembles a
natural-language profile. Example output with `PROFILE_MODE = 'both'`:

```
You are roleplaying as a real survey respondent with the following profile:

Demographics:
- Age: 34
- Gender: Female
- Country: United Kingdom
- Education: Bachelor's degree
- Subjective income: Middle income
- Political orientation: 4 out of 10 (1=very left, 10=very right)

Psychological profile:
- Trust in scientists: 4 out of 5
- Environmental self-identity: High
- Social Dominance Orientation: Low
- New Ecological Paradigm (NEP): High pro-ecological worldview
```

With `PROFILE_MODE = 'demographics'`, the psychological profile section is omitted.
With `PROFILE_MODE = 'psychology'`, the demographics section is omitted.

### The Task Prompts

`build_prompts(row_id, persona_text)` returns a dictionary with four task prompt
pairs. Each task has a **Stage 1 system+user prompt** (asks the LLM to reason first)
and a **Stage 2 user prompt** (asks for the final JSON answer based on the reasoning).

---

#### Task 1 — Belief (Climate Attitude Assessment)

**Stage 1 System Message:**
```
You are simulating a survey respondent. Your task is to reason carefully about
how this specific person would answer survey questions, based on their profile.
Stay in character throughout.
```

**Stage 1 User Message (CoT):**
```
Persona:
[persona text inserted here]

Task: Climate Attitude Assessment

Please reason step by step about how this person would rate their agreement
(on a scale of 0 to 100) with each of the following statements about climate change.
Consider their country, age, political views, values, and worldview.

Statements:
Q1: Climate change is happening.
Q2: Climate change is mostly caused by human activities.
Q3: Climate change will harm people in [country] a great deal.
Q4: I am personally worried about climate change.

Write a brief reasoning plan — do not produce the final numbers yet.
```

**Stage 1 User Message (VBN):**
```
Persona:
[persona text inserted here]

Task: Climate Attitude Assessment

Use the Value-Belief-Norm (VBN) framework to analyse this person's likely responses:

1. Values (AC — Awareness of Consequences / Altruistic/Biospheric values):
   What values does this person hold regarding the environment and society?

2. Beliefs (AR — Attribution of Responsibility):
   Does this person believe humans are responsible for climate change?
   How strong is this belief given their profile?

3. Personal Norms (PN):
   What moral obligations would this person feel about acknowledging climate change?

Synthesise these three components into a brief prediction of their responses.
```

**Stage 2 User Message (same for both CoT and VBN):**
```
Based on your reasoning above, provide your final answers.

Respond ONLY with a valid JSON object in this exact format — no extra text:
{"Q1": <integer 0-100>, "Q2": <integer 0-100>, "Q3": <integer 0-100>, "Q4": <integer 0-100>}
```

---

#### Task 2 — Policy (Policy Support Assessment)

Nine climate policy items are presented (e.g., carbon pricing, renewable energy
subsidies, international climate agreements). The format is identical to Task 1:
Stage 1 asks for reasoning, Stage 2 asks for a JSON object with Q1–Q9 on a 0–100 scale.

---

#### Task 3 — Share (Social Media Sharing Willingness)

**Stage 1 User Message:**
```
Persona:
[persona text inserted here]

Task: Social Media Sharing

Imagine this person has just watched a short online video about climate change.
Reason about whether they would be willing to share this video on their social
media accounts. Consider their values, information-sharing habits, and social identity.

Write a brief reasoning plan.
```

**Stage 2 User Message:**
```
Based on your reasoning above, provide your final answer.

Respond ONLY with a valid JSON object:
{"answer": "yes"} or {"answer": "no"}
```

---

#### Task 4 — Trees / WEPT (Pro-environmental Effort)

This task uses a **one-shot** format — the full instruction and persona appear in a
single prompt without a separate reasoning stage.

**User Message:**
```
Persona:
[persona text inserted here]

Task: Pro-environmental Effort (WEPT)

This person is offered the opportunity to read pages of a newsletter about
environmental conservation. They can choose to read between 0 and 8 pages.
Each page they read counts as a small pro-environmental action.

Based on this person's profile, how many pages (0 to 8) would they choose to read?

Respond ONLY with a valid JSON object:
{"pages": <integer 0-8>}
```

---

## 9. Stage 2 — LLM Calling

**File:** `call/llm_runner.py`

### What it does

For every respondent × every task × every repeat, the runner:

1. Calls the LLM **twice**: Stage 1 (reasoning plan) then Stage 2 (final JSON answer)
2. Parses and validates the JSON response from Stage 2
3. Saves each completed task to `temp_results.jsonl` in real time (enables resumption)
4. When all tasks finish, reconstructs the flat JSONL into the structured `results.json`

### Concurrency

Tasks run in parallel via `concurrent.futures.ThreadPoolExecutor`. Each "task" is one
respondent × one survey question block (Belief, Policy, Share, or Trees).
Workers default to 6 threads.

### Retry logic

If an API call fails or Stage 2 returns unparseable JSON, the runner retries up to
**6 times** with exponential backoff: 1 s → 2 s → 4 s → 8 s → 16 s → 32 s.
After 6 failures the task is skipped and a warning is printed. All successfully
completed tasks are preserved.

### What is written to temp_results.jsonl

Each line is a JSON record for one completed task:

```json
{
  "row_id": 42,
  "repeat": 1,
  "task_key": "belief",
  "stage1_content": "Q1: This person strongly believes climate change is happening because her high trust in scientists and pro-environmental identity lead her to accept the scientific consensus...",
  "stage2_content": "{\"Q1\": 88, \"Q2\": 75, \"Q3\": 82, \"Q4\": 79}",
  "parsed_answer": {"Q1": 88, "Q2": 75, "Q3": 82, "Q4": 79},
  "usage": {"prompt_tokens": 312, "completion_tokens": 95, "total_tokens": 407}
}
```

### What is saved to results.json

After all tasks complete, the flat JSONL is reconstructed into a structured file:

```json
{
  "metadata": {
    "model": "gpt-4o-mini",
    "method": "scenario_two_stage_v1",
    "repeat_times": 1,
    "llm_random_seed": 42,
    "selected_row_ids": "all",
    "random_sampling": {"enabled": false},
    "processed_at": "2026-03-15T22:35:16",
    "total_samples": 812,
    "max_tokens": {"stage1": 400, "stage2": 120}
  },
  "results": [
    {
      "row_id": 1,
      "repeat": 1,
      "belief":  {"Q1": 88, "Q2": 75, "Q3": 82, "Q4": 79},
      "policy":  {"Q1": 70, "Q2": 85, "Q3": 60, "Q4": 72, "Q5": 68, "Q6": 80, "Q7": 55, "Q8": 90, "Q9": 65},
      "share":   "yes",
      "trees":   6,
      "plans": {
        "belief":  "Q1: Strong belief because her high trust in scientists...",
        "policy":  "Generally supportive because she values collective action...",
        "share":   "Likely to share because she has a strong environmental identity...",
        "trees":   "Would read several pages given her high environmental engagement..."
      },
      "total_usage": {"prompt_tokens": 1240, "completion_tokens": 380, "total_tokens": 1620}
    }
  ]
}
```

---

## 10. Stage 3 — Evaluation

**Files:** `eval/run_cot.py` (or `eval/run_vbn.py`), `eval/metrics.py`, `eval/excel_output.py`

### What it does

Stage 3 loads `results.json` and the original CSV, aligns them by `row_id`,
computes all metrics, and writes four Excel output files.

### Step-by-step

1. **Load data** (`eval/data_loading.py`)
   - Reads `results.json` (LLM predictions)
   - Reads the original survey CSV (ground-truth answers)
   - Assigns sequential `row_id` values (1-indexed) to both so they can be joined

2. **Extract plans** — Stage 1 reasoning texts are extracted into a separate
   DataFrame for inspection (`evaluation_plans.xlsx`)

3. **Extract raw data** — The nested JSON predictions are expanded into a flat
   long-format DataFrame: one row per respondent × per question × per repeat

4. **Aggregate** — If `REPEAT_TIMES > 1`, repeats are averaged per respondent per
   question to produce a single predicted value per cell

5. **Compute metrics** — Each task type uses its own metric function (see Section 12)

6. **Write Excel files** — All results are written to structured Excel workbooks

### Ground-truth column mapping

| LLM output | CSV column |
|------------|------------|
| `belief Q1` | `Belief.in.CC_4` |
| `belief Q2` | `Belief.in.CC_1` |
| `belief Q3` | `Belief.in.CC_2` |
| `belief Q4` | `Belief.in.CC_5` |
| `policy Q1` | `CC_policy_2` |
| `policy Q2` | `CC_policy_7` |
| `policy Q3–Q9` | (further policy columns) |
| `share` | `Share` (0/1 encoded) |
| `trees Q2–Q9` | `WEPT1confirm` … `WEPT8confirm` |

---

## 11. Output Files

Every run creates a timestamped directory: `runs/run_YYYYMMDD_HHMMSS_{method}/`

| File | Description |
|------|-------------|
| `results.json` | Structured LLM predictions for all processed respondents |
| `temp_results.jsonl` | Raw per-task JSONL (kept for debugging and resumption) |
| `evaluation.xlsx` | Main evaluation workbook (multiple sheets — see below) |
| `evaluation_raw.xlsx` | Per-respondent per-question long-format table |
| `evaluation_plans.xlsx` | Stage 1 reasoning plans for each respondent |
| `evaluation_by_group.xlsx` | Metrics broken down by country |

### Sheets inside evaluation.xlsx

| Sheet | Description |
|-------|-------------|
| `Summary` | One-row-per-metric summary (used by terminal display) |
| `Belief_Detail` | Per-question MAE / Mean Error for Belief task |
| `Belief_Metrics` | Overall Belief metrics across three scales (original / decimal / discrete) |
| `Belief_Correlation` | Pearson r, Spearman r, p-values per question for Belief |
| `Policy_Detail` | Per-question MAE / Mean Error for Policy task |
| `Policy_Metrics` | Overall Policy metrics across three scales |
| `Policy_Correlation` | Pearson r, Spearman r, p-values per question for Policy |
| `Share_Metrics` | Accuracy, Precision, Recall, F1, N for Share task |
| `Share_CM` | 2×2 confusion matrix for Share task |
| `Trees_Detail` | Per-respondent Trees predictions and errors |
| `Trees_Continuous` | Continuous-scale MAE / Mean Error for Trees task |
| `Trees_Binary_CM` | 2×2 confusion matrix (read anything vs. read nothing) |
| `Trees_Ternary_CM` | 3×3 confusion matrix (none / some / full) |
| `Trees_Distribution` | Histogram of true vs. predicted Trees values (0–8 pages) |

---

## 12. Evaluation Metrics Explained

### 12.1 MAE and Mean Error (Belief, Policy, Trees)

These metrics apply to all tasks where predictions are continuous numbers.

**Mean Absolute Error (MAE):**

```
MAE = (1/N) × Σ |predicted_i − true_i|
```

Lower is better. Measures the average magnitude of prediction error regardless of direction.

**Mean Error (Bias):**

```
Mean_Error = (1/N) × Σ (predicted_i − true_i)
```

- Positive Mean Error → the LLM systematically over-predicts (guesses too high)
- Negative Mean Error → the LLM systematically under-predicts (guesses too low)
- A value near zero does not mean predictions are accurate — it may just mean
  over- and under-predictions cancel each other out. Always read MAE alongside Mean Error.

**Three scales are reported for each continuous task:**

| Scale | Description |
|-------|-------------|
| `original` | Raw values as answered and predicted (0–100 or 0–8) |
| `decimal` | Rounded to nearest 5 (e.g., 73 → 75) — reduces noise from fine-grained slider responses |
| `discrete` | Binned into 5 equal categories (e.g., 0–20, 21–40, 41–60, 61–80, 81–100) |

### 12.2 Pearson and Spearman Correlations (Belief, Policy)

These measure whether the LLM correctly identifies **relative differences** between
respondents, even if the absolute values are off.

**Pearson r:** Linear correlation between LLM predictions and true values.
Ranges from −1 to +1. Assumes a linear relationship.

**Spearman r:** Rank-order correlation. More robust to outliers and non-linear
relationships. Measures whether respondents who scored higher in the survey also
tend to get higher LLM predictions.

Both are reported per question and as an OVERALL score across all questions.
P-values are stored in Excel but not shown in the terminal summary.

**Interpretation guide:**

| Correlation | Interpretation |
|-------------|----------------|
| r > 0.5 | Strong — LLM reliably ranks respondents in the right order |
| 0.3 < r < 0.5 | Moderate — LLM shows meaningful but imperfect ordering |
| 0.1 < r < 0.3 | Weak — some signal but noisy |
| r < 0.1 | Very weak or no meaningful relationship |

### 12.3 Classification Metrics (Share Task)

The Share task is binary: the respondent either shared or did not share.

| Metric | Formula |
|--------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) — of all predicted "shares", how many were correct? |
| **Recall** | TP / (TP + FN) — of all actual "shares", how many did the LLM catch? |
| **F1** | 2 × Precision × Recall / (Precision + Recall) — harmonic mean |

"Positive" = the prediction or ground truth is "yes, will share".

### 12.4 Trees Metrics (WEPT Task)

Three views of the same task, each providing different information:

**Continuous:** Raw MAE and Mean Error on the 0–8 page count. Best for
understanding the average magnitude of prediction error.

**Binary:** Did the respondent read at least one page? (0 = No, 1–8 = Yes).
Accuracy and F1 capture whether the LLM can predict engagement vs. non-engagement.

**Ternary:** Three classes:
- None = 0 pages
- Some = 1–4 pages
- Full = 5–8 pages

The 3×3 confusion matrix shows which categories are most often confused.

---

## 13. CoT vs. VBN — Choosing a Method

### Chain-of-Thought (CoT)

The LLM is instructed to reason step by step, identifying relevant factors from the
persona and thinking through what response each factor implies for each question.

- **Prompt structure:** Universal template used across all four tasks.
  The model decides freely how to structure its reasoning.
- **Best for:** Quick exploratory runs; tasks where general world knowledge is
  the primary driver.

### Value-Belief-Norm (VBN)

Based on Stern et al.'s Value-Belief-Norm theory of pro-environmental behaviour.
The LLM is required to explicitly analyse the persona through three lenses before
producing an answer:

1. **Values (AC):** What does this person fundamentally care about?
2. **Beliefs (AR):** Does this person believe humans caused the problem?
3. **Personal Norms (PN):** What moral obligation does this person feel?

- **Prompt structure:** Four task-specific templates (one per survey task),
  each structured around the AC → AR → PN components.
- **Best for:** Research hypotheses about whether VBN theory predicts survey
  responses; more structured, interpretable reasoning traces.

### Choosing in practice

Run both methods on the same random sample (use identical seeds) and compare MAE
and correlation values. If VBN produces better correlations, the VBN framework may
be capturing something meaningful about the psychological mechanisms driving responses.
If CoT is equally good, the additional structure may not add value for your data.

---

## 14. Configuration Reference

When using `run_pipeline.py`, all settings come from `config.py`:

```python
# API / Model
API_KEY           = ""                             # Leave blank to read from env
BASE_URL          = "https://api.openai.com/v1"
LLM_MODEL         = "gpt-4o-mini"
MAX_TOKENS_STAGE1 = 400    # Reasoning plan; use 500 for VBN
MAX_TOKENS_STAGE2 = 120    # Final JSON answer
TEMPERATURE       = 0.0    # 0 = deterministic

# Execution
MAX_WORKERS       = 6      # Parallel threads
REPEAT_TIMES      = 1      # How many times to repeat each sample
LLM_RANDOM_SEED   = 42     # Seed passed to the model API

# Sampling
USE_RANDOM_SAMPLING = False
RANDOM_SAMPLE_SIZE  = 100
RANDOM_SAMPLE_SEED  = 42
SELECTED_ROW_IDS    = []   # Empty = process all rows
```

---

## 15. Resuming Interrupted Runs

If a run is interrupted (API error, network drop, Ctrl+C, etc.), it can be resumed
without re-processing already-completed tasks.

**How it works:** The runner reads `temp_results.jsonl` at startup. Any task already
recorded in this file is skipped. Only the remaining incomplete tasks are sent to the API.

**To resume:** Locate the existing run directory under `runs/` and re-run the same
command pointing at it, or simply re-run `run_pipeline.py cot` (or `vbn`) — the runner
will automatically detect and use the existing `temp_results.jsonl` in the same run
directory.

---

## 16. Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError: scipy` | scipy not installed | `pip install scipy` |
| `openai.AuthenticationError` | Invalid or missing API key | Check `.env` or `OPENAI_API_KEY` env var |
| `KeyError: 'Belief.in.CC_4'` | CSV column names don't match expected names | Verify column names in your CSV match those in `eval/run_cot.py` |
| JSON parse errors in Stage 2 | Model returned malformed JSON | Increase `MAX_TOKENS_STAGE2` (e.g., to 150); lower temperature to 0 |
| Very high MAE on all tasks | Too few reasoning tokens | Increase `MAX_TOKENS_STAGE1` (e.g., to 600) |
| Run is slow | Too few workers | Increase `Concurrency workers` to 10–16; check API rate limits |
| `PerformanceWarning: DataFrame is highly fragmented` | Pandas internals (cosmetic) | Safe to ignore — does not affect any results |
| Excel file missing a sheet | Task had zero valid predictions | Check `evaluation_raw.xlsx` for rows with missing predictions; look for Stage 2 parse errors in terminal output |
