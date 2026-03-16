# Climate Misinfo Survey LLM — Complete Guide

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

This project investigates whether a large language model (LLM) can replicate how
real people *respond to climate misinformation* when given a detailed description of
each person's demographic and psychological profile.

The underlying human study (Spampatti et al.) exposed participants from 12 countries
to climate misinformation tweets and tested whether **psychological inoculation**
(brief pre-emptive warnings about manipulation tactics) could protect people against
misinformation. Participants were randomly assigned to one of three conditions:

| Condition | Description |
|-----------|-------------|
| **Control** | No inoculation — saw tweets without prior warning |
| **Cognitive** | Received a fact-based inoculation about the tweet's misleading claims |
| **Socioaffective** | Received an emotion-based inoculation about the tweet's manipulation tactics |

This LLM pipeline builds a persona from each participant's profile and asks the LLM
to simulate that participant's responses to the same four survey tasks:

| Task | Description | Scale |
|------|-------------|-------|
| **Affect Science** | Rate emotional response (disgust/anger) to 10 science-denial tweets | 0–100 slider |
| **Affect Action** | Rate emotional response to 10 action-opposition tweets | 0–100 slider |
| **CCB** | Rate agreement with 3 climate change belief statements | 1–5 scale |
| **MIST** | Identify which of 20 news headlines are real vs. fake | Binary (real/fake) |

The LLM's predictions are then compared to the real participant responses using a
comprehensive set of statistical metrics.

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

> **Note on scipy:** `scipy` is required for Pearson/Spearman correlation and
> Signal Detection Theory calculations. If it is not listed in your local
> `requirements.txt`, install it with: `pip install scipy`.

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
climate-misinfo-survey-llm/
│
├── app.py                        # Interactive terminal UI (recommended entry point)
├── run_pipeline.py               # Command-line entry point (for scripting / test mode)
├── config.py                     # Default configuration values (used by run_pipeline.py)
├── requirements.txt
├── .env                          # Your API key — create this file manually
│
├── data/
│   └── Showdown_short.csv        # Pre-processed survey dataset (5,948 rows)
│
├── human-survey-data/            # Original Spampatti et al. study materials
│   ├── Data/
│   │   ├── RAW/                  # Raw per-country Qualtrics exports
│   │   └── Preprocessed/        # Cleaned and merged datasets
│   ├── Materials/
│   │   ├── Truth_DiscernmentRaw.txt   # MIST item texts (real and fake headlines)
│   │   ├── Tweet_Images/         # Photographs of the misinformation tweets
│   │   └── Main_Survey_Spampatti_et_al.qsf   # Original Qualtrics survey
│   └── Stimuli_Validation/       # Corpus and validation files for tweet stimuli
│
├── persona/
│   ├── run_cot.py                # Stage 1: persona + prompt generation (CoT method)
│   └── run_vbn.py                # Stage 1: persona + prompt generation (VBN method)
│
├── call/
│   └── llm_runner.py             # Stage 2: calls the LLM API, saves results
│
├── eval/
│   ├── run_cot.py                # Stage 3: full evaluation for CoT runs
│   ├── run_vbn.py                # Stage 3: full evaluation for VBN runs
│   ├── metrics.py                # All metric computation functions
│   ├── data_loading.py           # Load and parse results.json + original CSV
│   ├── group_analysis.py         # Per-country and per-condition group analysis
│   ├── excel_output.py           # Write all results to Excel files
│   └── utils.py                  # Scale helpers
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

**File:** `data/Showdown_short.csv`

This pre-processed dataset contains **5,948 rows**, one per participant, drawn from
the Spampatti et al. inoculation experiment conducted across 12 countries.

### Countries included

Australia, Canada, India, Ireland, New Zealand, Nigeria, Pakistan, Philippines,
Singapore, South Africa, United Kingdom, USA.

### Key columns used to build each persona

| Column | Description | Type |
|--------|-------------|------|
| `Country` | Country of residence | String (e.g., `"UK"`, `"USA"`) |
| `Age` | Age in years | Integer |
| `Gender` | 1=Male, 2=Female, 3=Other | Integer |
| `Education` | Highest educational level | Integer (coded) |
| `Political_ideology` | 1 (very left) – 10 (very right) | Integer |
| `CRT` | Cognitive Reflection Test score (analytical thinking) | Integer |
| `Threat_perception` | Perceived personal threat from climate change | Continuous |
| Baseline affect columns | Pre-experiment emotional responses (used as psychological context) | Continuous (0–100) |

### Key columns used only in evaluation (ground truth)

| Columns | Task |
|---------|------|
| `Sci_1_Affect_1` … `Sci_10_Affect_1` | Affect Science (10 tweets) |
| `Act_1_Affect_1` … `Act_10_Affect_1` | Affect Action (10 tweets) |
| `CCB_real`, `CCB_cause`, `CCB_cons` | CCB (3 belief items) |
| `T_Support_1–5`, `T_Deny_1–5`, `F_Support_1–5`, `F_Deny_1–5` | MIST (20 headlines) |

### Inoculation groups

The `Inoculation_Group` column identifies which condition each participant was
assigned to. Use the `Target condition` setting to filter by group:

| Value | Meaning |
|-------|---------|
| `All` | Include all participants regardless of condition (default) |
| `Control` | Only participants who received no inoculation |
| `Cognitive` | Only participants who received the fact-based inoculation |
| `Socioaffective` | Only participants who received the emotion-based inoculation |

---

## 5. Quick Start

```bash
# 1. Navigate to the project directory
cd climate-misinfo-survey-llm

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
  Climate Misinfo Survey - LLM Pipeline
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
| Max tokens Stage 1 | `500` | Controls length of the reasoning plan |
| Max tokens Stage 2 | `250` | Controls length of final answers (larger than belief survey because MIST has 20 items) |
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
| Target condition | `All` | `All` = all participants; `Control` = control only; `Cognitive` = cognitive inoculation only; `Socioaffective` = socioaffective inoculation only |

After you confirm, the pipeline runs and displays live progress:

```
  Run directory: runs/run_20260315_223516_cot

  Step 1/3 - Generating prompts...
  Step 2/3 - Calling LLM... 47/40 tasks (100%)
  Step 3/3 - Evaluating results...

  Pipeline complete!

  ── Continuous Metrics (MAE / Mean Error) ─────────────────
  Affect Science  MAE=41.93  Mean_Error=-39.41  N=100
  Affect Action   MAE=33.65  Mean_Error=-23.77  N=100
  CCB             MAE=0.86   Mean_Error=0.72    N=30

  ── Correlation (Pearson r / Spearman r) ──────────────────
  Affect Science  pearson_r=-0.03  spearman_r=-0.20  N=100
  Affect Action   pearson_r=-0.13  spearman_r=-0.17  N=100
  CCB             pearson_r=0.15   spearman_r=0.19   N=30

  ── MIST (Misinformation Susceptibility) ──────────────────
  MIST            Accuracy=0.5550  F1=0.5973  N=200
  MIST SDT        d'=4.21  c=0.22

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
# Full run — CoT method (all participants)
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

Stage 1 reads `data/Showdown_short.csv` and prepares a **lazy container** — it does
not generate any text upfront. Instead it stores callable functions that produce the
persona description and task prompts on demand as each participant is processed.

### Input

- `data/Showdown_short.csv`
- `TARGET_CONDITION` — which inoculation group to include (`'All'`, `'Control'`, `'Cognitive'`, or `'Socioaffective'`)
- `PROFILE_MODE` — which profile variables to include (`'demographics'`, `'psychology'`, or `'both'`)

### Output returned to the pipeline

A Python dictionary with the following keys:

```python
{
    "metadata": {
        "source_file":    "data/Showdown_short.csv",
        "condition":      "All",
        "total_samples":  5948,
        "profile_mode":   "both",
        "technique":      "scenario_two_stage_v1"
    },
    "row_ids":       [1, 2, 3, ...],             # list of row IDs to process
    "get_persona":   callable(row_id) -> str,    # generates persona text on demand
    "build_prompts": callable(row_id, persona_text) -> dict  # builds all 4 task prompts
}
```

### The Persona Description

`get_persona(row_id)` looks up the participant's row in the CSV and assembles a
natural-language profile. Example output with `PROFILE_MODE = 'both'`:

```
You are roleplaying as a real survey participant with the following profile:

Demographics:
- Age: 28
- Gender: Female
- Country: Nigeria
- Education: Bachelor's degree
- Political ideology: 3 out of 10 (1=very left, 10=very right)

Psychological profile:
- Cognitive Reflection Test (CRT) score: 2 out of 4 (higher = more analytical)
- Perceived personal threat from climate change: High
- Pre-experiment emotional response to science-denial content: 72 / 100
- Pre-experiment emotional response to action-opposition content: 65 / 100
```

With `PROFILE_MODE = 'demographics'`, only the demographics section appears.
With `PROFILE_MODE = 'psychology'`, only the psychological profile section appears.

### The Task Prompts

`build_prompts(row_id, persona_text)` returns a dictionary with four task prompt
pairs. Each task has a **Stage 1 system+user prompt** (asks the LLM to reason first)
and a **Stage 2 user prompt** (asks for the final JSON answer based on the reasoning).

---

#### Task 1 — Affect Science (Emotional Response to Science-Denial Tweets)

Participants were shown 10 tweets that deny or undermine climate science (e.g.,
tweets claiming scientists are fabricating data, or that natural cycles explain
warming). They rated their emotional response (anger/disgust) on a 0–100 slider.

**Stage 1 System Message:**
```
You are simulating a survey participant. Your task is to reason carefully about
how this specific person would emotionally respond to climate misinformation,
based on their profile. Stay in character throughout.
```

**Stage 1 User Message (CoT):**
```
Persona:
[persona text inserted here]

Task: Emotional Response to Science-Denial Content

This person is shown 10 tweets that deny or undermine climate science.
For each tweet, they rate their emotional response (anger, disgust, or upset)
on a 0–100 scale where:
  0 = No emotional reaction at all
  100 = Extremely angry / disgusted

Please reason step by step: given this person's background, political views,
threat perception, and analytical thinking, how emotionally reactive would they
be to science-denial content? Consider each tweet category and write a brief
reasoning plan. Do not produce the final numbers yet.

The 10 tweets cover these topics:
S1: [tweet text or description]
S2: [tweet text or description]
... (S1 through S10)
```

**Stage 1 User Message (VBN):**
```
Persona:
[persona text inserted here]

Task: Emotional Response to Science-Denial Content

Use the Value-Belief-Norm (VBN) framework to analyse this person's likely
emotional responses to tweets that deny climate science:

1. Values (AC — Biospheric/Altruistic values):
   How much does this person value environmental protection and societal well-being?
   Would science denial threaten something they care deeply about?

2. Beliefs (AR — Attribution of Responsibility):
   Does this person believe humans are causing climate change?
   Would they view science-denial tweets as dishonest or threatening?

3. Personal Norms (PN — Moral Obligation):
   Does this person feel a personal moral duty to defend climate science?
   Would this translate into stronger emotional reactions?

Synthesise these components into a prediction of emotional reactivity across
the 10 science-denial tweets.
```

**Stage 2 User Message (same for CoT and VBN):**
```
Based on your reasoning above, provide your final answers.

Rate this person's emotional response to each of the following tweets on a
scale of 0 to 100 (0 = no reaction, 100 = extremely upset/angry/disgusted).

[Tweet S1 text]
[Tweet S2 text]
... (all 10 tweets listed)

Respond ONLY with a valid JSON object — no extra text:
{"S1": <0-100>, "S2": <0-100>, ..., "S10": <0-100>}
```

---

#### Task 2 — Affect Action (Emotional Response to Action-Opposition Tweets)

Identical structure to Task 1, but the 10 tweets oppose climate action rather than
denying the science (e.g., tweets arguing climate policy is economically harmful, or
that individual action is pointless). Rated on the same 0–100 emotional scale.

The JSON output format is:
```json
{"A1": <0-100>, "A2": <0-100>, ..., "A10": <0-100>}
```

---

#### Task 3 — CCB (Climate Change Beliefs)

Three belief items from the Climate Change Beliefs scale, rated 1–5:

- `CCB_real`: "Climate change is real." (1=Strongly disagree, 5=Strongly agree)
- `CCB_cause`: "Climate change is caused by human activities."
- `CCB_cons`: "Climate change will have serious consequences for humanity."

**Stage 2 JSON format:**
```json
{"Q1": <1-5>, "Q2": <1-5>, "Q3": <1-5>}
```

---

#### Task 4 — MIST (Misinformation Susceptibility Test)

Participants are shown 20 news headlines and asked to identify each as "real news"
or "fake news". The 20 headlines are drawn from four categories:

| Category | Code | Description |
|----------|------|-------------|
| True-Supports | TS | Real news headline that supports the scientific consensus |
| True-Denies | TD | Real news headline that questions or complicates climate action |
| False-Supports | FS | Fake headline that appears to support climate action but is fabricated |
| False-Denies | FD | Fake headline that denies climate change (typical misinformation) |

The LLM must predict whether this participant would correctly identify each headline
as real or fake.

**Stage 2 JSON format:**
```json
{
  "TS1": "real", "TS2": "real", "TS3": "real", "TS4": "real", "TS5": "real",
  "TD1": "real", "TD2": "real", "TD3": "fake", "TD4": "real", "TD5": "real",
  "FS1": "fake", "FS2": "fake", "FS3": "real", "FS4": "fake", "FS5": "fake",
  "FD1": "fake", "FD2": "fake", "FD3": "fake", "FD4": "fake", "FD5": "fake"
}
```

Note: Max tokens Stage 2 is set to 250 for this task because the JSON response
contains 20 items.

---

## 9. Stage 2 — LLM Calling

**File:** `call/llm_runner.py`

### What it does

For every participant × every task × every repeat, the runner:

1. Calls the LLM **twice**: Stage 1 (reasoning plan) then Stage 2 (final JSON answer)
2. Parses and validates the JSON response from Stage 2
3. Saves each completed task to `temp_results.jsonl` in real time (enables resumption)
4. When all tasks finish, reconstructs the flat JSONL into the structured `results.json`

### Concurrency

Tasks run in parallel via `concurrent.futures.ThreadPoolExecutor`. Each "task" is one
participant × one survey question block (Affect Science, Affect Action, CCB, or MIST).
The number of workers defaults to 6 threads.

### Retry logic

If an API call fails or Stage 2 returns unparseable JSON, the runner retries up to
**6 times** with exponential backoff: 1 s → 2 s → 4 s → 8 s → 16 s → 32 s.
After 6 failures the task is skipped and a warning is printed.

### What is written to temp_results.jsonl

Each line is a JSON record for one completed task:

```json
{
  "row_id": 42,
  "repeat": 1,
  "task_key": "affect_science",
  "stage1_content": "This person has high threat perception and moderate CRT, so they would likely feel strongly upset by science-denial content...",
  "stage2_content": "{\"S1\": 82, \"S2\": 74, ...}",
  "parsed_answer": {"S1": 82, "S2": 74, "S3": 91, "S4": 68, "S5": 79, "S6": 85, "S7": 72, "S8": 88, "S9": 76, "S10": 83},
  "usage": {"prompt_tokens": 541, "completion_tokens": 148, "total_tokens": 689}
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
    "random_sampling": {"enabled": true, "size": 10, "seed": 42},
    "processed_at": "2026-03-15T22:35:16",
    "total_samples": 10,
    "max_tokens": {"stage1": 500, "stage2": 250}
  },
  "results": [
    {
      "row_id": 1,
      "repeat": 1,
      "affect_science": {"S1": 82, "S2": 74, "S3": 91, "S4": 68, "S5": 79, "S6": 85, "S7": 72, "S8": 88, "S9": 76, "S10": 83},
      "affect_action":  {"A1": 71, "A2": 65, "A3": 80, "A4": 58, "A5": 74, "A6": 69, "A7": 62, "A8": 77, "A9": 66, "A10": 72},
      "ccb":            {"Q1": 5, "Q2": 4, "Q3": 5},
      "mist":           {"TS1": "real", "TS2": "real", ..., "FD5": "fake"},
      "plans": {
        "affect_science": "High threat perception and strong climate beliefs mean...",
        "affect_action":  "Given moderate political ideology and high concern...",
        "ccb":            "Strongly believes in climate change given...",
        "mist":           "Moderate CRT suggests some ability to detect manipulation but..."
      },
      "total_usage": {"prompt_tokens": 2210, "completion_tokens": 590, "total_tokens": 2800}
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
   - Reads `data/Showdown_short.csv` (ground-truth answers)
   - Assigns sequential `row_id` values (1-indexed) so both can be joined

2. **Extract plans** — Stage 1 reasoning texts are saved separately to
   `evaluation_plans.xlsx` for human inspection

3. **Extract raw data** — The nested JSON predictions are expanded into a flat
   long-format DataFrame: one row per participant × per question × per repeat

4. **Aggregate** — If `REPEAT_TIMES > 1`, repeats are averaged per participant per
   question

5. **Compute metrics** — Five metric functions are called:
   - `evaluate_continuous()` → Affect Science and Affect Action (0–100 scale)
   - `evaluate_ccb()` → CCB (1–5 scale)
   - `evaluate_mist()` → MIST (binary + Signal Detection Theory)
   - `evaluate_affect_ternary()` → 3-class collapse of affect scores
   - `evaluate_correlation()` → Pearson and Spearman correlations
   - `compute_distribution()` → Value histograms

6. **Extract affect delta** — Change from baseline affect is computed for participants
   who had pre-experiment emotional baseline scores, allowing analysis of the
   inoculation effect on emotional responses

7. **Group analysis** — All metrics are re-computed broken down by Country and
   Inoculation_Group

8. **Write Excel files**

### Ground-truth column mapping

| LLM output | CSV column |
|------------|------------|
| `affect_science S1` | `Sci_1_Affect_1` |
| `affect_science S2–S10` | `Sci_2_Affect_1` … `Sci_10_Affect_1` |
| `affect_action A1` | `Act_1_Affect_1` |
| `affect_action A2–A10` | `Act_2_Affect_1` … `Act_10_Affect_1` |
| `ccb Q1` | `CCB_real` |
| `ccb Q2` | `CCB_cause` |
| `ccb Q3` | `CCB_cons` |
| `mist TS1–TS5` | `T_Support_1` … `T_Support_5` |
| `mist TD1–TD5` | `T_Deny_1` … `T_Deny_5` |
| `mist FS1–FS5` | `F_Support_1` … `F_Support_5` |
| `mist FD1–FD5` | `F_Deny_1` … `F_Deny_5` |

---

## 11. Output Files

Every run creates a timestamped directory: `runs/run_YYYYMMDD_HHMMSS_{method}/`

| File | Description |
|------|-------------|
| `results.json` | Structured LLM predictions for all processed participants |
| `temp_results.jsonl` | Raw per-task JSONL (kept for debugging and resumption) |
| `evaluation.xlsx` | Main evaluation workbook (multiple sheets — see below) |
| `evaluation_raw.xlsx` | Per-participant per-question long-format table |
| `evaluation_plans.xlsx` | Stage 1 reasoning plans for each participant |
| `evaluation_by_group.xlsx` | Metrics broken down by Country and Inoculation_Group |

### Sheets inside evaluation.xlsx

| Sheet | Description |
|-------|-------------|
| `Summary` | One-row-per-metric summary (used by terminal display) |
| `Affect_Science_Detail` | Per-question MAE / Mean Error for Affect Science task |
| `Affect_Science_Metrics` | Overall Affect Science metrics |
| `Affect_Action_Detail` | Per-question MAE / Mean Error for Affect Action task |
| `Affect_Action_Metrics` | Overall Affect Action metrics |
| `CCB_Detail` | Per-question MAE / Mean Error for CCB task |
| `CCB_Metrics` | Overall CCB metrics |
| `MIST_Metrics` | Per-category and overall accuracy/F1 for MIST task |
| `MIST_CM` | Confusion matrix for MIST task |
| `Ternary_Science` | 3-class confusion matrix for Affect Science (negative/neutral/positive) |
| `Ternary_Action` | 3-class confusion matrix for Affect Action |
| `Distribution` | Value histograms for Affect Science, Affect Action, and CCB |
| `Affect_Delta` | Change-from-baseline analysis for participants with pre-experiment scores |
| `Correlation_Science` | Pearson r, Spearman r, p-values for Affect Science |
| `Correlation_Action` | Pearson r, Spearman r, p-values for Affect Action |
| `Correlation_CCB` | Pearson r, Spearman r, p-values for CCB |

---

## 12. Evaluation Metrics Explained

### 12.1 MAE and Mean Error (Affect Science, Affect Action, CCB)

**Mean Absolute Error (MAE):**

```
MAE = (1/N) × Σ |predicted_i − true_i|
```

Lower is better. Measures the average magnitude of prediction error regardless of direction.

**Mean Error (Bias):**

```
Mean_Error = (1/N) × Σ (predicted_i − true_i)
```

- Positive Mean Error → the LLM over-predicts emotional responses or beliefs
- Negative Mean Error → the LLM under-predicts (e.g., predicts milder emotional
  reactions than the real participant had)

**Important note about Affect tasks:** The typical finding is a large negative Mean
Error (e.g., −30 to −50) on Affect Science and Affect Action. This means the LLM
predicts that people would be *more* upset by misinformation tweets than they actually
report being. This systematic bias is important to document and report.

### 12.2 Ternary Classification (Affect Science, Affect Action)

The 0–100 emotional response scale is collapsed into three categories:

| Category | Range | Meaning |
|----------|-------|---------|
| Negative (0) | 0–33 | Low emotional reaction |
| Neutral (1) | 34–66 | Moderate emotional reaction |
| Positive (2) | 67–100 | High emotional reaction |

A 3×3 confusion matrix is computed, allowing you to see whether the LLM is
correctly identifying high-reactivity vs. low-reactivity participants, even if the
exact numbers are off.

### 12.3 Pearson and Spearman Correlations (Affect Science, Affect Action, CCB)

These measure whether the LLM correctly identifies **relative differences** between
participants, even if the absolute values are systematically biased.

**Pearson r:** Linear correlation between LLM predictions and true values.

**Spearman r:** Rank-order correlation. More robust to outliers and non-linear
relationships. Particularly informative when MAE is large but the ordering of
participants might still be correct.

A positive correlation alongside a negative Mean Error would indicate: the LLM
correctly identifies which participants are more vs. less emotionally reactive,
but systematically predicts everyone as more upset than they actually are.

### 12.4 MIST Classification Metrics

The MIST task is binary: for each headline, the participant either correctly
identified it ("real" → true positive / "fake" → true negative) or made an error.

**Per-category accuracy:** Reported separately for TS, TD, FS, FD categories,
allowing you to see whether the LLM is better at predicting correct identification
of obvious misinformation (FD) vs. more subtle items (FS, TD).

**Overall metrics:**

| Metric | Formula |
|--------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |

Where "positive" = predicting the participant will *correctly* classify the headline.

### 12.5 Signal Detection Theory (MIST)

Signal Detection Theory (SDT) separates a participant's **sensitivity** (ability to
distinguish real from fake news) from their **response bias** (tendency to call
everything "real" or "fake").

**d' (d-prime) — Sensitivity:**

```
d' = Z(Hit Rate) − Z(False Alarm Rate)
```

Where:
- **Hit Rate** = proportion of real headlines correctly identified as real
- **False Alarm Rate** = proportion of fake headlines incorrectly called real
- **Z(·)** = inverse normal CDF (z-score)

A higher d' means better ability to distinguish real from fake news.
- d' ≈ 0: No ability to discriminate (chance-level performance)
- d' ≈ 1: Modest discrimination ability
- d' > 2: Good discrimination ability

**c (criterion) — Response Bias:**

```
c = −0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]
```

- c > 0: Conservative bias — tends to call things "fake" (cautious)
- c < 0: Liberal bias — tends to call things "real" (credulous)
- c ≈ 0: No systematic bias

SDT metrics allow you to assess whether the LLM's prediction errors are due to
poor sensitivity (the persona can't tell real from fake) vs. response bias (the
persona systematically calls things one way or the other).

### 12.6 Affect Delta (Change from Baseline)

For participants who completed baseline emotional measures before seeing the tweets,
the delta (change) is computed:

```
delta_i = post_affect_i − baseline_affect_i
```

This captures the inoculation effect: did the inoculation change how emotionally
reactive the participant was? The LLM must predict not just the absolute level but
the *direction and magnitude of change*, which is a substantially harder task.

---

## 13. CoT vs. VBN — Choosing a Method

### Chain-of-Thought (CoT)

The LLM is instructed to reason step by step, identifying relevant factors from the
persona and thinking through what response each factor implies.

- **Prompt structure:** Universal template across all four tasks.
- **Best for:** Quick exploratory runs; straightforward tasks where demographic
  and attitudinal factors are the primary predictors.

### Value-Belief-Norm (VBN)

Based on Stern et al.'s Value-Belief-Norm theory of pro-environmental behaviour.
For each task, the LLM analyses the persona through three specific lenses:

1. **Values (AC):** What does this person fundamentally care about?
2. **Beliefs (AR):** Does this person believe humans are responsible for climate change?
3. **Personal Norms (PN):** What moral obligation does this person feel to act or respond?

- **Prompt structure:** Four task-specific templates (one per survey task).
- **Best for:** Testing whether VBN theory explains variance in emotional responses
  and misinformation susceptibility; more interpretable reasoning traces.

### Choosing in practice

Run both methods on the same random sample using identical seeds and compare:
- **MAE and Mean Error** on the Affect tasks
- **d' and c** on the MIST task
- **Pearson/Spearman r** on all continuous tasks

If VBN improves d' but not MAE, the VBN framework may be better at capturing
who can discriminate real from fake news, but not better at predicting emotional
intensity.

---

## 14. Configuration Reference

When using `run_pipeline.py`, all settings come from `config.py`:

```python
# API / Model
API_KEY           = ""                             # Leave blank to read from env
BASE_URL          = "https://api.openai.com/v1"
LLM_MODEL         = "gpt-4o-mini"
MAX_TOKENS_STAGE1 = 500    # Reasoning plan
MAX_TOKENS_STAGE2 = 250    # Final JSON answer (larger than belief survey due to MIST's 20 items)
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
without re-processing completed tasks.

**How it works:** The runner reads `temp_results.jsonl` at startup. Any task already
recorded in this file is skipped. Only the remaining tasks are sent to the API.

**To resume:** Re-run the same command. The runner automatically detects the
existing `temp_results.jsonl` in the same run directory and continues from where
it left off.

Because the dataset is large (5,948 rows × 4 tasks = 23,792 tasks for a full run),
resumption support is essential. For large runs, consider increasing workers and
running overnight.

---

## 16. Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError: scipy` | scipy not installed | `pip install scipy` |
| `openai.AuthenticationError` | Invalid or missing API key | Check `.env` or `OPENAI_API_KEY` env var |
| `KeyError: 'Sci_1_Affect_1'` | CSV column names don't match | Verify column names match those in `eval/run_cot.py` |
| JSON parse errors in Stage 2 | Model returned malformed JSON | Increase `MAX_TOKENS_STAGE2` (e.g., to 300 for MIST); lower temperature to 0 |
| Very large negative Mean Error on Affect tasks | Systematic LLM overestimation of emotional reactions | This is an expected finding; document the bias separately |
| MIST d' is very high (> 3) but accuracy is only ~55% | LLM has large response bias (c ≠ 0) | Examine c value; consider adjusting prompt to balance real/fake predictions |
| Run is very slow on full dataset | 5,948 rows × 4 tasks = ~24,000 API calls | Increase workers to 10–16; check API rate limits; use random sampling for pilot tests |
| `PerformanceWarning: DataFrame is highly fragmented` | Pandas internals (cosmetic) | Safe to ignore — does not affect any results |
| Excel file missing a sheet | Task had zero valid predictions | Check `evaluation_raw.xlsx` for rows with missing predictions |
