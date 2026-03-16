from persona.questions import (
    QUESTION_1, REQUIRED_FORMAT_1,
    QUESTION_2, REQUIRED_FORMAT_2,
    QUESTION_3, REQUIRED_FORMAT_3,
    QUESTION_4_ONESHOT, REQUIRED_FORMAT_4,
)

# =========================================================================
# Two-stage System Prompts (Scenario / logical reasoning)
# =========================================================================
SYSTEM_PROMPT_STAGE1 = """You are preparing to answer a survey question as a specific person.
You will be given a profile that contains factual background and prior questionnaire responses.
Your goal is to think about how to complete the survey in a way that reflects the likely attitudes and behavioral tendencies implied by this person's demographic and psychological profile, as they would realistically respond in such a survey.
Do NOT answer the survey yet. Instead, produce a brief reasoning plan. Output JSON only. No extra keys, no comments, no markdown, no trailing text."""

SYSTEM_PROMPT_STAGE2 = """You are a survey participant. You will be given a profile that contains factual background and prior questionnaire responses. Your goal is to complete the survey in a way that reflects the likely attitudes and behavioral tendencies implied by this person's demographic profile, as they would realistically respond in such a survey. Prioritize consistency with the person's background and prior responses rather than general norms or idealized answers. Follow the response format EXACTLY as requested (e.g., JSON only). No extra keys, no comments, no markdown, no trailing text."""

# =========================================================================
# Two-stage User Prompt Templates
# =========================================================================

# Stage 1: Evidence -> Scenario -> Mapping plan
USER_PROMPT_TEMPLATE_STAGE1 = """Profile:
{persona}
--
Question:
{question}
--
Create a brief plan in THREE parts:
(1) evidence: 2–4 explicit profile items that are most relevant to the question (must be explicit; no guessing).
(2) key factors: 2–3 short factors derived directly from the evidence (no new facts).
(3) response guideline: 1–2 sentences guiding the answer while respecting the question scale and constraints (avoid giving a specific target number).

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "key_factors": ["...", "..."],
  "response_guideline": "..."
}}"""

# Stage 2: Use the plan JSON (filled later by runner) to answer survey question
# We intentionally keep a placeholder token __PLAN_JSON__ that your runner replaces
USER_PROMPT_TEMPLATE_STAGE2 = """Here is your profile:
{persona}
---
Here is a plan as a guidance:
__PLAN_JSON__
Using the plan above, answer the survey question below. If the plan is missing details, rely on the profile and the question. Do not add facts that are not in the profile. Answer the survey question below and follow the required response format EXACTLY.

{question}"""


def build_user_prompt_stage1(persona_text: str, question_text: str, required_format: str) -> str:
    return USER_PROMPT_TEMPLATE_STAGE1.format(
        persona=persona_text,
        question=question_text,
        required_format=required_format
    )


def build_user_prompt_stage2(persona_text: str, question_text: str) -> str:
    return USER_PROMPT_TEMPLATE_STAGE2.format(
        persona=persona_text,
        question=question_text
    )
