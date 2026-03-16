from persona.questions import (
    QUESTION_1, REQUIRED_FORMAT_1,
    QUESTION_2, REQUIRED_FORMAT_2,
    QUESTION_3, REQUIRED_FORMAT_3,
    QUESTION_4, REQUIRED_FORMAT_4,
)

# =========================================================================
# Two-stage System Prompts (VBN framework)
# =========================================================================
SYSTEM_PROMPT_STAGE1 = """You are an expert of the Value–Belief–Norm (VBN) framework for pro-environmental behavior.
You are preparing to answer a survey question as a specific person.
You will be given a profile that contains factual background and prior questionnaire responses.
Your goal is to extract VBN components (Awareness of Consequences - AC; Ascription of Responsibility - AR; Personal Norm - PN) from this profile.
Output JSON only. No extra keys, no comments, no markdown, no trailing text."""

SYSTEM_PROMPT_STAGE2 = """You are a survey participant. You will be given a profile that contains factual background, prior questionnaire responses, and an analyze plan extracted from it using the Value–Belief–Norm (VBN) framework.
VBN (Value–Belief–Norm) theory proposes that people's values shape their environmental beliefs (e.g., awareness of consequences and ascription of responsibility), which activate personal moral norms, and these norms in turn drive pro-environmental intentions and behavior.
Your goal is to complete the survey in a way that reflects the likely attitudes and behavioral tendencies implied by this person's demographic and psychological profile, as they would realistically respond in such a survey.
Use the given VBN plan as evidence about motivation. Follow the response format EXACTLY as requested (e.g., JSON only). No extra keys, no comments, no markdown, no trailing text."""

# =========================================================================
# Two-stage User Prompt Templates (TASK-SPECIFIC STAGE1)
# =========================================================================

USER_PROMPT_TEMPLATE_STAGE1_AFFECT_SCIENCE = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for emotional reactions to climate science-denying disinformation tweets:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for affect/emotional reaction to misinformation):
- AC: Low | Medium | High. Does the profile indicate they see climate science denial as causing serious harmful consequences?
- AR: Low | Medium | High. Does the profile suggest they feel people are responsible for correcting or rejecting misinformation?
- PN: Low | Medium | High. Does the profile indicate a personal moral norm about defending scientific truth?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

USER_PROMPT_TEMPLATE_STAGE1_AFFECT_ACTION = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for emotional reactions to tweets opposing climate action:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for affect/emotional reaction to anti-action misinformation):
- AC: Low | Medium | High. Does the profile indicate they see opposition to climate action as having serious harmful consequences?
- AR: Low | Medium | High. Does the profile suggest they feel responsible for supporting climate action against such opposition?
- PN: Low | Medium | High. Does the profile indicate a moral norm about taking climate action despite opposition?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

USER_PROMPT_TEMPLATE_STAGE1_CCB = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for judging climate change beliefs:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for belief/accuracy judgments):
- AC: Low | Medium | High. Does the profile indicate they see climate change as causing serious harmful consequences?
- AR: Low | Medium | High. Does the profile suggest they feel people (including themselves) are responsible to respond?
- PN: Low | Medium | High. Does the profile indicate a moral obligation or personal duty to act?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

USER_PROMPT_TEMPLATE_STAGE1_MIST = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for truth discernment about climate-related statements:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for truth discernment / misinformation detection):
- AC: Low | Medium | High. Does the profile indicate awareness that climate misinformation has serious consequences?
- AR: Low | Medium | High. Does the profile suggest they feel personally responsible for discerning truth from falsehood?
- PN: Low | Medium | High. Does the profile indicate a moral norm about epistemic accuracy and critical evaluation?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

# -------- Stage 1 template router --------
STAGE1_TEMPLATES = {
    "affect_science": USER_PROMPT_TEMPLATE_STAGE1_AFFECT_SCIENCE,
    "affect_action":  USER_PROMPT_TEMPLATE_STAGE1_AFFECT_ACTION,
    "ccb":            USER_PROMPT_TEMPLATE_STAGE1_CCB,
    "mist":           USER_PROMPT_TEMPLATE_STAGE1_MIST,
}

# -------- Stage 2 templates --------
_STAGE2_TEMPLATE = """
Here is your profile:
{persona}
---
Here is a VBN plan extracted from the profile:
__PLAN_JSON__
Answer the survey question below by applying the VBN plan.
Use the VBN plan as a reference framework, but allow nuanced responses based on the specific question context.
Follow the required response format EXACTLY.
{question}
"""

STAGE2_TEMPLATES = {
    "affect_science": _STAGE2_TEMPLATE,
    "affect_action":  _STAGE2_TEMPLATE,
    "ccb":            _STAGE2_TEMPLATE,
    "mist":           _STAGE2_TEMPLATE,
}


def build_user_prompt_stage1(task_type: str, persona_text: str, question_text: str, required_format: str) -> str:
    tpl = STAGE1_TEMPLATES.get(task_type)
    if tpl is None:
        raise ValueError(f"Unknown task_type for stage1 template: {task_type}")
    return tpl.format(persona=persona_text, question=question_text)


def build_user_prompt_stage2(task_type: str, persona_text: str, question_text: str) -> str:
    tpl = STAGE2_TEMPLATES.get(task_type)
    if tpl is None:
        raise ValueError(f"Unknown task_type for stage2 template: {task_type}")
    return tpl.format(persona=persona_text, question=question_text)
