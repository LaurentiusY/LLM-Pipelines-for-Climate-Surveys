from persona.questions import (
    QUESTION_1, REQUIRED_FORMAT_1,
    QUESTION_2, REQUIRED_FORMAT_2,
    QUESTION_3, REQUIRED_FORMAT_3,
    QUESTION_4_ONESHOT, REQUIRED_FORMAT_4,
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

# -------- Stage 1 templates: task-specific --------
USER_PROMPT_TEMPLATE_STAGE1_BELIEF = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for judging the accuracy of climate-change statements:
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

USER_PROMPT_TEMPLATE_STAGE1_POLICY = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for supporting public climate policies:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for policy support):
- AC: Low | Medium | High. Does this person think climate impacts are serious enough to justify collective policy intervention?
- AR: Low | Medium | High. Does this person think governments/industry/society are responsible for taking action (not just individuals)?
- PN: Low | Medium | High. Does this person feel "we should support action" as a matter of personal moral duty, even with tradeoffs?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer. Specially for policy supports, consider the potential financial/time constraints that would reduce support for costly policies.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

USER_PROMPT_TEMPLATE_STAGE1_SHARE = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for willingness to share climate information on social media:
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for pro-environmental information sharing):
- AC: Low | Medium | High. Does the profile indicate they see climate change as serious AND perceive this kind of message as useful/credible enough to share?
- AR: Low | Medium | High. Does the profile suggest they feel some responsibility to inform or influence others about climate issues?
- PN: Low | Medium | High. Does the profile indicate a moral norm that speaking up/sharing is the right thing to do?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

USER_PROMPT_TEMPLATE_STAGE1_TREES = """Profile:
{persona}
--
Question:
{question}
--
Return a VBN plan based on the profile for effort-based pro-environmental action (completing pages to plant trees):
1) evidence: list 2–4 most relevant explicit profile items (must be explicit; no guessing).
2) VBN labels (interpreted for effortful pro-environmental action):
- AC: Low | Medium | High. Do they believe tree planting/climate action yields meaningful benefits that worth finishing these somehow-boring tasks?
- AR: Low | Medium | High. Do they feel personally responsible to contribute effort, not just agree in principle?
- PN: Low | Medium | High. Do they feel an obligation to take an action when given a chance?
3) synthesis: 2-3 sentences (max 50 words) citing which evidence supports the labels, and 2-3 sentences inferring how these VBN factors drive the final answer.

Output JSON ONLY:
{{
  "evidence": ["...", "..."],
  "VBN": {{"AC":"Low|Medium|High","AR":"Low|Medium|High","PN":"Low|Medium|High"}},
  "synthesis": "..."
}}"""

# -------- Stage 1 template router --------
STAGE1_TEMPLATES = {
    "belief": USER_PROMPT_TEMPLATE_STAGE1_BELIEF,
    "policy": USER_PROMPT_TEMPLATE_STAGE1_POLICY,
    "share":  USER_PROMPT_TEMPLATE_STAGE1_SHARE,
    "trees":  USER_PROMPT_TEMPLATE_STAGE1_TREES,
}

# -------- Stage 2 templates --------
USER_PROMPT_TEMPLATE_STAGE2_BELIEF = """
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

USER_PROMPT_TEMPLATE_STAGE2_POLICY = """
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

USER_PROMPT_TEMPLATE_STAGE2_SHARE = """
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

USER_PROMPT_TEMPLATE_STAGE2_TREES = """
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
    "belief": USER_PROMPT_TEMPLATE_STAGE2_BELIEF,
    "policy": USER_PROMPT_TEMPLATE_STAGE2_POLICY,
    "share":  USER_PROMPT_TEMPLATE_STAGE2_SHARE,
    "trees":  USER_PROMPT_TEMPLATE_STAGE2_TREES,
}


def build_user_prompt_stage1(task_type: str, persona_text: str, question_text: str, required_format: str) -> str:
    # required_format retained for compatibility, not injected (by design)
    tpl = STAGE1_TEMPLATES.get(task_type)
    if tpl is None:
        raise ValueError(f"Unknown task_type for stage1 template: {task_type}")
    return tpl.format(persona=persona_text, question=question_text)


def build_user_prompt_stage2(task_type: str, persona_text: str, question_text: str) -> str:
    tpl = STAGE2_TEMPLATES.get(task_type)
    if tpl is None:
        raise ValueError(f"Unknown task_type for stage2 template: {task_type}")
    return tpl.format(persona=persona_text, question=question_text)
