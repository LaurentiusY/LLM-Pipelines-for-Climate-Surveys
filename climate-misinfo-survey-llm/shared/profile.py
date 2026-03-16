import pandas as pd


def generate_profile(row, mode='both'):
    """
    根据 mode 生成受访者画像 (Spampatti et al. misinfo survey).
    mode: 'demographics', 'psychology', 'both'
    """
    description_parts = []

    def clean_val(val):
        if pd.isna(val): return None
        try:
            f_val = float(val)
            if f_val.is_integer(): return int(f_val)
            return f_val
        except:
            return val

    def add_item(col_name, question_text, scale_def=None):
        if col_name in row.index:
            val = clean_val(row[col_name])
            if val is not None:
                if scale_def:
                    full_q = f"{question_text} ({scale_def})"
                else:
                    full_q = question_text
                description_parts.append(f"the person answered the question '{full_q}' as '{val}'")

    age = clean_val(row.get('Age'))
    gender_map = {1: "Male", 2: "Female", 3: "Other/Prefer not to say"}
    gender_raw = clean_val(row.get('Gender'))
    gender = gender_map.get(gender_raw, "person") if gender_raw else "person"

    country = row.get('Country', "their country")
    if pd.isna(country): country = "their country"

    edu_map = {
        1: "less than high school",
        2: "high school or equivalent",
        3: "some college or associate degree",
        4: "bachelor's degree or higher",
    }
    edu_raw = clean_val(row.get('Education'))
    education = edu_map.get(edu_raw) if edu_raw else None

    # Experimental condition
    condition = clean_val(row.get('Condition'))
    inoc_group = row.get('Inoculation_Group', None)
    if pd.isna(inoc_group):
        inoc_group = None

    intro_sentence = ""

    include_demo = mode in ['demographics', 'both']
    include_psych = mode in ['psychology', 'both']

    if include_demo:
        intro_parts = []
        if age:
            intro_parts.append(f"A {age}-year-old {gender}")
        else:
            intro_parts.append(f"A {gender}")

        intro_parts.append(f"living in {country}")
        if education:
            intro_parts.append(f"with an education level of {education}")
        intro_sentence = " ".join(intro_parts)

        add_item(
            "Pol_ideo",
            "What is your political orientation?",
            "1-Very liberal/left-wing to 10-Very conservative/right-wing"
        )

        add_item(
            "CRT",
            "Cognitive Reflection Test score (number of correct answers out of 4)",
            "0-4"
        )
    else:
        intro_sentence = f"A person living in {country}"

    # Experimental context (always included)
    if inoc_group == 'Control' or condition == 0:
        description_parts.append(
            "before viewing climate-related tweets, this person was in the control group "
            "and received no inoculation intervention"
        )
    elif inoc_group == 'Cognitive' or condition in [1, 3, 5]:
        description_parts.append(
            "before viewing climate-related tweets, this person received a cognitive "
            "inoculation intervention designed to build resistance against misinformation "
            "through logical reasoning and critical thinking"
        )
    elif inoc_group == 'Socioaffective' or condition in [2, 4, 6]:
        description_parts.append(
            "before viewing climate-related tweets, this person received a socioaffective "
            "inoculation intervention designed to build resistance against misinformation "
            "through trust in science and emotional awareness"
        )

    if include_psych:
        add_item("Threat_1", "How much do you feel threatened by climate misinformation?",
                 "1-Not at all to 7-Very much")
        add_item("Threat_2", "How much do you think climate misinformation is a problem for society?",
                 "1-Not at all to 7-Very much")
        add_item("Threat_3", "How worried are you about the spread of climate misinformation?",
                 "1-Not at all to 7-Very much")
        add_item("Threat_4", "How serious do you consider the threat of climate misinformation?",
                 "1-Not at all to 7-Very much")

        add_item("Affect_T1_1",
                 "Before viewing any tweets, how do you feel about climate change right now?",
                 "0-Very negative to 100-Very positive, 50-Neutral")

    if description_parts:
        return intro_sentence + " where " + ", ".join(description_parts) + "."
    else:
        return intro_sentence + "."
