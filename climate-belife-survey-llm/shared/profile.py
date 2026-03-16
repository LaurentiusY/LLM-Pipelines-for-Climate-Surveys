import pandas as pd


def generate_profile(row, mode='both'):
    """
    根据 mode 生成受访者画像。
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
    gender_map = {1: "Male", 2: "Female", 3: "Prefer not to say", 4: "Non-binary/third gender/other"}
    gender_raw = clean_val(row.get('Gender'))
    gender = gender_map.get(gender_raw, "person") if gender_raw else "person"

    country = row.get('country', "their country")
    if pd.isna(country): country = "their country"

    edu_map = {
        1: "primary education (0-6 years)", 2: "secondary education (7-12 years)",
        3: "college/university education (13-16 years)", 4: "advanced education (more than 17 years)",
        5: "unspecified education"
    }
    edu_raw = clean_val(row.get('Education.2'))
    education = edu_map.get(edu_raw) if edu_raw else None

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
        if education: intro_parts.append(f"with an education of {education}")
        intro_sentence = " ".join(intro_parts)

        add_item(
            "MacArthur_SES",
            f"Think of this ladder as representing where people stand in {country}. At the top of the ladder are the people who are the best off - those who have the most money, the most education, and the most respected jobs. At the bottom are those who have the least money, least education, the least respected jobs, or no job. The higher you are, the closer you are to the people at the top; the lower you are, the closer you are to the people at the very bottom. Where would you place yourself on this ladder relative to other people in {country}?",
            "1-Bottom (worst off) to 10-Top (best off)"
        )

        pol_scale = "0-Extremely liberal/left-wing to 100-Extremely conservative/right-wing"
        add_item("Politics2_1", "What is your political orientation for social issues?", pol_scale)
        add_item("Politics2_9", "What is your political orientation for economic issues?", pol_scale)

        if "Indirect_SES" in row and pd.notna(row["Indirect_SES"]):
            ses_items_map = {
                '1': "Separate room for kitchen", '2': "Washing machine", '3': "Vacuum cleaner",
                '4': "Freezer/deep freeze", '5': "Personal computer", '6': "Bathroom", '7': "Television"
            }
            try:
                raw_str = str(row["Indirect_SES"]).replace('.0', '')
                indices = [x.strip() for x in raw_str.split(',')]
                items = [ses_items_map[i] for i in indices if i in ses_items_map]
                if items:
                    description_parts.append(f"the person indicated they own/have access to: {', '.join(items)}")
            except:
                pass

        add_item(
            "PerceivedSciConsensu_1",
            "To the best of your knowledge, what percentage of climate scientists have concluded that human-caused climate change is happening?",
            "0-100 Percentage"
        )
    else:
        intro_sentence = f"A person living in {country}"

    if include_psych:
        add_item("Trust_sci1_1", "How much do you trust climate scientists as a source of information about global warming?",
                 "1-Strongly distrust to 5-Strongly trust")
        add_item("Trust_sci2_1", "How much do you trust climate scientists to provide accurate information about climate change?",
                 "1-Strongly distrust to 5-Strongly trust")
        add_item("Trust_gov_1", "How much do you trust your government to take appropriate action on climate change?",
                 "1-Strongly distrust to 5-Strongly trust")

        add_item("ID_hum_1", "How strongly do you identify as a member of humanity as a whole?",
                 "1-Not at all to 7-Very strongly")
        add_item("ID_GC_1", "How strongly do you identify as a global citizen?", "1-Not at all to 7-Very strongly")

        add_item("Enviro_ID_1", "I think of myself as someone who is very concerned about environmental issues",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_ID_2", "Being someone who is concerned about environmental issues is an important part of who I am",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_ID_3", "I am the type of person who acts in environmentally friendly ways",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_ID_4", "Acting in environmentally friendly ways is important to me",
                 "1-Strongly disagree to 7-Strongly agree")

        add_item("Enviro_motiv_1", "Protecting the environment is important because nature has value in its own right",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_11", "I feel personally obligated to do what I can to prevent climate change",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_12", "I feel guilty when I do things that harm the environment",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_13", "I would be willing to sacrifice some personal comfort to reduce environmental problems",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_14", "Preventing climate change is important for the sake of future generations",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_15", "Protecting the environment will benefit the economy in the long run",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_16", "I worry about the effects of climate change on my health",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_17", "Climate change is a threat to my family's wellbeing",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_18", "I support environmental protection mainly because it benefits people like me",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_19", "Preventing climate change is important for my own wellbeing",
                 "1-Strongly disagree to 7-Strongly agree")
        add_item("Enviro_motiv_20", "Acting pro-environmentally is something I do automatically without much thought",
                 "1-Strongly disagree to 7-Strongly agree")

        add_item("probe_CC_1", "How serious of a problem do you think climate change is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_GW_1", "How serious of a problem do you think global warming is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_GH_1", "How serious of a problem do you think greenhouse gas emissions are?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_CCrisis_1", "How serious of a problem do you think the climate crisis is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_GE_1", "How serious of a problem do you think the greenhouse effect is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_CE_1", "How serious of a problem do you think carbon emissions are?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_CP_1", "How serious of a problem do you think carbon pollution is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_CEmerg_1", "How serious of a problem do you think the climate emergency is?",
                 "1-Not at all serious to 5-Extremely serious")
        add_item("probe_CPoll_1", "How serious of a problem do you think CO2 pollution is?",
                 "1-Not at all serious to 5-Extremely serious")

    if description_parts:
        return intro_sentence + " where " + ", ".join(description_parts) + "."
    else:
        return intro_sentence + "."
