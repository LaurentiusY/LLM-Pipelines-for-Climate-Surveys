# ===========================================================================
# 20 disinformation tweet texts (transcribed from stimulus images)
# Science_01-10: Tweets denying climate science
# Action_01-10: Tweets opposing climate action
# ===========================================================================

SCIENCE_TWEETS = {
    "S1": (
        "As more wind and solar are added they raise electricity prices and destabilize "
        "electric grids. Because they are part-time unreliable weather dependent sources. "
        "We want full-time electricity. Not part-time like third world countries. All for "
        "silly expensive net zero. CA pays more."
    ),
    "S2": (
        "The current exceptional warming and cooling your seeing is due to the location "
        "of the Jet Stream. It's become very wavy due to the lack of Solar Energy going "
        "into the Oceans and nothing to do with Man Made CO2"
    ),
    "S3": (
        "Today's 'global warming' is estimated to be an otherwise unmeasurable 0.4\u00b0C "
        "(0.72\u00b0F) over the 1979-2000 average... despite 50% of all manmade emissions. "
        "No 2022 weather event was unprecedented or can be blamed on CO2 emissions."
    ),
    "S4": (
        "This is a portrait of climate fraud, posturing as the saviours of the world. "
        "They are a breed of crooks, getting rich by ripping off gullible western nations. "
        "The UN led climate hoax has been running since 1988. They want us to believe a "
        "pack of lies about earth's climate."
    ),
    "S5": (
        "Too often, academic reports on climate use highly skewed data that seem to have "
        "been carefully selected to support aggressive environmental regulations. One recent "
        "and much-cited Lancet report appears deliberately deceptive."
    ),
    "S6": (
        "The climate hoax devised by the UN, supported by rich elitists is endorsed by our "
        "treacherous leaders is an attack on freedoms & rights. Climate cultism is a form of "
        "global self hatred. It aims to punish western nations by transferring huge "
        "reparations to the developing world."
    ),
    "S7": (
        'Top NASA Climate Modeler Admits Predictions Are \u201cMathematically Impossible\u201d'
    ),
    "S8": (
        "Lots of links of studies of the Medieval Warm Period that climate science deniers "
        "(alarmists) want to pretend did not exist. Because there is no explanation for "
        "natural warming during this time. Studies point out temp was warmer back then, "
        "than now."
    ),
    "S9": (
        "According to global warming theory the poles should warm significantly if carbon "
        "dioxide is driving temperatures Just the opposite is occurring in the southern "
        "hemisphere."
    ),
    "S10": (
        "The evidence for manmade climate change is so thin they cannot debate it. They "
        "hide behind the lie of consensus. There is no room for consensus in science. The "
        "basis is a provable hypothesis. There is not a single peer reviewed study that "
        "proves manmade CO2 is causing warming."
    ),
}

ACTION_TWEETS = {
    "A1": (
        "At Climate Summit, Elites Chow Down on Gourmet Meats While Telling Us to Eat Bugs"
    ),
    "A2": (
        "FACT CHECK Results of the Biden administration's extreme climate agenda cutting "
        "emissions by 44% by 2030. Annual Jobs Lost: 1.2 MILLION. Lost Economic Growth: "
        "$7.7 TRILLION. Increase in Electric Bills: 23% Increase in Gas Prices: 2$ PER YEAR"
    ),
    "A3": (
        "The war on 'fossil fuels' is absurd considering the vast fields of coal/oil/gas "
        "everywhere on earth. The mantle is brimming over with it. A United Nations bid for "
        "control, cash & power has led to an energy crisis that looms as the biggest "
        "self-inflicted disaster in human history."
    ),
    "A4": (
        "Death and privation caused by the lack of affordable energy caused by Green Energy "
        "policies will not affect the Elites at all. They want us to eat bugs, do a lot less "
        "as they carry on with their lives just as they are doing now. Climate scamsters. "
        "They should lead by example."
    ),
    "A5": (
        "You are lying. Fossil fuels gave us cheap energy for decades so billions live "
        "longer healthier happier lives. Many technologies like carbon capture, filters fuel "
        "additives etc reduces emissions. Banning fossil fuels is creating fuel poverty and "
        "harming people"
    ),
    "A6": (
        "Energy literacy starts with the knowledge that renewable energy is only intermittent "
        "electricity generated from unreliable breezes and sunshine, as wind turbines and "
        "solar panels cannot manufacture anything for the 8 billion on this planet."
    ),
    "A7": (
        "Imagine sacrificing 500 high-paying coal jobs, ranging up to $60,000/yr, for the "
        "climate hoax. Even if you believe in the hoax, global emissions are up 5% from "
        "pre-pandemic levels -- 90% because of China. Emissions from a single mine are "
        "insignificant."
    ),
    "A8": (
        "Europe's transition to renewable energy and net zero carbon is not working, except "
        "to make life hard on average European citizens."
    ),
    "A9": (
        "Willfully-blind ignorance about the consequences of [the rush to green policies] "
        "\u2013 deep recessions, broken societies and millions more going hungry \u2013 "
        "doesn\u2019t make them any less immoral. The road to hell is paved with good "
        "intentions. Bingo."
    ),
    "A10": (
        "Solar and wind are far more expensive than established reliable stable secure "
        "electricity from pure hydro coal gas nuclear. That's why your shift to unreliable, "
        "unstable, expensive solar and wind; is devastating families; and exporting "
        "manufacturing jobs"
    ),
}


def _format_tweets(tweets_dict):
    return "\n".join(f'{k}: "{v}"' for k, v in tweets_dict.items())


# ===========================================================================
# Task 1 - Affect response to science-denial tweets (0-100)
# ===========================================================================
QUESTION_1 = f"""You are viewing a series of tweets about climate change that have been posted on social media. For each tweet, please rate how the tweet makes you feel on a scale from 0 to 100, where 0 means "very negative feelings", 50 means "neutral", and 100 means "very positive feelings".

Tweets:
{_format_tweets(SCIENCE_TWEETS)}

Respond in the following JSON format only, with no additional text: {{"S1": <number>, "S2": <number>, "S3": <number>, "S4": <number>, "S5": <number>, "S6": <number>, "S7": <number>, "S8": <number>, "S9": <number>, "S10": <number>}}"""

REQUIRED_FORMAT_1 = '{"S1": <number>, "S2": <number>, ..., "S10": <number>}'


# ===========================================================================
# Task 2 - Affect response to action-opposition tweets (0-100)
# ===========================================================================
QUESTION_2 = f"""You are viewing a series of tweets about climate change that have been posted on social media. For each tweet, please rate how the tweet makes you feel on a scale from 0 to 100, where 0 means "very negative feelings", 50 means "neutral", and 100 means "very positive feelings".

Tweets:
{_format_tweets(ACTION_TWEETS)}

Respond in the following JSON format only, with no additional text: {{"A1": <number>, "A2": <number>, "A3": <number>, "A4": <number>, "A5": <number>, "A6": <number>, "A7": <number>, "A8": <number>, "A9": <number>, "A10": <number>}}"""

REQUIRED_FORMAT_2 = '{"A1": <number>, "A2": <number>, ..., "A10": <number>}'


# ===========================================================================
# Task 3 - Climate Change Beliefs (1-5 scale, 3 subscales)
# ===========================================================================
QUESTION_3 = """Please indicate the extent to which you agree with the following statements on a scale from 1 to 5, where 1 means "strongly disagree" and 5 means "strongly agree".

Statements:
Q1: Climate change is really happening. (reality of climate change)
Q2: Human activities are a significant cause of climate change. (human causation)
Q3: Climate change will have serious negative consequences for humanity and the environment. (consequences)

Respond in the following JSON format only, with no additional text: {"Q1": <number>, "Q2": <number>, "Q3": <number>}"""

REQUIRED_FORMAT_3 = '{"Q1": <number>, "Q2": <number>, "Q3": <number>}'


# ===========================================================================
# Task 4 - MIST Truth Discernment (binary true/false, 20 items)
# Extracted from Truth_DiscernmentRaw.txt (Spampatti et al.)
# 4 categories x 5 items each:
#   TS = True Support (true facts supporting climate action)
#   TD = True Delay (true facts about costs/barriers of climate action)
#   FS = False Support (exaggerated pro-climate claims)
#   FD = False Delay (denial/downplaying claims)
# ===========================================================================

MIST_STATEMENTS = {
    # True statements supporting climate action (correct answer: TRUE)
    "TS1": "Human activities, such as burning fossil fuels, are the main cause of climate change.",
    "TS2": "A transition to renewable energy sources is crucial for mitigating climate change.",
    "TS3": "Climate change is leading to more intense and frequent natural disasters.",
    "TS4": "The world must reach net-zero emissions by mid-century to avoid the worst impacts of climate change.",
    "TS5": "The switch to electric vehicles can significantly reduce transportation sector emissions.",
    # True statements about climate delay/costs (correct answer: TRUE)
    "TD1": "Efforts to Mitigate Climate Change Can Require Significant Upfront Investments",
    "TD2": "Shifting to Clean Energy Sources Can Affect Energy Prices and Reliability",
    "TD3": "Phasing Out Fossil Fuels Can Have Economic Implications for Certain Industries",
    "TD4": "Implementing Renewable Energy Sources on a Large Scale Can be Costly",
    "TD5": "Adopting Sustainable Agricultural Practices Can Require Changes in Farming Methods",
    # False statements exaggerating climate support (correct answer: FALSE)
    "FS1": "Climate Catastrophe: Entire Cities to be Submerged by Rising Seas Within Decades",
    "FS2": "Global Warming Apocalypse: Temperatures to Soar by 10 Degrees Celsius This Century",
    "FS3": "Mass Extinction Imminent: Over Half of All Species to Disappear Due to Climate Change",
    "FS4": "Heat Death: Temperatures to Reach Lethal Limits, Making Earth Uninhabitable",
    "FS5": "Airpocalypse: Breathing to Become Impossible as Climate Change Chokes the World",
    # False statements downplaying/denying climate change (correct answer: FALSE)
    "FD1": "Climate change has little to no impact on human health",
    "FD2": "Greenhouse gases are not a significant contributor to air pollution",
    "FD3": "Global warming will not lead to widespread food shortages or famine",
    "FD4": "The impact of climate change on wildlife is overstated and not a concern",
    "FD5": "Greenhouse gas emissions have little to no impact on the earth's atmosphere.",
}


def _format_mist():
    lines = []
    for k, v in MIST_STATEMENTS.items():
        lines.append(f'{k}: "{v}"')
    return "\n".join(lines)


QUESTION_4 = f"""You will be shown a series of statements about climate change. For each statement, please indicate whether you think it is TRUE or FALSE.

Statements:
{_format_mist()}

Respond in the following JSON format only, with no additional text: {{"TS1": "true"|"false", "TS2": "true"|"false", "TS3": "true"|"false", "TS4": "true"|"false", "TS5": "true"|"false", "TD1": "true"|"false", "TD2": "true"|"false", "TD3": "true"|"false", "TD4": "true"|"false", "TD5": "true"|"false", "FS1": "true"|"false", "FS2": "true"|"false", "FS3": "true"|"false", "FS4": "true"|"false", "FS5": "true"|"false", "FD1": "true"|"false", "FD2": "true"|"false", "FD3": "true"|"false", "FD4": "true"|"false", "FD5": "true"|"false"}}"""

REQUIRED_FORMAT_4 = '{"TS1": "true"|"false", ..., "FD5": "true"|"false"}'
