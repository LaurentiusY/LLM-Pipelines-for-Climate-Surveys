# Task 1 - 气候态度
QUESTION_1 = """How accurate do you think these statements are? Answer on a scale from 0 to 100, where 0 means "not at all accurate" and 100 means "extremely accurate".
Questions:
Q1: Taking action to fight climate change is necessary to avoid a global catastrophe.
Q2: Human activities are causing climate change.
Q3: Climate change poses a serious threat to humanity.
Q4: Climate change is a global emergency.
Respond in the following JSON format only, with no additional text: {"Q1": <number>, "Q2": <number>, "Q3": <number>, "Q4": <number>}"""

REQUIRED_FORMAT_1 = '{"Q1": <number>, "Q2": <number>, "Q3": <number>, "Q4": <number>}'

# Task 2 - 政策支持
QUESTION_2 = """Many countries have introduced policies to help reduce carbon emissions and help to mitigate the climate crisis. This can include the implementation of laws and requirements which broadly aim to reduce various greenhouse gasses.
Please indicate your level of agreement with the following policy statements on a scale from 0 to 100, where 0 means "not at all", 50 means "moderately", and 100 means "very much so". If a policy does not apply to your situation, respond with "NA".
Policies:
Q1: Significantly expanding infrastructure for public transportation
Q2: Protecting forested and land areas
Q3: Increasing the number of charging stations for electric vehicles
Q4: Investing more in green jobs and businesses
Q5: Increasing taxes on airline companies to offset carbon emissions
Q6: Increasing taxes on carbon intense foods (for example meat, and dairy)
Q7: Raising carbon taxes on gas/fossil fuels/coal
Q8: Increasing the use of sustainable energy such as wind and solar energy
Q9: Introducing laws to keep waterways and oceans clean
Respond in the following JSON format only, with no additional text: {"Q1": <number or "NA">, "Q2": <number or "NA">, "Q3": <number or "NA">, "Q4": <number or "NA">, "Q5": <number or "NA">, "Q6": <number or "NA">, "Q7": <number or "NA">, "Q8": <number or "NA">, "Q9": <number or "NA">}"""

REQUIRED_FORMAT_2 = '{"Q1": <number or "NA">, "Q2": <number or "NA">, "Q3": <number or "NA">, "Q4": <number or "NA">, "Q5": <number or "NA">, "Q6": <number or "NA">, "Q7": <number or "NA">, "Q8": <number or "NA">, "Q9": <number or "NA">}'

# Task 3 - 社交媒体分享意愿  (FIXED QUOTES)
QUESTION_3 = """"Did you know that removing meat and dairy for only two out of three meals per day could decrease food-related carbon emissions by 60%? It is an easy way to fight #ClimateChange. Source: https://econ.st/3qjvOnn"
Q1: Are you willing to share this information (above) on your social media?
Respond in the following JSON format only, with no additional text: {"Q1": "yes" | "no" | "no_social_media"}"""

REQUIRED_FORMAT_3 = '{"Q1": "yes" | "no" | "no_social_media"}'

# Task 4 - 一次性种树任务（简化版）
QUESTION_4_ONESHOT = """
We would like you to complete a number identification task. Below, you see a series of two-digit numbers. You will need to identify the target numbers. Target numbers are all numbers that consist of an even first digit (i.e., 2, 4, 6, 8) and an odd second digit (i.e., 1, 3, 5, 7, 9). For example, "25" or "83" would be target numbers, but "17", "42", or "56" would not be target numbers.
As an example, you need to identify all numbers with an even first digit and an odd second digit from the following list:
19 54 67 71 85 44 14 92 75 74 78 73 24 23 26 81 75 64

Did you know that planting trees is one of the best ways to fight climate change? As trees grow, they remove carbon dioxide (a greenhouse gas) from the air. They store the carbon in the trees and soil, and then release oxygen into the air.
In the following pages, you will have the option to complete additional pages of the number-identification task. **For each page that you correctly complete, we will make a donation of one tree to the Eden Reforestation Project**, an organization that has planted over 830 million trees since its inauguration.
Please note, these trees **will actually be planted in the real world**. The more pages you complete, the more trees will be planted!
It is up to you to decide how much time and effort you want to invest in the task. There are a maximum of 8 pages that you can complete, and each page will contain 60 numbers.
**For each page that you complete, one tree will be planted on your behalf.** This is completely voluntary and will not impact your compensation.
Q1: How many pages (0-8) would you realistically choose to complete based on the persona?

Respond in the following JSON format only, with no additional text: {"trees": <number 0-8>}"""

REQUIRED_FORMAT_4 = '{"trees": <number 0-8>}'
