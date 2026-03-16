import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai-proxy.org/v1"

LLM_MODEL = "gpt-4o-mini"

MAX_TOKENS_STAGE1 = 500
MAX_TOKENS_STAGE2 = 120
TEMPERATURE = 0

MAX_WORKERS = 6

SELECTED_ROW_IDS = []  # e.g. [1, 2, 5] or [] for all

USE_RANDOM_SAMPLING = True
RANDOM_SAMPLE_SIZE = 100
RANDOM_SAMPLE_SEED = 42

REPEAT_TIMES = 1
LLM_RANDOM_SEED = 42
