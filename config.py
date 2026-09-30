import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

MAX_DISCOVERY_ROUNDS = 3
MAX_CANDIDATES_PER_ROUND = 5
MAX_EVALUATIONS = 8
RAG_MAX_REWRITES = 2
RECURSION_LIMIT = 100
INVEST_THRESHOLD = 70

load_dotenv(ROOT / ".env")
LLM_MODEL = os.getenv("LLM_MODEL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
