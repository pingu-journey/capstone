from langchain_openai import ChatOpenAI

from config import JUDGE_MODEL, LLM_MODEL
from tools.services import ConfigurationError


def _make(model: str, variable: str) -> ChatOpenAI:
    if not model:
        raise ConfigurationError(f"{variable}을 .env에 설정하세요.")
    return ChatOpenAI(model=model, temperature=0, max_retries=0, timeout=120)


def get_llm() -> ChatOpenAI:
    return _make(LLM_MODEL, "LLM_MODEL")


def get_judge_llm() -> ChatOpenAI:
    return _make(JUDGE_MODEL, "JUDGE_MODEL")
