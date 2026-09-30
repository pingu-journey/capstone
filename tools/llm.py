from langchain_openai import ChatOpenAI

from config import JUDGE_MODEL, LLM_MODEL


def _make(model: str, variable: str) -> ChatOpenAI:
    if not model:
        raise RuntimeError(f"{variable}을 .env에 설정하세요.")
    return ChatOpenAI(model=model, temperature=0)


def get_llm() -> ChatOpenAI:
    return _make(LLM_MODEL, "LLM_MODEL")


def get_judge_llm() -> ChatOpenAI:
    return _make(JUDGE_MODEL, "JUDGE_MODEL")
