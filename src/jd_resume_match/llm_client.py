from groq import RateLimitError, APIStatusError
import os
from langchain_groq import ChatGroq
from groq import RateLimitError, APIStatusError
FALLBACK_MODELS = [
    {"model": "openai/gpt-oss-120b", "reasoning_effort": "low"},
    {"model": "openai/gpt-oss-20b",  "reasoning_effort": "low"},
    {"model": "llama-3.1-8b-instant"},  # last resort, non-reasoning, fast
]

LAST_USED_MODEL = {"resume": None, "jd": None}  # tracks which model served the last call

def invoke_with_fallback(prompt: str, max_tokens: int = 2048, tag: str = "generic"):
    last_err = None
    for cfg in FALLBACK_MODELS:
        try:
            llm = ChatGroq(
                temperature=0,
                api_key=os.environ.get("GROQ_API_KEY"),
                reasoning_format="hidden",
                max_tokens=max_tokens,
                **cfg,
            )
            response = llm.invoke(prompt)
            if response.content.strip():
                print(f"[LLM] ({tag}) Used model: {cfg['model']}")
                LAST_USED_MODEL[tag] = cfg["model"]
                return response
            last_err = ValueError(f"Empty content from {cfg['model']}")
        except (RateLimitError, APIStatusError) as e:
            print(f"[LLM] ({tag}) {cfg['model']} failed: {e}")
            last_err = e
            continue
    raise RuntimeError(f"All fallback models failed. Last error: {last_err}")