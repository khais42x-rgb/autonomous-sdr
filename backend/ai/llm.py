import os
import time
import json
import logging
from typing import TypeVar, Type
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("LLM_API_KEY", ""),
    timeout=60.0,
    max_retries=2,
)

MODELS = {
    "small": os.getenv("MODEL_SMALL", "gpt-4o-mini"),
    "strong": os.getenv("MODEL_STRONG", "gpt-4o"),
}

PRICES = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o":      {"in": 2.50, "out": 10.00},
}

class AgentOutputError(Exception):
    pass

def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()

def _calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICES.get(model, {"in": 0.0, "out": 0.0})
    return (tokens_in * price["in"] + tokens_out * price["out"]) / 1_000_000

def call_structured(
    system: str,
    user: str,
    schema: Type[T],
    tier: str = "small",
    temperature: float = 0.2,
    max_repair_attempts: int = 1,
) -> tuple[T, dict]:
    model = MODELS.get(tier, MODELS["small"])
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{user}\n\nReturn ONLY valid JSON matching this exact schema:\n{schema_json}",
        },
    ]

    last_error = None

    for attempt in range(1 + max_repair_attempts):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as e:
            last_error = str(e)
            time.sleep(1)
            continue

        latency_ms = int((time.time() - t0) * 1000)
        text = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        cost_usd = _calculate_cost(model, tokens_in, tokens_out)

        meta = {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": latency_ms,
            "attempts": attempt + 1,
        }

        cleaned = strip_fences(text)
        try:
            validated = schema.model_validate_json(cleaned)
            return validated, meta
        except Exception as e:
            last_error = str(e)
            if attempt < max_repair_attempts:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Invalid JSON: {str(e)[:200]}. Return ONLY corrected JSON matching the schema.",
                })

    raise AgentOutputError(f"Invalid output after retries: {last_error}")