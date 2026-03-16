"""
LLM client — wraps the OpenAI API (gpt-4o-mini by default).

All agent modules import call_llm() and call_llm_stream() from here.
To switch models, change MODEL_NAME in config.py — nothing else needs editing.

Setup:
    pip install openai
    export OPENAI_API_KEY='sk-...'
"""

from __future__ import annotations
import os
from pathlib import Path
from openai import OpenAI

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed — fall back to system env vars

from config import OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, MAX_TOKENS

# --- Client singleton ---------------------------------------------------------

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OpenAI API key not found.\n"
            "Set it with:  export OPENAI_API_KEY='sk-...'\n"
            "Or add it to config.py:  OPENAI_API_KEY = 'sk-...'"
        )

    kwargs: dict = {"api_key": api_key}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL  # override for Azure / proxies

    _client = OpenAI(**kwargs)
    return _client


# --- Standard (non-streaming) call --------------------------------------------

def call_llm(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL_NAME,
) -> str:
    """
    Call the OpenAI chat completions API and return the assistant's text.

    Args:
        messages:    List of {"role": "user"|"assistant", "content": str} dicts.
        system:      Optional system prompt.
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens in the response.
        model:       Model identifier (default from config.MODEL_NAME).

    Returns:
        Assistant response as a plain string.
    """
    client = _get_client()

    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# --- Streaming call -----------------------------------------------------------

def call_llm_stream(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL_NAME,
):
    """
    Same as call_llm() but yields token strings as they stream in.

    Usage:
        for token in call_llm_stream(...):
            print(token, end="", flush=True)
    """
    client = _get_client()

    full_messages: list[dict] = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    with client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# --- Connection test ----------------------------------------------------------

def test_connection() -> bool:
    """Quick sanity check — run before a long experiment."""
    try:
        reply = call_llm(
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"[OpenAI] Connection OK — {MODEL_NAME} replied: {reply!r}")
        return True
    except Exception as e:
        print(f"[OpenAI] Connection FAILED: {e}")
        print(f"  Is OPENAI_API_KEY set?  Run: export OPENAI_API_KEY='sk-...'")
        return False


if __name__ == "__main__":
    test_connection()