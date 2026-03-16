"""
Thin wrapper around LM Studio's OpenAI-compatible local API.

LM Studio runs a local server that mirrors the OpenAI chat/completions
endpoint, so we use the `openai` Python library pointed at localhost.

All agent modules import `call_llm()` from here — model/URL changes
only need to happen in config.py.
"""

from __future__ import annotations
from openai import OpenAI
from config import (
    LM_STUDIO_BASE_URL, LM_STUDIO_API_KEY,
    MODEL_NAME, MAX_TOKENS,
)

# ─── Client (module-level singleton) ─────────────────────────────────────────

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,   # LM Studio ignores this; any value works
        )
    return _client


# ─── Public API ───────────────────────────────────────────────────────────────

def call_llm(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL_NAME,
) -> str:
    """
    Call the local LM Studio server and return the assistant's text response.

    Args:
        messages:    List of {"role": "user"|"assistant", "content": str} dicts.
        system:      Optional system prompt — prepended as a {"role": "system"} message.
        temperature: Sampling temperature.
        max_tokens:  Maximum tokens in the response.
        model:       Model identifier (must match the name shown in LM Studio).

    Returns:
        The assistant's response as a plain string.
    """
    client = _get_client()

    # Build full message list: system goes first if provided
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


def call_llm_stream(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
    model: str = MODEL_NAME,
):
    """
    Same as call_llm() but yields token strings as they arrive from LM Studio.
    Use in a for-loop:
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


# ─── Connection test ──────────────────────────────────────────────────────────

def test_connection() -> bool:
    """
    Quick sanity-check: ping the local server with a trivial prompt.
    Run this before starting a long experiment to catch config issues early.
    """
    try:
        reply = call_llm(
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"[LM Studio] Connection OK — model replied: {reply!r}")
        return True
    except Exception as e:
        print(f"[LM Studio] Connection FAILED: {e}")
        print(f"  → Is LM Studio running at {LM_STUDIO_BASE_URL}?")
        print(f"  → Is the model '{MODEL_NAME}' loaded in LM Studio?")
        return False


if __name__ == "__main__":
    test_connection()