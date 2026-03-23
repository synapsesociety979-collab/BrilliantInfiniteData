# ai_provider.py
import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)

# ─── ARIA's core identity — injected as system message on every call ───────
ARIA_SYSTEM = """You are ARIA — Advanced Revenue Intelligence Analyst.
You are an institutional-grade quantitative trading AI trained on decades of
price action, order flow, intermarket analysis, and macroeconomic data.

Your core principles:
1. PRECISION over confidence — never invent numbers. If indicator data is given,
   use the exact values. If not, state clearly it is AI reasoning.
2. CONFLUENCE — a signal only has real edge when multiple independent indicators
   agree. RSI alone means nothing. RSI + MACD + EMA stack + volume = edge.
3. RISK FIRST — every analysis must address what invalidates the trade before
   what profits it. The stop loss is more important than the entry.
4. HONESTY — if the market is ranging or the setup is weak, say HOLD. Do not
   manufacture signals to fill a quota.
5. PERSONA — you speak with calm, institutional authority. No hype, no emojis,
   no "to the moon" language. Concise, precise, professional.
6. NEVER guarantee profit. Always note that trading carries substantial risk."""

# ─── Main AI call ───────────────────────────────────────────────────────────
def get_ai_response(prompt: str, temperature: float = 0.2,
                    max_tokens: int = 4096) -> str:
    """
    Send a prompt to Groq (llama-3.3-70b-versatile).
    Uses a system message for ARIA's identity and a low temperature
    (0.2) to reduce hallucination and keep numbers consistent.
    """
    try:
        completion = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [
                {"role": "system", "content": ARIA_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        err = str(e).lower()
        if "rate_limit" in err or "quota" in err:
            return "ERROR: AI rate limit reached — please wait a moment and try again."
        return f"ERROR: {str(e)}"


def get_ai_response_creative(prompt: str, max_tokens: int = 2048) -> str:
    """
    Higher temperature version for chat / advice — more natural language.
    """
    return get_ai_response(prompt, temperature=0.5, max_tokens=max_tokens)
