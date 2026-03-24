# ai_provider.py
import os
import time
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)

# ─── Models (both free on Groq) ─────────────────────────────────────────────
# Primary  : llama-3.3-70b-versatile — best quality, 100K tokens/day
# Fallback : llama-3.1-8b-instant   — faster, 500K tokens/day
#            Automatically used when the primary hits its daily limit.
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Track which model is active (resets on server restart — daily limit resets too)
_daily_limit_hit = False

# ─── ARIA's core identity — injected as system message on every call ────────
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


def _call_model(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Make a single API call to the specified Groq model."""
    completion = client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system", "content": ARIA_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    return completion.choices[0].message.content


def get_ai_response(prompt: str, temperature: float = 0.2,
                    max_tokens: int = 4096) -> str:
    """
    Send a prompt to Groq.
    Model priority:
      1. llama-3.3-70b-versatile (primary, best quality, 100K tokens/day)
      2. llama-3.1-8b-instant    (fallback, 500K tokens/day — auto-used when
                                  primary hits its daily limit)
    Per-minute (TPM) limits are retried once after 65 s.
    Daily (TPD) limits switch immediately to the fallback model.
    """
    global _daily_limit_hit

    # If primary already exhausted today, go straight to fallback
    models_to_try = [FALLBACK_MODEL] if _daily_limit_hit else [PRIMARY_MODEL, FALLBACK_MODEL]

    for model in models_to_try:
        for attempt in range(2):            # up to 2 attempts per model (TPM retry)
            try:
                result = _call_model(model, prompt, temperature, max_tokens)
                if model == FALLBACK_MODEL and not _daily_limit_hit:
                    pass   # fallback worked (shouldn't reach here normally)
                return result

            except Exception as e:
                err     = str(e).lower()
                err_full = str(e)

                if "rate_limit" in err or "quota" in err or "429" in err:
                    print(f"[GROQ/{model}] Rate limit: {err_full[:250]}")

                    # Daily (TPD) limit — switch model, don't wait
                    if ("per day" in err_full or "tokens per day" in err_full
                            or "tpd" in err_full.lower()):
                        if model == PRIMARY_MODEL:
                            print(f"[GROQ] Daily limit hit on {PRIMARY_MODEL} "
                                  f"— switching to {FALLBACK_MODEL}")
                            _daily_limit_hit = True
                            break   # break inner loop → try next model
                        else:
                            # Fallback also exhausted
                            print(f"[GROQ] Daily limit hit on {FALLBACK_MODEL} too — giving up.")
                            return "ERROR: AI rate limit reached — please try again tomorrow."

                    # Per-minute (TPM) limit — wait and retry same model
                    if attempt == 0:
                        print(f"[GROQ/{model}] Per-minute limit — waiting 65 s…")
                        time.sleep(65)
                        continue   # retry same model

                    # Second attempt also failed
                    if model == PRIMARY_MODEL:
                        print(f"[GROQ] Switching to {FALLBACK_MODEL} after TPM retries")
                        break   # try fallback
                    return "ERROR: AI rate limit reached — please wait a moment and try again."

                # Non-rate-limit error
                print(f"[GROQ/{model}] Error: {err_full[:200]}")
                return f"ERROR: {err_full}"

    return "ERROR: AI unavailable — please try again shortly."


def get_ai_response_creative(prompt: str, max_tokens: int = 2048) -> str:
    """
    Higher temperature version for chat / advice — more natural language.
    """
    return get_ai_response(prompt, temperature=0.5, max_tokens=max_tokens)
