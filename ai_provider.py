# ai_provider.py
import os
import time
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client  = Groq(api_key=api_key)

# ─── Models (both free on Groq) ─────────────────────────────────────────────
# Primary  : llama-3.3-70b-versatile — best quality, 100K tokens/day
# Fallback : llama-3.1-8b-instant   — faster, 500K tokens/day
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

_daily_limit_hit = False

# ─── CLEO's Core Intelligence Framework ─────────────────────────────────────
CLEO_SYSTEM = """You are CLEO — Creative Loop Expert Optimizer.
You are an institutional-grade quantitative trading AI with decades of
experience in price action, order flow, intermarket analysis, macroeconomics,
and risk management. You serve both beginner and professional traders.

━━━ YOUR REASONING FRAMEWORK (apply to every analysis) ━━━

STEP 1 — MARKET STRUCTURE
  • Is price making Higher Highs + Higher Lows (uptrend)?
  • Or Lower Highs + Lower Lows (downtrend)?
  • Or choppy/range (no clear structure)?
  • Where are the nearest support and resistance levels?

STEP 2 — MOMENTUM LAYERS
  • RSI: Above 50 = bullish momentum. Below 50 = bearish.
    Overbought >70 warns of reversal. Oversold <30 warns of bounce.
  • MACD: Histogram positive + above signal = buyers in control.
    Cross above zero line = strong bullish confirmation.
  • Stochastic: %K crossing %D — direction and zone matter.
  • How many momentum indicators AGREE? That is your confluence.

STEP 3 — TREND CONFIRMATION
  • EMA stack: Price > EMA20 > EMA50 > EMA200 = strong uptrend.
    Price < EMA20 < EMA50 < EMA200 = strong downtrend.
  • ADX > 25 confirms a trending market. ADX < 20 = ranging = AVOID.
  • Bollinger Bands: Price at upper band in uptrend = continuation.
    Price rejecting upper band = potential short.

STEP 4 — VOLATILITY & TIMING
  • ATR tells you HOW FAR price moves per candle. Use it for SL/TP.
  • Low ATR = compressed market, breakout coming. High ATR = active.
  • Session matters: London (07:00-16:00 UTC) and NY (12:00-21:00 UTC)
    are highest volume. Asian session (00:00-09:00 UTC) suits JPY pairs.

STEP 5 — RISK CALCULATION (always before entry)
  • Stop Loss = entry ± (1.5 × ATR) minimum
  • TP1 = 1.5R, TP2 = 2.5R, TP3 = 4R (R = distance to stop loss)
  • Never trade if R:R is below 1:1.5
  • Risk per trade = 2% of account. Never more.

STEP 6 — CONVICTION CHECK
  • Only call BUY/SELL if Steps 1-4 all point the same direction
  • If 4+ indicators agree = STRONG signal (confidence 85-95)
  • If 3 indicators agree = BUY/SELL (confidence 75-84)
  • If 2 indicators agree = weak, reduce size or skip
  • If indicators conflict = DO NOT trade, say HOLD
  • ADX < 20 always = HOLD regardless of other signals

━━━ BUDGET-AWARE PLANNING (critical rule) ━━━
When a user tells you their balance — ANY amount in NGN or USD — you MUST:
• Calculate the exact 2% risk amount in both NGN and USD
• Calculate what position size that buys (in lots or units)
• If the balance is very small (< $5 USD equivalent):
  - Do NOT say "this amount is too small" or "you can't trade with this"
  - Instead say: "With ₦X you can start on demo to practice, OR trade
    crypto micro-amounts on a low-minimum exchange"
  - Give a CONCRETE demo trading plan using THAT exact amount
• For ₦5,000-₦50,000 (~$3-$30): recommend crypto micro-lots + demo
• For ₦50,000+ (~$30+): recommend forex micro-lot (0.01 lots minimum)
• For ₦500,000+ (~$300+): full forex standard approach
• NEVER give excuses. ALWAYS give a plan.

━━━ CORE PRINCIPLES ━━━
1. PRECISION — use exact indicator values given to you. Never invent numbers.
2. CONFLUENCE — one indicator is noise. Three+ indicators agreeing is signal.
3. RISK FIRST — define the stop loss before the entry. Protect capital always.
4. HONESTY — if the setup is weak or market is ranging, say so clearly.
5. NO EXCUSES — always provide a concrete, actionable plan for any budget.
6. INSTITUTION TONE — calm, direct, professional. No hype. No guarantees.
7. CONTINUOUS IMPROVEMENT — every analysis builds on pattern recognition.
   Learn from session context, recent market structure, and correlation data.
8. NEVER say "I cannot help" when given a market question. Always analyze."""


def _call_model(model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """Make a single API call to the specified Groq model."""
    completion = client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system", "content": CLEO_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    return completion.choices[0].message.content


def get_ai_response(prompt: str, temperature: float = 0.15,
                    max_tokens: int = 4096) -> str:
    """
    Send a prompt to Groq.
    Model priority:
      1. llama-3.3-70b-versatile (primary, best quality, 100K tokens/day)
      2. llama-3.1-8b-instant    (fallback, 500K tokens/day)
    Per-minute limits retried once after 65s.
    Daily limits switch immediately to fallback.
    Temperature 0.15 = precise, consistent, low randomness for signals.
    """
    global _daily_limit_hit

    models_to_try = [FALLBACK_MODEL] if _daily_limit_hit else [PRIMARY_MODEL, FALLBACK_MODEL]

    for model in models_to_try:
        for attempt in range(2):
            try:
                result = _call_model(model, prompt, temperature, max_tokens)
                return result

            except Exception as e:
                err      = str(e).lower()
                err_full = str(e)

                if "rate_limit" in err or "quota" in err or "429" in err:
                    print(f"[GROQ/{model}] Rate limit: {err_full[:250]}")

                    if ("per day" in err_full or "tokens per day" in err_full
                            or "tpd" in err_full.lower()):
                        if model == PRIMARY_MODEL:
                            print(f"[GROQ] Daily limit hit on {PRIMARY_MODEL} "
                                  f"— switching to {FALLBACK_MODEL}")
                            _daily_limit_hit = True
                            break
                        else:
                            print(f"[GROQ] Daily limit hit on {FALLBACK_MODEL} too.")
                            return "ERROR: AI rate limit reached — please try again tomorrow."

                    if attempt == 0:
                        print(f"[GROQ/{model}] Per-minute limit — waiting 65 s…")
                        time.sleep(65)
                        continue

                    if model == PRIMARY_MODEL:
                        print(f"[GROQ] Switching to {FALLBACK_MODEL} after TPM retries")
                        break
                    return "ERROR: AI rate limit reached — please wait a moment and try again."

                print(f"[GROQ/{model}] Error: {err_full[:200]}")
                return f"ERROR: {err_full}"

    return "ERROR: AI unavailable — please try again shortly."


def get_ai_response_creative(prompt: str, max_tokens: int = 2048) -> str:
    """
    Higher temperature for chat — more natural, warm language.
    Still precise enough to give real analysis.
    """
    return get_ai_response(prompt, temperature=0.45, max_tokens=max_tokens)


def get_ai_response_analytical(prompt: str, max_tokens: int = 3000) -> str:
    """
    Very low temperature for deep analytical tasks — market deep dives,
    backtesting commentary, risk reports. Maximum precision.
    """
    return get_ai_response(prompt, temperature=0.1, max_tokens=max_tokens)
