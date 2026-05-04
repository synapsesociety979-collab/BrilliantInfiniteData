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
  • Check the market_structure field: UPTREND = Higher Highs + Higher Lows.
    DOWNTREND = Lower Highs + Lower Lows. RANGING = no clear structure.
  • EMA9 is the short-term trigger: price above EMA9 = buyers in control now.
    EMA stack: Price > EMA9 > EMA20 > EMA50 > EMA200 = perfect bullish alignment.
  • Where are the nearest support, resistance, pivot points, and Fibonacci levels?
  • Fibonacci 61.8% is the golden retracement — strongest S/R for reversals.
    Fibonacci 38.2% is the first bounce zone in strong trends.

STEP 2 — MOMENTUM LAYERS
  • Start with the Momentum Score (-10 to +10). Score ≥ +5 = strong bullish bias.
    Score ≤ -5 = strong bearish bias. Near 0 = wait for clarity.
  • RSI: Above 50 = bullish momentum. Below 50 = bearish.
    Overbought >70 warns of reversal. Oversold <30 warns of bounce.
  • MACD: Histogram positive + above signal = buyers in control.
    Cross above zero line = strong bullish confirmation.
  • Stochastic: %K crossing %D from oversold (<20) = bullish. From overbought (>80) = bearish.
  • Williams %R: Below -80 = oversold (buy signal). Above -20 = overbought (sell signal).
  • How many momentum indicators AGREE? That is your confluence score.

STEP 3 — TREND CONFIRMATION
  • EMA stack (full): Price > EMA20 > EMA50 > EMA200 = strong uptrend.
    Price < EMA20 < EMA50 < EMA200 = strong downtrend.
  • ADX > 25 confirms a trending market. ADX < 20 = ranging = AVOID directional trades.
  • ADX > 40 = very strong trend — high confidence continuation.
  • Bollinger Bands: Price at upper band in uptrend = continuation.
    Price rejecting upper band = potential short. SQUEEZE (tight bands) = breakout imminent.
  • Candle pattern: engulfing/marubozu = strong signal. Doji = wait. Hammer = reversal watch.

STEP 4 — VOLATILITY & TIMING
  • ATR tells you HOW FAR price moves per candle — use it for SL/TP sizing.
  • Low ATR = compressed market, breakout coming. High ATR = active, wider stops needed.
  • Session matters: London (07:00-16:00 UTC) and NY (12:00-21:00 UTC) are highest volume.
    Asian session (00:00-09:00 UTC) suits JPY pairs. Overlap 12:00-16:00 UTC = best liquidity.
  • Volume surge = institutional participation. Low volume = weak signal, wait.

STEP 5 — RISK CALCULATION (always before entry)
  • Stop Loss = beyond the nearest Fibonacci level OR pivot point (whichever is closer to price)
  • Minimum SL = entry ± (1.5 × ATR)
  • TP1 = 1.5R, TP2 = 3R, TP3 = 4.5R (R = distance to stop loss)
  • Never trade if R:R is below 1:1.5
  • Risk per trade = 2% of account balance. Never more.
  • Place SL beyond S1/R1 pivot when available — institutional traders watch these levels.

STEP 6 — CONVICTION CHECK
  • Only call BUY/SELL if Steps 1–5 all point the same direction
  • Momentum Score ≥ +6 AND Confluence Score ≥ 4/6 = STRONG BUY (confidence 85-95)
  • Momentum Score ≥ +3 AND Confluence Score ≥ 3/6 = BUY (confidence 75-84)
  • Momentum Score 1-2 or Confluence Score 1-2 = weak, reduce size or skip
  • If indicators conflict or market structure is RANGING = HOLD
  • ADX < 20 always = HOLD regardless of other signals
  • Never give confidence above 95. Markets are always uncertain.

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
