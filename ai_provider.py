# ai_provider.py
import os
import io
import time
import asyncio
from groq import Groq
import edge_tts

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

STEP 4 — VOLATILITY, TIMING & SESSION INTELLIGENCE
  • ATR tells you HOW FAR price moves per candle — use it for SL/TP sizing.
  • Low ATR = compressed market, breakout coming. High ATR = active, wider stops needed.
  • Volume surge = institutional participation. Low volume = weak signal, wait.

  SESSION WINDOWS (UTC) — memorise these:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Asian Early   00:00–04:00 UTC │ Best: USDJPY, AUDJPY — range 30-60 pip  │
  │ Asian/London  04:00–07:00 UTC │ Best: JPY, AUD — build toward LO break  │
  │ London Open   07:00–10:00 UTC │ Best: EURUSD, GBPUSD — 60-100 pip range │
  │ London Mid    10:00–13:00 UTC │ Best: EUR/GBP crosses — H4 signals       │
  │ NY/LO Overlap 13:00–16:00 UTC │ PEAK: ALL pairs — 80-150 pip range ✓✓   │
  │ New York      16:00–21:00 UTC │ Best: USD pairs + crypto — H1 signals    │
  │ NY Close      21:00–00:00 UTC │ AVOID forex; crypto still trades         │
  └──────────────────────────────────────────────────────────────────────────┘

  TIMEFRAME SELECTION RULES:
  • M15 (15-min chart): Scalp only — London/NY Open first 30 min. Hold 15-60 min.
  • H1  (1-hour chart):  Standard intraday. All active sessions. Hold 2-8 hours.
  • H4  (4-hour chart):  Swing setups. Asian session or low-volatility windows. Hold 1-3 days.
  • D1  (daily chart):   Major swing trades. Enter on daily close/open. Hold 3-10 days.

  ENTRY WINDOW RULES (apply to every signal):
  • Always specify the UTC window when entry is valid (e.g., "Enter 09:00–09:30 UTC")
  • Entry deadline = latest time to enter (beyond this, wait for next session)
  • Time exit = if TP1 not reached by this UTC time, close for BE or small loss
  • London Open trades: deadline is 10:00 UTC. Time exit: 14:00 UTC.
  • NY Overlap trades: deadline is 15:00 UTC. Time exit: 17:00 UTC.
  • H4/D1 trades: entry within 4 hours. Time exit: next session close.

  EXPECTED TP1 TIMING (ATR-based rules of thumb):
  • H1 signal with 20-pip TP1: expect 1-3 hours in London, 45 min in Overlap
  • H4 signal with 60-pip TP1: expect 4-12 hours
  • D1 signal with 150-pip TP1: expect 1-3 days

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

━━━ BUDGET-AWARE PLANNING (critical rule — based on real lot-size math) ━━━

REAL MINIMUMS YOU MUST KNOW:
• Forex 0.01 micro lot on EURUSD with 20-pip SL = $2.00 ACTUAL risk
  → To risk only 2% of account: need $100 account (₦160,000 at current rate)
  → If someone has $30 and opens 0.01 lots with 20-pip SL, they risk 6.7% — DANGEROUS
• Crypto spot on Binance/Bybit: $1 minimum order
  → To risk only 2% per trade: need $50 account ($1 / 2% = $50)
• These are NOT opinions — these are mathematical facts about how forex lots work.

TIER RULES (apply these exactly):
Tier 1 — Under $5 (~₦8,000):
  → Demo ONLY. 2% = $0.10. No real exchange accepts this. Be direct about it.
  → Say: "Your 2% risk is $0.10 — no broker accepts this. Practice on demo first."
  → Give a demo plan. Tell them how much more to save.

Tier 2 — $5-$50 (~₦8,000-₦80,000):
  → Demo + crypto spot as savings vehicle (NOT active trading yet)
  → Can BUY fractional crypto to hold long-term, but not trade with SL/TP
  → Say: "You can hold BTC/ETH as investment. For active trading, save to $50."
  → Never recommend forex at this level.

Tier 3 — $50-$100 (~₦80,000-₦160,000):
  → Crypto spot ONLY with real money (2% = $1-2, enough for crypto orders)
  → Forex: demo only — 0.01 lots risks $2 minimum which is 2-4% of account
  → Say: "Crypto real money ✅. Forex still on demo until you hit $100."

Tier 4 — $100-$300 (~₦160,000-₦480,000):
  → Forex micro (0.01 lots) NOW viable — 2% = $2-6, covers minimum lot
  → Keep SL ≤ 20 pips to stay within 2% risk with 0.01 lots
  → Both forex and crypto real money ready

Tier 5 — $300-$1,000 (~₦480,000-₦1,600,000):
  → Comfortable forex micro trading, can scale to 0.02-0.05 lots
  → Full access to all 30 CLEO pairs

Tier 6 — $1,000+ (~₦1,600,000+):
  → Full institutional approach, standard lot sizing formula
  → Consider prop firm challenges (FTMO, MyFundedFX)

NEVER say "just use 2% and trade 0.01 lots" to someone with less than $100.
NEVER recommend forex to anyone with less than $100 in their account.
ALWAYS tell the user the EXACT minimum they need and how far they are from it.

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
    """Make a single API call (single-turn) to the specified Groq model."""
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


def _call_model_multiturn(
    model: str,
    messages: list,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Multi-turn chat call — receives full messages array with roles."""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    completion = client.chat.completions.create(
        model       = model,
        messages    = all_messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    return completion.choices[0].message.content


def _run_with_fallback(
    caller,          # callable(*args) → str
    *args,
    temperature: float = 0.45,
    max_tokens: int = 2048,
) -> str:
    """Shared retry / fallback logic for both single-turn and multi-turn calls."""
    global _daily_limit_hit

    models_to_try = [FALLBACK_MODEL] if _daily_limit_hit else [PRIMARY_MODEL, FALLBACK_MODEL]

    for model in models_to_try:
        for attempt in range(2):
            try:
                return caller(model, *args, temperature=temperature, max_tokens=max_tokens)

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


def get_ai_response(prompt: str, temperature: float = 0.15,
                    max_tokens: int = 4096) -> str:
    """
    Single-turn prompt → response.
    Model priority:
      1. llama-3.3-70b-versatile (primary, best quality, 100K tokens/day)
      2. llama-3.1-8b-instant    (fallback, 500K tokens/day)
    Temperature 0.15 = precise, consistent, low randomness for signals.
    """
    def caller(model, prompt, *, temperature, max_tokens):
        return _call_model(model, prompt, temperature, max_tokens)

    return _run_with_fallback(caller, prompt, temperature=temperature, max_tokens=max_tokens)


def get_ai_response_creative(prompt: str, max_tokens: int = 2048) -> str:
    """Higher temperature for chat — more natural, warm language."""
    return get_ai_response(prompt, temperature=0.45, max_tokens=max_tokens)


def get_ai_response_analytical(prompt: str, max_tokens: int = 3000) -> str:
    """Very low temperature for deep analytical tasks — maximum precision."""
    return get_ai_response(prompt, temperature=0.1, max_tokens=max_tokens)


def get_ai_response_chat(
    messages: list,
    system_prompt: str,
    max_tokens: int = 2048,
) -> str:
    """
    Proper multi-turn chat using Groq's native messages format.
    `messages` = list of {"role": "user"/"assistant", "content": str}
    This gives CLEO true in-session memory — it sees the full conversation
    as a native message array, not just text embedded in the system prompt.
    """
    def caller(model, messages, system_prompt, *, temperature, max_tokens):
        return _call_model_multiturn(model, messages, system_prompt, temperature, max_tokens)

    return _run_with_fallback(
        caller, messages, system_prompt,
        temperature=0.45, max_tokens=max_tokens,
    )


# ─── Voice: Speech-to-Text (Groq Whisper) ────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribe audio bytes to text using Groq Whisper large-v3.
    Accepts any format Whisper supports: webm, mp3, mp4, wav, ogg, m4a, flac.
    Returns the transcription string.
    """
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    transcription = client.audio.transcriptions.create(
        model = "whisper-large-v3",
        file  = audio_file,
        response_format = "text",
    )
    if isinstance(transcription, str):
        return transcription.strip()
    return getattr(transcription, "text", str(transcription)).strip()


# ─── Voice: Text-to-Speech (Microsoft Edge TTS — free, no API key) ───────────
#
# Uses Microsoft's Edge neural TTS voices via the edge-tts package.
# Quality is on par with Azure Cognitive Services TTS (same backend).
# No API key required. Returns MP3 bytes.

TTS_VOICES = {
    "cleo":   "en-US-AriaNeural",      # default CLEO voice — warm, professional female
    "female": "en-US-JennyNeural",     # friendly female
    "male":   "en-US-GuyNeural",       # clear male
    "warm":   "en-GB-SoniaNeural",     # British female — warm tone
    "deep":   "en-US-EricNeural",      # deeper male voice
}


async def text_to_speech(text: str, voice: str = "cleo") -> bytes:
    """
    Convert text to speech using Microsoft Edge neural TTS (via edge-tts).
    Returns raw MP3 bytes. No API key required. Must be awaited.

    voice: 'cleo' (default warm female) | 'female' | 'male' | 'warm' | 'deep'
           OR a raw Edge voice name like 'en-US-AriaNeural'.
    """
    voice_id   = TTS_VOICES.get(voice, voice)
    clean_text = text.strip()[:4000]
    if not clean_text:
        return b""
    communicate = edge_tts.Communicate(clean_text, voice_id)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()
