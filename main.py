# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, time, requests, random, math
from datetime import datetime, timedelta
from ai_provider import get_ai_response, get_ai_response_creative
from sqlalchemy.orm import Session

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import (
    load_history_df,
    run_backtest_from_signals,
    signals_from_sma,
    router as backtest_router,
)
from models import (
    init_db,
    get_db,
    User,
    DemoAccount,
    DemoTrade,
    ChatMessage as DBChatMessage,
    TradeJournalEntry as DBJournalEntry,
    WatchlistItem,
    UserActivity,
    Conversation,
    UserMemory,
    BotConfig,
    BotOrder,
)
from trading_engine import RiskEngine, MarketFilter, TradeManager
from market_data import (
    get_symbol_analysis, format_for_ai_prompt, fetch_realtime_quote,
    FALLBACK_PRICES, FALLBACK_ATR_PCT,
)

# ----------------------------
# DB init on startup
# ----------------------------
init_db()

# ----------------------------
# Alpha Vantage Integration
# ----------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_alpha_vantage_forex(from_symbol: str, to_symbol: str = "USD") -> Dict:
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_symbol}&to_currency={to_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get("Realtime Currency Exchange Rate", {})
        if data:
            return {
                "rate": float(data.get("5. Exchange Rate") or 0),
                "bid": float(data.get("8. Bid Price") or 0),
                "ask": float(data.get("9. Ask Price") or 0),
                "last_refreshed": data.get("6. Last Refreshed"),
            }
    except Exception as e:
        print(f"AV Forex Error: {e}")
    return {}


def fetch_alpha_vantage_crypto(symbol: str) -> Dict:
    base = symbol.replace("USDT", "").replace("BUSD", "")
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get("Realtime Currency Exchange Rate", {})
        if data:
            return {
                "price": float(data.get("5. Exchange Rate") or 0),
                "last_refreshed": data.get("6. Last Refreshed"),
            }
    except Exception as e:
        print(f"AV Crypto Error: {e}")
    return {}


def get_live_price(symbol: str) -> float:
    # First try to get price from full candle analysis (already cached)
    analysis = get_symbol_analysis(symbol)
    if analysis.get("live_price"):
        return float(analysis["live_price"])
    # Fallback to real-time quote
    if "USDT" in symbol or "BUSD" in symbol:
        base = symbol.replace("USDT", "").replace("BUSD", "")
        d = fetch_realtime_quote(base, "USD")
    else:
        d = fetch_realtime_quote(symbol[:3], symbol[3:] or "USD")
    return float(d.get("price") or 0)


def get_ngn_rate() -> float:
    d = fetch_alpha_vantage_forex("USD", "NGN")
    return float(d.get("rate") or 1600.0)


def get_market_context() -> str:
    context = []
    for f in ["EUR", "GBP", "JPY", "AUD", "CAD"]:
        d = fetch_alpha_vantage_forex(f)
        if d.get("rate"):
            context.append(f"{f}/USD={d['rate']:.5f}")
    for c in ["BTC", "ETH", "SOL", "BNB"]:
        d = fetch_alpha_vantage_crypto(c)
        if d.get("price"):
            context.append(f"{c}=${d['price']:,.2f}")
    return " | ".join(context) if context else "Live data unavailable"


# ----------------------------
# Cache
# ----------------------------
PREDICTIONS_CACHE: Dict[str, Any] = {}
ADVICE_CACHE: Dict[str, Any] = {}
CACHE_TTL = 600

# ----------------------------
# Symbol config
# ----------------------------
TRADING_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "EURCAD",
    "AUDCAD",
    "EURAUD",
    "GBPAUD",
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "SHIBUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "UNIUSDT",
]

SYMBOL_BASE_PRICES = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2700,
    "USDJPY": 149.50,
    "AUDUSD": 0.6530,
    "USDCAD": 1.3600,
    "NZDUSD": 0.6050,
    "USDCHF": 0.8950,
    "EURGBP": 0.8550,
    "EURJPY": 162.0,
    "GBPJPY": 189.5,
    "AUDJPY": 97.5,
    "EURCAD": 1.4750,
    "AUDCAD": 0.8880,
    "EURAUD": 1.6600,
    "GBPAUD": 1.9450,
    "BTCUSDT": 82000.0,
    "ETHUSDT": 1900.0,
    "BNBUSDT": 580.0,
    "XRPUSDT": 2.1,
    "SOLUSDT": 130.0,
    "ADAUSDT": 0.75,
    "DOGEUSDT": 0.16,
    "DOTUSDT": 6.5,
    "MATICUSDT": 0.55,
    "LTCUSDT": 95.0,
    "SHIBUSDT": 0.000013,
    "TRXUSDT": 0.22,
    "AVAXUSDT": 25.0,
    "LINKUSDT": 13.5,
    "UNIUSDT": 8.5,
}


# ----------------------------
# DB helpers
# ----------------------------
def get_or_create_user(username: str, db: Session) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_active = datetime.utcnow()
        db.commit()
    return user


def log_activity(
    user: User,
    action: str,
    symbol: str = None,
    details: dict = None,
    db: Session = None,
):
    if db:
        entry = UserActivity(
            user_id=user.id, action=action, symbol=symbol, details=details
        )
        db.add(entry)
        db.commit()


def get_user_profile_summary(user: User, db: Session) -> str:
    """Build a text summary of user behavior for AI personalization."""
    journal = (
        db.query(DBJournalEntry)
        .filter(DBJournalEntry.user_id == user.id)
        .order_by(DBJournalEntry.logged_at.desc())
        .limit(20)
        .all()
    )
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    activity = (
        db.query(UserActivity)
        .filter(UserActivity.user_id == user.id)
        .order_by(UserActivity.created_at.desc())
        .limit(30)
        .all()
    )

    wins = [j for j in journal if j.result == "WIN"]
    losses = [j for j in journal if j.result == "LOSS"]
    win_rate = (len(wins) / len(journal) * 100) if journal else 0
    most_traded = {}
    for j in journal:
        most_traded[j.symbol] = most_traded.get(j.symbol, 0) + 1
    top_symbol = max(most_traded, key=most_traded.get) if most_traded else "N/A"
    recent_actions = [f"{a.action} {a.symbol or ''}" for a in activity[:10]]
    watched = [w.symbol for w in watchlist]

    return (
        f"Username: {user.username} | Balance: {user.balance_ngn:,.0f} NGN | "
        f"Risk Tolerance: {user.risk_tolerance} | Style: {user.trading_style} | "
        f"Preferred Pairs: {user.preferred_pairs or watched or ['any']} | "
        f"Total Trades: {len(journal)} | Win Rate: {win_rate:.1f}% | Top Symbol: {top_symbol} | "
        f"Recent Activity: {', '.join(recent_actions[:5])}"
    )


# ----------------------------
# AI — Enhanced Predictions
# ----------------------------
def _get_session_priority_symbols(h: int) -> List[str]:
    """Return symbols ordered by current market session — most active first."""
    if 7 <= h < 12:
        priority = [
            "EURUSD",
            "GBPUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
            "EURCAD",
            "EURAUD",
            "GBPAUD",
            "BTCUSDT",
            "ETHUSDT",
            "USDCHF",
            "AUDUSD",
            "USDJPY",
            "SOLUSDT",
            "BNBUSDT",
        ]
    elif 12 <= h < 17:
        priority = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "USDCAD",
            "USDCHF",
            "EURGBP",
            "GBPJPY",
            "EURJPY",
            "XRPUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "LINKUSDT",
        ]
    elif 17 <= h < 21:
        priority = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "LINKUSDT",
            "UNIUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "EURUSD",
            "GBPUSD",
            "USDCAD",
            "MATICUSDT",
            "TRXUSDT",
        ]
    else:
        priority = [
            "USDJPY",
            "GBPJPY",
            "AUDJPY",
            "EURJPY",
            "AUDCAD",
            "AUDUSD",
            "NZDUSD",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LTCUSDT",
            "XRPUSDT",
            "SHIBUSDT",
            "TRXUSDT",
        ]
    # Add remaining symbols not already in list
    remaining = [s for s in TRADING_SYMBOLS if s not in priority]
    return priority + remaining


def generate_market_predictions(
    investment_amount_ngn: float = 0.0, user_profile: str = ""
) -> Dict[str, Any]:
    global PREDICTIONS_CACHE
    now = time.time()
    cache_key = f"preds_{int(investment_amount_ngn)}"
    if cache_key in PREDICTIONS_CACHE and (
        now - PREDICTIONS_CACHE[cache_key]["ts"] < CACHE_TTL
    ):
        return PREDICTIONS_CACHE[cache_key]["data"]

    ngn_rate = get_ngn_rate()
    inv_usd = investment_amount_ngn / ngn_rate if investment_amount_ngn > 0 else 0
    utc = datetime.utcnow()
    h = utc.hour

    if 7 <= h < 12:
        session = "London Session — EURUSD, GBPUSD, EURGBP most active"
    elif 12 <= h < 17:
        session = "London/NY Overlap — PEAK liquidity, all pairs active"
    elif 17 <= h < 21:
        session = "New York Session — USD pairs and crypto most active"
    else:
        session = "Asian Session — JPY pairs: USDJPY, GBPJPY, AUDJPY most active"

    personalization = f"\nUSER PROFILE: {user_profile}" if user_profile else ""

    # Fetch REAL live data for session-priority symbols (up to 12 to respect rate limits)
    priority_symbols = _get_session_priority_symbols(h)
    real_data_symbols = priority_symbols[:12]
    live_data_blocks = []
    has_real_data = False

    print(f"[ARIA] Fetching real market data for {len(real_data_symbols)} symbols...")
    analyses_by_symbol: Dict[str, Dict] = {}   # ← store for post-AI validation
    for sym in real_data_symbols:
        analysis = get_symbol_analysis(sym)
        block = format_for_ai_prompt(analysis)
        live_data_blocks.append(block)
        analyses_by_symbol[sym] = analysis
        if analysis.get("live_price"):
            has_real_data = True

    # Remaining symbols — build price + ATR context so AI can do real analysis
    remaining_symbols = [s for s in TRADING_SYMBOLS if s not in real_data_symbols]
    remaining_lines = []
    for sym in remaining_symbols:
        ref_price = FALLBACK_PRICES.get(sym, 0)
        atr_pct   = FALLBACK_ATR_PCT.get(sym, 0.01)
        atr_abs   = round(ref_price * atr_pct, 6)
        sl_buy    = round(ref_price - atr_abs * 1.5, 6)
        tp1_buy   = round(ref_price + atr_abs * 1.5, 6)
        tp2_buy   = round(ref_price + atr_abs * 3.0, 6)
        sl_sell   = round(ref_price + atr_abs * 1.5, 6)
        tp1_sell  = round(ref_price - atr_abs * 1.5, 6)
        tp2_sell  = round(ref_price - atr_abs * 3.0, 6)
        cat = "crypto" if "USDT" in sym else "forex"
        remaining_lines.append(
            f"{sym} ({cat}) | ref_price={ref_price} | ATR≈{atr_abs} | "
            f"BUY SL={sl_buy} TP1={tp1_buy} TP2={tp2_buy} | "
            f"SELL SL={sl_sell} TP1={tp1_sell} TP2={tp2_sell}"
        )
    remaining_text = "\n".join(remaining_lines)

    live_data_section = "\n\n".join(live_data_blocks)
    data_source_note = (
        "LIVE Alpha Vantage data"
        if has_real_data
        else "AI pattern reasoning (live data unavailable)"
    )

    prompt = f"""SESSION: {session} | UTC: {utc.strftime("%H:%M")} | Data: {data_source_note}
Budget: {investment_amount_ngn:,.0f} NGN (~${inv_usd:,.2f} USD @ {ngn_rate:.2f}){personalization}

GOAL: Analyze ALL 30 symbols below and return EVERY symbol that has a tradeable signal.
Target: 5-12 signals. No artificial limit. If a clear setup exists, include it.

━━━ SECTION A — LIVE INDICATOR DATA (use exact values shown) ━━━
{"No live data available this cycle — see Section B for all symbols." if not has_real_data else ""}
{live_data_section if has_real_data else ""}

SECTION A GATES (apply ONLY to symbols above with live data):
✗ Confluence score 0-2/6 → exclude
✗ ADX < 20 → exclude (ranging market)
✗ MACD histogram and RSI point opposite directions → exclude
✗ R:R below 1:1.8 → exclude
Confidence: score 5-6/6→88-97 | 4/6→80-87 | 3/6→73-79

━━━ SECTION B — AI REASONING (reference prices + ATR provided) ━━━
For these symbols you do NOT have live candle data. Use your institutional knowledge
of each asset's recent price action, macro drivers, and intermarket relationships.
SL/TP levels below are pre-calculated from estimated ATR — use them.

{remaining_text if remaining_text else "(all symbols have live data)"}

SECTION B GATES (apply to all Section B symbols + any Section A symbols that lost live data):
✗ Skip only if RSI, MACD, AND EMA all point different directions (total disagreement)
✗ R:R below 1:1.5 → exclude
✓ If at least 2 of (RSI trend, MACD direction, EMA stack, price vs key level) agree → include
Confidence cap: 78 max for AI reasoning
Confluence: estimate as "X/3" (out of RSI/MACD/EMA only, since no other indicators)

━━━ SIGNAL RULES (both sections) ━━━
- data_source = "live_data" for Section A, "ai_reasoning" for Section B
- Entry = current/reference price, or a slightly better pullback level
- Use ATR SL/TP levels provided — do not invent different numbers
- back_out_trigger = price level that invalidates the thesis
- Session fit: {session.split('—')[0].strip()}

POSITION SIZING: 2% risk per trade
Risk: {investment_amount_ngn * 0.02:,.0f} NGN = ${investment_amount_ngn * 0.02 / ngn_rate:.2f} USD

Return ONLY a valid JSON array. No HOLD signals. No text outside brackets.
[
  {{
    "symbol": "SYMBOL",
    "signal": "STRONG_BUY|BUY|SELL|STRONG_SELL",
    "confidence": 73-97,
    "confluence_score": "X/6 or X/3",
    "data_source": "live_data|ai_reasoning",
    "category": "forex|crypto",
    "timeframe": "scalp(5-15m)|intraday(1-4h)|swing(daily)",
    "session_fit": "excellent|good|fair|poor",
    "live_price": "number",
    "entry_price": "number",
    "stop_loss": "number",
    "take_profit_1": "number",
    "take_profit_2": "number",
    "take_profit_3": "number",
    "risk_reward": "1:X.X",
    "hold_time": "duration",
    "position_size_ngn": "{investment_amount_ngn * 0.02:,.0f} NGN",
    "position_size_usd": "${investment_amount_ngn * 0.02 / ngn_rate:.2f}",
    "back_out_trigger": "invalidation price",
    "indicators": {{
      "rsi": "value/estimate + reading",
      "macd": "direction + histogram",
      "stochastic": "if available",
      "adx": "if available",
      "williams_r": "if available",
      "ema_bias": "price vs EMAs",
      "bollinger": "band position",
      "pattern": "pattern if visible"
    }},
    "key_levels": {{
      "support": "level",
      "resistance": "level",
      "pivot": "if available"
    }},
    "rationale": "2-3 sentences: cite specific indicators/macro drivers, entry logic, and main risk."
  }}
]"""

    try:
        content = get_ai_response(prompt)
        if not content or content.startswith("ERROR:"):
            return {"success": False, "error": content or "Empty response"}
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)

        # ── PYTHON-LEVEL QUALITY GATES (hard rules AI cannot override) ──────────
        validated = []
        for s in data:
            if not isinstance(s, dict):
                continue
            sym = s.get("symbol", "")
            sig = s.get("signal", "")
            conf = int(s.get("confidence", 0))
            src  = s.get("data_source", "ai_reasoning")

            # Gate 1: minimum confidence
            if conf < 73:
                continue

            # Gate 2: HOLD signals excluded
            if sig == "HOLD":
                continue

            # Gate 3: live-data signals must pass Python confluence check
            if src == "live_data" and sym in analyses_by_symbol:
                calc = analyses_by_symbol[sym].get("confluence", {})
                calc_dir    = calc.get("direction", "NEUTRAL")   # BUY/SELL/NEUTRAL
                calc_score  = int(calc.get("score", 0))
                calc_tradeable = calc.get("tradeable", True)

                # Must have at least 3 indicators agreeing
                if calc_score < 3:
                    print(f"[GATE] Rejected {sym} {sig}: confluence {calc_score}/6 < 3")
                    continue

                # Signal direction must match calculated confluence direction
                is_buy  = sig in ("BUY", "STRONG_BUY")
                is_sell = sig in ("SELL", "STRONG_SELL")
                if is_buy  and calc_dir == "SELL":
                    print(f"[GATE] Rejected {sym} {sig}: AI says BUY but Python says SELL")
                    continue
                if is_sell and calc_dir == "BUY":
                    print(f"[GATE] Rejected {sym} {sig}: AI says SELL but Python says BUY")
                    continue
                if calc_dir == "NEUTRAL" and calc_score < 3:
                    print(f"[GATE] Rejected {sym} {sig}: NEUTRAL market (score={calc_score})")
                    continue

                # ADX gate: market must be trending (not ranging)
                if not calc_tradeable:
                    print(f"[GATE] Rejected {sym} {sig}: ADX <20 (ranging market)")
                    continue

                # Confidence must match confluence score
                if calc_score <= 3 and conf > 80:
                    s["confidence"] = 76   # cap overconfident AI signals
                elif calc_score == 4 and conf > 87:
                    s["confidence"] = 85
                elif calc_score >= 5 and conf > 97:
                    s["confidence"] = 95

                # Stamp the verified confluence score on the signal
                s["confluence_score"] = f"{calc_score}/6"
                s["confluence_direction"] = calc_dir

            elif src == "ai_reasoning":
                # Cap AI-only signals at 78%
                if conf > 78:
                    s["confidence"] = 78
                # Normalise confluence_score string (AI sometimes returns "4/3" etc.)
                raw_cs = str(s.get("confluence_score", "0/3"))
                try:
                    parts = raw_cs.split("/")
                    num   = int(parts[0])
                    denom = int(parts[1]) if len(parts) > 1 else 3
                    num   = min(num, denom)          # clamp (4/3 → 3/3)
                    s["confluence_score"] = f"{num}/{denom}"
                except Exception:
                    s["confluence_score"] = raw_cs

            validated.append(s)

        data = validated
        # ── END PYTHON GATES ────────────────────────────────────────────────────

        strong = [s for s in data if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]]
        live_count = len([s for s in data if s.get("data_source") == "live_data"])

        result = {
            "success": True,
            "generated_at": utc.isoformat(),
            "active_session": session,
            "data_quality": {
                "source": data_source_note,
                "live_data_symbols": real_data_symbols,
                "signals_from_live_data": live_count,
                "signals_from_ai_reasoning": len(data) - live_count,
            },
            "investment_context_ngn": investment_amount_ngn,
            "usd_ngn_rate": ngn_rate,
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data),
            "strong_signals_count": len(strong),
            "top_picks": strong[:5],
            "disclaimer": "Signals are data-driven using live Alpha Vantage feeds. Trading involves substantial risk.",
        }
        PREDICTIONS_CACHE[cache_key] = {"data": result, "ts": now}
        return result
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}


# ----------------------------
# Personalized Signal for 1 Symbol (FIXED)
# ----------------------------
def get_personalized_signal(symbol: str, user: User, db: Session) -> Dict:
    """Generate a focused AI signal for a single symbol using REAL live data + user personalization."""
    symbol = symbol.upper()
    ngn_rate = get_ngn_rate()
    inv_usd = user.balance_ngn / ngn_rate if user.balance_ngn > 0 else 0
    profile = get_user_profile_summary(user, db)
    utc = datetime.utcnow()

    # Fetch real live data first
    analysis = get_symbol_analysis(symbol)
    live_data_block = format_for_ai_prompt(analysis)
    live_price = analysis.get("live_price", 0)
    has_live = bool(live_price)

    prompt = f"""━━━ MARKET DATA FOR {symbol} ━━━
{live_data_block}

━━━ USER CONTEXT ━━━
{profile}
Balance: {user.balance_ngn:,.0f} NGN (~${inv_usd:,.2f} USD)
Style: {user.trading_style} | Risk: {user.risk_tolerance}
UTC: {utc.strftime("%Y-%m-%d %H:%M")}
Position size (2% risk): {user.balance_ngn * 0.02:,.0f} NGN = ${user.balance_ngn * 0.02 / ngn_rate:.2f} USD

━━━ CONFLUENCE-BASED CONFIDENCE ━━━
{"Use the confluence score from the data block to calibrate confidence:" if has_live else "AI reasoning mode — cap confidence at 78."}
{"Score 5-6/6 → 88-97 (STRONG) | Score 4/6 → 80-87 | Score 3/6 → 73-79 | Score 0-2 → HOLD" if has_live else ""}

━━━ QUALITY GATES (fail any = return HOLD) ━━━
{"✗ ADX < 20 → HOLD (ranging) | ✗ MACD/RSI conflict → HOLD | ✗ Stoch/Williams disagree → HOLD | ✗ R:R < 1:1.8 → HOLD" if has_live else ""}

━━━ PERSONALISATION ━━━
- Timeframe must match {user.trading_style} style
- SL distance: {"tighter (0.8–1.0×ATR)" if user.risk_tolerance == "low" else ("standard (1.5×ATR)" if user.risk_tolerance == "medium" else "wider (2.0×ATR)")} for {user.risk_tolerance} risk
- {"Use EXACT indicator values — do not invent numbers." if has_live else "Conservative analysis only — no live data."}

Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0-97,
  "confluence_score": "X/6 or N/A",
  "data_source": "{"live_data" if has_live else "ai_reasoning"}",
  "category": "forex|crypto",
  "timeframe": "scalp|intraday|swing",
  "session_fit": "excellent|good|fair|poor",
  "live_price": "{live_price if live_price else "unavailable"}",
  "entry_price": "exact number",
  "stop_loss": "exact number (ATR-based)",
  "take_profit_1": "exact number",
  "take_profit_2": "exact number",
  "take_profit_3": "exact number",
  "risk_reward": "1:X.X",
  "hold_time": "duration for {user.trading_style}",
  "position_size_ngn": "{user.balance_ngn * 0.02:,.0f} NGN",
  "position_size_usd": "${user.balance_ngn * 0.02 / ngn_rate:.2f} USD",
  "back_out_trigger": "exact invalidation price",
  "indicators": {{
    "rsi": "exact value + zone",
    "macd": "histogram value + direction",
    "stochastic": "K/D + zone",
    "adx": "value + trend strength",
    "williams_r": "value + zone",
    "ema_bias": "price vs EMA20/50/200",
    "bollinger": "band position",
    "pattern": "detected pattern"
  }},
  "key_levels": {{
    "support": "exact level",
    "resistance": "exact level",
    "pivot": "pivot level if available"
  }},
  "personalized_note": "1 sentence addressing this user's specific style, risk, and history",
  "rationale": "3 sentences: (1) what the confluence/indicators say, (2) why this entry is valid, (3) what the risk is"
}}"""

    try:
        content = get_ai_response(prompt)
        if not content or content.startswith("ERROR:"):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "error": "AI unavailable",
                "live_price": live_price,
            }
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)

        # ── Python-level quality gate for single-symbol ──────────────────────
        if has_live:
            calc   = analysis.get("confluence", {})
            c_dir  = calc.get("direction", "NEUTRAL")
            c_score = int(calc.get("score", 0))
            c_trade = calc.get("tradeable", True)
            sig    = data.get("signal", "HOLD")
            is_buy  = sig in ("BUY", "STRONG_BUY")
            is_sell = sig in ("SELL", "STRONG_SELL")

            gate_fail = None
            if c_score < 3:
                gate_fail = f"confluence {c_score}/6 below minimum (need ≥3)"
            elif is_buy and c_dir == "SELL":
                gate_fail = "AI BUY conflicts with calculated SELL confluence"
            elif is_sell and c_dir == "BUY":
                gate_fail = "AI SELL conflicts with calculated BUY confluence"
            elif not c_trade:
                gate_fail = "ADX <20 — ranging market, no directional trade"

            if gate_fail:
                data["signal"]     = "HOLD"
                data["confidence"] = 0
                data["gate_rejection"] = gate_fail
                data["confluence_score"] = f"{c_score}/6"
                data["note"] = "Overridden to HOLD by Python quality gate. Wait for clearer setup."
            else:
                # Stamp verified confluence data
                data["confluence_score"]     = f"{c_score}/6"
                data["confluence_direction"] = c_dir
                # Cap confidence to match confluence score
                conf = int(data.get("confidence", 0))
                if c_score <= 3 and conf > 80:   data["confidence"] = 76
                elif c_score == 4 and conf > 87: data["confidence"] = 85
                elif c_score >= 5 and conf > 97: data["confidence"] = 95
        else:
            # AI-only mode — cap at 78
            if int(data.get("confidence", 0)) > 78:
                data["confidence"] = 78
        # ── End gate ──────────────────────────────────────────────────────────

        data["personalized"] = True
        data["raw_live_analysis"] = analysis
        return data
    except Exception as e:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "error": str(e),
            "live_price": live_price,
        }


# ----------------------------
# Simulator
# ----------------------------
def simulate_ohlcv(
    symbol: str, num_candles: int = 150, interval_minutes: int = 15, seed: int = None
) -> List[Dict]:
    s = symbol.upper()
    base_price = SYMBOL_BASE_PRICES.get(s, 1.0)
    is_crypto = "USDT" in s
    annual_sigma = 0.65 if is_crypto else 0.08
    dt = (interval_minutes / 60) / 8760
    sigma = annual_sigma * math.sqrt(dt)
    rng = random.Random(seed or int(time.time()))
    candles, price = [], base_price
    now = datetime.utcnow() - timedelta(minutes=interval_minutes * num_candles)

    for i in range(num_candles):
        z = rng.gauss(0, 1)
        pct = 0.00005 * dt + sigma * z
        close = price * (1 + pct)
        cv = abs(rng.gauss(0, sigma * 0.4))
        high = max(price, close) * (1 + abs(rng.gauss(0, cv)))
        low = min(price, close) * (1 - abs(rng.gauss(0, cv)))
        bvol = 2000 if is_crypto else 200000
        vol = rng.uniform(bvol * 0.5, bvol * 2.5)
        dp = 4 if is_crypto else 5
        ts = now + timedelta(minutes=interval_minutes * i)
        candles.append(
            {
                "time": int(ts.timestamp()),
                "datetime": ts.strftime("%Y-%m-%d %H:%M"),
                "open": round(price, dp),
                "high": round(high, dp),
                "low": round(low, dp),
                "close": round(close, dp),
                "volume": round(vol, 2),
            }
        )
        price = close
    return candles


def run_strategy_on_candles(
    candles: List[Dict], initial_balance: float = 10000.0
) -> Dict:
    if len(candles) < 51:
        return {"error": "Need 51+ candles"}
    closes = [c["close"] for c in candles]

    def sma(d, p):
        return [None] * (p - 1) + [sum(d[i : i + p]) / p for i in range(len(d) - p + 1)]

    s20, s50 = sma(closes, 20), sma(closes, 50)
    balance, position, trades, equity_curve = initial_balance, None, [], []

    for i in range(50, len(candles)):
        c = candles[i]
        if s20[i] and s50[i] and s20[i - 1] and s50[i - 1]:
            if s20[i - 1] <= s50[i - 1] and s20[i] > s50[i] and position is None:
                position = {
                    "entry": c["close"],
                    "time": c["time"],
                    "datetime": c["datetime"],
                }
            elif s20[i - 1] >= s50[i - 1] and s20[i] < s50[i] and position is not None:
                pnl_pct = (c["close"] - position["entry"]) / position["entry"]
                pnl_usd = balance * 0.1 * pnl_pct
                balance += pnl_usd
                trades.append(
                    {
                        "type": "BUY",
                        "entry_price": position["entry"],
                        "entry_time": position["time"],
                        "entry_datetime": position["datetime"],
                        "exit_price": c["close"],
                        "exit_time": c["time"],
                        "exit_datetime": c["datetime"],
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "pnl_usd": round(pnl_usd, 2),
                        "result": "WIN" if pnl_usd > 0 else "LOSS",
                    }
                )
                position = None
        equity_curve.append(
            {"time": c["time"], "datetime": c["datetime"], "equity": round(balance, 2)}
        )

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    return {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "total_pnl_usd": round(balance - initial_balance, 2),
        "total_pnl_pct": round((balance - initial_balance) / initial_balance * 100, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_usd": round(sum(t["pnl_usd"] for t in wins) / len(wins), 2)
        if wins
        else 0,
        "avg_loss_usd": round(sum(t["pnl_usd"] for t in losses) / len(losses), 2)
        if losses
        else 0,
        "strategy": "SMA 20/50 Crossover",
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Trading Bot — ARIA v3.1")
app.include_router(backtest_router, prefix="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Pydantic Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    username: Optional[str] = "guest"
    conversation_id: Optional[int] = None  # omit to auto-create a new thread


class RiskCalcRequest(BaseModel):
    balance_ngn: float
    risk_percent: float = 2.0
    entry_price: float
    stop_loss_price: float
    symbol: str


class TradeIdeaRequest(BaseModel):
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    rationale: Optional[str] = ""
    username: Optional[str] = "guest"


class TradeJournalEntryRequest(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    result: str
    pnl_usd: float
    notes: Optional[str] = ""


class UserProfileUpdate(BaseModel):
    balance_ngn: Optional[float] = None
    risk_tolerance: Optional[str] = None
    trading_style: Optional[str] = None
    preferred_pairs: Optional[List[str]] = None


class DemoTradeRequest(BaseModel):
    symbol: str
    volume: float
    trade_type: str


class UserAccountRequest(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_server: Optional[str] = None


# ----------------------------
# Health
# ----------------------------
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "ARIA — AI Trading Bot",
        "version": "3.1",
        "pairs_monitored": len(TRADING_SYMBOLS),
        "persistence": "PostgreSQL",
        "features": [
            "30 Pairs (15 Forex + 15 Crypto)",
            "Personalized Signals per User",
            "NGN Position Sizing",
            "Risk Calculator",
            "Visual Simulator",
            "Trade Journal (DB)",
            "Watchlist (DB)",
            "Chat History (DB)",
            "User Profiles + Activity Tracking",
            "Trade Idea Scorer",
            "Demo Account (DB)",
        ],
    }


# ----------------------------
# User Profile
# ----------------------------
@app.post("/user/{username}")
def create_or_update_user(
    username: str, profile: UserProfileUpdate, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    if profile.balance_ngn is not None:
        user.balance_ngn = profile.balance_ngn
    if profile.risk_tolerance is not None:
        user.risk_tolerance = profile.risk_tolerance
    if profile.trading_style is not None:
        user.trading_style = profile.trading_style
    if profile.preferred_pairs is not None:
        user.preferred_pairs = profile.preferred_pairs
    db.commit()
    db.refresh(user)
    return {
        "status": "success",
        "user": {
            "username": user.username,
            "balance_ngn": user.balance_ngn,
            "risk_tolerance": user.risk_tolerance,
            "trading_style": user.trading_style,
            "preferred_pairs": user.preferred_pairs,
            "created_at": user.created_at.isoformat(),
        },
    }


@app.get("/user/{username}")
def get_user_profile(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    wl = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).all()
    wins = [j for j in journal if j.result == "WIN"]
    return {
        "username": user.username,
        "balance_ngn": user.balance_ngn,
        "risk_tolerance": user.risk_tolerance,
        "trading_style": user.trading_style,
        "preferred_pairs": user.preferred_pairs,
        "watchlist": [w.symbol for w in wl],
        "stats": {
            "total_trades": len(journal),
            "wins": len(wins),
            "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(sum(j.pnl_usd for j in journal), 2),
        },
        "last_active": user.last_active.isoformat(),
        "member_since": user.created_at.isoformat(),
    }


# ----------------------------
# Predictions
# ----------------------------
@app.get("/predictions")
def get_predictions_public(
    amount_ngn: float = 0.0,
    username: Optional[str] = None,
    db: Session = Depends(get_db),
):
    profile = ""
    if username:
        user = get_or_create_user(username, db)
        profile = get_user_profile_summary(user, db)
        log_activity(user, "viewed_predictions", db=db)
    return generate_market_predictions(amount_ngn, user_profile=profile)


@app.get("/get_predictions")
def get_personalized_predictions(
    username: str, symbol: str, db: Session = Depends(get_db)
):
    """FIXED: Personalized signal for a specific symbol without needing a CSV file."""
    user = get_or_create_user(username, db)
    log_activity(user, "viewed_signal", symbol=symbol.upper(), db=db)
    signal = get_personalized_signal(symbol, user, db)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "signal": signal,
        "ngn_rate": get_ngn_rate(),
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/market_analysis")
def market_analysis(db: Session = Depends(get_db)):
    preds = generate_market_predictions()
    if not preds.get("success"):
        return {"error": "Analysis failed", "details": preds.get("error")}
    signals = preds.get("signals", [])
    forex = [s for s in signals if s.get("category") == "forex"]
    crypto = [s for s in signals if s.get("category") == "crypto"]
    strong = [s for s in signals if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]]
    return {
        "market_health": "healthy",
        "active_session": preds.get("active_session"),
        "total_signals": len(signals),
        "forex_signals": len(forex),
        "crypto_signals": len(crypto),
        "strong_signals": len(strong),
        "best_opportunities": strong[:5],
        "usd_ngn_rate": preds.get("usd_ngn_rate"),
        "generated_at": preds.get("generated_at"),
    }


@app.get("/advice/{symbol}")
def get_trading_advice(
    symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)
):
    symbol = symbol.upper()
    now = time.time()
    if symbol in ADVICE_CACHE and (now - ADVICE_CACHE[symbol]["ts"] < 3600):
        return ADVICE_CACHE[symbol]["data"]
    if username:
        user = get_or_create_user(username, db)
        log_activity(user, "viewed_advice", symbol=symbol, db=db)

    # Fetch full live indicator analysis (not just price)
    analysis  = get_symbol_analysis(symbol)
    live_block = format_for_ai_prompt(analysis)
    live_price = analysis.get("live_price", "unavailable")
    conf       = analysis.get("confluence", {})
    has_live   = not analysis.get("ai_only_mode", True)

    prompt = f"""━━━ INSTITUTIONAL DEEP ANALYSIS REQUEST: {symbol} ━━━

{"LIVE INDICATOR DATA (use exact values):" if has_live else "AI reasoning mode — no live candles:"}
{live_block}

{"CONFLUENCE: " + str(conf.get("direction","?")) + " | Score " + str(conf.get("score",0)) + "/6 | Tradeable: " + str(conf.get("tradeable","?")) if has_live else ""}

Provide a comprehensive institutional-grade analysis. Use the EXACT indicator values above.
{"Do NOT say you lack real-time data — the live data is shown above." if has_live else "State clearly this is AI pattern reasoning."}

Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "live_price": "{live_price}",
  "data_source": "{"live_data" if has_live else "ai_reasoning"}",
  "confluence_score": "{conf.get("score",0)}/6",
  "recommendation": "BUY|SELL|HOLD",
  "confidence": 0-97,
  "trade_setup": {{
    "entry": "exact price",
    "stop_loss": "ATR-based level from data",
    "take_profit_1": "1:1.5 target",
    "take_profit_2": "1:3 target",
    "take_profit_3": "1:4.5 target",
    "risk_reward": "1:X",
    "hold_time": "duration"
  }},
  "technical_analysis": {{
    "trend": "bullish/bearish/ranging",
    "rsi": "EXACT value from live data + interpretation",
    "macd": "EXACT histogram value + direction",
    "stochastic": "K/D values + zone",
    "adx": "EXACT ADX + trend strength",
    "williams_r": "value + zone",
    "ema_stack": "EXACT EMA alignment",
    "bollinger": "band position",
    "pivot_points": "P / R1 / S1 from data",
    "key_support": "exact level from data",
    "key_resistance": "exact level from data",
    "pattern": "detected candle pattern"
  }},
  "market_context": "1-2 sentences on macro/session drivers for {symbol} right now",
  "risk_assessment": {{
    "risk_level": "low/medium/high",
    "max_position_pct": "2%",
    "invalidation": "exact price that kills the thesis",
    "news_risk": "any upcoming events that could move {symbol}"
  }},
  "rationale": "3 sentences using exact indicator values from the live data block"
}}"""
    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        result = {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "live_data_used": has_live,
            "advice": data,
        }
        ADVICE_CACHE[symbol] = {"data": result, "ts": now}
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════
#  MEMORY HELPERS — ARIA remembers facts about users
# ═══════════════════════════════════════════════════


def get_user_memories(user: User, db: Session) -> str:
    """Return a formatted block of everything ARIA remembers about this user."""
    memories = db.query(UserMemory).filter(UserMemory.user_id == user.id).all()
    if not memories:
        return "No memories yet — learn about the user from this conversation."
    lines = [f"- {m.key}: {m.value}" for m in memories]
    return "\n".join(lines)


def extract_and_save_memories(
    user: User, user_message: str, aria_response: str, db: Session
):
    """
    Ask the AI to extract any personal facts from the exchange and save them.
    Only runs if the message might contain personal info (cheap heuristic).
    """
    keywords = [
        "my name",
        "i am",
        "i'm",
        "i work",
        "i live",
        "i want",
        "i trade",
        "my goal",
        "my balance",
        "i have",
        "call me",
        "know that",
        "remember",
    ]
    if not any(k in user_message.lower() for k in keywords):
        return

    existing = get_user_memories(user, db)
    extract_prompt = f"""You are a memory extraction assistant.
Extract ONLY clear personal facts the user stated about themselves.

USER MESSAGE: {user_message}
ARIA RESPONSE: {aria_response}
ALREADY KNOWN: {existing}

Return a JSON array of new or updated facts ONLY (skip anything already known):
[
  {{"key": "name", "value": "Simeon"}},
  {{"key": "occupation", "value": "software engineer"}},
  {{"key": "trading_goal", "value": "grow ₦500,000 to ₦2,000,000"}},
  {{"key": "location", "value": "Lagos"}},
  {{"key": "preferred_pairs", "value": "BTCUSDT, EURUSD"}},
  {{"key": "experience_level", "value": "beginner"}}
]
Return [] if nothing new was stated. Output ONLY valid JSON. No explanation."""

    try:
        content = get_ai_response(extract_prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            return
        facts = json.loads(m.group(0))
        for fact in facts:
            key = str(fact.get("key", "")).strip().lower().replace(" ", "_")
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            existing_mem = (
                db.query(UserMemory)
                .filter(UserMemory.user_id == user.id, UserMemory.key == key)
                .first()
            )
            if existing_mem:
                existing_mem.value = value
                existing_mem.updated_at = datetime.utcnow()
            else:
                db.add(
                    UserMemory(
                        user_id=user.id, key=key, value=value, source="user_stated"
                    )
                )
        db.commit()

        # Also update display_name if "name" was found
        name_fact = next((f for f in facts if f.get("key") == "name"), None)
        if name_fact:
            user.display_name = name_fact["value"]
            db.commit()
    except Exception as e:
        print(f"[MEMORY] extraction error: {e}")


def auto_title_conversation(first_message: str) -> str:
    """Generate a short title from the first message (max 50 chars)."""
    clean = first_message.strip().replace("\n", " ")
    if len(clean) <= 50:
        return clean
    words = clean.split()
    title = ""
    for w in words:
        if len(title) + len(w) + 1 > 47:
            break
        title = (title + " " + w).strip()
    return title + "..."


# ═══════════════════════════════════════════════════
#  CONVERSATION MANAGEMENT
# ═══════════════════════════════════════════════════


@app.post("/conversations/{username}")
def create_conversation(
    username: str, title: Optional[str] = None, db: Session = Depends(get_db)
):
    """Start a new named conversation thread (like a new chat in ChatGPT sidebar)."""
    user = get_or_create_user(username, db)
    conv = Conversation(user_id=user.id, title=title or "New Chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {
        "conversation_id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
    }


@app.get("/conversations/{username}")
def list_conversations(username: str, db: Session = Depends(get_db)):
    """List all conversations for a user — like the ChatGPT sidebar."""
    user = get_or_create_user(username, db)
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    result = []
    for c in convs:
        last_msg = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.conversation_id == c.id)
            .order_by(DBChatMessage.created_at.desc())
            .first()
        )
        msg_count = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.conversation_id == c.id)
            .count()
        )
        result.append(
            {
                "conversation_id": c.id,
                "title": c.title,
                "message_count": msg_count,
                "last_message": last_msg.content[:80] + "..."
                if last_msg and len(last_msg.content) > 80
                else (last_msg.content if last_msg else ""),
                "last_role": last_msg.role if last_msg else None,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
        )
    return {"username": username, "total_chats": len(result), "conversations": result}


@app.get("/conversations/{username}/{conv_id}")
def get_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Get full message history of a specific conversation."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.conversation_id == conv_id)
        .order_by(DBChatMessage.created_at.asc())
        .all()
    )
    return {
        "conversation_id": conv.id,
        "title": conv.title,
        "username": username,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "time": m.created_at.isoformat(),
            }
            for m in msgs
        ],
        "total_messages": len(msgs),
        "created_at": conv.created_at.isoformat(),
    }


@app.patch("/conversations/{username}/{conv_id}/rename")
def rename_conversation(
    username: str, conv_id: int, title: str, db: Session = Depends(get_db)
):
    """Rename a conversation."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.title = title
    db.commit()
    return {"status": "success", "conversation_id": conv_id, "new_title": title}


@app.delete("/conversations/{username}/{conv_id}")
def delete_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"status": "deleted", "conversation_id": conv_id}


@app.delete("/conversations/{username}")
def delete_all_conversations(username: str, db: Session = Depends(get_db)):
    """Delete ALL conversations for a user."""
    user = get_or_create_user(username, db)
    db.query(Conversation).filter(Conversation.user_id == user.id).delete()
    db.commit()
    return {"status": "success", "message": f"All conversations deleted for {username}"}


# ═══════════════════════════════════════════════════
#  MAIN CHAT ENDPOINT — Full memory + conversation
# ═══════════════════════════════════════════════════


@app.post("/chat")
def chat_with_aria(chat: ChatMessage, db: Session = Depends(get_db)):
    """
    Send a message to ARIA.
    - Pass conversation_id to continue an existing thread.
    - Omit conversation_id to auto-create a new thread.
    ARIA remembers facts about the user across ALL conversations.
    """
    user = get_or_create_user(chat.username, db)
    log_activity(user, "sent_chat", db=db)

    # ── resolve / create conversation ──
    conv_id = getattr(chat, "conversation_id", None)
    conv = None
    if conv_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
            .first()
        )

    if not conv:
        # Auto-create a new conversation
        conv = Conversation(
            user_id=user.id, title=auto_title_conversation(chat.message)
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    # ── load this conversation's history (last 12 turns) ──
    history = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.conversation_id == conv_id)
        .order_by(DBChatMessage.created_at.desc())
        .limit(12)
        .all()
    )
    history_text = "\n".join(
        [f"{m.role.upper()}: {m.content}" for m in reversed(history)]
    )

    # ── load long-term memories ──
    memories = get_user_memories(user, db)
    name = user.display_name or user.username

    # ── market context ──
    ngn_rate = get_ngn_rate()
    preds = generate_market_predictions(user.balance_ngn)
    signals = preds.get("signals", [])
    profile = get_user_profile_summary(user, db)

    # ── detect symbols mentioned in the message and fetch live data ──
    SYMBOL_ALIASES = {
        # Standard slash format
        "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
        "AUD/USD": "AUDUSD", "USD/CAD": "USDCAD", "NZD/USD": "NZDUSD",
        "USD/CHF": "USDCHF", "EUR/GBP": "EURGBP", "EUR/JPY": "EURJPY",
        "GBP/JPY": "GBPJPY", "AUD/JPY": "AUDJPY", "EUR/CAD": "EURCAD",
        "AUD/CAD": "AUDCAD", "EUR/AUD": "EURAUD", "GBP/AUD": "GBPAUD",
        # Reversed slash format (users often type these)
        "USD/EUR": "EURUSD", "USD/GBP": "GBPUSD", "JPY/USD": "USDJPY",
        "CAD/USD": "USDCAD", "CHF/USD": "USDCHF", "GBP/EUR": "EURGBP",
        "JPY/EUR": "EURJPY", "JPY/GBP": "GBPJPY", "JPY/AUD": "AUDJPY",
        "CAD/EUR": "EURCAD", "CAD/AUD": "AUDCAD", "AUD/EUR": "EURAUD",
        "AUD/GBP": "GBPAUD",
        # Dot format
        "EUR.USD": "EURUSD", "GBP.USD": "GBPUSD", "USD.JPY": "USDJPY",
        # Short names / natural language
        "CABLE": "GBPUSD", "FIBER": "EURUSD", "LOONIE": "USDCAD",
        "SWISSY": "USDCHF", "KIWI": "NZDUSD", "AUSSIE": "AUDUSD",
        "EURO": "EURUSD",
        # Crypto names & tickers
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETHEREUM": "ETHUSDT", "ETH": "ETHUSDT",
        "BNB": "BNBUSDT", "BINANCE COIN": "BNBUSDT",
        "XRP": "XRPUSDT", "RIPPLE": "XRPUSDT",
        "SOL": "SOLUSDT", "SOLANA": "SOLUSDT",
        "ADA": "ADAUSDT", "CARDANO": "ADAUSDT",
        "DOGE": "DOGEUSDT", "DOGECOIN": "DOGEUSDT",
        "DOT": "DOTUSDT", "POLKADOT": "DOTUSDT",
        "MATIC": "MATICUSDT", "POLYGON": "MATICUSDT",
        "LTC": "LTCUSDT", "LITECOIN": "LTCUSDT",
        "SHIB": "SHIBUSDT", "SHIBA": "SHIBUSDT",
        "TRX": "TRXUSDT", "TRON": "TRXUSDT",
        "AVAX": "AVAXUSDT", "AVALANCHE": "AVAXUSDT",
        "LINK": "LINKUSDT", "CHAINLINK": "LINKUSDT",
        "UNI": "UNIUSDT", "UNISWAP": "UNIUSDT",
    }
    msg_upper = chat.message.upper()
    mentioned_symbols = []
    # Check direct symbol names first (exact match or crypto without USDT)
    for sym in TRADING_SYMBOLS:
        if sym in msg_upper:
            mentioned_symbols.append(sym)
        elif sym.endswith("USDT") and sym[:-4] in msg_upper.split():
            # Only add ticker-only match if it's a standalone word (avoids "ADA" matching "CANADA")
            if sym not in mentioned_symbols:
                mentioned_symbols.append(sym)
    # Check aliases (longer aliases first to avoid partial clobbers)
    for alias in sorted(SYMBOL_ALIASES.keys(), key=len, reverse=True):
        sym = SYMBOL_ALIASES[alias]
        if alias in msg_upper and sym not in mentioned_symbols:
            mentioned_symbols.append(sym)
    mentioned_symbols = mentioned_symbols[:3]  # max 3 per message

    # Fetch live indicator data for each mentioned symbol
    live_data_blocks_chat = []
    for sym in mentioned_symbols:
        try:
            analysis = get_symbol_analysis(sym)
            block = format_for_ai_prompt(analysis)
            conf = analysis.get("confluence", {})
            live_data_blocks_chat.append(
                f"── {sym} LIVE DATA ──\n{block}\n"
                f"CONFLUENCE: {conf.get('direction','?')} | Score {conf.get('score',0)}/6 | "
                f"Tradeable: {conf.get('tradeable','?')}"
            )
        except Exception as ex:
            live_data_blocks_chat.append(f"── {sym}: data fetch error ({ex})")

    live_data_section_chat = "\n\n".join(live_data_blocks_chat) if live_data_blocks_chat else ""

    # ── build system prompt ──
    system_prompt = f"""You are ARIA — Advanced Revenue Intelligence Analyst.
You are a world-class AI trading strategist with a warm, confident personality.
You have ACCESS TO REAL-TIME LIVE MARKET DATA — use it whenever it is provided below.

━━━ WHO YOU ARE TALKING TO ━━━
Name: {name}
{profile}
Live USD/NGN Rate: {ngn_rate:.2f}
Time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC

━━━ WHAT YOU REMEMBER ABOUT {name.upper()} ━━━
{memories}

{"━━━ LIVE MARKET DATA FOR MENTIONED SYMBOL(S) ━━━" if live_data_section_chat else ""}
{live_data_section_chat}
{"IMPORTANT: Use the EXACT indicator values above. Do not say you lack real-time data — you have it." if live_data_section_chat else ""}

━━━ CURRENT TOP SIGNALS (from live analysis engine) ━━━
{json.dumps([{
    "symbol": s.get("symbol"), "signal": s.get("signal"),
    "confidence": s.get("confidence"), "entry": s.get("entry_price"),
    "sl": s.get("stop_loss"), "tp1": s.get("take_profit_1"),
    "data_source": s.get("data_source"),
} for s in signals[:6]], indent=2)}

━━━ THIS CONVERSATION ━━━
{history_text if history_text else "New conversation."}

━━━ YOUR RULES ━━━
1. Address {name} by name naturally — not every sentence
2. If LIVE DATA is shown above for any symbol, YOU MUST derive a signal from it — do NOT say "I don't have a signal for this pair" or "it's not in my top signals". The top-signals list is supplementary. Your primary analysis comes from the live data block.
3. NEVER say "I don't have real-time data" when data is shown in this prompt — you have it
4. If a user asks about a pair NOT in the live data block above, pick the closest signal from the top-signals list and note the similarity
5. Any NGN amount → instantly show USD: ₦X = ${{X / {ngn_rate:.2f}:.2f}} USD
6. Recommend 2% risk per trade for {user.trading_style} / {user.risk_tolerance} risk
7. Show exact maths — no approximations without the calculation
8. REQUIRED signal format (use every time you give a signal):
   • Pair + Direction (BUY/SELL)
   • Entry: [exact price from data]
   • Stop Loss: [ATR-based level] ([X pips / X%])
   • TP1: [level] — [estimated X–Y hours]
   • TP2: [level] — [estimated X–Y hours/days]
   • TP3: [level] — [estimated X–Y days]
   • Confluence: [score]/6 — [2-word strength label]
   • Exit if: [exact invalidation condition]
   • Data: [live / AI-reasoning]
9. Hold time MUST be in real time units: minutes (< 1h), hours (1h–48h), or days (> 48h). Never say "short-term" without a number.
10. Tone: warm, direct, institutional — no hype, no guaranteed profit language
11. End every trade recommendation with: ⚠️ Risk: [one-sentence disclaimer]"""

    full_prompt = f"{system_prompt}\n\n{name}: {chat.message}\nARIA:"
    response = get_ai_response_creative(full_prompt)

    # ── save messages to DB ──
    db.add(
        DBChatMessage(
            user_id=user.id, conversation_id=conv_id, role="user", content=chat.message
        )
    )
    db.add(
        DBChatMessage(
            user_id=user.id, conversation_id=conv_id, role="aria", content=response
        )
    )

    # ── update conversation timestamp ──
    conv.updated_at = datetime.utcnow()

    # ── auto-title if this is the first message ──
    if conv.title == "New Chat" and not history:
        conv.title = auto_title_conversation(chat.message)

    db.commit()

    # ── extract + save memories in background (non-blocking) ──
    try:
        extract_and_save_memories(user, chat.message, response, db)
    except Exception as e:
        print(f"[MEMORY] non-critical error: {e}")

    return {
        "success": True,
        "username": chat.username,
        "display_name": name,
        "conversation_id": conv_id,
        "conversation_title": conv.title,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── legacy single-user history endpoint (kept for backwards compat) ──
@app.get("/chat/history/{username}")
def get_chat_history(username: str, limit: int = 30, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    msgs = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.user_id == user.id)
        .order_by(DBChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return {
        "username": username,
        "history": [
            {
                "role": m.role,
                "content": m.content,
                "conversation_id": m.conversation_id,
                "time": m.created_at.isoformat(),
            }
            for m in msgs
        ],
        "total": len(msgs),
    }


@app.delete("/chat/history/{username}")
def clear_chat_history(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    db.query(DBChatMessage).filter(DBChatMessage.user_id == user.id).delete()
    db.commit()
    return {"status": "success", "message": f"All chat history cleared for {username}"}


# ═══════════════════════════════════════════════════
#  MEMORY ENDPOINTS — See / manage what ARIA remembers
# ═══════════════════════════════════════════════════


@app.get("/memory/{username}")
def get_memory(username: str, db: Session = Depends(get_db)):
    """See everything ARIA currently remembers about you."""
    user = get_or_create_user(username, db)
    memories = db.query(UserMemory).filter(UserMemory.user_id == user.id).all()
    return {
        "username": username,
        "display_name": user.display_name,
        "memory_count": len(memories),
        "memories": [
            {
                "key": m.key,
                "value": m.value,
                "source": m.source,
                "updated_at": m.updated_at.isoformat(),
            }
            for m in memories
        ],
        "note": "ARIA uses these facts to personalise every conversation.",
    }


@app.post("/memory/{username}")
def add_memory(username: str, key: str, value: str, db: Session = Depends(get_db)):
    """Manually tell ARIA something to remember about you."""
    user = get_or_create_user(username, db)
    key = key.strip().lower().replace(" ", "_")
    existing = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user.id, UserMemory.key == key)
        .first()
    )
    if existing:
        existing.value = value
        existing.updated_at = datetime.utcnow()
    else:
        db.add(UserMemory(user_id=user.id, key=key, value=value, source="user_stated"))
    if key == "name":
        user.display_name = value
    db.commit()
    return {"status": "saved", "key": key, "value": value}


@app.delete("/memory/{username}/{key}")
def delete_memory(username: str, key: str, db: Session = Depends(get_db)):
    """Tell ARIA to forget a specific fact."""
    user = get_or_create_user(username, db)
    deleted = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user.id, UserMemory.key == key.lower())
        .delete()
    )
    db.commit()
    if deleted:
        return {"status": "forgotten", "key": key}
    raise HTTPException(status_code=404, detail=f"No memory found with key '{key}'")


@app.delete("/memory/{username}")
def clear_all_memory(username: str, db: Session = Depends(get_db)):
    """Clear ALL of ARIA's memories about this user."""
    user = get_or_create_user(username, db)
    db.query(UserMemory).filter(UserMemory.user_id == user.id).delete()
    user.display_name = None
    db.commit()
    return {"status": "success", "message": f"All memories cleared for {username}"}


# ----------------------------
# Risk Calculator
# ----------------------------
@app.post("/risk_calculator")
def calculate_risk(req: RiskCalcRequest):
    ngn_rate = get_ngn_rate()
    balance_usd = req.balance_ngn / ngn_rate
    risk_ngn = req.balance_ngn * (req.risk_percent / 100)
    risk_usd = risk_ngn / ngn_rate
    dist = abs(req.entry_price - req.stop_loss_price)
    pct_dist = dist / req.entry_price if req.entry_price > 0 else 0
    pos_usd = risk_usd / pct_dist if pct_dist > 0 else 0
    units = pos_usd / req.entry_price if req.entry_price > 0 else 0
    is_crypto = "USDT" in req.symbol.upper()
    pip_dist = dist if is_crypto else dist * 10000

    return {
        "symbol": req.symbol.upper(),
        "balance_ngn": req.balance_ngn,
        "balance_usd": round(balance_usd, 2),
        "risk_percent": req.risk_percent,
        "risk_amount_ngn": round(risk_ngn, 2),
        "risk_amount_usd": round(risk_usd, 4),
        "entry_price": req.entry_price,
        "stop_loss_price": req.stop_loss_price,
        "stop_distance_pct": round(pct_dist * 100, 4),
        "recommended_position_usd": round(pos_usd, 2),
        "units": round(units, 6),
        "pip_distance": round(pip_dist, 1) if not is_crypto else None,
        "note": f"Risking {req.risk_percent}% of your NGN balance per this trade.",
    }


# ----------------------------
# Trade Idea Scorer
# ----------------------------
@app.post("/score_trade")
def score_trade_idea(idea: TradeIdeaRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(idea.username, db)
    log_activity(user, "scored_trade", symbol=idea.symbol, db=db)
    dist_sl = abs(idea.entry - idea.stop_loss)
    dist_tp = abs(idea.take_profit - idea.entry)
    rr = dist_tp / dist_sl if dist_sl > 0 else 0
    profile = get_user_profile_summary(user, db)

    prompt = f"""Elite institutional trade reviewer. Evaluate this trade idea for user {idea.username}.
USER PROFILE: {profile}

TRADE:
Symbol: {idea.symbol.upper()} | Direction: {idea.direction.upper()}
Entry: {idea.entry} | Stop Loss: {idea.stop_loss} | Take Profit: {idea.take_profit}
Calculated R/R: 1:{round(rr, 2)} | Rationale: {idea.rationale}

Return ONLY valid JSON:
{{
  "score": 0-100, "grade": "A+|A|B|C|D|F", "verdict": "TAKE IT|RISKY BUT CONSIDER|AVOID",
  "rr_ratio": "1:{round(rr, 2)}", "rr_assessment": "excellent(>1:3)|good(1:2-3)|weak(<1:2)",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvements": ["improvement1", "improvement2"],
  "adjusted_entry": "better entry if applicable",
  "adjusted_sl": "better SL if applicable",
  "adjusted_tp": "better TP if applicable",
  "personalized_feedback": "feedback specific to this user's style and history",
  "summary": "2-sentence professional verdict"
}}"""
    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        return {"success": True, "trade_idea": idea.dict(), "analysis": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Watchlist — DB persistent
# ----------------------------
@app.post("/watchlist/{username}")
def add_to_watchlist(username: str, symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper()
    if symbol not in TRADING_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"{symbol} not supported")
    user = get_or_create_user(username, db)
    exists = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol)
        .first()
    )
    if not exists:
        db.add(WatchlistItem(user_id=user.id, symbol=symbol))
        db.commit()
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    return {"status": "success", "watchlist": [w.symbol for w in items]}


@app.delete("/watchlist/{username}/{symbol}")
def remove_from_watchlist(username: str, symbol: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol.upper()
    ).delete()
    db.commit()
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    return {"status": "success", "watchlist": [w.symbol for w in items]}


@app.get("/watchlist/{username}")
def get_watchlist(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    preds = generate_market_predictions(user.balance_ngn)
    sig_map = {s.get("symbol"): s for s in preds.get("signals", [])}
    result = []
    for w in items:
        sig = sig_map.get(w.symbol, {})
        price = get_live_price(w.symbol)
        result.append(
            {
                "symbol": w.symbol,
                "live_price": price,
                "signal": sig.get("signal", "N/A"),
                "confidence": sig.get("confidence", "N/A"),
                "entry_price": sig.get("entry_price", "N/A"),
                "stop_loss": sig.get("stop_loss", "N/A"),
                "hold_time": sig.get("hold_time", "N/A"),
                "added_at": w.added_at.isoformat(),
            }
        )
    return {"username": username, "watchlist": result, "total": len(result)}


# ----------------------------
# Trade Journal — DB persistent
# ----------------------------
@app.post("/journal/{username}")
def add_journal_entry(
    username: str, entry: TradeJournalEntryRequest, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    record = DBJournalEntry(
        user_id=user.id,
        symbol=entry.symbol.upper(),
        direction=entry.direction.upper(),
        entry_price=entry.entry_price,
        exit_price=entry.exit_price,
        volume=entry.volume,
        result=entry.result.upper(),
        pnl_usd=entry.pnl_usd,
        notes=entry.notes,
    )
    db.add(record)
    log_activity(
        user,
        "logged_trade",
        symbol=entry.symbol.upper(),
        details={"result": entry.result, "pnl": entry.pnl_usd},
        db=db,
    )
    db.commit()
    db.refresh(record)
    return {
        "status": "success",
        "entry_id": record.id,
        "logged_at": record.logged_at.isoformat(),
    }


@app.get("/journal/{username}")
def get_journal(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    entries = (
        db.query(DBJournalEntry)
        .filter(DBJournalEntry.user_id == user.id)
        .order_by(DBJournalEntry.logged_at.desc())
        .all()
    )
    if not entries:
        return {"username": username, "entries": [], "stats": {}}
    wins = [e for e in entries if e.result == "WIN"]
    losses = [e for e in entries if e.result == "LOSS"]
    total_pnl = sum(e.pnl_usd for e in entries)
    sym_freq = {}
    for e in entries:
        sym_freq[e.symbol] = sym_freq.get(e.symbol, 0) + 1
    best = max(entries, key=lambda e: e.pnl_usd)
    worst = min(entries, key=lambda e: e.pnl_usd)

    return {
        "username": username,
        "entries": [
            {
                "id": e.id,
                "symbol": e.symbol,
                "direction": e.direction,
                "entry_price": e.entry_price,
                "exit_price": e.exit_price,
                "volume": e.volume,
                "result": e.result,
                "pnl_usd": e.pnl_usd,
                "notes": e.notes,
                "logged_at": e.logged_at.isoformat(),
            }
            for e in entries
        ],
        "stats": {
            "total_trades": len(entries),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(entries) * 100, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(sum(e.pnl_usd for e in wins) / len(wins), 2)
            if wins
            else 0,
            "avg_loss_usd": round(sum(e.pnl_usd for e in losses) / len(losses), 2)
            if losses
            else 0,
            "most_traded_symbol": max(sym_freq, key=sym_freq.get),
            "best_trade": {"symbol": best.symbol, "pnl_usd": best.pnl_usd},
            "worst_trade": {"symbol": worst.symbol, "pnl_usd": worst.pnl_usd},
        },
    }


# ----------------------------
# Demo Account — DB persistent
# ----------------------------
@app.post("/demo/open_account/{username}")
def open_demo_account(
    username: str, initial_balance: float = 10000.0, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    existing = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if existing:
        return {
            "status": "exists",
            "message": f"Demo account already exists. Balance: ${existing.balance:,.2f}",
            "balance": existing.balance,
        }
    acct = DemoAccount(user_id=user.id, balance=initial_balance)
    db.add(acct)
    db.commit()
    db.refresh(acct)
    return {
        "status": "success",
        "message": f"Demo account opened for {username}",
        "balance": acct.balance,
        "account_id": acct.id,
    }


@app.post("/demo/execute_trade/{username}")
def execute_demo_trade(
    username: str, req: DemoTradeRequest, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        acct = DemoAccount(user_id=user.id, balance=10000.0)
        db.add(acct)
        db.commit()
        db.refresh(acct)

    symbol = req.symbol.upper()
    entry_price = get_live_price(symbol)
    if entry_price == 0.0:
        raise HTTPException(
            status_code=400, detail=f"Could not fetch live price for {symbol}"
        )

    trade = DemoTrade(
        demo_account_id=acct.id,
        symbol=symbol,
        entry_price=entry_price,
        current_price=entry_price,
        volume=req.volume,
        trade_type=req.trade_type.upper(),
    )
    db.add(trade)
    log_activity(
        user,
        "opened_demo_trade",
        symbol=symbol,
        details={"entry": entry_price, "volume": req.volume},
        db=db,
    )
    db.commit()
    db.refresh(trade)
    return {
        "status": "success",
        "trade_id": trade.id,
        "symbol": symbol,
        "entry_price": entry_price,
        "volume": req.volume,
        "type": trade.trade_type,
    }


@app.post("/demo/close_trade/{username}/{trade_id}")
def close_demo_trade(username: str, trade_id: int, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        raise HTTPException(status_code=404, detail="No demo account found")
    trade = (
        db.query(DemoTrade)
        .filter(DemoTrade.id == trade_id, DemoTrade.demo_account_id == acct.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    exit_price = get_live_price(trade.symbol)
    if exit_price == 0:
        exit_price = trade.current_price
    pnl = (
        (exit_price - trade.entry_price) * trade.volume
        if trade.trade_type == "BUY"
        else (trade.entry_price - exit_price) * trade.volume
    )
    trade.exit_price = exit_price
    trade.current_price = exit_price
    trade.pnl = pnl
    trade.is_active = False
    trade.closed_at = datetime.utcnow()
    acct.balance += pnl
    db.commit()
    return {
        "status": "closed",
        "symbol": trade.symbol,
        "entry": trade.entry_price,
        "exit": exit_price,
        "pnl_usd": round(pnl, 2),
        "new_balance": round(acct.balance, 2),
    }


@app.get("/demo/account/{username}")
def get_demo_account(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {
            "error": "No demo account found. POST /demo/open_account/{username} to create one."
        }

    active = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True)
        .all()
    )
    closed = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == False)
        .all()
    )

    # Refresh PnL for active trades
    total_floating_pnl = 0
    active_list = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (
            (price - t.entry_price) * t.volume
            if t.trade_type == "BUY"
            else (t.entry_price - price) * t.volume
        )
        t.current_price = price
        t.pnl = pnl
        total_floating_pnl += pnl
        active_list.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "type": t.trade_type,
                "entry_price": t.entry_price,
                "current_price": price,
                "volume": t.volume,
                "pnl_usd": round(pnl, 2),
                "opened_at": t.opened_at.isoformat(),
            }
        )
    db.commit()

    wins = [t for t in closed if t.pnl > 0]
    return {
        "username": username,
        "balance_usd": round(acct.balance, 2),
        "floating_pnl_usd": round(total_floating_pnl, 2),
        "equity_usd": round(acct.balance + total_floating_pnl, 2),
        "active_trades": active_list,
        "trade_history": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "type": t.trade_type,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl_usd": round(t.pnl, 2),
                "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            }
            for t in closed[-20:]
        ],
        "stats": {
            "total_closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "realized_pnl_usd": round(sum(t.pnl for t in closed), 2),
        },
    }


@app.get("/demo/ai_feedback/{username}")
def get_demo_ai_feedback(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {"error": "No demo account found"}
    active = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True)
        .all()
    )
    if not active:
        return {"error": "No active trades to analyze"}

    summary = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (
            (price - t.entry_price) * t.volume
            if t.trade_type == "BUY"
            else (t.entry_price - price) * t.volume
        )
        summary.append(
            f"{t.trade_type} {t.volume} {t.symbol} @ {t.entry_price} | Now: {price} | PnL: ${pnl:.2f}"
        )

    profile = get_user_profile_summary(user, db)
    prompt = f"""Elite trading risk manager reviewing DEMO TRADES.
USER: {profile}
ACTIVE TRADES: {json.dumps(summary)}
DEMO BALANCE: ${acct.balance:,.2f}

Provide a direct, calculative review:
1. Which trades are profitable and which are at risk?
2. Should any trade be closed NOW? Justify with specific levels.
3. Adjust SL/TP for active trades with exact new levels.
4. Is the overall portfolio over-leveraged or well-managed?
5. One specific improvement for this trader's style."""

    feedback = get_ai_response(prompt)
    return {
        "username": username,
        "balance": acct.balance,
        "active_trades": summary,
        "ai_strategic_advice": feedback,
    }


# ----------------------------
# Account Monitor
# ----------------------------
@app.get("/account/monitor/{username}")
def monitor_account(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    ngn_rate = get_ngn_rate()
    demo_acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).all()
    wins = [j for j in journal if j.result == "WIN"]
    total_pnl = sum(j.pnl_usd for j in journal)

    return {
        "username": username,
        "balance_ngn": user.balance_ngn,
        "balance_usd": round(user.balance_ngn / ngn_rate, 2),
        "usd_ngn_rate": ngn_rate,
        "demo_account": {
            "balance_usd": demo_acct.balance if demo_acct else None,
            "exists": demo_acct is not None,
        },
        "trading_stats": {
            "total_trades": len(journal),
            "wins": len(wins),
            "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(total_pnl, 2),
            "total_pnl_ngn": round(total_pnl * ngn_rate, 2),
        },
        "profile": {
            "risk_tolerance": user.risk_tolerance,
            "trading_style": user.trading_style,
            "preferred_pairs": user.preferred_pairs,
            "member_since": user.created_at.isoformat(),
        },
    }


# ----------------------------
# Simulator
# ----------------------------
@app.get("/live_data/{symbol}")
def get_live_technical_data(
    symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)
):
    """Returns real live OHLCV-based technical indicators for any symbol."""
    symbol = symbol.upper()
    if username:
        user = get_or_create_user(username, db)
        log_activity(user, "viewed_live_data", symbol=symbol, db=db)
    analysis = get_symbol_analysis(symbol)
    return {
        "symbol": symbol,
        "success": bool(analysis.get("live_price")),
        "data_source": analysis.get("data_source", "unavailable"),
        "live_price": analysis.get("live_price"),
        "last_candle_time": analysis.get("last_candle_time"),
        "candles_used": analysis.get("candles_used"),
        "confluence": analysis.get("confluence", {}),
        "indicators": analysis.get("indicators", {}),
        "key_levels": analysis.get("key_levels", {}),
        "trend_bias": analysis.get("trend_bias"),
        "fetched_at": datetime.utcnow().isoformat(),
        "cache_ttl_minutes": 15,
        "note": "Data from Alpha Vantage daily candles. Cached 15 min to respect rate limits.",
    }


@app.get("/simulator/candles/{symbol}")
def get_simulator_candles(
    symbol: str, candles: int = 120, interval: int = 15, seed: int = None
):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(
        symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed
    )
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "total_candles": len(candle_data),
        "chart_type": "candlestick",
        "candles": candle_data,
        "meta": {
            "base_price": SYMBOL_BASE_PRICES.get(symbol),
            "note": "Simulated via GBM — demo only.",
        },
    }


@app.get("/simulator/run/{symbol}")
def run_simulator(
    symbol: str,
    candles: int = 150,
    interval: int = 15,
    balance: float = 10000.0,
    seed: int = None,
):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(
        symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed
    )
    results = run_strategy_on_candles(candle_data, initial_balance=balance)
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    entry_markers = [
        {
            "time": t["entry_time"],
            "datetime": t["entry_datetime"],
            "price": t["entry_price"],
            "label": "BUY",
            "color": "#00ff88",
        }
        for t in results["trades"]
    ]
    exit_markers = [
        {
            "time": t["exit_time"],
            "datetime": t["exit_datetime"],
            "price": t["exit_price"],
            "label": t["result"],
            "color": "#00bfff" if t["result"] == "WIN" else "#ff4444",
        }
        for t in results["trades"]
    ]
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "strategy": results["strategy"],
        "performance": {
            "initial_balance_usd": results["initial_balance"],
            "final_balance_usd": results["final_balance"],
            "total_pnl_usd": results["total_pnl_usd"],
            "total_pnl_pct": results["total_pnl_pct"],
            "total_trades": results["total_trades"],
            "wins": results["wins"],
            "losses": results["losses"],
            "win_rate_pct": results["win_rate_pct"],
        },
        "chart_data": {
            "candles": candle_data,
            "equity_curve": results["equity_curve"],
            "entry_markers": entry_markers,
            "exit_markers": exit_markers,
        },
        "trades": results["trades"],
        "disclaimer": "Simulated results. Past performance is not indicative of future results.",
    }


# ----------------------------
# Connect Account
# ----------------------------
@app.post("/connect_account")
def connect_account(req: UserAccountRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(req.username, db)
    return {
        "status": "success",
        "message": f"Account connected for {req.username}",
        "user_id": user.id,
    }


# ════════════════════════════════════════════════════════════════
#  ARIA AUTO-TRADER  —  Bot Config · Risk Engine · Market Filter
#                        Trade Manager · MT5 Bridge endpoints
# ════════════════════════════════════════════════════════════════

import secrets as _secrets


# ── Pydantic models ──────────────────────────────────────────────
class BotConfigRequest(BaseModel):
    risk_percent: Optional[float] = 1.0
    max_concurrent_trades: Optional[int] = 3
    max_daily_loss_pct: Optional[float] = 5.0
    max_weekly_loss_pct: Optional[float] = 10.0
    max_lot_size: Optional[float] = 1.0
    min_lot_size: Optional[float] = 0.01
    min_confidence: Optional[float] = 75.0
    max_spread_pips: Optional[float] = 3.0
    avoid_news_minutes: Optional[int] = 30
    min_atr_percentile: Optional[float] = 30.0
    allowed_sessions: Optional[List[str]] = None
    allowed_pairs: Optional[List[str]] = None
    use_break_even: Optional[bool] = True
    break_even_trigger_rr: Optional[float] = 1.0
    use_trailing_stop: Optional[bool] = True
    trail_trigger_rr: Optional[float] = 1.5
    trail_step_atr: Optional[float] = 0.5
    use_partial_close: Optional[bool] = True
    partial_close_pct: Optional[float] = 50.0
    max_hold_hours: Optional[int] = 48
    mt5_account_number: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_broker: Optional[str] = None


class ManualSignalRequest(BaseModel):
    symbol: str
    direction: str  # BUY | SELL
    entry_price: Optional[float] = None
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    account_balance_usd: float
    signal_confidence: Optional[float] = 80.0
    timeframe: Optional[str] = "H1"


class BridgeExecutedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    filled_price: float
    spread_pips: Optional[float] = None
    executed_at: Optional[str] = None


class BridgeRejectedRequest(BaseModel):
    order_id: int
    reason: str
    rejected_at: Optional[str] = None


class PositionUpdateItem(BaseModel):
    mt5_ticket: int
    symbol: str
    current_price: float
    floating_pnl: float
    volume: float
    current_sl: Optional[float] = None
    current_tp: Optional[float] = None


class PositionUpdateRequest(BaseModel):
    positions: List[PositionUpdateItem]
    account: Optional[Dict[str, Any]] = None


class PartialClosedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    close_price: Optional[float] = None
    success: bool


class ClosedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    closed_at: Optional[str] = None


class BridgeConnectRequest(BaseModel):
    account_info: Optional[Dict[str, Any]] = None
    bridge_version: Optional[str] = "1.0"


# ── Helper: verify bridge key ─────────────────────────────────────
def _verify_bridge_key(username: str, provided_key: str, db: Session) -> BotConfig:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Bot not configured for this user")
    if cfg.bridge_api_key and cfg.bridge_api_key != provided_key:
        raise HTTPException(status_code=403, detail="Invalid bridge API key")
    return cfg


# ════════════════════════════════════════════════════════════════
#  1. BOT CONFIGURATION
# ════════════════════════════════════════════════════════════════


@app.post("/bot/configure/{username}")
def configure_bot(username: str, req: BotConfigRequest, db: Session = Depends(get_db)):
    """
    Create or update the trading bot configuration for a user.
    Also generates a unique bridge API key if one doesn't exist yet.
    """
    user = get_or_create_user(username, db)
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()

    if not cfg:
        cfg = BotConfig(user_id=user.id, bridge_api_key=_secrets.token_urlsafe(32))
        db.add(cfg)

    fields = req.dict(exclude_none=True)
    for k, v in fields.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if cfg.allowed_sessions is None:
        cfg.allowed_sessions = ["london", "overlap", "newyork"]

    cfg.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(cfg)

    return {
        "success": True,
        "message": "Bot configured successfully",
        "bridge_api_key": cfg.bridge_api_key,
        "note": "Keep your bridge_api_key secret — paste it in the .env of your mt5_bridge.py",
        "config": {
            "is_active": cfg.is_active,
            "risk_percent": cfg.risk_percent,
            "max_concurrent_trades": cfg.max_concurrent_trades,
            "max_daily_loss_pct": cfg.max_daily_loss_pct,
            "max_weekly_loss_pct": cfg.max_weekly_loss_pct,
            "min_confidence": cfg.min_confidence,
            "max_spread_pips": cfg.max_spread_pips,
            "allowed_sessions": cfg.allowed_sessions,
            "allowed_pairs": cfg.allowed_pairs or "All 30 pairs",
            "use_trailing_stop": cfg.use_trailing_stop,
            "use_break_even": cfg.use_break_even,
            "use_partial_close": cfg.use_partial_close,
            "mt5_account_number": cfg.mt5_account_number,
            "mt5_server": cfg.mt5_server,
        },
    }


@app.post("/bot/start/{username}")
def start_bot(username: str, db: Session = Depends(get_db)):
    """Activate the bot — it will start processing signals."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail="Configure the bot first via POST /bot/configure/{username}",
        )
    cfg.is_active = True
    cfg.updated_at = datetime.utcnow()
    db.commit()
    return {
        "success": True,
        "message": "ARIA bot is now ACTIVE — monitoring markets 24/7",
    }


@app.post("/bot/stop/{username}")
def stop_bot(username: str, db: Session = Depends(get_db)):
    """Pause the bot — no new orders will be queued. Open trades are unaffected."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Bot not configured")
    cfg.is_active = False
    cfg.updated_at = datetime.utcnow()
    db.commit()
    return {
        "success": True,
        "message": "ARIA bot PAUSED — no new orders will be placed",
    }


@app.get("/bot/status/{username}")
def bot_status(username: str, db: Session = Depends(get_db)):
    """Full bot status: config, active orders, today's P&L."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        return {
            "configured": False,
            "message": "Bot not configured. Call POST /bot/configure/{username} first.",
        }

    active_orders = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status.in_(["QUEUED", "SENT", "EXECUTED", "ACTIVE"]),
        )
        .all()
    )

    today = datetime.utcnow().date()
    closed_today = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
            BotOrder.closed_at >= datetime.combine(today, datetime.min.time()),
        )
        .all()
    )

    daily_pnl = sum(o.realised_pnl_usd or 0 for o in closed_today)
    total_closed = (
        db.query(BotOrder)
        .filter(BotOrder.user_id == user.id, BotOrder.status == "CLOSED")
        .count()
    )
    total_wins = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
            BotOrder.realised_pnl_usd > 0,
        )
        .count()
    )
    win_rate = round(total_wins / total_closed * 100, 1) if total_closed > 0 else 0.0

    return {
        "configured": True,
        "is_active": cfg.is_active,
        "status_label": "🟢 ACTIVE" if cfg.is_active else "⏸ PAUSED",
        "config": {
            "risk_percent": cfg.risk_percent,
            "max_concurrent_trades": cfg.max_concurrent_trades,
            "max_daily_loss_pct": cfg.max_daily_loss_pct,
            "min_confidence": cfg.min_confidence,
            "max_spread_pips": cfg.max_spread_pips,
            "allowed_sessions": cfg.allowed_sessions,
            "use_trailing_stop": cfg.use_trailing_stop,
            "use_break_even": cfg.use_break_even,
        },
        "active_orders": [
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry": o.filled_price or o.requested_entry,
                "sl": o.current_sl or o.stop_loss,
                "tp1": o.take_profit_1,
                "status": o.status,
                "floating_pnl": o.floating_pnl_usd,
                "ticket": o.mt5_ticket,
            }
            for o in active_orders
        ],
        "todays_performance": {
            "trades_closed": len(closed_today),
            "realised_pnl": round(daily_pnl, 2),
            "pnl_sign": "+" if daily_pnl >= 0 else "",
        },
        "all_time_performance": {
            "total_closed": total_closed,
            "wins": total_wins,
            "losses": total_closed - total_wins,
            "win_rate_pct": win_rate,
        },
    }


# ════════════════════════════════════════════════════════════════
#  2. PLACE A TRADE  (passes through Risk Engine + Market Filter)
# ════════════════════════════════════════════════════════════════


@app.post("/bot/trade/{username}")
def place_bot_trade(
    username: str, req: ManualSignalRequest, db: Session = Depends(get_db)
):
    """
    Queue a trade after passing it through:
      1. Market Filter  — spread, session, news, volatility, confidence
      2. Risk Engine    — lot size, daily loss limit, concurrent position cap
    If everything passes, an order is written to DB with status=QUEUED.
    The MT5 bridge will pick it up within seconds.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=400, detail="Configure bot first")
    if not cfg.is_active:
        raise HTTPException(
            status_code=400, detail="Bot is paused. Call /bot/start/{username} first."
        )

    symbol = req.symbol.upper()
    direction = req.direction.upper()
    confidence = req.signal_confidence or 80.0

    # ── Get live market data for filter checks ──────────────────
    analysis = get_symbol_analysis(symbol)
    indicators = analysis.get("indicators", {})
    atr = indicators.get("atr")
    atr_avg = indicators.get("atr_avg") or (atr * 1.2 if atr else None)

    # ── 1. Market Filter ────────────────────────────────────────
    filter_result = MarketFilter.run_all_checks(
        symbol=symbol,
        signal_confidence=confidence,
        config=cfg,
        current_spread_pips=None,  # bridge will re-check live spread at execution
        atr=atr,
        atr_avg=atr_avg,
    )

    if not filter_result["approved"]:
        return {
            "success": False,
            "approved": False,
            "blocks": filter_result["blocks"],
            "warnings": filter_result.get("warnings", []),
            "message": "Trade BLOCKED by Market Filter — no order queued",
        }

    # ── 2. Risk Engine — daily loss check ───────────────────────
    loss_check = RiskEngine.check_daily_loss(user.id, cfg, db)
    if not loss_check["allowed"]:
        return {
            "success": False,
            "approved": False,
            "blocks": [loss_check["reason"]],
            "message": "Trade BLOCKED by Risk Engine — daily/weekly loss limit reached",
        }

    # ── 3. Risk Engine — concurrent positions check ─────────────
    conc_check = RiskEngine.check_concurrent_positions(user.id, symbol, cfg, db)
    if not conc_check["allowed"]:
        return {
            "success": False,
            "approved": False,
            "blocks": [conc_check["reason"]],
            "message": "Trade BLOCKED — too many concurrent positions",
        }

    # ── 4. Risk Engine — calculate lot size ─────────────────────
    entry = req.entry_price or analysis.get("live_price") or 0
    if not entry:
        raise HTTPException(
            status_code=400,
            detail="entry_price not provided and live price unavailable",
        )

    sizing = RiskEngine.calculate_lot_size(
        account_balance_usd=req.account_balance_usd,
        risk_percent=cfg.risk_percent,
        entry=entry,
        stop_loss=req.stop_loss,
        symbol=symbol,
        min_lot=cfg.min_lot_size,
        max_lot=cfg.max_lot_size,
    )
    if "error" in sizing:
        raise HTTPException(status_code=400, detail=sizing["error"])

    # ── 5. Queue the order ───────────────────────────────────────
    order = BotOrder(
        user_id=user.id,
        symbol=symbol,
        direction=direction,
        signal_confidence=confidence,
        signal_source="aria",
        timeframe=req.timeframe,
        requested_entry=entry,
        stop_loss=req.stop_loss,
        take_profit_1=req.take_profit_1,
        take_profit_2=req.take_profit_2,
        take_profit_3=req.take_profit_3,
        lot_size=sizing["lot_size"],
        risk_percent=sizing["risk_percent"],
        risk_usd=sizing["risk_usd"],
        sl_pips=sizing["sl_pips"],
        current_sl=req.stop_loss,
        filter_passed=True,
        filter_block_reasons=filter_result.get("warnings"),
        status="QUEUED",
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    log_activity(user, "bot_order_queued", symbol=symbol, db=db)

    return {
        "success": True,
        "approved": True,
        "order_id": order.id,
        "status": "QUEUED",
        "message": f"✅ Order queued — MT5 bridge will execute within seconds",
        "trade_details": {
            "symbol": symbol,
            "direction": direction,
            "lot_size": sizing["lot_size"],
            "entry": entry,
            "stop_loss": req.stop_loss,
            "take_profit_1": req.take_profit_1,
            "take_profit_2": req.take_profit_2,
        },
        "risk_details": {
            "risk_percent": sizing["risk_percent"],
            "risk_usd": sizing["risk_usd"],
            "sl_pips": sizing["sl_pips"],
        },
        "filter_warnings": filter_result.get("warnings", []),
    }


@app.post("/bot/auto_signal/{username}")
def auto_signal(
    username: str,
    symbol: str,
    account_balance_usd: float,
    db: Session = Depends(get_db),
):
    """
    ARIA generates a signal for the symbol, runs it through all filters,
    and queues it automatically if it passes.  No manual entry/SL/TP needed —
    ARIA calculates everything from live indicator data.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg or not cfg.is_active:
        return {"success": False, "message": "Bot not configured or not active"}

    sym = symbol.upper()
    analysis = get_symbol_analysis(sym)
    if not analysis.get("live_price"):
        return {"success": False, "message": f"No live data available for {sym}"}

    indicators = analysis.get("indicators", {})
    prompt = (
        f"You are ARIA. Give an auto-trade signal for {sym}.\n"
        f"Live data: {format_for_ai_prompt(sym, analysis)}\n\n"
        f"Reply ONLY with this JSON (no text before/after):\n"
        f'{{"direction":"BUY or SELL","confidence":0-100,"entry":{analysis["live_price"]},'
        f'"stop_loss":float,"take_profit_1":float,"take_profit_2":float,"reasoning":"one sentence"}}'
    )
    try:
        raw = get_ai_response(prompt)
        # Extract JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"success": False, "message": "AI did not return valid JSON"}
        signal = json.loads(match.group())
    except Exception as e:
        return {"success": False, "message": f"AI signal failed: {e}"}

    req = ManualSignalRequest(
        symbol=sym,
        direction=signal["direction"],
        entry_price=float(signal["entry"]),
        stop_loss=float(signal["stop_loss"]),
        take_profit_1=float(signal["take_profit_1"]),
        take_profit_2=signal.get("take_profit_2"),
        account_balance_usd=account_balance_usd,
        signal_confidence=float(signal.get("confidence", 78)),
        timeframe="H1",
    )
    result = place_bot_trade(username=username, req=req, db=db)
    result["ai_reasoning"] = signal.get("reasoning", "")
    return result


# ════════════════════════════════════════════════════════════════
#  3. MT5 BRIDGE ENDPOINTS  (called by mt5_bridge.py on Windows)
# ════════════════════════════════════════════════════════════════


@app.post("/bot/bridge_connect/{username}")
def bridge_connect(
    username: str,
    req: BridgeConnectRequest,
    x_bridge_key: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """MT5 bridge calls this on startup to register itself."""
    from fastapi import Header

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "status": "connected",
        "message": f"Bridge registered for {username}",
        "account": req.account_info,
        "server_time": datetime.utcnow().isoformat(),
    }


@app.get("/bot/queue/{username}")
def get_order_queue(username: str, bridge_key: str, db: Session = Depends(get_db)):
    """
    MT5 bridge polls this endpoint every 5 seconds.
    Returns orders with status=QUEUED and marks them SENT.
    """
    cfg = _verify_bridge_key(username, bridge_key, db)
    user = db.query(User).filter(User.username == username).first()

    queued = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "QUEUED",
        )
        .all()
    )

    orders_out = []
    for o in queued:
        orders_out.append(
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry_price": o.requested_entry,
                "stop_loss": o.stop_loss,
                "take_profit_1": o.take_profit_1,
                "take_profit_2": o.take_profit_2,
            }
        )
        o.status = "SENT"
        o.sent_at = datetime.utcnow()

    db.commit()
    return {"orders": orders_out, "count": len(orders_out)}


@app.post("/bot/executed/{username}")
def order_executed(
    username: str,
    bridge_key: str,
    req: BridgeExecutedRequest,
    db: Session = Depends(get_db),
):
    """Bridge reports a successful MT5 execution."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.mt5_ticket = req.mt5_ticket
    order.filled_price = req.filled_price
    order.current_sl = order.stop_loss
    order.status = "ACTIVE"
    order.executed_at = (
        datetime.fromisoformat(req.executed_at)
        if req.executed_at
        else datetime.utcnow()
    )

    db.commit()
    return {"success": True, "message": f"Order {req.order_id} marked ACTIVE"}


@app.post("/bot/rejected/{username}")
def order_rejected(
    username: str,
    bridge_key: str,
    req: BridgeRejectedRequest,
    db: Session = Depends(get_db),
):
    """Bridge reports a failed MT5 execution."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = "REJECTED"
    order.reject_reason = req.reason
    db.commit()
    return {
        "success": True,
        "message": f"Order {req.order_id} marked REJECTED: {req.reason}",
    }


@app.post("/bot/position_update/{username}")
def position_update(
    username: str,
    bridge_key: str,
    req: PositionUpdateRequest,
    db: Session = Depends(get_db),
):
    """
    Bridge sends current position state every 10 seconds.
    We run TradeManager on each position and return instructions.
    """
    cfg = _verify_bridge_key(username, bridge_key, db)
    instructions = []

    for pos in req.positions:
        order = db.query(BotOrder).filter(BotOrder.mt5_ticket == pos.mt5_ticket).first()
        if not order:
            continue

        # Update floating state
        order.current_price = pos.current_price
        order.floating_pnl_usd = pos.floating_pnl
        if pos.current_sl:
            order.current_sl = pos.current_sl

        # Run Trade Manager
        decision = TradeManager.evaluate_position(order, pos.current_price, cfg)
        action = decision.get("action", "HOLD")

        if action == "HOLD":
            pass

        elif action in ("MOVE_SL_BREAKEVEN", "TRAIL_SL"):
            new_sl = decision.get("new_sl")
            if new_sl:
                order.current_sl = new_sl
                if action == "MOVE_SL_BREAKEVEN":
                    order.sl_at_breakeven = True
                else:
                    order.trailing_active = True
                instructions.append(
                    {
                        "order_id": order.id,
                        "mt5_ticket": pos.mt5_ticket,
                        "symbol": order.symbol,
                        "action": action,
                        "new_sl": new_sl,
                        "details": decision.get("details"),
                    }
                )

        elif action == "PARTIAL_CLOSE":
            order.tp1_closed = True
            order.sl_at_breakeven = True
            order.current_sl = decision.get("new_sl") or order.current_sl
            instructions.append(
                {
                    "order_id": order.id,
                    "mt5_ticket": pos.mt5_ticket,
                    "symbol": order.symbol,
                    "action": "PARTIAL_CLOSE",
                    "close_pct": cfg.partial_close_pct,
                    "new_sl": decision.get("new_sl"),
                    "details": decision.get("details"),
                }
            )

        elif action == "CLOSE":
            instructions.append(
                {
                    "order_id": order.id,
                    "mt5_ticket": pos.mt5_ticket,
                    "symbol": order.symbol,
                    "action": "CLOSE",
                    "close_reason": decision.get("close_reason"),
                    "details": decision.get("details"),
                }
            )

    db.commit()
    return {"instructions": instructions, "processed": len(req.positions)}


@app.post("/bot/partial_closed/{username}")
def partial_closed(
    username: str,
    bridge_key: str,
    req: PartialClosedRequest,
    db: Session = Depends(get_db),
):
    """Bridge confirms a partial close was executed."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.tp1_closed = True
    db.commit()
    return {
        "success": True,
        "message": f"TP1 partial close confirmed for order {req.order_id}",
    }


@app.post("/bot/closed/{username}")
def position_closed(
    username: str, bridge_key: str, req: ClosedRequest, db: Session = Depends(get_db)
):
    """Bridge reports a position was fully closed."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    entry = order.filled_price or order.requested_entry or 0
    close = req.close_price or 0

    pnl = (
        TradeManager.calculate_pnl(
            direction=order.direction,
            entry=entry,
            close=close,
            lot_size=order.lot_size,
            symbol=order.symbol,
        )
        if (entry and close)
        else None
    )

    order.status = "CLOSED"
    order.close_price = close
    order.close_reason = req.close_reason
    order.realised_pnl_usd = pnl
    order.closed_at = (
        datetime.fromisoformat(req.closed_at) if req.closed_at else datetime.utcnow()
    )

    # Also log to trade journal
    user = db.query(User).filter(User.id == order.user_id).first()
    if user and entry and close:
        journal = DBJournalEntry(
            user_id=user.id,
            symbol=order.symbol,
            direction=order.direction,
            entry_price=entry,
            exit_price=close,
            volume=order.lot_size,
            result="WIN"
            if (pnl or 0) > 0
            else ("LOSS" if (pnl or 0) < 0 else "BREAK_EVEN"),
            pnl_usd=pnl or 0,
            notes=f"Auto-trade | Reason: {req.close_reason} | Ticket: {req.mt5_ticket}",
        )
        db.add(journal)

    db.commit()
    return {
        "success": True,
        "order_id": order.id,
        "symbol": order.symbol,
        "pnl_usd": pnl,
        "result": "WIN" if (pnl or 0) > 0 else "LOSS",
        "message": f"Order {req.order_id} closed — {'WIN' if (pnl or 0) > 0 else 'LOSS'} ${abs(pnl or 0):.2f}",
    }


# ════════════════════════════════════════════════════════════════
#  4. PERFORMANCE & HISTORY
# ════════════════════════════════════════════════════════════════


@app.get("/bot/history/{username}")
def bot_trade_history(username: str, limit: int = 50, db: Session = Depends(get_db)):
    """Full closed trade history for the bot."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    orders = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
        )
        .order_by(BotOrder.closed_at.desc())
        .limit(limit)
        .all()
    )

    trades = []
    for o in orders:
        trades.append(
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry": o.filled_price,
                "close": o.close_price,
                "pnl_usd": o.realised_pnl_usd,
                "result": "WIN" if (o.realised_pnl_usd or 0) > 0 else "LOSS",
                "close_reason": o.close_reason,
                "risk_percent": o.risk_percent,
                "risk_usd": o.risk_usd,
                "mt5_ticket": o.mt5_ticket,
                "opened_at": o.executed_at.isoformat() if o.executed_at else None,
                "closed_at": o.closed_at.isoformat() if o.closed_at else None,
            }
        )

    total = len(orders)
    wins = sum(1 for o in orders if (o.realised_pnl_usd or 0) > 0)
    total_pnl = sum(o.realised_pnl_usd or 0 for o in orders)

    return {
        "username": username,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "trades": trades,
    }


@app.get("/bot/performance/{username}")
def bot_performance(username: str, db: Session = Depends(get_db)):
    """Aggregated performance stats: all-time, monthly, weekly breakdown."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    all_closed = (
        db.query(BotOrder)
        .filter(BotOrder.user_id == user.id, BotOrder.status == "CLOSED")
        .order_by(BotOrder.closed_at)
        .all()
    )

    def stats(orders):
        if not orders:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "pnl_usd": 0}
        wins = sum(1 for o in orders if (o.realised_pnl_usd or 0) > 0)
        pnl = sum(o.realised_pnl_usd or 0 for o in orders)
        return {
            "trades": len(orders),
            "wins": wins,
            "losses": len(orders) - wins,
            "win_rate": round(wins / len(orders) * 100, 1),
            "pnl_usd": round(pnl, 2),
        }

    now = datetime.utcnow()
    today = now.date()
    this_week_start = today - timedelta(days=today.weekday())
    this_month_start = today.replace(day=1)

    today_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() == today
    ]
    week_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() >= this_week_start
    ]
    month_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() >= this_month_start
    ]

    # Symbol breakdown
    by_symbol = {}
    for o in all_closed:
        if o.symbol not in by_symbol:
            by_symbol[o.symbol] = []
        by_symbol[o.symbol].append(o)
    symbol_stats = {sym: stats(orders) for sym, orders in by_symbol.items()}

    return {
        "username": username,
        "all_time": stats(all_closed),
        "this_month": stats(month_orders),
        "this_week": stats(week_orders),
        "today": stats(today_orders),
        "by_symbol": symbol_stats,
        "generated_at": now.isoformat(),
    }


@app.post("/bot/emergency_stop/{username}")
def emergency_stop(username: str, db: Session = Depends(get_db)):
    """
    EMERGENCY — immediately:
    1. Pauses the bot (no new orders)
    2. Cancels all QUEUED orders
    3. Returns a list of ACTIVE positions the user must manually close in MT5
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if cfg:
        cfg.is_active = False

    queued = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "QUEUED",
        )
        .all()
    )
    cancelled_count = len(queued)
    for o in queued:
        o.status = "CANCELLED"
        o.reject_reason = "EMERGENCY_STOP"

    active = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status.in_(["SENT", "EXECUTED", "ACTIVE"]),
        )
        .all()
    )

    db.commit()

    return {
        "success": True,
        "bot_paused": True,
        "cancelled_queued": cancelled_count,
        "active_positions": [
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "ticket": o.mt5_ticket,
                "lot_size": o.lot_size,
            }
            for o in active
        ],
        "action_required": (
            f"⚠️ Bot stopped. {len(active)} live position(s) still open in MT5. "
            "Close them manually in your MT5 terminal or restart the bridge to allow auto-close."
        )
        if active
        else "✅ All clear. No open positions.",
    }
    import uvicorn
    import os

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port)
