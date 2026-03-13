# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, time, requests, random, math
from datetime import datetime, timedelta
from ai_provider import get_ai_response
from sqlalchemy.orm import Session

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import load_history_df, run_backtest_from_signals, signals_from_sma, router as backtest_router
from models import (
    init_db, get_db, User, DemoAccount, DemoTrade,
    ChatMessage as DBChatMessage, TradeJournalEntry as DBJournalEntry,
    WatchlistItem, UserActivity, Conversation, UserMemory
)
from market_data import (
    get_symbol_analysis, format_for_ai_prompt,
    fetch_realtime_quote
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
                "last_refreshed": data.get("6. Last Refreshed")
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
                "last_refreshed": data.get("6. Last Refreshed")
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
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCAD", "AUDCAD", "EURAUD", "GBPAUD",
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

SYMBOL_BASE_PRICES = {
    "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 149.50, "AUDUSD": 0.6530,
    "USDCAD": 1.3600, "NZDUSD": 0.6050, "USDCHF": 0.8950, "EURGBP": 0.8550,
    "EURJPY": 162.0, "GBPJPY": 189.5, "AUDJPY": 97.5, "EURCAD": 1.4750,
    "AUDCAD": 0.8880, "EURAUD": 1.6600, "GBPAUD": 1.9450,
    "BTCUSDT": 82000.0, "ETHUSDT": 1900.0, "BNBUSDT": 580.0, "XRPUSDT": 2.1,
    "SOLUSDT": 130.0, "ADAUSDT": 0.75, "DOGEUSDT": 0.16, "DOTUSDT": 6.5,
    "MATICUSDT": 0.55, "LTCUSDT": 95.0, "SHIBUSDT": 0.000013, "TRXUSDT": 0.22,
    "AVAXUSDT": 25.0, "LINKUSDT": 13.5, "UNIUSDT": 8.5
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

def log_activity(user: User, action: str, symbol: str = None, details: dict = None, db: Session = None):
    if db:
        entry = UserActivity(user_id=user.id, action=action, symbol=symbol, details=details)
        db.add(entry)
        db.commit()

def get_user_profile_summary(user: User, db: Session) -> str:
    """Build a text summary of user behavior for AI personalization."""
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).order_by(DBJournalEntry.logged_at.desc()).limit(20).all()
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    activity = db.query(UserActivity).filter(UserActivity.user_id == user.id).order_by(UserActivity.created_at.desc()).limit(30).all()

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
        priority = ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "EURCAD", "EURAUD", "GBPAUD",
                    "BTCUSDT", "ETHUSDT", "USDCHF", "AUDUSD", "USDJPY", "SOLUSDT", "BNBUSDT"]
    elif 12 <= h < 17:
        priority = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSDT", "ETHUSDT", "SOLUSDT", "USDCAD",
                    "USDCHF", "EURGBP", "GBPJPY", "EURJPY", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT"]
    elif 17 <= h < 21:
        priority = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT",
                    "UNIUSDT", "ADAUSDT", "DOGEUSDT", "EURUSD", "GBPUSD", "USDCAD", "MATICUSDT", "TRXUSDT"]
    else:
        priority = ["USDJPY", "GBPJPY", "AUDJPY", "EURJPY", "AUDCAD", "AUDUSD", "NZDUSD",
                    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOTUSDT", "LTCUSDT", "XRPUSDT", "SHIBUSDT", "TRXUSDT"]
    # Add remaining symbols not already in list
    remaining = [s for s in TRADING_SYMBOLS if s not in priority]
    return priority + remaining


def generate_market_predictions(investment_amount_ngn: float = 0.0, user_profile: str = "") -> Dict[str, Any]:
    global PREDICTIONS_CACHE
    now = time.time()
    cache_key = f"preds_{int(investment_amount_ngn)}"
    if cache_key in PREDICTIONS_CACHE and (now - PREDICTIONS_CACHE[cache_key]["ts"] < CACHE_TTL):
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
    for sym in real_data_symbols:
        analysis = get_symbol_analysis(sym)
        block = format_for_ai_prompt(analysis)
        live_data_blocks.append(block)
        if analysis.get("live_price"):
            has_real_data = True

    # Remaining symbols — AI will use pattern reasoning
    remaining_symbols = [s for s in TRADING_SYMBOLS if s not in real_data_symbols]
    remaining_text = ", ".join(remaining_symbols)

    live_data_section = "\n\n".join(live_data_blocks)
    data_source_note = "LIVE Alpha Vantage data" if has_real_data else "AI pattern reasoning (live data unavailable)"

    prompt = f"""You are ARIA — elite institutional quantitative trading analyst. Data source: {data_source_note}

SESSION: {session} | UTC: {utc.strftime('%H:%M')} | Budget: {investment_amount_ngn:,.0f} NGN (~${inv_usd:,.2f} USD @ {ngn_rate:.2f}){personalization}

━━━ REAL LIVE MARKET DATA (use these EXACT indicator values) ━━━
{live_data_section}

━━━ ADDITIONAL SYMBOLS (use pattern reasoning) ━━━
{remaining_text}

━━━ YOUR ANALYSIS RULES ━━━
For symbols WITH live data above:
- Use the EXACT RSI, MACD, EMA, Bollinger, ATR, Support/Resistance values provided
- You are INTERPRETING real data — do NOT make up different values
- Use the provided ATR-based SL/TP levels as your base, adjust only slightly
- REJECT the symbol if indicators show conflicting signals

For symbols without live data:
- Use institutional pattern reasoning
- Be MORE conservative (lower confidence: 73-80 max)

QUALITY GATES — skip any signal that fails:
✗ RSI outside 28-72 range → skip
✗ MACD and RSI disagree on direction → skip
✗ Price between EMA20 and EMA50 with no clear bias → skip
✗ Volume below average with no squeeze → skip
✗ R:R below 1:2 → skip

POSITION SIZING: 2% risk per trade
- Risk per trade: {investment_amount_ngn * 0.02:,.0f} NGN = ${investment_amount_ngn * 0.02 / ngn_rate:.2f} USD

Return ONLY a valid JSON array. Include only signals with confidence >= 73. Exclude HOLD signals.
[
  {{
    "symbol": "SYMBOL",
    "signal": "STRONG_BUY|BUY|SELL|STRONG_SELL",
    "confidence": 73-97,
    "data_source": "live_data|ai_reasoning",
    "category": "forex|crypto",
    "timeframe": "scalp(5-15m)|intraday(1-4h)|swing(daily)",
    "session_fit": "excellent|good|fair",
    "live_price": "actual price from data",
    "entry_price": "specific entry level",
    "stop_loss": "ATR-based stop",
    "take_profit_1": "1:1 target",
    "take_profit_2": "1:2 target",
    "take_profit_3": "1:3 target",
    "risk_reward": "1:X.X",
    "hold_time": "duration",
    "position_size_ngn": "{investment_amount_ngn * 0.02:,.0f} NGN",
    "position_size_usd": "${investment_amount_ngn * 0.02 / ngn_rate:.2f}",
    "back_out_trigger": "exact price/condition that invalidates this trade",
    "indicators": {{
      "rsi": "value from live data + interpretation",
      "macd": "reading from live data",
      "ema_bias": "price vs EMA stack",
      "bollinger": "band position",
      "volume": "reading",
      "pattern": "chart pattern detected"
    }},
    "key_levels": {{
      "support": "level",
      "resistance": "level"
    }},
    "rationale": "3 sentences referencing the EXACT indicator values from live data"
  }}
]
Output ONLY the JSON array. No text outside the array."""

    try:
        content = get_ai_response(prompt)
        if not content or content.startswith("ERROR:"):
            return {"success": False, "error": content or "Empty response"}
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        data = [s for s in data if isinstance(s, dict) and int(s.get("confidence", 0)) >= 73]
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
                "signals_from_ai_reasoning": len(data) - live_count
            },
            "investment_context_ngn": investment_amount_ngn,
            "usd_ngn_rate": ngn_rate,
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data),
            "strong_signals_count": len(strong),
            "top_picks": strong[:5],
            "disclaimer": "Signals are data-driven using live Alpha Vantage feeds. Trading involves substantial risk."
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

    prompt = f"""You are ARIA, an elite institutional trading analyst.

━━━ REAL LIVE MARKET DATA FOR {symbol} ━━━
{live_data_block}

━━━ USER CONTEXT ━━━
{profile}
Balance: {user.balance_ngn:,.0f} NGN (~${inv_usd:,.2f} USD)
Trading Style: {user.trading_style} | Risk Tolerance: {user.risk_tolerance}
Time: {utc.strftime('%Y-%m-%d %H:%M')} UTC
Position size (2% risk): {user.balance_ngn * 0.02:,.0f} NGN = ${user.balance_ngn * 0.02 / ngn_rate:.2f} USD

━━━ ANALYSIS INSTRUCTIONS ━━━
{"Use the EXACT indicator values from the live data block above. Do NOT invent different values." if has_live else "No live data available — use institutional pattern reasoning."}

Provide a personalized deep analysis for {symbol} considering:
- The user's {user.trading_style} style (adjust hold time and entry precision accordingly)
- Their {user.risk_tolerance} risk tolerance (adjust position size and SL distance)
- Their trade history and win rate in their profile above
- {"The EXACT RSI, MACD, EMA, ATR, support/resistance from live data" if has_live else "Conservative estimates since no live data"}

Return ONLY a valid JSON object:
{{
  "symbol": "{symbol}",
  "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0-99,
  "data_source": "{"live_data" if has_live else "ai_reasoning"}",
  "category": "forex|crypto",
  "timeframe": "scalp|intraday|swing",
  "session_fit": "excellent|good|fair|poor",
  "live_price": "{live_price if live_price else 'unavailable'}",
  "entry_price": "specific price (use live data levels)",
  "stop_loss": "ATR-based SL (use ATR from live data if available)",
  "take_profit_1": "1:1 target",
  "take_profit_2": "1:2 target",
  "take_profit_3": "1:3 target",
  "risk_reward": "1:X",
  "hold_time": "duration matching user's {user.trading_style} style",
  "position_size_ngn": "{user.balance_ngn * 0.02:,.0f} NGN (2% of balance)",
  "position_size_usd": "${user.balance_ngn * 0.02 / ngn_rate:.2f} USD",
  "back_out_trigger": "exact price or condition invalidating this setup",
  "indicators": {{
    "rsi": "EXACT value from live data + interpretation",
    "macd": "EXACT reading from live data",
    "ema_bias": "EXACT EMA position from live data",
    "bollinger": "EXACT band position from live data",
    "volume": "EXACT volume reading",
    "pattern": "chart pattern detected"
  }},
  "key_levels": {{
    "support": "EXACT level from live data",
    "resistance": "EXACT level from live data"
  }},
  "personalized_note": "Specific note for this user's trading style, history, and risk profile",
  "rationale": "3 sentences using EXACT indicator values from live data"
}}"""

    try:
        content = get_ai_response(prompt)
        if not content or content.startswith("ERROR:"):
            return {"symbol": symbol, "signal": "HOLD", "confidence": 0, "error": "AI unavailable", "live_price": live_price}
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        data["personalized"] = True
        data["raw_live_analysis"] = analysis
        return data
    except Exception as e:
        return {"symbol": symbol, "signal": "HOLD", "confidence": 0, "error": str(e), "live_price": live_price}

# ----------------------------
# Simulator
# ----------------------------
def simulate_ohlcv(symbol: str, num_candles: int = 150, interval_minutes: int = 15, seed: int = None) -> List[Dict]:
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
        low  = min(price, close) * (1 - abs(rng.gauss(0, cv)))
        bvol = 2000 if is_crypto else 200000
        vol  = rng.uniform(bvol * 0.5, bvol * 2.5)
        dp   = 4 if is_crypto else 5
        ts   = now + timedelta(minutes=interval_minutes * i)
        candles.append({
            "time": int(ts.timestamp()), "datetime": ts.strftime("%Y-%m-%d %H:%M"),
            "open": round(price, dp), "high": round(high, dp),
            "low": round(low, dp), "close": round(close, dp), "volume": round(vol, 2)
        })
        price = close
    return candles

def run_strategy_on_candles(candles: List[Dict], initial_balance: float = 10000.0) -> Dict:
    if len(candles) < 51:
        return {"error": "Need 51+ candles"}
    closes = [c["close"] for c in candles]

    def sma(d, p):
        return [None] * (p - 1) + [sum(d[i:i+p]) / p for i in range(len(d) - p + 1)]

    s20, s50 = sma(closes, 20), sma(closes, 50)
    balance, position, trades, equity_curve = initial_balance, None, [], []

    for i in range(50, len(candles)):
        c = candles[i]
        if s20[i] and s50[i] and s20[i-1] and s50[i-1]:
            if s20[i-1] <= s50[i-1] and s20[i] > s50[i] and position is None:
                position = {"entry": c["close"], "time": c["time"], "datetime": c["datetime"]}
            elif s20[i-1] >= s50[i-1] and s20[i] < s50[i] and position is not None:
                pnl_pct = (c["close"] - position["entry"]) / position["entry"]
                pnl_usd = balance * 0.1 * pnl_pct
                balance += pnl_usd
                trades.append({
                    "type": "BUY", "entry_price": position["entry"],
                    "entry_time": position["time"], "entry_datetime": position["datetime"],
                    "exit_price": c["close"], "exit_time": c["time"], "exit_datetime": c["datetime"],
                    "pnl_pct": round(pnl_pct * 100, 3), "pnl_usd": round(pnl_usd, 2),
                    "result": "WIN" if pnl_usd > 0 else "LOSS"
                })
                position = None
        equity_curve.append({"time": c["time"], "datetime": c["datetime"], "equity": round(balance, 2)})

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    return {
        "initial_balance": initial_balance, "final_balance": round(balance, 2),
        "total_pnl_usd": round(balance - initial_balance, 2),
        "total_pnl_pct": round((balance - initial_balance) / initial_balance * 100, 2),
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_usd": round(sum(t["pnl_usd"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss_usd": round(sum(t["pnl_usd"] for t in losses) / len(losses), 2) if losses else 0,
        "strategy": "SMA 20/50 Crossover", "trades": trades, "equity_curve": equity_curve
    }

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Trading Bot — ARIA v3.1")
app.include_router(backtest_router, prefix="/api")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ----------------------------
# Pydantic Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    username: Optional[str] = "guest"
    conversation_id: Optional[int] = None   # omit to auto-create a new thread

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
        "status": "healthy", "service": "ARIA — AI Trading Bot", "version": "3.1",
        "pairs_monitored": len(TRADING_SYMBOLS),
        "persistence": "PostgreSQL",
        "features": [
            "30 Pairs (15 Forex + 15 Crypto)", "Personalized Signals per User",
            "NGN Position Sizing", "Risk Calculator", "Visual Simulator",
            "Trade Journal (DB)", "Watchlist (DB)", "Chat History (DB)",
            "User Profiles + Activity Tracking", "Trade Idea Scorer", "Demo Account (DB)"
        ]
    }

# ----------------------------
# User Profile
# ----------------------------
@app.post("/user/{username}")
def create_or_update_user(username: str, profile: UserProfileUpdate, db: Session = Depends(get_db)):
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
    return {"status": "success", "user": {"username": user.username, "balance_ngn": user.balance_ngn,
        "risk_tolerance": user.risk_tolerance, "trading_style": user.trading_style,
        "preferred_pairs": user.preferred_pairs, "created_at": user.created_at.isoformat()}}

@app.get("/user/{username}")
def get_user_profile(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    wl = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).all()
    wins = [j for j in journal if j.result == "WIN"]
    return {
        "username": user.username, "balance_ngn": user.balance_ngn,
        "risk_tolerance": user.risk_tolerance, "trading_style": user.trading_style,
        "preferred_pairs": user.preferred_pairs, "watchlist": [w.symbol for w in wl],
        "stats": {
            "total_trades": len(journal), "wins": len(wins), "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(sum(j.pnl_usd for j in journal), 2)
        },
        "last_active": user.last_active.isoformat(), "member_since": user.created_at.isoformat()
    }

# ----------------------------
# Predictions
# ----------------------------
@app.get("/predictions")
def get_predictions_public(amount_ngn: float = 0.0, username: Optional[str] = None, db: Session = Depends(get_db)):
    profile = ""
    if username:
        user = get_or_create_user(username, db)
        profile = get_user_profile_summary(user, db)
        log_activity(user, "viewed_predictions", db=db)
    return generate_market_predictions(amount_ngn, user_profile=profile)

@app.get("/get_predictions")
def get_personalized_predictions(username: str, symbol: str, db: Session = Depends(get_db)):
    """FIXED: Personalized signal for a specific symbol without needing a CSV file."""
    user = get_or_create_user(username, db)
    log_activity(user, "viewed_signal", symbol=symbol.upper(), db=db)
    signal = get_personalized_signal(symbol, user, db)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "signal": signal,
        "ngn_rate": get_ngn_rate(),
        "generated_at": datetime.utcnow().isoformat()
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
        "market_health": "healthy", "active_session": preds.get("active_session"),
        "total_signals": len(signals), "forex_signals": len(forex), "crypto_signals": len(crypto),
        "strong_signals": len(strong), "best_opportunities": strong[:5],
        "usd_ngn_rate": preds.get("usd_ngn_rate"),
        "generated_at": preds.get("generated_at")
    }

@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)):
    symbol = symbol.upper()
    now = time.time()
    if symbol in ADVICE_CACHE and (now - ADVICE_CACHE[symbol]["ts"] < 3600):
        return ADVICE_CACHE[symbol]["data"]
    if username:
        user = get_or_create_user(username, db)
        log_activity(user, "viewed_advice", symbol=symbol, db=db)
    live_price = get_live_price(symbol)
    prompt = f"""Deep institutional analysis for {symbol}. Live price: {live_price}.
Return ONLY valid JSON:
{{
  "symbol": "{symbol}", "live_price": "{live_price}",
  "recommendation": "BUY|SELL|HOLD", "confidence": 0-99,
  "trade_setup": {{
    "entry": "price", "stop_loss": "ATR-based",
    "take_profit_1": "1:1", "take_profit_2": "1:2", "take_profit_3": "1:3",
    "risk_reward": "1:X", "hold_time": "duration"
  }},
  "technical_analysis": {{
    "trend": "bullish/bearish/ranging", "rsi": "value + interpretation",
    "macd": "signal", "ema_stack": "alignment", "volume": "assessment",
    "key_support": "level", "key_resistance": "level", "pattern": "detected pattern"
  }},
  "risk_assessment": {{"risk_level": "low/medium/high", "max_position_pct": "X%", "invalidation": "condition"}},
  "rationale": "3 institutional sentences with specific levels"
}}"""
    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        result = {"success": True, "generated_at": datetime.utcnow().isoformat(), "advice": data}
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


def extract_and_save_memories(user: User, user_message: str, aria_response: str, db: Session):
    """
    Ask the AI to extract any personal facts from the exchange and save them.
    Only runs if the message might contain personal info (cheap heuristic).
    """
    keywords = ["my name", "i am", "i'm", "i work", "i live", "i want", "i trade",
                "my goal", "my balance", "i have", "call me", "know that", "remember"]
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
            key   = str(fact.get("key",   "")).strip().lower().replace(" ", "_")
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            existing_mem = db.query(UserMemory).filter(
                UserMemory.user_id == user.id,
                UserMemory.key == key
            ).first()
            if existing_mem:
                existing_mem.value = value
                existing_mem.updated_at = datetime.utcnow()
            else:
                db.add(UserMemory(user_id=user.id, key=key, value=value, source="user_stated"))
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
def create_conversation(username: str, title: Optional[str] = None, db: Session = Depends(get_db)):
    """Start a new named conversation thread (like a new chat in ChatGPT sidebar)."""
    user = get_or_create_user(username, db)
    conv = Conversation(user_id=user.id, title=title or "New Chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {
        "conversation_id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat()
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
        msg_count = db.query(DBChatMessage).filter(DBChatMessage.conversation_id == c.id).count()
        result.append({
            "conversation_id": c.id,
            "title":           c.title,
            "message_count":   msg_count,
            "last_message":    last_msg.content[:80] + "..." if last_msg and len(last_msg.content) > 80 else (last_msg.content if last_msg else ""),
            "last_role":       last_msg.role if last_msg else None,
            "created_at":      c.created_at.isoformat(),
            "updated_at":      c.updated_at.isoformat(),
        })
    return {
        "username":      username,
        "total_chats":   len(result),
        "conversations": result
    }


@app.get("/conversations/{username}/{conv_id}")
def get_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Get full message history of a specific conversation."""
    user = get_or_create_user(username, db)
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == user.id
    ).first()
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
        "title":           conv.title,
        "username":        username,
        "messages": [
            {
                "id":      m.id,
                "role":    m.role,
                "content": m.content,
                "time":    m.created_at.isoformat()
            }
            for m in msgs
        ],
        "total_messages": len(msgs),
        "created_at":     conv.created_at.isoformat(),
    }


@app.patch("/conversations/{username}/{conv_id}/rename")
def rename_conversation(username: str, conv_id: int, title: str, db: Session = Depends(get_db)):
    """Rename a conversation."""
    user = get_or_create_user(username, db)
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == user.id
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.title = title
    db.commit()
    return {"status": "success", "conversation_id": conv_id, "new_title": title}


@app.delete("/conversations/{username}/{conv_id}")
def delete_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages."""
    user = get_or_create_user(username, db)
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id,
        Conversation.user_id == user.id
    ).first()
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
        conv = db.query(Conversation).filter(
            Conversation.id == conv_id,
            Conversation.user_id == user.id
        ).first()

    if not conv:
        # Auto-create a new conversation
        conv = Conversation(user_id=user.id, title=auto_title_conversation(chat.message))
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
    preds    = generate_market_predictions(user.balance_ngn)
    signals  = preds.get("signals", [])
    profile  = get_user_profile_summary(user, db)

    # ── build system prompt ──
    system_prompt = f"""You are ARIA — Advanced Revenue Intelligence Analyst.
You are a world-class AI trading strategist with a warm, confident personality.
You remember everything about the people you work with and use that knowledge naturally.

━━━ WHO YOU ARE TALKING TO ━━━
Name: {name}
{profile}
Live USD/NGN Rate: {ngn_rate:.2f}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC

━━━ WHAT YOU REMEMBER ABOUT {name.upper()} ━━━
{memories}

━━━ CURRENT TOP MARKET SIGNALS ━━━
{json.dumps([{{
    "symbol": s.get("symbol"),
    "signal": s.get("signal"),
    "confidence": s.get("confidence"),
    "entry": s.get("entry_price"),
    "data_source": s.get("data_source")
}} for s in signals[:5]], indent=2)}

━━━ THIS CONVERSATION (recent history) ━━━
{history_text if history_text else "This is the start of this conversation."}

━━━ YOUR RULES ━━━
1. Address {name} by name naturally (not every sentence — just when it feels right)
2. If {name} tells you something personal, acknowledge it warmly and remember it
3. If {name} mentions an amount in NGN, instantly convert it: ₦X = ${{X / {ngn_rate:.2f}:.2f}} USD
4. Recommend 2% risk per trade for their style ({user.trading_style}) and tolerance ({user.risk_tolerance})
5. Always show exact calculations — never say "roughly" without showing the maths
6. If you don't know something live (price, news), say so clearly
7. End responses with a brief risk note only if giving a trade recommendation
8. Be warm, direct, and specific — avoid vague generalities"""

    full_prompt = f"{system_prompt}\n\n{name}: {chat.message}\nARIA:"
    response = get_ai_response(full_prompt)

    # ── save messages to DB ──
    db.add(DBChatMessage(user_id=user.id, conversation_id=conv_id, role="user", content=chat.message))
    db.add(DBChatMessage(user_id=user.id, conversation_id=conv_id, role="aria", content=response))

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
        "success":         True,
        "username":        chat.username,
        "display_name":    name,
        "conversation_id": conv_id,
        "conversation_title": conv.title,
        "response":        response,
        "timestamp":       datetime.utcnow().isoformat()
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
                "role":            m.role,
                "content":         m.content,
                "conversation_id": m.conversation_id,
                "time":            m.created_at.isoformat()
            }
            for m in msgs
        ],
        "total": len(msgs)
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
        "username":     username,
        "display_name": user.display_name,
        "memory_count": len(memories),
        "memories": [
            {
                "key":        m.key,
                "value":      m.value,
                "source":     m.source,
                "updated_at": m.updated_at.isoformat()
            }
            for m in memories
        ],
        "note": "ARIA uses these facts to personalise every conversation."
    }


@app.post("/memory/{username}")
def add_memory(username: str, key: str, value: str, db: Session = Depends(get_db)):
    """Manually tell ARIA something to remember about you."""
    user = get_or_create_user(username, db)
    key = key.strip().lower().replace(" ", "_")
    existing = db.query(UserMemory).filter(
        UserMemory.user_id == user.id, UserMemory.key == key
    ).first()
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
    deleted = db.query(UserMemory).filter(
        UserMemory.user_id == user.id,
        UserMemory.key == key.lower()
    ).delete()
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
        "symbol": req.symbol.upper(), "balance_ngn": req.balance_ngn,
        "balance_usd": round(balance_usd, 2), "risk_percent": req.risk_percent,
        "risk_amount_ngn": round(risk_ngn, 2), "risk_amount_usd": round(risk_usd, 4),
        "entry_price": req.entry_price, "stop_loss_price": req.stop_loss_price,
        "stop_distance_pct": round(pct_dist * 100, 4),
        "recommended_position_usd": round(pos_usd, 2),
        "units": round(units, 6),
        "pip_distance": round(pip_dist, 1) if not is_crypto else None,
        "note": f"Risking {req.risk_percent}% of your NGN balance per this trade."
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
    exists = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol).first()
    if not exists:
        db.add(WatchlistItem(user_id=user.id, symbol=symbol))
        db.commit()
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    return {"status": "success", "watchlist": [w.symbol for w in items]}

@app.delete("/watchlist/{username}/{symbol}")
def remove_from_watchlist(username: str, symbol: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol.upper()).delete()
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
        result.append({
            "symbol": w.symbol, "live_price": price,
            "signal": sig.get("signal", "N/A"), "confidence": sig.get("confidence", "N/A"),
            "entry_price": sig.get("entry_price", "N/A"), "stop_loss": sig.get("stop_loss", "N/A"),
            "hold_time": sig.get("hold_time", "N/A"), "added_at": w.added_at.isoformat()
        })
    return {"username": username, "watchlist": result, "total": len(result)}

# ----------------------------
# Trade Journal — DB persistent
# ----------------------------
@app.post("/journal/{username}")
def add_journal_entry(username: str, entry: TradeJournalEntryRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    record = DBJournalEntry(
        user_id=user.id, symbol=entry.symbol.upper(), direction=entry.direction.upper(),
        entry_price=entry.entry_price, exit_price=entry.exit_price, volume=entry.volume,
        result=entry.result.upper(), pnl_usd=entry.pnl_usd, notes=entry.notes
    )
    db.add(record)
    log_activity(user, "logged_trade", symbol=entry.symbol.upper(),
                 details={"result": entry.result, "pnl": entry.pnl_usd}, db=db)
    db.commit()
    db.refresh(record)
    return {"status": "success", "entry_id": record.id, "logged_at": record.logged_at.isoformat()}

@app.get("/journal/{username}")
def get_journal(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    entries = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).order_by(DBJournalEntry.logged_at.desc()).all()
    if not entries:
        return {"username": username, "entries": [], "stats": {}}
    wins   = [e for e in entries if e.result == "WIN"]
    losses = [e for e in entries if e.result == "LOSS"]
    total_pnl = sum(e.pnl_usd for e in entries)
    sym_freq = {}
    for e in entries:
        sym_freq[e.symbol] = sym_freq.get(e.symbol, 0) + 1
    best  = max(entries, key=lambda e: e.pnl_usd)
    worst = min(entries, key=lambda e: e.pnl_usd)

    return {
        "username": username,
        "entries": [{"id": e.id, "symbol": e.symbol, "direction": e.direction,
                     "entry_price": e.entry_price, "exit_price": e.exit_price,
                     "volume": e.volume, "result": e.result, "pnl_usd": e.pnl_usd,
                     "notes": e.notes, "logged_at": e.logged_at.isoformat()} for e in entries],
        "stats": {
            "total_trades": len(entries), "wins": len(wins), "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(entries) * 100, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(sum(e.pnl_usd for e in wins) / len(wins), 2) if wins else 0,
            "avg_loss_usd": round(sum(e.pnl_usd for e in losses) / len(losses), 2) if losses else 0,
            "most_traded_symbol": max(sym_freq, key=sym_freq.get),
            "best_trade": {"symbol": best.symbol, "pnl_usd": best.pnl_usd},
            "worst_trade": {"symbol": worst.symbol, "pnl_usd": worst.pnl_usd}
        }
    }

# ----------------------------
# Demo Account — DB persistent
# ----------------------------
@app.post("/demo/open_account/{username}")
def open_demo_account(username: str, initial_balance: float = 10000.0, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    existing = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if existing:
        return {"status": "exists", "message": f"Demo account already exists. Balance: ${existing.balance:,.2f}", "balance": existing.balance}
    acct = DemoAccount(user_id=user.id, balance=initial_balance)
    db.add(acct)
    db.commit()
    db.refresh(acct)
    return {"status": "success", "message": f"Demo account opened for {username}", "balance": acct.balance, "account_id": acct.id}

@app.post("/demo/execute_trade/{username}")
def execute_demo_trade(username: str, req: DemoTradeRequest, db: Session = Depends(get_db)):
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
        raise HTTPException(status_code=400, detail=f"Could not fetch live price for {symbol}")

    trade = DemoTrade(
        demo_account_id=acct.id, symbol=symbol, entry_price=entry_price,
        current_price=entry_price, volume=req.volume, trade_type=req.trade_type.upper()
    )
    db.add(trade)
    log_activity(user, "opened_demo_trade", symbol=symbol, details={"entry": entry_price, "volume": req.volume}, db=db)
    db.commit()
    db.refresh(trade)
    return {"status": "success", "trade_id": trade.id, "symbol": symbol, "entry_price": entry_price, "volume": req.volume, "type": trade.trade_type}

@app.post("/demo/close_trade/{username}/{trade_id}")
def close_demo_trade(username: str, trade_id: int, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        raise HTTPException(status_code=404, detail="No demo account found")
    trade = db.query(DemoTrade).filter(DemoTrade.id == trade_id, DemoTrade.demo_account_id == acct.id).first()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    exit_price = get_live_price(trade.symbol)
    if exit_price == 0:
        exit_price = trade.current_price
    pnl = (exit_price - trade.entry_price) * trade.volume if trade.trade_type == "BUY" else (trade.entry_price - exit_price) * trade.volume
    trade.exit_price = exit_price
    trade.current_price = exit_price
    trade.pnl = pnl
    trade.is_active = False
    trade.closed_at = datetime.utcnow()
    acct.balance += pnl
    db.commit()
    return {"status": "closed", "symbol": trade.symbol, "entry": trade.entry_price, "exit": exit_price, "pnl_usd": round(pnl, 2), "new_balance": round(acct.balance, 2)}

@app.get("/demo/account/{username}")
def get_demo_account(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {"error": "No demo account found. POST /demo/open_account/{username} to create one."}

    active = db.query(DemoTrade).filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True).all()
    closed = db.query(DemoTrade).filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == False).all()

    # Refresh PnL for active trades
    total_floating_pnl = 0
    active_list = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (price - t.entry_price) * t.volume if t.trade_type == "BUY" else (t.entry_price - price) * t.volume
        t.current_price = price
        t.pnl = pnl
        total_floating_pnl += pnl
        active_list.append({"id": t.id, "symbol": t.symbol, "type": t.trade_type,
                             "entry_price": t.entry_price, "current_price": price,
                             "volume": t.volume, "pnl_usd": round(pnl, 2), "opened_at": t.opened_at.isoformat()})
    db.commit()

    wins = [t for t in closed if t.pnl > 0]
    return {
        "username": username, "balance_usd": round(acct.balance, 2),
        "floating_pnl_usd": round(total_floating_pnl, 2),
        "equity_usd": round(acct.balance + total_floating_pnl, 2),
        "active_trades": active_list,
        "trade_history": [{"id": t.id, "symbol": t.symbol, "type": t.trade_type,
                           "entry": t.entry_price, "exit": t.exit_price,
                           "pnl_usd": round(t.pnl, 2), "closed_at": t.closed_at.isoformat() if t.closed_at else None}
                          for t in closed[-20:]],
        "stats": {
            "total_closed_trades": len(closed), "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "realized_pnl_usd": round(sum(t.pnl for t in closed), 2)
        }
    }

@app.get("/demo/ai_feedback/{username}")
def get_demo_ai_feedback(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {"error": "No demo account found"}
    active = db.query(DemoTrade).filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True).all()
    if not active:
        return {"error": "No active trades to analyze"}

    summary = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (price - t.entry_price) * t.volume if t.trade_type == "BUY" else (t.entry_price - price) * t.volume
        summary.append(f"{t.trade_type} {t.volume} {t.symbol} @ {t.entry_price} | Now: {price} | PnL: ${pnl:.2f}")

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
    return {"username": username, "balance": acct.balance, "active_trades": summary, "ai_strategic_advice": feedback}

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
        "username": username, "balance_ngn": user.balance_ngn,
        "balance_usd": round(user.balance_ngn / ngn_rate, 2),
        "usd_ngn_rate": ngn_rate,
        "demo_account": {"balance_usd": demo_acct.balance if demo_acct else None, "exists": demo_acct is not None},
        "trading_stats": {
            "total_trades": len(journal), "wins": len(wins),
            "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(total_pnl, 2),
            "total_pnl_ngn": round(total_pnl * ngn_rate, 2)
        },
        "profile": {"risk_tolerance": user.risk_tolerance, "trading_style": user.trading_style,
                    "preferred_pairs": user.preferred_pairs, "member_since": user.created_at.isoformat()}
    }

# ----------------------------
# Simulator
# ----------------------------
@app.get("/live_data/{symbol}")
def get_live_technical_data(symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)):
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
        "indicators": analysis.get("indicators", {}),
        "key_levels": analysis.get("key_levels", {}),
        "trend_bias": analysis.get("trend_bias"),
        "fetched_at": datetime.utcnow().isoformat(),
        "cache_ttl_minutes": 15,
        "note": "Data from Alpha Vantage 5-min OHLCV candles. Cached 15 min to respect rate limits."
    }


@app.get("/simulator/candles/{symbol}")
def get_simulator_candles(symbol: str, candles: int = 120, interval: int = 15, seed: int = None):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed)
    return {
        "symbol": symbol, "interval_minutes": interval, "total_candles": len(candle_data),
        "chart_type": "candlestick", "candles": candle_data,
        "meta": {"base_price": SYMBOL_BASE_PRICES.get(symbol), "note": "Simulated via GBM — demo only."}
    }

@app.get("/simulator/run/{symbol}")
def run_simulator(symbol: str, candles: int = 150, interval: int = 15, balance: float = 10000.0, seed: int = None):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed)
    results = run_strategy_on_candles(candle_data, initial_balance=balance)
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    entry_markers = [{"time": t["entry_time"], "datetime": t["entry_datetime"], "price": t["entry_price"], "label": "BUY", "color": "#00ff88"} for t in results["trades"]]
    exit_markers  = [{"time": t["exit_time"],  "datetime": t["exit_datetime"],  "price": t["exit_price"],  "label": t["result"], "color": "#00bfff" if t["result"] == "WIN" else "#ff4444"} for t in results["trades"]]
    return {
        "symbol": symbol, "interval_minutes": interval, "strategy": results["strategy"],
        "performance": {
            "initial_balance_usd": results["initial_balance"], "final_balance_usd": results["final_balance"],
            "total_pnl_usd": results["total_pnl_usd"], "total_pnl_pct": results["total_pnl_pct"],
            "total_trades": results["total_trades"], "wins": results["wins"], "losses": results["losses"],
            "win_rate_pct": results["win_rate_pct"]
        },
        "chart_data": {"candles": candle_data, "equity_curve": results["equity_curve"], "entry_markers": entry_markers, "exit_markers": exit_markers},
        "trades": results["trades"],
        "disclaimer": "Simulated results. Past performance is not indicative of future results."
    }

# ----------------------------
# Connect Account
# ----------------------------
@app.post("/connect_account")
def connect_account(req: UserAccountRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(req.username, db)
    return {"status": "success", "message": f"Account connected for {req.username}", "user_id": user.id}
