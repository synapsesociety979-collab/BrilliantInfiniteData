# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, time, requests, random, math
from datetime import datetime, timedelta
from ai_provider import get_ai_response

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import load_history_df, run_backtest_from_signals, signals_from_sma, router as backtest_router

# ----------------------------
# Alpha Vantage Integration
# ----------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_alpha_vantage_forex(from_symbol: str, to_symbol: str = "USD") -> Dict:
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_symbol}&to_currency={to_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        if rate_data:
            return {
                "rate": float(rate_data.get("5. Exchange Rate") or 0),
                "bid": float(rate_data.get("8. Bid Price") or 0),
                "ask": float(rate_data.get("9. Ask Price") or 0),
                "last_refreshed": rate_data.get("6. Last Refreshed")
            }
    except Exception as e:
        print(f"Alpha Vantage Forex Error: {e}")
    return {}

def fetch_alpha_vantage_crypto(symbol: str) -> Dict:
    base_symbol = symbol.replace("USDT", "")
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_symbol}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        rate_data = data.get("Realtime Currency Exchange Rate", {})
        if rate_data:
            return {
                "price": float(rate_data.get("5. Exchange Rate") or 0),
                "last_refreshed": rate_data.get("6. Last Refreshed")
            }
    except Exception as e:
        print(f"Alpha Vantage Crypto Error: {e}")
    return {}

def get_ngn_rate() -> float:
    data = fetch_alpha_vantage_forex("USD", "NGN")
    return float(data.get("rate") or 1600.0)

def get_market_context() -> str:
    context = []
    key_forex = ["EUR", "GBP", "JPY", "AUD", "CAD"]
    key_crypto = ["BTC", "ETH", "SOL", "BNB"]
    for f in key_forex:
        data = fetch_alpha_vantage_forex(f)
        if data and data.get("rate"):
            context.append(f"{f}/USD: {data['rate']} (updated: {data.get('last_refreshed', 'N/A')})")
    for c in key_crypto:
        data = fetch_alpha_vantage_crypto(c)
        if data and data.get("price"):
            context.append(f"{c}/USD: {data['price']} (updated: {data.get('last_refreshed', 'N/A')})")
    return " | ".join(context) if context else "Live market data currently unavailable."

# ----------------------------
# Cache Configuration
# ----------------------------
PREDICTIONS_CACHE = {}
ADVICE_CACHE = {}
CACHE_DURATION_SECONDS = 600

# ----------------------------
# Trading symbols
# ----------------------------
TRADING_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCAD", "AUDCAD", "EURAUD", "GBPAUD",
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "MATICUSDT", "LTCUSDT", "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

# Default starting prices for GBM simulator
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
# AI Generation — Enhanced Accuracy
# ----------------------------
def generate_market_predictions(investment_amount_ngn: float = 0.0) -> Dict[str, Any]:
    global PREDICTIONS_CACHE
    current_time = time.time()
    cache_key = f"preds_{investment_amount_ngn}"

    if cache_key in PREDICTIONS_CACHE and (current_time - PREDICTIONS_CACHE[cache_key]["timestamp"] < CACHE_DURATION_SECONDS):
        return PREDICTIONS_CACHE[cache_key]["data"]

    live_data = get_market_context()
    ngn_rate = get_ngn_rate()
    investment_usd = investment_amount_ngn / ngn_rate if investment_amount_ngn > 0 else 0
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    if 7 <= hour < 12:
        session = "London Session (high liquidity — best for EURUSD, GBPUSD, EURGBP)"
    elif 12 <= hour < 17:
        session = "London/New York Overlap (PEAK liquidity — all pairs active)"
    elif 17 <= hour < 21:
        session = "New York Session (best for USD pairs and crypto)"
    else:
        session = "Asian Session (best for JPY pairs: USDJPY, GBPJPY, AUDJPY)"

    prompt = f"""You are an ELITE INSTITUTIONAL QUANTITATIVE ANALYST with 20 years of market experience.

LIVE MARKET DATA: {live_data}
CURRENT SESSION: {session} (UTC {now_utc.strftime('%H:%M')})
INVESTOR BUDGET: {investment_amount_ngn:,.0f} NGN = ~${investment_usd:,.2f} USD (Rate: {ngn_rate:.2f} NGN/USD)

STRICT ANALYSIS PROTOCOL — Apply ALL of the following for each asset:
1. MULTI-TIMEFRAME TREND: Confirm direction on 15m, 1h, AND 4h. Only trade WITH all 3 timeframes aligned.
2. KEY TECHNICAL INDICATORS:
   - RSI(14): Overbought>70 (sell bias), Oversold<30 (buy bias)
   - MACD(12,26,9): Signal line crossover direction
   - EMA(20,50,200): Price vs EMAs for trend strength
   - Bollinger Bands: Squeeze = breakout imminent; upper/lower band touches
   - Volume: Confirm moves with above-average volume
3. CHART PATTERNS: Identify any active pattern (H&S, double top/bottom, flags, wedges, triangles)
4. KEY S/R LEVELS: Use recent swing highs/lows and round numbers
5. SESSION FILTER: Prioritize pairs active in the current session above
6. RISK/REWARD: MINIMUM 1:2 ratio required — reject any setup below this
7. POSITION SIZING: Risk exactly 2% of {investment_amount_ngn:,.0f} NGN per trade

ASSETS TO ANALYZE:
FOREX (15): EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF, EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, AUDCAD, EURAUD, GBPAUD
CRYPTO (15): BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, DOTUSDT, MATICUSDT, LTCUSDT, SHIBUSDT, TRXUSDT, AVAXUSDT, LINKUSDT, UNIUSDT

Output ONLY a valid JSON array. Only include signals where confidence >= 72:
[
  {{
    "symbol": "PAIR",
    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 72-99,
    "category": "forex|crypto",
    "timeframe": "scalp(5-15m)|intraday(1-4h)|swing(daily)",
    "session_fit": "excellent|good|fair",
    "entry_price": "exact price",
    "stop_loss": "price (ATR-based)",
    "take_profit_1": "1:1 target",
    "take_profit_2": "1:2 target",
    "take_profit_3": "1:3 target",
    "risk_reward": "1:X",
    "hold_time": "e.g. 2-4 hours",
    "position_size_ngn": "2% risk amount in NGN",
    "position_size_usd": "equivalent in USD",
    "back_out_trigger": "Specific invalidation condition",
    "indicators": {{
      "rsi": "value + bias",
      "macd": "bullish/bearish crossover",
      "ema_trend": "above/below 20/50/200 EMA",
      "pattern": "chart pattern if any"
    }},
    "rationale": "3-sentence institutional justification referencing specific levels"
  }}
]
CRITICAL: Pure JSON only. No text before or after. Reject all setups below 1:2 R/R."""

    try:
        content = get_ai_response(prompt)
        if not content:
            return {"success": False, "error": "AI returned empty content"}
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}

        content = content.strip()
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            content = re.sub(r"```json|```", "", content).strip()

        data = json.loads(content)
        data = [s for s in data if isinstance(s, dict) and s.get("confidence", 0) >= 72]

        strong = [s for s in data if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]]
        result = {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "active_session": session,
            "investment_context_ngn": investment_amount_ngn,
            "usd_ngn_rate": ngn_rate,
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data),
            "strong_signals_count": len(strong),
            "top_picks": strong[:5],
            "disclaimer": "Trading involves substantial risk. Signals are AI-generated based on pattern confluence — not guaranteed."
        }
        PREDICTIONS_CACHE[cache_key] = {"data": result, "timestamp": current_time}
        return result
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}

# ----------------------------
# Helper: Dynamic AI Signal Filtering
# ----------------------------
def get_filtered_ai_signals(symbol: str, confidence_threshold: float = 72.0) -> List[Dict]:
    symbol = symbol.upper()
    try:
        df = load_history_df(symbol)
    except Exception:
        return [{"symbol": symbol, "signal": "HOLD", "confidence": 0, "note": "No history found"}]

    preds_resp = generate_market_predictions()
    if not preds_resp.get("success"):
        return [{"symbol": symbol, "signal": "HOLD", "confidence": 0, "note": preds_resp.get("error")}]

    raw_preds = preds_resp.get("signals", [])
    symbol_preds = [p for p in raw_preds if p.get("symbol", "").upper() == symbol]
    high_conf = [p for p in symbol_preds if p.get("confidence", 0) >= confidence_threshold]

    filtered_preds = []
    for p in high_conf:
        try:
            sma_signals = signals_from_sma(df)
            metrics, _, _ = run_backtest_from_signals(df, sma_signals)
            if metrics.get("total_return_pct", 0) > 0:
                filtered_preds.append(p)
        except Exception:
            filtered_preds.append(p)

    return filtered_preds if filtered_preds else [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]

# ----------------------------
# Simulator — Geometric Brownian Motion
# ----------------------------
def simulate_ohlcv(symbol: str, num_candles: int = 100, interval_minutes: int = 15, seed: int = None) -> List[Dict]:
    """Generate realistic OHLCV candles using Geometric Brownian Motion."""
    s = symbol.upper()
    base_price = SYMBOL_BASE_PRICES.get(s, 1.0)
    is_crypto = "USDT" in s

    # Volatility per candle (annualised sigma converted to per-candle)
    annual_sigma = 0.65 if is_crypto else 0.08
    dt = (interval_minutes / 60) / 8760  # fraction of a year
    sigma = annual_sigma * math.sqrt(dt)
    drift = 0.0001  # slight upward drift

    rng = random.Random(seed if seed else int(time.time()))
    candles = []
    price = base_price
    now = datetime.utcnow() - timedelta(minutes=interval_minutes * num_candles)

    for i in range(num_candles):
        # GBM step
        z = rng.gauss(0, 1)
        pct_change = drift * dt + sigma * z
        close = price * (1 + pct_change)

        # Build OHLC within the candle
        candle_vol = abs(rng.gauss(0, sigma * 0.5))
        high = max(price, close) * (1 + abs(rng.gauss(0, candle_vol)))
        low = min(price, close) * (1 - abs(rng.gauss(0, candle_vol)))
        open_p = price

        # Volume (simulate realistic volume spikes)
        base_vol = 1000 if is_crypto else 100000
        volume = rng.uniform(base_vol * 0.5, base_vol * 2.5)

        ts = now + timedelta(minutes=interval_minutes * i)
        candles.append({
            "time": int(ts.timestamp()),
            "datetime": ts.strftime("%Y-%m-%d %H:%M"),
            "open": round(open_p, 5 if not is_crypto else 4),
            "high": round(high, 5 if not is_crypto else 4),
            "low": round(low, 5 if not is_crypto else 4),
            "close": round(close, 5 if not is_crypto else 4),
            "volume": round(volume, 2)
        })
        price = close

    return candles

def run_strategy_on_candles(candles: List[Dict], initial_balance: float = 10000.0) -> Dict:
    """Run SMA(20/50) crossover strategy on candles, return equity curve + trades."""
    if len(candles) < 51:
        return {"error": "Not enough candles for strategy (need 51+)"}

    closes = [c["close"] for c in candles]

    def sma(data, period):
        return [None] * (period - 1) + [
            sum(data[i:i + period]) / period for i in range(len(data) - period + 1)
        ]

    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)

    balance = initial_balance
    position = None  # None | {"entry": price, "entry_idx": int, "type": "BUY"}
    trades = []
    equity_curve = []

    for i in range(50, len(candles)):
        c = candles[i]
        s20 = sma20[i]
        s20_prev = sma20[i - 1]
        s50 = sma50[i]
        s50_prev = sma50[i - 1]

        if s20 is None or s50 is None or s20_prev is None or s50_prev is None:
            equity_curve.append({"time": c["time"], "datetime": c["datetime"], "equity": round(balance, 2)})
            continue

        # BUY signal: SMA20 crosses above SMA50
        if s20_prev <= s50_prev and s20 > s50 and position is None:
            position = {"entry": c["close"], "entry_idx": i, "type": "BUY", "time": c["time"], "datetime": c["datetime"]}

        # SELL signal: SMA20 crosses below SMA50
        elif s20_prev >= s50_prev and s20 < s50 and position is not None:
            pnl_pct = (c["close"] - position["entry"]) / position["entry"]
            pnl_usd = balance * 0.1 * pnl_pct  # risk 10% per trade in simulator
            balance += pnl_usd
            trades.append({
                "type": "BUY",
                "entry_time": position["time"],
                "entry_datetime": position["datetime"],
                "entry_price": position["entry"],
                "exit_time": c["time"],
                "exit_datetime": c["datetime"],
                "exit_price": c["close"],
                "pnl_pct": round(pnl_pct * 100, 3),
                "pnl_usd": round(pnl_usd, 2),
                "result": "WIN" if pnl_usd > 0 else "LOSS"
            })
            position = None

        equity_curve.append({"time": c["time"], "datetime": c["datetime"], "equity": round(balance, 2)})

    # Performance stats
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    total_pnl = balance - initial_balance
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    avg_win = (sum(t["pnl_usd"] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t["pnl_usd"] for t in losses) / len(losses)) if losses else 0

    return {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "total_pnl_pct": round((total_pnl / initial_balance) * 100, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "strategy": "SMA 20/50 Crossover",
        "trades": trades,
        "equity_curve": equity_curve
    }

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Multi-User Trading Bot Backend")
app.include_router(backtest_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# In-memory stores
# ----------------------------
accounts_db: Dict[str, Any] = {}
demo_accounts_db: Dict[str, Any] = {}
users_db: Dict[str, Any] = {}
chat_history_db: Dict[str, List[Dict]] = {}
watchlist_db: Dict[str, List[str]] = {}
trade_journal_db: Dict[str, List[Dict]] = {}

# ----------------------------
# Pydantic Models
# ----------------------------
class Trade(BaseModel):
    symbol: str
    entry_price: float
    current_price: float
    volume: float
    type: str
    pnl: float = 0.0
    is_demo: bool = True

class AccountState(BaseModel):
    username: str
    balance: float
    currency: str = "NGN"
    trades: List[Trade] = []

class DemoAccount(BaseModel):
    username: str
    demo_balance: float = 10000.0
    currency: str = "USD"
    active_demo_trades: List[Trade] = []
    trade_history: List[Trade] = []

class UserAccount(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    username: Optional[str] = "guest"

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

class TradeJournalEntry(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    result: str
    pnl_usd: float
    notes: Optional[str] = ""

# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Trading Bot Backend",
        "version": "3.0",
        "pairs_monitored": len(TRADING_SYMBOLS),
        "features": [
            "30 Forex+Crypto Signals", "NGN Position Sizing", "Risk Calculator",
            "Visual Simulator", "Trade Journal", "Watchlist", "Trade Idea Scorer",
            "AI Chat with History", "Demo Account", "Account Monitor"
        ]
    }

# ----------------------------
# Predictions
# ----------------------------
@app.get("/predictions")
def get_predictions_public(amount_ngn: float = 0.0):
    return generate_market_predictions(amount_ngn)

@app.get("/get_predictions")
def get_predictions(username: str, symbol: str):
    predictions = get_filtered_ai_signals(symbol)
    return {"username": username, "symbol": symbol.upper(), "predictions": predictions}

@app.get("/market_analysis")
def market_analysis():
    predictions = generate_market_predictions()
    if predictions.get("success"):
        signals = predictions.get("signals", [])
        top_picks = [s for s in signals if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]][:5]
        forex_signals = [s for s in signals if s.get("category") == "forex"]
        crypto_signals = [s for s in signals if s.get("category") == "crypto"]
        return {
            "market_health": "healthy",
            "active_session": predictions.get("active_session"),
            "best_opportunities": top_picks,
            "forex_signals": len(forex_signals),
            "crypto_signals": len(crypto_signals),
            "total_signals_analyzed": predictions.get("total_signals", 0)
        }
    return {"error": "Analysis failed", "details": predictions.get("error")}

@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    symbol = symbol.upper()
    current_time = time.time()
    if symbol in ADVICE_CACHE:
        cache_data, ts = ADVICE_CACHE[symbol]
        if current_time - ts < 3600:
            return cache_data

    prompt = f"""Perform deep institutional analysis on {symbol}.
Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "recommendation": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "trade_setup": {{
    "entry": "price", "stop_loss": "price",
    "take_profit_1": "price", "take_profit_2": "price", "take_profit_3": "price",
    "risk_reward": "1:X", "hold_time": "duration"
  }},
  "technical_analysis": {{
    "trend": "bullish/bearish/ranging",
    "rsi": "value + interpretation",
    "macd": "signal",
    "key_support": "level",
    "key_resistance": "level",
    "pattern": "chart pattern"
  }},
  "risk_assessment": {{
    "risk_level": "low/medium/high",
    "max_position_size": "% of capital",
    "invalidation": "condition that cancels this setup"
  }},
  "rationale": "3 sentences"
}}"""
    try:
        content = get_ai_response(prompt)
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}
        content = re.sub(r"```json|```", "", content.strip()).strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
        result = {"success": True, "generated_at": datetime.utcnow().isoformat(), "advice": data}
        ADVICE_CACHE[symbol] = (result, current_time)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

# ----------------------------
# AI Chat — with per-user history
# ----------------------------
@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    username = chat.username or "guest"
    if username not in chat_history_db:
        chat_history_db[username] = []

    history = chat_history_db[username]
    preds_resp = generate_market_predictions()
    signals = preds_resp.get("signals", [])
    ngn_rate = get_ngn_rate()

    system_prompt = f"""You are ARIA — Advanced Revenue Intelligence Analyst — an elite AI trading strategist.

LIVE CONTEXT:
- Active Market Signals: {json.dumps(signals[:6], indent=2)}
- USD/NGN Rate: {ngn_rate:.2f}
- UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}

YOUR CAPABILITIES:
1. Precise NGN/USD position sizing and lot size calculations
2. Multi-timeframe technical analysis (RSI, MACD, EMA, BB, Volume)
3. Risk management: max 1-3% per trade, never exceed
4. Trade idea evaluation and feedback
5. Market psychology and sentiment interpretation
6. Portfolio allocation advice
7. Explain any trading concept in simple terms

CONVERSATION HISTORY (last 6 turns):
{json.dumps(history[-6:], indent=2)}

RULES:
- Always show calculations when financial figures are mentioned
- Format responses clearly with numbered points when giving advice
- If user mentions an amount in NGN, convert to USD and calculate position size
- End advice with a one-line risk reminder"""

    full_prompt = f"{system_prompt}\n\nUser: {chat.message}\nARIA:"
    ai_response = get_ai_response(full_prompt)

    # Save to history
    history.append({"role": "user", "content": chat.message, "time": datetime.utcnow().isoformat()})
    history.append({"role": "aria", "content": ai_response, "time": datetime.utcnow().isoformat()})
    if len(history) > 40:
        chat_history_db[username] = history[-40:]

    return {
        "success": True,
        "username": username,
        "response": ai_response,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/chat/history/{username}")
def get_chat_history(username: str):
    return {
        "username": username,
        "history": chat_history_db.get(username, []),
        "total_messages": len(chat_history_db.get(username, []))
    }

@app.delete("/chat/history/{username}")
def clear_chat_history(username: str):
    chat_history_db[username] = []
    return {"status": "success", "message": f"Chat history cleared for {username}"}

# ----------------------------
# Risk Calculator
# ----------------------------
@app.post("/risk_calculator")
def calculate_risk(req: RiskCalcRequest):
    ngn_rate = get_ngn_rate()
    balance_usd = req.balance_ngn / ngn_rate
    risk_amount_ngn = req.balance_ngn * (req.risk_percent / 100)
    risk_amount_usd = risk_amount_ngn / ngn_rate

    price_diff = abs(req.entry_price - req.stop_loss_price)
    risk_pct_per_unit = price_diff / req.entry_price

    if risk_pct_per_unit > 0:
        position_size_usd = risk_amount_usd / risk_pct_per_unit
        units = position_size_usd / req.entry_price
    else:
        position_size_usd = 0
        units = 0

    is_crypto = "USDT" in req.symbol.upper()
    pip_value = price_diff if is_crypto else price_diff * 10000

    return {
        "symbol": req.symbol.upper(),
        "balance_ngn": req.balance_ngn,
        "balance_usd": round(balance_usd, 2),
        "risk_percent": req.risk_percent,
        "risk_amount_ngn": round(risk_amount_ngn, 2),
        "risk_amount_usd": round(risk_amount_usd, 2),
        "entry_price": req.entry_price,
        "stop_loss_price": req.stop_loss_price,
        "stop_distance_pct": round(risk_pct_per_unit * 100, 4),
        "recommended_position_size_usd": round(position_size_usd, 2),
        "units_to_buy": round(units, 6),
        "pip_distance": round(pip_value, 1) if not is_crypto else None,
        "note": "Position sized to risk exactly your specified % of balance."
    }

# ----------------------------
# Trade Idea Scorer
# ----------------------------
@app.post("/score_trade")
def score_trade_idea(idea: TradeIdeaRequest):
    rr = abs(idea.take_profit - idea.entry) / abs(idea.entry - idea.stop_loss) if abs(idea.entry - idea.stop_loss) > 0 else 0
    direction = idea.direction.upper()

    prompt = f"""You are an elite institutional trade reviewer.

TRADE IDEA SUBMITTED:
- Symbol: {idea.symbol.upper()}
- Direction: {direction}
- Entry: {idea.entry}
- Stop Loss: {idea.stop_loss}
- Take Profit: {idea.take_profit}
- Calculated R/R: 1:{round(rr, 2)}
- Trader's Rationale: {idea.rationale}

EVALUATE THIS TRADE and return ONLY valid JSON:
{{
  "score": 0-100,
  "grade": "A+|A|B|C|D|F",
  "verdict": "TAKE IT|RISKY BUT OK|AVOID",
  "risk_reward": "1:{round(rr, 2)}",
  "rr_assessment": "excellent(>1:3)|good(1:2-3)|weak(<1:2)|negative",
  "strengths": ["point1", "point2"],
  "weaknesses": ["point1", "point2"],
  "improvements": ["suggestion1", "suggestion2"],
  "adjusted_entry": "better entry if applicable",
  "adjusted_sl": "tighter SL if applicable",
  "adjusted_tp": "better TP if applicable",
  "summary": "2 sentence professional opinion"
}}"""

    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
        return {"success": True, "trade_idea": idea.dict(), "analysis": data}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ----------------------------
# Watchlist
# ----------------------------
@app.post("/watchlist/{username}")
def add_to_watchlist(username: str, symbol: str):
    symbol = symbol.upper()
    if symbol not in TRADING_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"{symbol} is not a supported trading pair")
    if username not in watchlist_db:
        watchlist_db[username] = []
    if symbol not in watchlist_db[username]:
        watchlist_db[username].append(symbol)
    return {"status": "success", "watchlist": watchlist_db[username]}

@app.delete("/watchlist/{username}/{symbol}")
def remove_from_watchlist(username: str, symbol: str):
    symbol = symbol.upper()
    if username in watchlist_db and symbol in watchlist_db[username]:
        watchlist_db[username].remove(symbol)
    return {"status": "success", "watchlist": watchlist_db.get(username, [])}

@app.get("/watchlist/{username}")
def get_watchlist(username: str):
    symbols = watchlist_db.get(username, [])
    preds = generate_market_predictions()
    signals_map = {s.get("symbol"): s for s in preds.get("signals", [])}

    items = []
    for sym in symbols:
        sig = signals_map.get(sym, {})
        live = fetch_alpha_vantage_crypto(sym) if "USDT" in sym else fetch_alpha_vantage_forex(sym[:3], sym[3:])
        items.append({
            "symbol": sym,
            "live_price": live.get("price") or live.get("rate"),
            "signal": sig.get("signal", "N/A"),
            "confidence": sig.get("confidence", "N/A"),
            "entry_price": sig.get("entry_price", "N/A"),
            "stop_loss": sig.get("stop_loss", "N/A"),
            "hold_time": sig.get("hold_time", "N/A")
        })

    return {"username": username, "watchlist": items, "total": len(items)}

# ----------------------------
# Trade Journal
# ----------------------------
@app.post("/journal/{username}")
def add_journal_entry(username: str, entry: TradeJournalEntry):
    if username not in trade_journal_db:
        trade_journal_db[username] = []
    record = entry.dict()
    record["logged_at"] = datetime.utcnow().isoformat()
    trade_journal_db[username].append(record)
    return {"status": "success", "entry": record}

@app.get("/journal/{username}")
def get_journal(username: str):
    entries = trade_journal_db.get(username, [])
    if not entries:
        return {"username": username, "entries": [], "stats": {}}

    wins = [e for e in entries if e.get("result", "").upper() == "WIN"]
    losses = [e for e in entries if e.get("result", "").upper() == "LOSS"]
    total_pnl = sum(e.get("pnl_usd", 0) for e in entries)
    win_rate = (len(wins) / len(entries) * 100) if entries else 0
    avg_win = (sum(e["pnl_usd"] for e in wins) / len(wins)) if wins else 0
    avg_loss = (sum(e["pnl_usd"] for e in losses) / len(losses)) if losses else 0
    best = max(entries, key=lambda e: e.get("pnl_usd", 0))
    worst = min(entries, key=lambda e: e.get("pnl_usd", 0))

    return {
        "username": username,
        "entries": entries,
        "stats": {
            "total_trades": len(entries),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "best_trade": best,
            "worst_trade": worst
        }
    }

# ----------------------------
# Visual Simulator (Graph-Ready)
# ----------------------------
@app.get("/simulator/candles/{symbol}")
def get_simulator_candles(symbol: str, candles: int = 120, interval: int = 15, seed: int = None):
    """Returns simulated OHLCV candlestick data ready for charting."""
    symbol = symbol.upper()
    if symbol not in SYMBOL_BASE_PRICES and symbol not in TRADING_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol: {symbol}")
    candle_data = simulate_ohlcv(symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed)
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "total_candles": len(candle_data),
        "chart_type": "candlestick",
        "candles": candle_data,
        "meta": {
            "base_price": SYMBOL_BASE_PRICES.get(symbol),
            "note": "Simulated data using Geometric Brownian Motion — for demo trading practice only."
        }
    }

@app.get("/simulator/run/{symbol}")
def run_simulator(symbol: str, candles: int = 150, interval: int = 15, balance: float = 10000.0, seed: int = None):
    """Full trading simulation: candles + equity curve + trade markers + performance stats."""
    symbol = symbol.upper()
    if symbol not in SYMBOL_BASE_PRICES and symbol not in TRADING_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol: {symbol}")

    candle_data = simulate_ohlcv(symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed)
    results = run_strategy_on_candles(candle_data, initial_balance=balance)

    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])

    # Build entry/exit markers separately for chart overlay
    entry_markers = [
        {"time": t["entry_time"], "datetime": t["entry_datetime"], "price": t["entry_price"], "label": "BUY", "color": "#00ff88"}
        for t in results["trades"]
    ]
    exit_markers = [
        {"time": t["exit_time"], "datetime": t["exit_datetime"], "price": t["exit_price"],
         "label": "WIN" if t["result"] == "WIN" else "LOSS",
         "color": "#00bfff" if t["result"] == "WIN" else "#ff4444"}
        for t in results["trades"]
    ]

    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "simulation_candles": len(candle_data),
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
            "avg_win_usd": results["avg_win_usd"],
            "avg_loss_usd": results["avg_loss_usd"],
        },
        "chart_data": {
            "candles": candle_data,
            "equity_curve": results["equity_curve"],
            "entry_markers": entry_markers,
            "exit_markers": exit_markers
        },
        "trades": results["trades"],
        "disclaimer": "Simulated results do not guarantee future performance."
    }

# ----------------------------
# Demo Account
# ----------------------------
@app.post("/demo/open_account/{username}")
def open_demo_account(username: str, initial_balance: float = 10000.0):
    demo_accounts_db[username] = DemoAccount(username=username, demo_balance=initial_balance)
    return {"status": "success", "message": f"Demo account opened for {username} with ${initial_balance:,.2f}"}

@app.post("/demo/execute_trade/{username}")
def execute_demo_trade(username: str, symbol: str, volume: float, trade_type: str):
    if username not in demo_accounts_db:
        open_demo_account(username)
    account = demo_accounts_db[username]
    symbol = symbol.upper()
    live_data = fetch_alpha_vantage_crypto(symbol) if "USDT" in symbol else fetch_alpha_vantage_forex(symbol[:3], symbol[3:] or "USD")
    entry_price = float(live_data.get("price") or live_data.get("rate") or 0.0)
    if entry_price == 0.0:
        raise HTTPException(status_code=400, detail="Could not fetch live price for entry")
    new_trade = Trade(symbol=symbol, entry_price=entry_price, current_price=entry_price, volume=volume, type=trade_type.upper(), is_demo=True)
    account.active_demo_trades.append(new_trade)
    return {"status": "success", "trade": new_trade}

@app.get("/demo/ai_feedback/{username}")
def get_demo_ai_feedback(username: str):
    if username not in demo_accounts_db:
        return {"error": "No demo account found"}
    account = demo_accounts_db[username]
    for trade in account.active_demo_trades:
        live_data = fetch_alpha_vantage_crypto(trade.symbol) if "USDT" in trade.symbol else fetch_alpha_vantage_forex(trade.symbol[:3], trade.symbol[3:])
        trade.current_price = float(live_data.get("price") or live_data.get("rate") or trade.current_price)
        trade.pnl = (trade.current_price - trade.entry_price) * trade.volume if trade.type == "BUY" else (trade.entry_price - trade.current_price) * trade.volume

    summary = [f"{t.type} {t.volume} {t.symbol} @ {t.entry_price}, now {t.current_price}, PnL: ${t.pnl:.2f}" for t in account.active_demo_trades]
    prompt = f"""Elite Trading Risk Manager reviewing DEMO TRADES for {username}:
{json.dumps(summary)}
Balance: ${account.demo_balance:,.2f} USD

1. Evaluate performance — is this profitable strategy?
2. Should they replicate these in a LIVE account? What to adjust?
3. Identify any dangerous positions.
4. Suggest exact SL/TP improvements.
Be precise and calculative."""

    feedback = get_ai_response(prompt)
    return {"username": username, "demo_performance": account.active_demo_trades, "ai_strategic_advice": feedback}

# ----------------------------
# Account Monitor
# ----------------------------
def update_account_pnl(username: str):
    if username not in accounts_db:
        return
    account = accounts_db[username]
    for trade in account.trades:
        live_data = fetch_alpha_vantage_crypto(trade.symbol) if "USDT" in trade.symbol else fetch_alpha_vantage_forex(trade.symbol[:3], trade.symbol[3:])
        current_price = float(live_data.get("price") or live_data.get("rate") or trade.current_price)
        trade.current_price = current_price
        trade.pnl = (current_price - trade.entry_price) * trade.volume if trade.type == "BUY" else (trade.entry_price - current_price) * trade.volume

@app.get("/account/monitor/{username}")
def monitor_account(username: str):
    if username not in accounts_db:
        accounts_db[username] = AccountState(username=username, balance=1000000.0, trades=[
            Trade(symbol="BTCUSDT", entry_price=43000.0, current_price=43500.0, volume=0.01, type="BUY"),
            Trade(symbol="EURUSD", entry_price=1.0850, current_price=1.0900, volume=10000, type="BUY")
        ])
    update_account_pnl(username)
    account = accounts_db[username]
    ngn_rate = get_ngn_rate()
    total_pnl_usd = sum(t.pnl for t in account.trades)
    total_pnl_ngn = total_pnl_usd * ngn_rate
    return {
        "username": username,
        "balance_ngn": account.balance,
        "balance_usd": round(account.balance / ngn_rate, 2),
        "active_trades": account.trades,
        "total_pnl_usd": round(total_pnl_usd, 2),
        "total_pnl_ngn": round(total_pnl_ngn, 2),
        "usd_to_ngn_rate": ngn_rate,
        "status": "alert" if total_pnl_usd < -50 else "healthy"
    }

@app.post("/connect_account")
def connect_account(user: UserAccount):
    users_db[user.username] = user
    return {"status": "success", "message": f"User {user.username} connected"}
