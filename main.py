# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, time, requests
from datetime import datetime
from ai_provider import get_ai_response

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import load_history_df, run_backtest_from_signals, signals_from_sma, router as backtest_router

# ----------------------------
# Alpha Vantage Integration
# ----------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_alpha_vantage_forex(from_symbol: str, to_symbol: str = "USD") -> Dict:
    """Fetch real-time Forex data from Alpha Vantage."""
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
    """Fetch real-time Crypto data from Alpha Vantage."""
    # Convert BTCUSDT to BTC
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
    """Get the current USD/NGN exchange rate."""
    data = fetch_alpha_vantage_forex("USD", "NGN")
    return float(data.get("rate") or 1600.0)

def get_market_context() -> str:
    """Gathers real-time context for key symbols to feed the AI."""
    context = []
    # Sample a few key assets to stay within rate limits if needed
    key_forex = ["EUR", "GBP", "JPY"]
    key_crypto = ["BTC", "ETH"]
    
    for f in key_forex:
        data = fetch_alpha_vantage_forex(f)
        if data:
            context.append(f"{f}/USD: Rate {data.get('rate')} (Last Update: {data.get('last_refreshed')})")
            
    for c in key_crypto:
        data = fetch_alpha_vantage_crypto(c)
        if data:
            context.append(f"{c}/USD: Price {data.get('price')} (Last Update: {data.get('last_refreshed')})")
            
    return " | ".join(context) if context else "Live market data currently unavailable."

# ----------------------------
# Cache Configuration
# ----------------------------
PREDICTIONS_CACHE = {}
ADVICE_CACHE = {}
CACHE_DURATION_SECONDS = 600  # 10 minutes for predictions

# ----------------------------
# Trading symbols for AI and backtesting
# ----------------------------
TRADING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "EURUSD", "GBPUSD", "USDJPY",
    "AUDUSD", "USDCAD", "NZDUSD", "USDCHF", "SOLUSDT", "ADAUSDT", "DOGEUSDT"
]

# ----------------------------
# AI Generation Functions
# ----------------------------
def generate_market_predictions(investment_amount_ngn: float = 0.0) -> Dict[str, Any]:
    """Generates high-accuracy market predictions with investment-specific advice."""
    global PREDICTIONS_CACHE
    
    current_time = time.time()
    # Cache key includes investment amount
    cache_key = f"preds_{investment_amount_ngn}"
    
    if cache_key in PREDICTIONS_CACHE and (current_time - PREDICTIONS_CACHE[cache_key]["timestamp"] < CACHE_DURATION_SECONDS):
        return PREDICTIONS_CACHE[cache_key]["data"]

    live_data = get_market_context()
    ngn_rate = get_ngn_rate()
    investment_usd = investment_amount_ngn / ngn_rate if investment_amount_ngn > 0 else 0

    prompt = f"""INSTITUTIONAL MARKET ANALYSIS - TARGET ACCURACY: 70%+
LIVE MARKET CONTEXT: {live_data}
INVESTMENT CONTEXT: {investment_amount_ngn} NGN (Approx. {investment_usd:.2f} USD)

Analyze these pairs with 70%+ ACCURACY RIGOR. For each signal, you MUST include:
1. Exact entry, stop-loss, and 3 take-profit targets.
2. HOLD TIME: Specify exactly how many minutes or hours to hold the trade.
3. RISK ADVICE: Based on {investment_amount_ngn} NGN, calculate the exact position size to use in Naira.
4. BACK-OUT STRATEGY: Define a specific 'Exit Trigger' (e.g., 'If price stays below X for 15 mins, BACK OUT').

FOREX: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
CRYPTO: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT

Return JSON array:
[
  {{
    "symbol": "PAIR",
    "signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence": 70-100,
    "hold_time": "X minutes/hours",
    "position_size_ngn": "amount to use from your budget",
    "entry_price": "price",
    "stop_loss": "price",
    "take_profit_1": "price",
    "back_out_trigger": "Specific condition to exit early",
    "rationale": "Calculative justification"
  }}
]
CRITICAL: ONLY include signals with 70%+ confidence. Pure JSON only."""

    try:
        content = get_ai_response(prompt)
        if not content:
            return {"success": False, "error": "AI returned empty content"}
        
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}

        content = content.strip()
        # Extract JSON array
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            content = re.sub(r"```json|```", "", content).strip()
            
        data = json.loads(content)
        
        # Filter for 70%+ accuracy
        data = [s for s in data if s.get("confidence", 0) >= 70]
        
        result = {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "investment_context_ngn": investment_amount_ngn,
            "usd_ngn_rate": ngn_rate,
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data) if isinstance(data, list) else 0,
            "disclaimer": "Trading involves substantial risk. Target accuracy is based on technical confluence."
        }
        
        PREDICTIONS_CACHE[cache_key] = {"data": result, "timestamp": current_time}
        return result
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}

# ----------------------------
# Helper: Dynamic AI Signal Filtering
# ----------------------------
def get_filtered_ai_signals(symbol: str, confidence_threshold: float = 70.0) -> List[Dict]:
    """Returns high-quality AI predictions for any trading symbol dynamically."""
    symbol = symbol.upper()
    try:
        df = load_history_df(symbol)
    except Exception:
        return [{"symbol": symbol, "signal": "HOLD", "confidence": 0, "note": "No history found"}]

    preds_resp = generate_market_predictions()
    if not preds_resp.get("success"):
        return [{"symbol": symbol, "signal": "HOLD", "confidence": 0, "note": preds_resp.get("error")}]
        
    raw_preds = preds_resp.get("signals", [])
    symbol_preds = [p for p in raw_preds if p["symbol"].upper() == symbol]
    high_conf = [p for p in symbol_preds if p.get("confidence", 0) >= confidence_threshold]

    filtered_preds = []
    for p in high_conf:
        try:
            sma_signals = signals_from_sma(df)
            metrics, _, _ = run_backtest_from_signals(df, sma_signals)
            if metrics.get('total_return_pct', 0) > 0:
                filtered_preds.append(p)
        except Exception:
            filtered_preds.append(p)

    return filtered_preds if filtered_preds else [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]

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
# Account & Trade Monitoring
# ----------------------------
class Trade(BaseModel):
    symbol: str
    entry_price: float
    current_price: float
    volume: float
    type: str  # BUY/SELL
    pnl: float = 0.0
    is_demo: bool = True

class AccountState(BaseModel):
    username: str
    balance: float
    currency: str = "NGN"
    trades: List[Trade] = []

accounts_db: Dict[str, AccountState] = {}

class DemoAccount(BaseModel):
    username: str
    demo_balance: float = 10000.0
    currency: str = "USD"
    active_demo_trades: List[Trade] = []
    trade_history: List[Trade] = []

demo_accounts_db: Dict[str, DemoAccount] = {}

@app.post("/demo/open_account/{username}")
def open_demo_account(username: str, initial_balance: float = 10000.0):
    demo_accounts_db[username] = DemoAccount(username=username, demo_balance=initial_balance)
    return {"status": "success", "message": f"Demo account opened for {username} with ${initial_balance}"}

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
        
    new_trade = Trade(
        symbol=symbol,
        entry_price=entry_price,
        current_price=entry_price,
        volume=volume,
        type=trade_type.upper(),
        is_demo=True
    )
    
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
        if trade.type == "BUY":
            trade.pnl = (trade.current_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - trade.current_price) * trade.volume

    trades_summary = [
        f"{t.type} {t.volume} {t.symbol} at {t.entry_price}, current {t.current_price}, PnL: {t.pnl}"
        for t in account.active_demo_trades
    ]
    
    prompt = f"""You are an Elite Trading Risk Manager. 
Analyze these DEMO TRADES for user {username}:
{json.dumps(trades_summary)}

User Balance: {account.demo_balance} USD

TASK:
1. Evaluate their current demo performance.
2. Provide CRITICAL advice: Should they replicate these trades in their MAIN Forex account?
3. Warn about potential market turns.
4. Suggest exact adjustments to stop losses or take profits for their REAL account.

Be calculative and direct."""

    feedback = get_ai_response(prompt)
    return {
        "username": username,
        "demo_performance": account.active_demo_trades,
        "ai_strategic_advice": feedback
    }

def update_account_pnl(username: str):
    if username not in accounts_db:
        return
    
    account = accounts_db[username]
    for trade in account.trades:
        live_data = fetch_alpha_vantage_crypto(trade.symbol) if "USDT" in trade.symbol else fetch_alpha_vantage_forex(trade.symbol[:3], trade.symbol[3:])
        current_price = float(live_data.get("price") or live_data.get("rate") or trade.current_price)
        
        trade.current_price = current_price
        if trade.type == "BUY":
            trade.pnl = (current_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - current_price) * trade.volume

@app.get("/account/monitor/{username}")
def monitor_account(username: str):
    if username not in accounts_db:
        accounts_db[username] = AccountState(
            username=username,
            balance=1000000.0,
            trades=[
                Trade(symbol="BTCUSDT", entry_price=43000.0, current_price=43500.0, volume=0.01, type="BUY"),
                Trade(symbol="EURUSD", entry_price=1.0850, current_price=1.0900, volume=10000, type="BUY")
            ]
        )
    
    update_account_pnl(username)
    account = accounts_db[username]
    
    ngn_rate = get_ngn_rate()
    
    total_pnl_usd = sum(t.pnl for t in account.trades)
    total_pnl_ngn = total_pnl_usd * ngn_rate
    
    return {
        "username": username,
        "balance_ngn": account.balance,
        "balance_usd": account.balance / ngn_rate,
        "active_trades": account.trades,
        "total_pnl_usd": total_pnl_usd,
        "total_pnl_ngn": total_pnl_ngn,
        "usd_to_ngn_rate": ngn_rate,
        "status": "alert" if total_pnl_usd < -50 else "healthy"
    }

class UserAccount(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

class AnalyzeRequest(BaseModel):
    prompt: str = "test prompt"

users_db = {}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Trading Bot Backend"}

@app.get("/predictions")
def get_predictions_public(amount_ngn: float = 0.0):
    return generate_market_predictions(amount_ngn)

@app.get("/get_predictions")
def get_predictions(username: str, symbol: str):
    predictions = get_filtered_ai_signals(symbol)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "predictions": predictions
    }

@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    symbol = symbol.upper()
    current_time = time.time()
    
    if symbol in ADVICE_CACHE:
        cache_data, timestamp = ADVICE_CACHE[symbol]
        if current_time - timestamp < 3600:
            return cache_data

    prompt = f"Perform deep analysis on {symbol} and return JSON with recommendation, confidence, trade_setup, technical_analysis, and risk_assessment."
    try:
        content = get_ai_response(prompt)
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}
        content = content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
        result = {"success": True, "generated_at": datetime.utcnow().isoformat(), "advice": data}
        
        ADVICE_CACHE[symbol] = (result, current_time)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = []

@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    try:
        preds_resp = generate_market_predictions()
        signals = preds_resp.get("signals", [])
        
        system_prompt = f"""You are an ELITE AI Trading Strategist and Financial Consultant. 
Your reasoning must be highly calculative, professional, and institutional-grade.

CORE CAPABILITIES:
1. GLOBAL CURRENCY HANDLING: Expert in global currencies, including Nigerian Naira (NGN).
2. CALCULATIVE ADVICE: Calculate position sizes, potential profits, and risks precisely.
3. CONVERSION EXPERTISE: Use approximately 1 USD = 1,600 NGN or latest market rates.
4. RISK MANAGEMENT: Never suggest more than 1-5% risk per trade.

Current Market Signals: {json.dumps(signals[:5])}.

Always provide specific advice with calculations if the user provides financial data."""
        
        full_prompt = f"{system_prompt}\nUser: {chat.message}\nAI:"
        ai_response = get_ai_response(full_prompt)
        return {"success": True, "response": ai_response, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/connect_account")
def connect_account(user: UserAccount):
    users_db[user.username] = user
    return {"status": "success", "message": f"User {user.username} connected"}

@app.get("/market_analysis")
def market_analysis():
    predictions = generate_market_predictions()
    if predictions.get("success"):
        signals = predictions.get("signals", [])
        top_picks = [s for s in signals if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]][:3]
        return {
            "market_health": "healthy",
            "best_opportunities": top_picks,
            "total_signals_analyzed": predictions.get("total_signals", 0)
        }
    return {"error": "Analysis failed", "details": predictions.get("error")}
