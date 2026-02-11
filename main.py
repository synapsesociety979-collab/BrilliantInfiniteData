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
                "rate": rate_data.get("5. Exchange Rate"),
                "bid": rate_data.get("8. Bid Price"),
                "ask": rate_data.get("9. Ask Price"),
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
                "price": rate_data.get("5. Exchange Rate"),
                "last_refreshed": rate_data.get("6. Last Refreshed")
            }
    except Exception as e:
        print(f"Alpha Vantage Crypto Error: {e}")
    return {}

def get_market_context() -> str:
    """Gathers real-time context for key symbols to feed the AI."""
    context = []
    # Sample a few key assets to stay within rate limits if needed
    key_forex = ["EUR", "GBP", "JPY"]
    key_crypto = ["BTC", "ETH"]
    
    for f in key_forex:
        data = fetch_alpha_vantage_forex(f)
        if data:
            context.append(f"{f}/USD: Rate {data['rate']} (Last Update: {data['last_refreshed']})")
            
    for c in key_crypto:
        data = fetch_alpha_vantage_crypto(c)
        if data:
            context.append(f"{c}/USD: Price {data['price']} (Last Update: {data['last_refreshed']})")
            
    return " | ".join(context) if context else "Live market data currently unavailable."

# ----------------------------
# Cache Configuration
# ----------------------------
PREDICTIONS_CACHE = {
    "data": None,
    "timestamp": 0
}
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
# AI Generation Functions
# ----------------------------
def generate_market_predictions() -> Dict[str, Any]:
    """Generates world-class market predictions using advanced analysis via Gemini with caching."""
    global PREDICTIONS_CACHE
    
    current_time = time.time()
    if PREDICTIONS_CACHE["data"] and (current_time - PREDICTIONS_CACHE["timestamp"] < CACHE_DURATION_SECONDS):
        return PREDICTIONS_CACHE["data"]

    live_data = get_market_context()

    prompt = f"""INSTITUTIONAL MARKET ANALYSIS REPORT
LIVE MARKET DATA CONTEXT: {live_data}

Analyze these pairs with MAXIMUM RIGOR and INSTITUTIONAL-GRADE CALCULATIONS for top-tier trading signals. 

FOREX: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
CRYPTO: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT

For each signal, provide deep calculative reasoning including:
1. Volatility-adjusted stop loss levels.
2. Exact pip/percentage profit targets.
3. Global currency context (consider how strength in USD affects NGN and other local currencies).

Return JSON array with this structure ONLY:
[
  {{
    "symbol": "PAIR",
    "signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence": 0-100,
    "confluence_factors": 5-15,
    "entry_price": "optimal entry",
    "stop_loss": "calculated SL",
    "take_profit_1": "TP1",
    "take_profit_2": "TP2",
    "take_profit_3": "TP3",
    "risk_reward_ratio": "1:X",
    "position_size": "1-5%",
    "timeframe": "scalp/intraday/swing/position",
    "trend": {{
      "short": "bullish/bearish/neutral", 
      "medium": "bullish/bearish/neutral", 
      "long": "bullish/bearish/neutral"
    }},
    "technical": {{
      "rsi": "value",
      "macd": "bullish/bearish",
      "ema_alignment": "bullish/bearish",
      "volume": "trend",
      "pattern": "pattern"
    }},
    "key_levels": {{
      "resistance_1": "level", "resistance_2": "level",
      "support_1": "level", "support_2": "level"
    }},
    "sentiment": "sentiment",
    "rationale": "Deep institutional analysis with calculative justification"
  }}
]

CRITICAL: Pure JSON ONLY. Only include signals where 5+ factors align. Confidence = win probability."""
    
    try:
        content = get_ai_response(prompt)
        if not content:
            return {"success": False, "error": "AI returned empty content"}
        
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}

        content = content.strip()
        json_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            content = re.sub(r"```json|```", "", content).strip()
            
        data = json.loads(content)
        strong_signals = [s for s in data if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]] if isinstance(data, list) else []
        
        result = {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data) if isinstance(data, list) else 0,
            "strong_signals_count": len(strong_signals),
            "top_picks": strong_signals[:3] if strong_signals else [],
            "disclaimer": "Trading involves substantial risk."
        }
        
        PREDICTIONS_CACHE["data"] = result
        PREDICTIONS_CACHE["timestamp"] = current_time
        return result
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}

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

class AccountState(BaseModel):
    username: str
    balance: float
    currency: str = "NGN"
    trades: List[Trade] = []

accounts_db: Dict[str, AccountState] = {}

def update_account_pnl(username: str):
    """Update PnL for all trades in an account using live prices."""
    if username not in accounts_db:
        return
    
    account = accounts_db[username]
    for trade in account.trades:
        # Fetch live price
        live_data = fetch_alpha_vantage_crypto(trade.symbol) if "USDT" in trade.symbol else fetch_alpha_vantage_forex(trade.symbol[:3], trade.symbol[3:])
        current_price = float(live_data.get("price") or live_data.get("rate") or trade.current_price)
        
        trade.current_price = current_price
        if trade.type == "BUY":
            trade.pnl = (current_price - trade.entry_price) * trade.volume
        else:
            trade.pnl = (trade.entry_price - current_price) * trade.volume

@app.get("/account/monitor/{username}")
def monitor_account(username: str):
    """Monitor account balance and trades with live currency conversion."""
    if username not in accounts_db:
        # Mocking an account for demo if not found
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
    
    # Get live NGN rate
    ngn_data = fetch_alpha_vantage_forex("USD", "NGN")
    usd_to_ngn = float(ngn_data.get("rate") or 1500.0)
    
    total_pnl_usd = sum(t.pnl for t in account.trades)
    total_pnl_ngn = total_pnl_usd * usd_to_ngn
    
    return {
        "username": username,
        "balance_ngn": account.balance,
        "balance_usd": account.balance / usd_to_ngn,
        "active_trades": account.trades,
        "total_pnl_usd": total_pnl_usd,
        "total_pnl_ngn": total_pnl_ngn,
        "usd_to_ngn_rate": usd_to_ngn,
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
chat_history_db = {}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Trading Bot Backend"}

@app.post("/analyze")
def analyze_market_endpoint(request: AnalyzeRequest):
    try:
        result = get_ai_response(request.prompt)
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions_public():
    return generate_market_predictions()

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
    """Get comprehensive institutional-grade trading advice with caching."""
    symbol = symbol.upper()
    current_time = time.time()
    
    # Cache check
    if symbol in ADVICE_CACHE:
        cache_data, timestamp = ADVICE_CACHE[symbol]
        if current_time - timestamp < 3600:  # 1 hour cache for deep advice
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
        
        # Update cache
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
        
        # Enhanced system prompt for better reasoning and currency handling
        system_prompt = f"""You are an ELITE AI Trading Strategist and Financial Consultant. 
Your reasoning must be highly calculative, professional, and institutional-grade.

CORE CAPABILITIES:
1. GLOBAL CURRENCY HANDLING: You are an expert in all global currencies, including Nigerian Naira (NGN), USD, EUR, etc.
2. CALCULATIVE ADVICE: When users mention specific amounts (e.g., 'I have 500,000 Naira'), calculate position sizes, potential profits, and risks precisely.
3. CONVERSION EXPERTISE: Always consider current market exchange rates (approximately 1 USD = 1,500 NGN or latest market rates) when giving advice.
4. RISK MANAGEMENT: Never suggest more than 1-5% risk per trade.

Current Market Signals: {json.dumps(signals[:5])}.

Always provide specific, numbered advice with calculations if the user provides financial data."""
        
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
        return {
            "market_health": "healthy",
            "best_opportunities": predictions.get("top_picks", []),
            "total_signals_analyzed": predictions.get("total_signals", 0)
        }
    return {"error": "Analysis failed", "details": predictions.get("error")}
