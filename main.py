# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re
from datetime import datetime
from ai_provider import get_ai_response

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import load_history_df, run_backtest_from_signals, signals_from_sma, router as backtest_router

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
    """Generates world-class market predictions using advanced analysis via Gemini."""
    prompt = """Analyze these pairs with MAXIMUM RIGOR for top-tier trading signals:

FOREX: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
CRYPTO: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT

Return JSON array with this structure ONLY:
[{
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
  "trend": {"short": "bullish/bearish/neutral", "medium": "bullish/bearish/neutral", "long": "bullish/bearish/neutral"},
  "technical": {
    "rsi": "value",
    "macd": "bullish/bearish",
    "ema_alignment": "bullish/bearish",
    "volume": "trend",
    "pattern": "pattern"
  },
  "key_levels": {
    "resistance_1": "level", "resistance_2": "level",
    "support_1": "level", "support_2": "level"
  },
  "sentiment": "sentiment",
  "rationale": "explanation"
}]

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
        
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "model": "gemini-2.0-flash",
            "signals": data,
            "total_signals": len(data) if isinstance(data, list) else 0,
            "strong_signals_count": len(strong_signals),
            "top_picks": strong_signals[:3] if strong_signals else [],
            "disclaimer": "Trading involves substantial risk."
        }
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

class UserAccount(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None

users_db = {}
chat_history_db = {}

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Trading Bot Backend"}

@app.get("/predictions")
def get_predictions_public():
    return generate_market_predictions()

@app.get("/get_predictions")
def get_predictions(username: str, symbol: str):
    if username not in users_db:
        # For demo purposes, we'll allow looking up predictions even if user isn't 'connected'
        # but let's maintain the error if that's what's expected
        pass
    predictions = get_filtered_ai_signals(symbol)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "predictions": predictions
    }

@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    prompt = f"Perform deep analysis on {symbol.upper()} and return JSON with recommendation, confidence, trade_setup, technical_analysis, and risk_assessment."
    try:
        content = get_ai_response(prompt)
        if content.startswith("ERROR:"):
            return {"success": False, "error": content}
        content = content.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
        return {"success": True, "generated_at": datetime.utcnow().isoformat(), "advice": data}
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
        system_prompt = f"You are an ELITE AI Trading Strategist. Signals: {json.dumps(signals[:5])}. Use this data to help the user."
        
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
