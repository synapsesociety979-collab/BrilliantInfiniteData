# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re
from datetime import datetime
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from backtest_api import load_history_df, run_backtest_from_signals, signals_from_sma, router as backtest_router

# ----------------------------
# Trading symbols for AI and backtesting
# ----------------------------
TRADING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "EURUSD", "GBPUSD", "USDJPY",
    "AUDUSD", "USDCAD"
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

    raw_preds = generate_market_predictions().get("signals", [])
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
# OpenAI Client
# ----------------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Multi-User Trading Bot Backend")
app.include_router(backtest_router, prefix="/api")

# ----------------------------
# CORS Middleware
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# User Model & DB
# ----------------------------
class UserAccount(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None


users_db = {}
chat_history_db = {}


# ----------------------------
# Connect Account
# ----------------------------
@app.post("/connect_account")
def connect_account(user: UserAccount):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    users_db[user.username] = user
    return {"status": "success", "message": f"User {user.username} connected successfully"}


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Trading Bot Backend"}


# ----------------------------
# Generate AI Market Predictions - ENHANCED FOR MAX ACCURACY
# ----------------------------
def generate_market_predictions() -> Dict[str, Any]:
    """Generates world-class market predictions using advanced analysis."""
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an ELITE institutional trading AI. Your analysis combines:
- Quantitative models (statistical, ML-based pattern recognition)
- Advanced Technical (RSI divergences, MACD histogram, Stochastic, Volume profile, Ichimoku, Keltner channels)
- Multi-timeframe Confluence (5m/1h/4h/daily alignment)
- 20+ chart patterns with formation psychology
- Microstructure analysis (order flow, accumulation/distribution)
- Fundamental Economics (central banks, macro cycles, yield curves)
- Sentiment Analysis (fear/greed index, volatility, positioning)
- Machine Learning: Historical pattern matching from billions of price bars

OUTPUT: Pure JSON with NO text explanations. Each signal MUST have 5+ factor confluence.
GOAL: Maximum accuracy predictions that institutional traders can rely on.
CONSTRAINT: Only generate signals when risk-reward is minimum 1:1.5 and confluence is 5+."""
                },
                {
                    "role": "user",
                    "content": """Analyze these pairs with MAXIMUM RIGOR for top-tier trading signals:

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
  "take_profit_1": "TP1 (70% probability)",
  "take_profit_2": "TP2 (50% probability)",
  "take_profit_3": "TP3 (30% probability)",
  "risk_reward_ratio": "1:X",
  "position_size": "1-5% of capital",
  "timeframe": "scalp/intraday/swing/position",
  "trend": {"short": "bullish/bearish/neutral", "medium": "bullish/bearish/neutral", "long": "bullish/bearish/neutral"},
  "technical": {
    "rsi": "0-100 with signal",
    "macd": "bullish/bearish/divergence",
    "ema_alignment": "bullish/bearish/mixed",
    "volume": "increasing/decreasing",
    "pattern": "pattern name or none"
  },
  "key_levels": {
    "resistance_1": "level", "resistance_2": "level",
    "support_1": "level", "support_2": "level"
  },
  "sentiment": "extreme_fear/fear/neutral/greed/extreme_greed",
  "rationale": "2-3 sentences explaining WHY this signal has edge"
}]

CRITICAL: Only include signals where 5+ factors align. Confidence = actual win probability."""
                }
            ],
            temperature=0.5,
            max_tokens=6000
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        
        strong_signals = [s for s in data if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]] if isinstance(data, list) else []
        
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "model": "gpt-4o-mini",
            "analysis_type": "elite_institutional",
            "signals": data,
            "total_signals": len(data) if isinstance(data, list) else 0,
            "strong_signals_count": len(strong_signals),
            "top_picks": strong_signals[:3] if strong_signals else [],
            "disclaimer": "Trading involves substantial risk. Always use proper risk management."
        }

    except json.JSONDecodeError as e:
        return {"success": False, "error": "Failed to parse AI response", "details": str(e)}
    except Exception as e:
        return {"success": False, "error": "Prediction generation failed", "details": str(e)}


# ----------------------------
# Get Predictions (User-specific)
# ----------------------------
@app.get("/get_predictions")
def get_predictions(username: str, symbol: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    predictions = get_filtered_ai_signals(symbol)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "predictions": predictions
    }


# ----------------------------
# Get Predictions (Public)
# ----------------------------
@app.get("/predictions")
def get_predictions_public():
    return generate_market_predictions()


# ----------------------------
# Get Trading Advice for Specific Symbol
# ----------------------------
@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    """Get comprehensive institutional-grade trading advice."""
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an elite trading strategist. Provide DEEP institutional analysis."""
                },
                {
                    "role": "user",
                    "content": f"""Perform deep analysis on {symbol.upper()}:

Return JSON (no markdown):
{{
    "symbol": "{symbol.upper()}",
    "recommendation": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence": 0-100,
    "trade_setup": {{
        "entry_price": "optimal entry",
        "entry_zone": {{"low": "zone_low", "high": "zone_high"}},
        "stop_loss": "calculated SL",
        "take_profit_targets": [
            {{"level": "TP1", "probability": "70%"}},
            {{"level": "TP2", "probability": "50%"}},
            {{"level": "TP3", "probability": "30%"}}
        ],
        "risk_reward_ratio": "calculated"
    }},
    "technical_analysis": {{
        "trend": "strong_uptrend/uptrend/sideways/downtrend",
        "rsi_14": "value with signal",
        "macd": "bullish/bearish",
        "moving_averages": "bullish/bearish/mixed",
        "volume": "trend description"
    }},
    "key_levels": {{
        "resistance_1": "level", "resistance_2": "level",
        "support_1": "level", "support_2": "level"
    }},
    "market_sentiment": "extreme_fear/fear/neutral/greed/extreme_greed",
    "trade_management": {{
        "entry_timing": "immediate/pullback/breakout",
        "position_sizing": "% of capital",
        "exit_strategy": "trail_stop/fixed/hybrid"
    }},
    "risk_assessment": {{
        "risk_level": "low/medium/high",
        "key_risks": ["risk1", "risk2"],
        "invalidation_price": "price where thesis breaks"
    }}
}}"""
                }
            ],
            temperature=0.6,
            max_tokens=2500
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        
        return {"success": True, "generated_at": datetime.utcnow().isoformat(), "advice": data}
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": "Failed to parse AI response", "details": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Chat Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = []


# ----------------------------
# AI Chat
# ----------------------------
@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        system_prompt = "You are an expert AI trading assistant. Provide actionable BUY/SELL/HOLD advice based on market analysis."
        messages = [{"role": "system", "content": system_prompt}]
        if chat.conversation_history:
            for msg in chat.conversation_history[-10:]:
                messages.append(msg)
        messages.append({"role": "user", "content": chat.message})

        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1000)
        ai_response = response.choices[0].message.content
        return {"success": True, "response": ai_response, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Chat with User Context
# ----------------------------
@app.post("/chat/{username}")
def chat_with_ai_user(username: str, chat: ChatMessage):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    if username not in chat_history_db:
        chat_history_db[username] = []

    system_prompt = f"You are an AI trading assistant for {username}. Provide actionable trading advice."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history_db[username][-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": chat.message})

    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.7, max_tokens=1000)
        ai_response = response.choices[0].message.content
        chat_history_db[username].append({"role": "user", "content": chat.message})
        chat_history_db[username].append({"role": "assistant", "content": ai_response})
        if len(chat_history_db[username]) > 50:
            chat_history_db[username] = chat_history_db[username][-50:]
        return {"success": True, "response": ai_response, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Execute Trade
# ----------------------------
@app.post("/execute_trade")
def execute_trade(username: str, symbol: str, action: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "success", "message": f"Executed {action.upper()} on {symbol}"}


# ----------------------------
# Market Analysis
# ----------------------------
@app.get("/market_analysis")
def market_analysis():
    """Get overall market analysis and trends."""
    predictions = generate_market_predictions()
    if predictions.get("success"):
        strong_signals = predictions.get("top_picks", [])
        return {
            "market_health": "analyzing",
            "best_opportunities": strong_signals,
            "total_signals_analyzed": predictions.get("total_signals", 0),
            "strong_signals_count": predictions.get("strong_signals_count", 0),
            "analysis_timestamp": predictions.get("generated_at")
        }
    return {"error": "Failed to generate market analysis"}
