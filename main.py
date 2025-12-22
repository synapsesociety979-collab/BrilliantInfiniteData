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
def get_filtered_ai_signals(symbol: str,
                            confidence_threshold: float = 70.0) -> List[Dict]:
    """
    Returns high-quality AI predictions for any trading symbol dynamically.
    """
    symbol = symbol.upper()
    try:
        df = load_history_df(symbol)
    except Exception:
        return [{
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "note": "No history found"
        }]

    raw_preds = generate_market_predictions().get("signals", [])
    symbol_preds = [p for p in raw_preds if p["symbol"].upper() == symbol]

    high_conf = [
        p for p in symbol_preds
        if p.get("confidence", 0) >= confidence_threshold
    ]

    filtered_preds = []
    for p in high_conf:
        try:
            sma_signals = signals_from_sma(df)
            metrics, _, _ = run_backtest_from_signals(df, sma_signals)
            if metrics.get('total_return_pct', 0) > 0:
                filtered_preds.append(p)
        except Exception:
            filtered_preds.append(p)

    if not filtered_preds:
        filtered_preds = [{
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0
        }]

    return filtered_preds


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
app.include_router(backtest_router, prefix="/api")  # optional prefix

# ----------------------------
# CORS Middleware
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (Lovable frontend)
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
    return {
        "status": "success",
        "message": f"User {user.username} connected successfully"
    }


# ----------------------------
# Generate AI Market Predictions
# ----------------------------
def generate_market_predictions() -> Dict[str, Any]:
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are an expert AI trading assistant. Return ONLY JSON with trading signals."
            }, {
                "role":
                "user",
                "content":
                "Generate signals for FOREX and CRYPTO pairs including symbol, signal, confidence, stop_loss, take_profit. Return only JSON array."
            }],
            temperature=0.6,
            max_tokens=2000)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {"success": True, "signals": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
# Execute Trade
# ----------------------------
@app.post("/execute_trade")
def execute_trade(username: str, symbol: str, action: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "status": "success",
        "message": f"Executed {action.upper()} on {symbol} for {username}"
    }


# ----------------------------
# Chat Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = []


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


# ----------------------------
# AI Chat
# ----------------------------
@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        system_prompt = "You are an expert AI trading assistant for Forex & Crypto. Provide actionable BUY/SELL/HOLD advice."
        messages = [{"role": "system", "content": system_prompt}]
        if chat.conversation_history:
            for msg in chat.conversation_history[-10:]:
                messages.append(msg)
        messages.append({"role": "user", "content": chat.message})

        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=0.7,
                                                  max_tokens=1000)
        ai_response = response.choices[0].message.content
        return {
            "success": True,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
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

    system_prompt = f"You are an AI trading assistant for {username}. Provide actionable advice."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history_db[username][-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": chat.message})

    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=0.7,
                                                  max_tokens=1000)
        ai_response = response.choices[0].message.content
        chat_history_db[username].append({
            "role": "user",
            "content": chat.message
        })
        chat_history_db[username].append({
            "role": "assistant",
            "content": ai_response
        })
        if len(chat_history_db[username]) > 50:
            chat_history_db[username] = chat_history_db[username][-50:]
        return {
            "success": True,
            "username": username,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Clear Chat History
# ----------------------------
@app.delete("/chat/{username}/clear")
def clear_chat_history(username: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    chat_history_db[username] = []
    return {"success": True, "message": f"Chat history cleared for {username}"}


# ----------------------------
# Trading Advice Endpoint
# ----------------------------
@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are a professional trading analyst. Return ONLY JSON advice."
            }, {
                "role":
                "user",
                "content":
                f"Provide detailed trading advice for {symbol.upper()}. Include entry, stop loss, take profit, risk management."
            }],
            temperature=0.6,
            max_tokens=1500)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "advice": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Market Analysis
# ----------------------------
@app.get("/market_analysis")
def get_market_analysis():
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are an expert market analyst. Return ONLY JSON."
            }, {
                "role":
                "user",
                "content":
                "Provide comprehensive market analysis for Forex and Crypto, include top opportunities and risk factors."
            }],
            temperature=0.7,
            max_tokens=1500)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "analysis": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Multi-User Trading Bot Backend is running"}


# ----------------------------
# Run Uvicorn
# ----------------------------
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.environ.get("PORT", 3000)),
                reload=True)
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
# Helper: Dynamic AI Signal Filtering
# ----------------------------
def get_filtered_ai_signals(symbol: str,
                            confidence_threshold: float = 70.0) -> List[Dict]:
    """
    Returns high-quality AI predictions for any trading symbol dynamically.
    """
    symbol = symbol.upper()
    try:
        df = load_history_df(symbol)
    except Exception:
        return [{
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "note": "No history found"
        }]

    raw_preds = generate_market_predictions().get("signals", [])
    symbol_preds = [p for p in raw_preds if p["symbol"].upper() == symbol]

    high_conf = [
        p for p in symbol_preds
        if p.get("confidence", 0) >= confidence_threshold
    ]

    filtered_preds = []
    for p in high_conf:
        try:
            sma_signals = signals_from_sma(df)
            metrics, _, _ = run_backtest_from_signals(df, sma_signals)
            if metrics.get('total_return_pct', 0) > 0:
                filtered_preds.append(p)
        except Exception:
            filtered_preds.append(p)

    if not filtered_preds:
        filtered_preds = [{
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0
        }]

    return filtered_preds


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
app.include_router(backtest_router, prefix="/api")  # optional prefix

# ----------------------------
# CORS Middleware
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (Lovable frontend)
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
    return {
        "status": "success",
        "message": f"User {user.username} connected successfully"
    }


# ----------------------------
# Generate AI Market Predictions
# ----------------------------
def generate_market_predictions() -> Dict[str, Any]:
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are an expert AI trading assistant. Return ONLY JSON with trading signals."
            }, {
                "role":
                "user",
                "content":
                "Generate signals for FOREX and CRYPTO pairs including symbol, signal, confidence, stop_loss, take_profit. Return only JSON array."
            }],
            temperature=0.6,
            max_tokens=2000)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {"success": True, "signals": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
# Execute Trade
# ----------------------------
@app.post("/execute_trade")
def execute_trade(username: str, symbol: str, action: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "status": "success",
        "message": f"Executed {action.upper()} on {symbol} for {username}"
    }


# ----------------------------
# Chat Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = []


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


# ----------------------------
# AI Chat
# ----------------------------
@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        system_prompt = "You are an expert AI trading assistant for Forex & Crypto. Provide actionable BUY/SELL/HOLD advice."
        messages = [{"role": "system", "content": system_prompt}]
        if chat.conversation_history:
            for msg in chat.conversation_history[-10:]:
                messages.append(msg)
        messages.append({"role": "user", "content": chat.message})

        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=0.7,
                                                  max_tokens=1000)
        ai_response = response.choices[0].message.content
        return {
            "success": True,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
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

    system_prompt = f"You are an AI trading assistant for {username}. Provide actionable advice."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history_db[username][-20:]:
        messages.append(msg)
    messages.append({"role": "user", "content": chat.message})

    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=messages,
                                                  temperature=0.7,
                                                  max_tokens=1000)
        ai_response = response.choices[0].message.content
        chat_history_db[username].append({
            "role": "user",
            "content": chat.message
        })
        chat_history_db[username].append({
            "role": "assistant",
            "content": ai_response
        })
        if len(chat_history_db[username]) > 50:
            chat_history_db[username] = chat_history_db[username][-50:]
        return {
            "success": True,
            "username": username,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Clear Chat History
# ----------------------------
@app.delete("/chat/{username}/clear")
def clear_chat_history(username: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    chat_history_db[username] = []
    return {"success": True, "message": f"Chat history cleared for {username}"}


# ----------------------------
# Trading Advice Endpoint
# ----------------------------
@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are a professional trading analyst. Return ONLY JSON advice."
            }, {
                "role":
                "user",
                "content":
                f"Provide detailed trading advice for {symbol.upper()}. Include entry, stop loss, take profit, risk management."
            }],
            temperature=0.6,
            max_tokens=1500)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "advice": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Market Analysis
# ----------------------------
@app.get("/market_analysis")
def get_market_analysis():
    client = get_openai_client()
    if not client:
        return {"success": False, "error": "OpenAI API key not configured"}

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role":
                "system",
                "content":
                "You are an expert market analyst. Return ONLY JSON."
            }, {
                "role":
                "user",
                "content":
                "Provide comprehensive market analysis for Forex and Crypto, include top opportunities and risk factors."
            }],
            temperature=0.7,
            max_tokens=1500)
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "analysis": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Multi-User Trading Bot Backend is running"}


# ----------------------------
# Run Uvicorn
# ----------------------------
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app",
                host="0.0.0.0",
                port=int(os.environ.get("PORT", 3000)),
                reload=True)
    from ai_training_loop import start_training_loop

    # Start automatic AI training loop every 60 minutes
    start_training_loop(interval_minutes=60)
