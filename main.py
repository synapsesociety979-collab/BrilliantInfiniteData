# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI

# ----------------------------
# OpenAI API Setup (Lazy initialization)
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

# ----------------------------
# ADD THIS: CORS MIDDLEWARE
# ----------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (Lovable frontend)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# User Model
# ----------------------------
class UserAccount(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None


# In-memory storage for users
users_db = {}


# ----------------------------
# Connect User Account
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

import json
import re
from datetime import datetime
from typing import Dict, Any


def generate_market_predictions() -> Dict[str, Any]:
    """
    Generates market predictions for crypto and Forex using OpenAI
    with enhanced analysis and structured output.
    """
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "error": "OpenAI API key not configured",
            "details": "Please set the OPENAI_API_KEY environment variable"
        }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert financial analyst AI. Return ONLY valid JSON with no markdown formatting.
Analyze market conditions and provide actionable trading signals with detailed reasoning."""
                },
                {
                    "role": "user",
                    "content": """Generate comprehensive trade signals for the following assets:

Forex pairs: EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD
Crypto pairs: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT

For each signal, provide:
{
  "symbol": "PAIR_NAME",
  "signal": "BUY/SELL/HOLD",
  "confidence": 0-100,
  "entry_price": "suggested entry price",
  "stop_loss": "stop loss level",
  "take_profit": "take profit level",
  "risk_reward_ratio": "ratio as string",
  "timeframe": "short-term/medium-term/long-term",
  "analysis": "brief market analysis reason",
  "market_sentiment": "bullish/bearish/neutral"
}

Return as a JSON array of objects."""
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "model": "gpt-4o-mini",
            "signals": data,
            "total_signals": len(data) if isinstance(data, list) else 0
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": "Failed to parse AI response",
            "details": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Prediction generation failed",
            "details": str(e)
        }


# ----------------------------
# Get Predictions Endpoint
# ----------------------------
@app.get("/get_predictions")
def get_predictions(username: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    ai_output = generate_market_predictions()

    return {
        "username": username,
        "request_time": datetime.utcnow().isoformat(),
        "predictions": ai_output
    }


# ----------------------------
# Get Predictions (No Auth Required)
# ----------------------------
@app.get("/predictions")
def get_predictions_public():
    """Get AI predictions without requiring user authentication"""
    ai_output = generate_market_predictions()
    return ai_output


# ----------------------------
# Execute Trade Endpoint (Placeholder)
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
# AI Chat Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


# In-memory chat history storage
chat_history_db = {}


# ----------------------------
# AI Trading Assistant Chat
# ----------------------------
@app.post("/chat")
def chat_with_ai(chat: ChatMessage):
    """Chat with the AI trading assistant"""
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "error": "OpenAI API key not configured"
        }
    
    try:
        system_prompt = """You are an expert AI trading assistant specializing in Forex and cryptocurrency markets. 
You help traders with:
- Market analysis and insights
- Trading strategies and techniques
- Risk management advice
- Explaining technical indicators
- Answering questions about trading pairs (EURUSD, BTCUSDT, etc.)
- Providing educational content about trading

Be helpful, professional, and give actionable advice. Always remind users that trading involves risk.
Keep responses concise but informative."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if chat.conversation_history:
            for msg in chat.conversation_history[-10:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": chat.message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "success": True,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ----------------------------
# AI Chat with User Context
# ----------------------------
@app.post("/chat/{username}")
def chat_with_ai_user(username: str, chat: ChatMessage):
    """Chat with AI trading assistant with user context and history"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "error": "OpenAI API key not configured"
        }
    
    try:
        if username not in chat_history_db:
            chat_history_db[username] = []
        
        system_prompt = f"""You are an expert AI trading assistant for {username}. 
You specialize in Forex and cryptocurrency markets and help with:
- Personalized market analysis and insights
- Trading strategies tailored to the user
- Risk management advice
- Explaining technical indicators
- Answering questions about trading pairs
- Providing educational content

Be helpful, professional, and give actionable advice. Remember previous conversations.
Always remind users that trading involves risk. Keep responses concise but informative."""

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in chat_history_db[username][-20:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": chat.message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        chat_history_db[username].append({"role": "user", "content": chat.message})
        chat_history_db[username].append({"role": "assistant", "content": ai_response})
        
        if len(chat_history_db[username]) > 50:
            chat_history_db[username] = chat_history_db[username][-50:]
        
        return {
            "success": True,
            "username": username,
            "response": ai_response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ----------------------------
# Clear Chat History
# ----------------------------
@app.delete("/chat/{username}/clear")
def clear_chat_history(username: str):
    """Clear chat history for a user"""
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    if username in chat_history_db:
        chat_history_db[username] = []
    
    return {
        "success": True,
        "message": f"Chat history cleared for {username}"
    }


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Multi-User Trading Bot Backend is running"}


import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 3000)),
        reload=True
    )

