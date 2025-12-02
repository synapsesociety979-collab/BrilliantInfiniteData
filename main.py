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
        system_prompt = """You are an expert AI trading analyst and advisor specializing in Forex and cryptocurrency markets.

YOUR ROLE:
You analyze market trends, technical indicators, and market sentiment to provide CLEAR BUY/SELL/HOLD recommendations.

WHEN ASKED ABOUT ANY TRADING PAIR, YOU MUST:
1. Analyze current market conditions and trends
2. Consider technical indicators (RSI, MACD, Moving Averages, Support/Resistance)
3. Evaluate market sentiment (bullish/bearish/neutral)
4. Provide a CLEAR recommendation: BUY, SELL, or HOLD
5. Give specific entry price, stop loss, and take profit levels
6. Explain your reasoning briefly

RESPONSE FORMAT FOR TRADING ADVICE:
- Recommendation: [BUY/SELL/HOLD]
- Confidence: [0-100%]
- Entry Price: [price level]
- Stop Loss: [price level]
- Take Profit: [price level]
- Risk/Reward Ratio: [ratio]
- Analysis: [brief market analysis]

MARKETS YOU COVER:
- Forex: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
- Crypto: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT

Be direct and actionable. Users want clear advice on whether to BUY or SELL.
Always include a risk warning at the end."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if chat.conversation_history:
            for msg in chat.conversation_history[-10:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": chat.message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
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
# Get Trading Advice for Specific Symbol
# ----------------------------
@app.get("/advice/{symbol}")
def get_trading_advice(symbol: str):
    """Get specific BUY/SELL advice for a trading pair"""
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "error": "OpenAI API key not configured"
        }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert trading analyst. Analyze the given trading pair and provide a detailed trading recommendation.
Return ONLY valid JSON with no markdown formatting."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze {symbol.upper()} and provide a trading recommendation.

Return JSON in this exact format:
{{
    "symbol": "{symbol.upper()}",
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "entry_price": "suggested entry price",
    "stop_loss": "stop loss level",
    "take_profit": "take profit level", 
    "risk_reward_ratio": "1:2 format",
    "timeframe": "short-term/medium-term/long-term",
    "market_sentiment": "bullish/bearish/neutral",
    "trend": "uptrend/downtrend/sideways",
    "key_levels": {{
        "support": "support level",
        "resistance": "resistance level"
    }},
    "technical_indicators": {{
        "rsi": "overbought/oversold/neutral",
        "macd": "bullish/bearish",
        "moving_averages": "above/below"
    }},
    "analysis": "detailed analysis explaining the recommendation",
    "risk_warning": "brief risk warning"
}}"""
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "advice": data
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
            "error": str(e)
        }


# ----------------------------
# Market Analysis Summary
# ----------------------------
@app.get("/market_analysis")
def get_market_analysis():
    """Get overall market analysis and trends"""
    client = get_openai_client()
    if not client:
        return {
            "success": False,
            "error": "OpenAI API key not configured"
        }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert market analyst. Provide current market analysis. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": """Provide a comprehensive market analysis covering:

Return JSON in this format:
{
    "market_overview": "brief overall market summary",
    "forex_analysis": {
        "trend": "bullish/bearish/mixed",
        "key_drivers": ["driver1", "driver2"],
        "top_opportunities": [
            {"pair": "EURUSD", "recommendation": "BUY/SELL", "reason": "brief reason"}
        ]
    },
    "crypto_analysis": {
        "trend": "bullish/bearish/mixed", 
        "market_sentiment": "fear/greed/neutral",
        "bitcoin_dominance": "high/low/stable",
        "top_opportunities": [
            {"pair": "BTCUSDT", "recommendation": "BUY/SELL", "reason": "brief reason"}
        ]
    },
    "risk_factors": ["risk1", "risk2"],
    "trading_tips": ["tip1", "tip2"],
    "best_trades_today": [
        {"symbol": "PAIR", "action": "BUY/SELL", "confidence": 85}
    ]
}"""
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json|```", "", content).strip()
        data = json.loads(content)
        
        return {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "analysis": data
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

