# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI

# ----------------------------
# OpenAI API Setup
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

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
from typing import List, Dict, Any


def generate_market_predictions() -> Dict[str, Any]:
    """
    Generates market predictions for crypto and Forex using OpenAI
    with enhanced analysis and structured output.
    """
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
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Multi-User Trading Bot Backend is running"}


import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 3000)),
        reload=True
    )

