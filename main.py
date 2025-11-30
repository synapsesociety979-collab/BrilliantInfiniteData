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
    return {"status": "success", "message": f"User {user.username} connected successfully"}

# ----------------------------
# Generate AI Market Predictions
# ----------------------------


# ✅ **Here is the CORRECT version you must replace it with**


import json
import re

def generate_market_predictions():
    """
    Generates market predictions for crypto and Forex using OpenAI
    and returns real JSON (Python list/dict).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY pure JSON. No markdown, no ```json."
                },
                {
                    "role": "user",
                    "content": (
                        "Generate trade signals for Forex pairs (EURUSD, USDJPY, GBPUSD) "
                        "and crypto pairs (BTCUSDT, ETHUSDT). "
                        "Each signal must include: symbol, signal (BUY/SELL/HOLD), "
                        "confidence (0-100), stop_loss, take_profit. "
                        "Return only JSON array."
                    )
                }
            ],
        )

        # Extract content correctly
        content = response.choices[0].message.content.strip()

        # Remove markdown formatting if present
        content = re.sub(r"```json|```", "", content).strip()

        # Convert string → Python JSON object
        data = json.loads(content)

        return data

    except Exception as e:
        return {"error": str(e)}


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
        "predictions": ai_output
    }

# ----------------------------
# Execute Trade Endpoint (Placeholder)
# ----------------------------
@app.post("/execute_trade")
def execute_trade(username: str, symbol: str, action: str):
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Placeholder: Integrate with MT5 or Binance API here
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
