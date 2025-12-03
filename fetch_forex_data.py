# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from openai import OpenAI
import json
import re
from datetime import datetime
from threading import Thread
import time
import requests
import pandas as pd
from time import sleep
from fetch_all_history import run_fetch_all
from ai_training_loop import run_training_once


# ----------------------------
# Output folder for historical data
# ----------------------------
HISTORY_FOLDER = "backtest_api/history_data"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# ----------------------------
# Symbols
# ----------------------------
FOREX_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "AUDNZD"
]

CRYPTO_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "TRXUSDT"
]

# ----------------------------
# API Keys
# ----------------------------
ALPHAVANTAGE_API_KEY = "AOUMNS4XUG8QQ2Q5"
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Multi-User Trading Bot Backend")

# ----------------------------
# CORS Middleware
# ----------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# User Model & In-Memory DB
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
# OpenAI Client
# ----------------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


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
# Fetch Forex CSV
# ----------------------------
def fetch_forex_csv(symbol: str):
    from_symbol = symbol[:3]
    to_symbol = symbol[3:]
    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
        "datatype": "csv"
    }
    try:
        response = requests.get(ALPHAVANTAGE_URL, params=params)
        response.raise_for_status()
        filepath = os.path.join(HISTORY_FOLDER, f"{symbol}.csv")
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"[Forex] Saved {symbol}.csv")
        sleep(12)
    except Exception as e:
        print(f"[Forex] Error fetching {symbol}: {e}")


# ----------------------------
# Fetch Crypto CSV
# ----------------------------
def fetch_crypto_csv(symbol: str, interval="1h", limit=1000):
    url = f"{BINANCE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        klines = resp.json()
        df = pd.DataFrame(klines,
                          columns=[
                              "timestamp", "open", "high", "low", "close",
                              "volume", "close_time", "quote_asset_volume",
                              "number_of_trades", "taker_buy_base",
                              "taker_buy_quote", "ignore"
                          ])
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        filepath = os.path.join(HISTORY_FOLDER, f"{symbol}.csv")
        df.to_csv(filepath, index=False)
        print(f"[Crypto] Saved {symbol}.csv")
        sleep(1)
    except Exception as e:
        print(f"[Crypto] Error fetching {symbol}: {e}")


# ----------------------------
# Fetch All Historical Data
# ----------------------------
@app.post("/fetch_all_data")
def fetch_all_data():
    for fx in FOREX_SYMBOLS:
        fetch_forex_csv(fx)
    for cr in CRYPTO_SYMBOLS:
        fetch_crypto_csv(cr)
    return {"status": "success", "message": "All historical data fetched"}


@app.post("/fetch_forex")
def fetch_forex():
    for fx in FOREX_SYMBOLS:
        fetch_forex_csv(fx)
    return {"status": "success", "message": "Forex data fetched"}


@app.post("/fetch_crypto")
def fetch_crypto():
    for cr in CRYPTO_SYMBOLS:
        fetch_crypto_csv(cr)
    return {"status": "success", "message": "Crypto data fetched"}


# ----------------------------
# Your existing AI endpoints and trading endpoints
# ----------------------------
# Here you can paste all your previous endpoints like:
# /predictions, /get_predictions, /execute_trade, /chat, etc.
# I can integrate them next exactly as before if you want.


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
def run_fetch_all():
    print("Fetching all Forex data...")
    for fx in FOREX_SYMBOLS:
        fetch_forex_csv(fx)

    print("Fetching all Crypto data...")
    for cr in CRYPTO_SYMBOLS:
        fetch_crypto_csv(cr)

    print("All historical data fetched successfully!")
    return True
