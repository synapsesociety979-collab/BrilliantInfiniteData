# backtest_api.py
import os
import io
import json
from typing import Optional, List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
import math

router = APIRouter()

# ---- OpenAI client (uses OPENAI_API_KEY env var) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # we'll keep going; /ai_backtest will error clearly when called
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---- storage folder for uploaded CSVs ----
DATA_DIR = "history_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# Utilities: read/write CSV
# ---------------------------
def save_history_csv(symbol: str, csv_bytes: bytes):
    fn = os.path.join(DATA_DIR, f"{symbol}.csv")
    with open(fn, "wb") as f:
        f.write(csv_bytes)
    return fn

def load_history_df(symbol: str) -> pd.DataFrame:
    fn = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"No history for symbol: {symbol}")
    df = pd.read_csv(fn, parse_dates=True)
    # Normalize column names to lowercase
    df.columns = [c.strip() for c in df.columns]
    # Expect columns: time/timestamp/date, open, high, low, close, volume
    # Try to infer time column
    possible_time_cols = [c for c in df.columns if c.lower() in ("time","timestamp","date","datetime")]
    if possible_time_cols:
        df = df.rename(columns={possible_time_cols[0]: "time"})
        df['time'] = pd.to_datetime(df['time'], unit=None, errors='coerce')
    if 'time' not in df.columns:
        # if index is datetime
        try:
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={'index': 'time'})
        except Exception:
            pass
    # Ensure numeric columns exist
    for col in ['open','high','low','close','volume']:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    # Keep only relevant columns
    df = df[['time','open','high','low','close','volume']]
    df = df.sort_values('time').reset_index(drop=True)
    return df

# ---------------------------
# Simple backtester engine
# ---------------------------
def run_backtest_from_signals(df: pd.DataFrame,
                              signals: List[str],
                              init_cash: float = 1000.0,
                              risk_per_trade: float = 0.01,
                              slippage: float = 0.000,   # as fraction of price
                              commission: float = 0.0):
    """
    signals: list of 'BUY','SELL','HOLD' aligned to df rows.
    Simple position: full-size long only; a BUY opens 1 position, SELL closes it.
    Returns metrics dict and equity_curve (list of equity per bar).
    """
    if len(signals) != len(df):
        raise ValueError("signals length must equal dataframe length")

    cash = init_cash
    position = 0.0   # number of units
    entry_price = None
    equity_curve = []
    trade_results = []
    peak = init_cash
    drawdown = 0.0

    for i, row in df.iterrows():
        price = float(row['close'])
        sig = signals[i].upper() if isinstance(signals[i], str) else "HOLD"

        # close signal
        if sig == "SELL" and position > 0:
            # exit
            exit_price = price * (1 + slippage)
            proceeds = position * exit_price
            cost = commission
            pnl = proceeds - (position * entry_price) - cost
            cash += proceeds - cost
            trade_results.append({"entry": entry_price, "exit": exit_price, "pnl": pnl})
            position = 0
            entry_price = None

        # buy signal: invest risk_per_trade fraction of equity into asset
        elif sig == "BUY" and position == 0:
            available = cash * risk_per_trade
            if available > 0:
                buy_price = price * (1 + slippage)
                units = available / buy_price
                cost = commission
                position = units
                entry_price = buy_price
                cash -= (units * buy_price) + cost

        # update equity
        market_value = position * price
        equity = cash + market_value
        equity_curve.append(equity)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        drawdown = max(drawdown, dd)

    # If position still open at end, close at last price
    if position > 0:
        price = float(df.iloc[-1]['close'])
        exit_price = price * (1 + slippage)
        proceeds = position * exit_price
        cost = commission
        pnl = proceeds - (position * entry_price) - cost
        cash += proceeds - cost
        trade_results.append({"entry": entry_price, "exit": exit_price, "pnl": pnl})
        position = 0
        entry_price = None
        equity = cash
        equity_curve[-1] = equity

    total_return = (cash - init_cash) / init_cash * 100
    wins = sum(1 for t in trade_results if t['pnl'] > 0)
    losses = sum(1 for t in trade_results if t['pnl'] <= 0)
    win_rate = (wins / len(trade_results)) * 100 if trade_results else 0.0
    gross_profit = sum(t['pnl'] for t in trade_results if t['pnl'] > 0)
    gross_loss = sum(t['pnl'] for t in trade_results if t['pnl'] <= 0)

    metrics = {
        "init_cash": init_cash,
        "final_cash": cash,
        "net_profit": cash - init_cash,
        "total_return_pct": total_return,
        "trades": len(trade_results),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "max_drawdown_pct": drawdown * 100,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss
    }

    return metrics, equity_curve, trade_results

# ---------------------------
# Basic SMA strategy generator (rule-based)
# ---------------------------
def signals_from_sma(df: pd.DataFrame, short: int = 20, long: int = 50):
    s1 = df['close'].rolling(short).mean()
    s2 = df['close'].rolling(long).mean()
    sigs = []
    prev_state = "HOLD"
    for i in range(len(df)):
        if i < long:
            sigs.append("HOLD")
            continue
        if s1.iloc[i] > s2.iloc[i] and s1.iloc[i-1] <= s2.iloc[i-1]:
            sigs.append("BUY")
        elif s1.iloc[i] < s2.iloc[i] and s1.iloc[i-1] >= s2.iloc[i-1]:
            sigs.append("SELL")
        else:
            sigs.append("HOLD")
    return sigs

# ---------------------------
# AI-driven signals (calls OpenAI)
# ---------------------------
def generate_signals_with_openai(df: pd.DataFrame, sample_bars: int = 150) -> List[str]:
    """
    Sends recent candle data to OpenAI and requests a list of signals aligned to the bars.
    WARNING: tokens limits — we cap to last `sample_bars` bars by default.
    The model must be able to handle the prompt; if the model can't handle it, reduce bars.
    """
    if client is None:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing).")

    n = min(sample_bars, len(df))
    sub = df.tail(n).copy().reset_index(drop=True)
    # Build a compact representation: list of arrays or lines "time,open,high,low,close,volume"
    lines = []
    for i, r in sub.iterrows():
        t = r['time'].isoformat() if not pd.isna(r['time']) else str(i)
        lines.append(f"{i},{r['open']:.6f},{r['high']:.6f},{r['low']:.6f},{r['close']:.6f},{int(r['volume'])}")

    data_blob = "\n".join(lines)

    prompt = (
        "You are a helpful trading assistant. I will provide recent price bars in the format:\n"
        "INDEX,OPEN,HIGH,LOW,CLOSE,VOLUME (one bar per line).\n"
        f"There are {n} bars, index 0 is oldest and {n-1} newest.\n"
        "Please output a JSON array of length {n} with values 'BUY', 'SELL' or 'HOLD' for each bar, "
        "representing the signal generated at the close of that bar. Output ONLY the JSON array (no explanation).\n\n"
        "Here are the bars:\n"
        f"{data_blob}\n\n"
        "Respond with only a JSON array like [\"HOLD\",\"HOLD\",\"BUY\",...].\n"
    )

    # Call OpenAI
    try:
        # Use chat completions
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # follow your allowed model
            messages=[
                {"role":"system", "content":"You are a JSON-generation assistant."},
                {"role":"user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        content = resp.choices[0].message.content.strip()
        # Remove code fences if any
        content = content.replace("```json", "").replace("```", "").strip()
        signals = json.loads(content)
        # If shorter than df tail, pad with HOLD at front
        if len(signals) < n:
            signals = ["HOLD"]*(n-len(signals)) + signals
        # Expand to full length of df by prefixing holds
        prefix = ["HOLD"] * (len(df) - n)
        full_signals = prefix + signals
        return full_signals
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

# ---------------------------
# Routes
# ---------------------------

@router.post("/upload_history")
async def upload_history(symbol: str, file: UploadFile = File(...)):
    """
    Upload a CSV for a symbol. Required CSV cols: time,open,high,low,close,volume
    Example form-data: symbol=BTCUSDT, file=@BTCUSDT.csv
    """
    contents = await file.read()
    try:
        # quick validation
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
    save_history_csv(symbol, contents)
    return {"status":"ok", "symbol": symbol, "rows": len(df)}

@router.get("/history/{symbol}")
def get_history(symbol: str, head: int = 50):
    try:
        df = load_history_df(symbol)
    except Exception as e:
        raise HTTPException(404, str(e))
    return {"symbol": symbol, "rows": len(df), "head": df.tail(head).to_dict(orient="records")}

@router.post("/backtest")
def backtest_endpoint(symbol: str,
                      strategy: str = "sma",
                      short: int = 20,
                      long: int = 50,
                      init_cash: float = 1000.0,
                      risk_per_trade: float = 0.01,
                      sample_bars: int = 150):
    """
    Backtest endpoint.
    - strategy: "sma" or "ai"
    - for "sma" uses short/long
    - for "ai" sends last `sample_bars` to OpenAI for signals then backtests
    """
    try:
        df = load_history_df(symbol)
    except Exception as e:
        raise HTTPException(404, str(e))

    if strategy.lower() == "sma":
        signals = signals_from_sma(df, short=short, long=long)
    elif strategy.lower() == "ai":
        try:
            signals = generate_signals_with_openai(df, sample_bars=sample_bars)
        except Exception as e:
            raise HTTPException(500, f"AI signal generation failed: {e}")
    else:
        raise HTTPException(400, "Unknown strategy")

    metrics, equity_curve, trades = run_backtest_from_signals(
        df, signals, init_cash=init_cash, risk_per_trade=risk_per_trade
    )

    # Add timestamps for equity curve
    timestamps = [str(t) for t in df['time'].astype(str).tolist()]
    equity_timeseries = [{"time": timestamps[i], "equity": equity_curve[i]} for i in range(len(equity_curve))]

    return {
        "symbol": symbol,
        "strategy": strategy,
        "metrics": metrics,
        "equity_curve": equity_timeseries,
        "trades": trades
    }

# small health route for plugin
@router.get("/backtest/health")
def backtest_health():
    return {"status":"ok", "message":"backtest router active"}
