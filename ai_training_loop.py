# ai_training_loop.py
import os
import json
import pandas as pd
from datetime import datetime
import requests
from backtest_api import load_history_df, run_backtest_from_signals
from fetch_all_history import run_fetch_all


# Optional: Binance API (or any crypto data provider)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")


def fetch_crypto_history(symbol="BTCUSDT", interval="1h", limit=1000):
    """
    Fetch crypto historical OHLCV data from Binance API
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    resp = requests.get(url)
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
    df.to_csv(f"backtest_api/history_data/{symbol}.csv", index=False)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_all_history(symbols=TRADING_SYMBOLS):
    """
    Fetch historical data for all symbols
    """
    for symbol in symbols:
        if "USDT" in symbol:  # crypto
            try:
                fetch_crypto_history(symbol)
                print(f"Fetched crypto data for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        else:
            # Forex: assume CSV already exists or use AlphaVantage/TwelveData API
            print(
                f"Forex symbol {symbol}, ensure CSV exists in history_data folder."
            )


def ai_training_loop():
    """
    Automated loop: get AI signals, backtest, and store results
    """
    results = []
    for symbol in TRADING_SYMBOLS:
        try:
            # 1. Fetch filtered AI predictions
            predictions = get_filtered_ai_signals(symbol=symbol)

            # 2. Load historical data
            df = load_history_df(symbol)

            # 3. Backtest each prediction
            for p in predictions:
                metrics, _, _ = run_backtest_from_signals(df, [p])
                p["historical_success_pct"] = metrics.get("success_pct", 0)
                p["backtested_total_return"] = metrics.get(
                    "total_return_pct", 0)

            # 4. Save results per symbol
            results.append({"symbol": symbol, "predictions": predictions})

        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
            from main import AI_TRAINED_SIGNALS

            AI_TRAINED_SIGNALS[symbol] = predictions


    # 5. Save results to JSON
    filename = f"ai_training_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("training_results", exist_ok=True)
    filepath = os.path.join("training_results", filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"AI training loop completed. Results saved to {filepath}")
    return results
    


# Optional: periodic loop
import asyncio
import threading


def start_training_loop(interval_minutes=60):

    async def run_periodically():
        while True:
            print("Running AI training loop...")
            fetch_all_history()
            ai_training_loop()
            await asyncio.sleep(interval_minutes * 60)

    threading.Thread(target=lambda: asyncio.run(run_periodically()),
                     daemon=True).start()
    def run_training_once():
        # 1. fetch latest history
        run_fetch_all()

        # 2. train
        results = ai_training_loop()

        return results

