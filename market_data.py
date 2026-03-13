# market_data.py — Real live market data + technical indicator engine
import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

# Cache: symbol -> {data, timestamp}
_DATA_CACHE: Dict[str, dict] = {}
CACHE_TTL = 900  # 15 minutes — respects free tier rate limits


# --------------------------------------------------
# Alpha Vantage fetchers
# --------------------------------------------------

def _fetch_av(params: dict, timeout: int = 12) -> dict:
    params["apikey"] = ALPHA_VANTAGE_API_KEY
    try:
        r = requests.get(BASE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fetch_forex_candles(from_sym: str, to_sym: str, interval: str = "5min") -> Optional[pd.DataFrame]:
    """Fetch real intraday OHLCV for a forex pair from Alpha Vantage."""
    key = f"fx_{from_sym}{to_sym}_{interval}"
    if key in _DATA_CACHE and time.time() - _DATA_CACHE[key]["ts"] < CACHE_TTL:
        return _DATA_CACHE[key]["df"]

    data = _fetch_av({
        "function": "FX_INTRADAY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "interval": interval,
        "outputsize": "compact",
    })

    ts_key = f"Time Series FX ({interval})"
    if ts_key not in data:
        return None

    rows = []
    for dt_str, vals in data[ts_key].items():
        rows.append({
            "time": pd.to_datetime(dt_str),
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["4. close"]),
            "volume": float(vals.get("5. volume", 0) or 0),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    _DATA_CACHE[key] = {"df": df, "ts": time.time()}
    return df


def fetch_crypto_candles(symbol: str, market: str = "USD", interval: str = "5min") -> Optional[pd.DataFrame]:
    """Fetch real intraday OHLCV for a crypto pair from Alpha Vantage."""
    base = symbol.replace("USDT", "").replace("BUSD", "")
    key = f"crypto_{base}{market}_{interval}"
    if key in _DATA_CACHE and time.time() - _DATA_CACHE[key]["ts"] < CACHE_TTL:
        return _DATA_CACHE[key]["df"]

    data = _fetch_av({
        "function": "CRYPTO_INTRADAY",
        "symbol": base,
        "market": market,
        "interval": interval,
        "outputsize": "compact",
    })

    ts_key = f"Time Series Crypto ({interval})"
    if ts_key not in data:
        return None

    rows = []
    for dt_str, vals in data[ts_key].items():
        rows.append({
            "time": pd.to_datetime(dt_str),
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["4. close"]),
            "volume": float(vals.get("5. volume", 0) or 0),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    _DATA_CACHE[key] = {"df": df, "ts": time.time()}
    return df


def fetch_realtime_quote(from_sym: str, to_sym: str = "USD") -> dict:
    """Fetch real-time bid/ask for forex or USD rate for crypto."""
    data = _fetch_av({
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_sym,
        "to_currency": to_sym,
    })
    rate_data = data.get("Realtime Currency Exchange Rate", {})
    if not rate_data:
        return {}
    return {
        "price": float(rate_data.get("5. Exchange Rate", 0) or 0),
        "bid":   float(rate_data.get("8. Bid Price", 0) or 0),
        "ask":   float(rate_data.get("9. Ask Price", 0) or 0),
        "last_refreshed": rate_data.get("6. Last Refreshed", ""),
    }


# --------------------------------------------------
# Technical Indicator Engine (calculated from real OHLCV)
# --------------------------------------------------

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else 50.0


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    dp = 6
    return (round(float(macd_line.iloc[-1]), dp),
            round(float(signal_line.iloc[-1]), dp),
            round(float(histogram.iloc[-1]), dp))


def calc_ema(close: pd.Series, period: int) -> float:
    ema = close.ewm(span=period, adjust=False).mean()
    return round(float(ema.iloc[-1]), 6)


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    dp = 6
    return (round(float(upper.iloc[-1]), dp),
            round(float(mid.iloc[-1]), dp),
            round(float(lower.iloc[-1]), dp))


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return round(float(atr.iloc[-1]), 6)


def calc_volume_signal(volume: pd.Series) -> str:
    if len(volume) < 20:
        return "insufficient data"
    avg = volume.rolling(20).mean().iloc[-1]
    last = volume.iloc[-1]
    if last > avg * 1.5:
        return f"HIGH ({last:,.0f} vs avg {avg:,.0f}) — strong confirmation"
    elif last > avg:
        return f"ABOVE average ({last:,.0f} vs avg {avg:,.0f})"
    else:
        return f"BELOW average ({last:,.0f} vs avg {avg:,.0f}) — weak confirmation"


def calc_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
    recent = df.tail(lookback)
    support = round(float(recent["low"].min()), 6)
    resistance = round(float(recent["high"].max()), 6)
    return support, resistance


def calc_trend(close: pd.Series) -> str:
    if len(close) < 50:
        return "insufficient data"
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    if price > ema20 > ema50:
        return "BULLISH (price > EMA20 > EMA50)"
    elif price < ema20 < ema50:
        return "BEARISH (price < EMA20 < EMA50)"
    elif price > ema50:
        return "MILDLY BULLISH (price above EMA50)"
    else:
        return "MILDLY BEARISH (price below EMA50)"


def detect_candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "none"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    total_range = last["high"] - last["low"]

    if total_range == 0:
        return "doji"
    if body / total_range < 0.1:
        return "doji (indecision)"
    if lower_wick > 2 * body and upper_wick < body:
        return "hammer/pin bar (bullish reversal signal)"
    if upper_wick > 2 * body and lower_wick < body:
        return "shooting star (bearish reversal signal)"
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["close"] > prev["open"]:
        return "bullish engulfing"
    if last["close"] < last["open"] and prev["close"] > prev["open"] and last["close"] < prev["open"]:
        return "bearish engulfing"
    return "no clear pattern"


# --------------------------------------------------
# Main analysis function
# --------------------------------------------------

def get_symbol_analysis(symbol: str) -> dict:
    """
    Returns a full technical analysis dict with REAL live data.
    Falls back gracefully if Alpha Vantage is unavailable.
    """
    symbol = symbol.upper()
    cache_key = f"analysis_{symbol}"
    if cache_key in _DATA_CACHE and time.time() - _DATA_CACHE[cache_key]["ts"] < CACHE_TTL:
        return _DATA_CACHE[cache_key]["data"]

    is_crypto = any(x in symbol for x in ["USDT", "BUSD", "BTC", "ETH"])
    df = None

    if is_crypto:
        base = symbol.replace("USDT", "").replace("BUSD", "")
        df = fetch_crypto_candles(symbol, market="USD", interval="5min")
    else:
        from_sym = symbol[:3]
        to_sym = symbol[3:]
        df = fetch_forex_candles(from_sym, to_sym, interval="5min")

    if df is None or len(df) < 30:
        return {
            "symbol": symbol,
            "data_source": "unavailable",
            "live_price": None,
            "error": "Could not fetch live data — AI will use pattern reasoning instead"
        }

    close = df["close"]
    live_price = round(float(close.iloc[-1]), 6)

    rsi = calc_rsi(close)
    macd_line, signal_line, histogram = calc_macd(close)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    ema200 = calc_ema(close, 200) if len(close) >= 200 else None
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    atr = calc_atr(df)
    volume_sig = calc_volume_signal(df["volume"])
    support, resistance = calc_support_resistance(df)
    trend = calc_trend(close)
    pattern = detect_candle_pattern(df)

    # Suggested SL/TP from ATR
    sl_buy  = round(live_price - atr * 1.5, 6)
    sl_sell = round(live_price + atr * 1.5, 6)
    tp1_buy = round(live_price + atr * 1.5, 6)
    tp2_buy = round(live_price + atr * 3.0, 6)
    tp3_buy = round(live_price + atr * 4.5, 6)

    # RSI interpretation
    if rsi < 30:
        rsi_reading = f"{rsi} — OVERSOLD (strong buy bias)"
    elif rsi < 40:
        rsi_reading = f"{rsi} — approaching oversold (mild buy bias)"
    elif rsi > 70:
        rsi_reading = f"{rsi} — OVERBOUGHT (strong sell bias)"
    elif rsi > 60:
        rsi_reading = f"{rsi} — approaching overbought (mild sell bias)"
    else:
        rsi_reading = f"{rsi} — neutral zone"

    # MACD interpretation
    if histogram > 0 and histogram > df["close"].diff().abs().mean() * 0.01:
        macd_reading = f"bullish (histogram={histogram:+.6f}, MACD above signal)"
    elif histogram < 0:
        macd_reading = f"bearish (histogram={histogram:+.6f}, MACD below signal)"
    else:
        macd_reading = f"neutral (histogram={histogram:+.6f})"

    # BB position
    bb_width = bb_upper - bb_lower
    bb_pct = (live_price - bb_lower) / bb_width if bb_width > 0 else 0.5
    if bb_pct > 0.9:
        bb_reading = f"price near upper band ({live_price} vs upper {bb_upper}) — overbought zone"
    elif bb_pct < 0.1:
        bb_reading = f"price near lower band ({live_price} vs lower {bb_lower}) — oversold zone"
    elif bb_width < close.std() * 0.5:
        bb_reading = f"SQUEEZE detected (tight bands) — breakout imminent"
    else:
        bb_reading = f"price at {bb_pct:.0%} of band width (mid: {bb_mid})"

    # EMA bias
    ema_parts = [f"EMA20={ema20}", f"EMA50={ema50}"]
    if ema200:
        ema_parts.append(f"EMA200={ema200}")
    ema_reading = f"Price={live_price} | {' | '.join(ema_parts)} | Trend: {trend}"

    result = {
        "symbol": symbol,
        "data_source": "Alpha Vantage live",
        "live_price": live_price,
        "candles_used": len(df),
        "last_candle_time": df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "indicators": {
            "rsi_14": rsi_reading,
            "macd_12_26_9": macd_reading,
            "ema_stack": ema_reading,
            "bollinger_bands": bb_reading,
            "volume": volume_sig,
            "atr_14": atr,
            "candle_pattern": pattern,
        },
        "key_levels": {
            "support": support,
            "resistance": resistance,
            "atr_based_sl_for_buy": sl_buy,
            "atr_based_sl_for_sell": sl_sell,
            "atr_based_tp1": tp1_buy,
            "atr_based_tp2": tp2_buy,
            "atr_based_tp3": tp3_buy,
        },
        "trend_bias": trend,
    }

    _DATA_CACHE[cache_key] = {"data": result, "ts": time.time()}
    return result


def format_for_ai_prompt(analysis: dict) -> str:
    """Convert analysis dict into a compact string for the AI prompt."""
    if analysis.get("error") or not analysis.get("live_price"):
        return f"{analysis['symbol']}: No live data available — use pattern reasoning."

    ind = analysis.get("indicators", {})
    lv  = analysis.get("key_levels", {})
    lines = [
        f"=== {analysis['symbol']} | Live: {analysis['live_price']} | Source: {analysis['data_source']} ===",
        f"RSI(14): {ind.get('rsi_14', 'N/A')}",
        f"MACD: {ind.get('macd_12_26_9', 'N/A')}",
        f"EMA: {ind.get('ema_stack', 'N/A')}",
        f"Bollinger: {ind.get('bollinger_bands', 'N/A')}",
        f"Volume: {ind.get('volume', 'N/A')}",
        f"ATR(14): {ind.get('atr_14', 'N/A')} | Pattern: {ind.get('candle_pattern', 'N/A')}",
        f"Support: {lv.get('support')} | Resistance: {lv.get('resistance')}",
        f"ATR SL(buy): {lv.get('atr_based_sl_for_buy')} | TP1: {lv.get('atr_based_tp1')} | TP2: {lv.get('atr_based_tp2')} | TP3: {lv.get('atr_based_tp3')}",
    ]
    return "\n".join(lines)
