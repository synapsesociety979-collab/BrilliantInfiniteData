# market_data.py — Real live market data + technical indicator engine
# Free Alpha Vantage endpoints used:
#   FX_DAILY                   — daily OHLCV for forex pairs
#   DIGITAL_CURRENCY_DAILY     — daily OHLCV for crypto
#   CURRENCY_EXCHANGE_RATE     — real-time price overlay
#
# FALLBACK: When Alpha Vantage is rate-limited or unavailable,
#   ARIA uses estimated reference prices + AI pattern reasoning.
#   The AI will clearly label which mode it's using.

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

_DATA_CACHE: Dict[str, dict] = {}
CACHE_TTL     = 900    # 15 min cache for live data
FALLBACK_TTL  = 300    # 5 min cache for AI-only mode

# ── Disk-backed cache ─────────────────────────────────────────────
# Survives server restarts so we don't blow through AV rate limits
# on every redeploy.
import json as _json
_DISK_CACHE_PATH = "/tmp/aria_market_cache.json"

def _load_disk_cache() -> None:
    """Load persisted cache from disk into _DATA_CACHE on startup."""
    global _DATA_CACHE
    try:
        with open(_DISK_CACHE_PATH, "r") as f:
            raw = _json.load(f)
        now = time.time()
        # Only keep entries that haven't fully expired (use 2× TTL so we
        # still have something to serve while we try to refresh)
        for k, v in raw.items():
            age = now - v.get("ts", 0)
            ttl = FALLBACK_TTL if v["data"].get("ai_only_mode") else CACHE_TTL
            if age < ttl * 2:
                _DATA_CACHE[k] = v
        print(f"[CACHE] Loaded {len(_DATA_CACHE)} entries from disk")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[CACHE] Disk load error (non-fatal): {e}")

def _save_disk_cache() -> None:
    """Persist the current in-memory cache to disk."""
    try:
        with open(_DISK_CACHE_PATH, "w") as f:
            _json.dump(_DATA_CACHE, f)
    except Exception as e:
        print(f"[CACHE] Disk save error (non-fatal): {e}")

_load_disk_cache()   # run on import

# ── AV rate-limit circuit breaker ────────────────────────────────
# If AV rate-limits us, stop hammering it for 30 min.
_AV_RATE_LIMITED_UNTIL: float = 0.0   # unix timestamp

def _av_is_rate_limited() -> bool:
    return time.time() < _AV_RATE_LIMITED_UNTIL

def _mark_av_rate_limited(minutes: int = 30) -> None:
    global _AV_RATE_LIMITED_UNTIL
    _AV_RATE_LIMITED_UNTIL = time.time() + minutes * 60
    print(f"[AV] Circuit breaker: pausing AV calls for {minutes} min")


# ─────────────────────────────────────────────────────────────────
#  Reference prices — used when Alpha Vantage is unavailable.
#  These keep ARIA aware of approximate price levels so her
#  SL/TP suggestions are in the right ballpark.
#  Updated periodically — exact values matter less than order of magnitude.
# ─────────────────────────────────────────────────────────────────
FALLBACK_PRICES: Dict[str, float] = {
    # Forex
    "EURUSD": 1.0820, "GBPUSD": 1.2940, "USDJPY": 150.20,
    "AUDUSD": 0.6310, "USDCAD": 1.3680, "NZDUSD": 0.5740,
    "USDCHF": 0.8870, "EURGBP": 0.8360, "EURJPY": 162.50,
    "GBPJPY": 194.40, "AUDJPY": 94.80,  "EURCAD": 1.4820,
    "AUDCAD": 0.8630, "EURAUD": 1.7160, "GBPAUD": 2.0510,
    # Crypto
    "BTCUSDT":  83500.0, "ETHUSDT":  1920.0,  "BNBUSDT":  590.0,
    "XRPUSDT":  2.15,    "SOLUSDT":  130.0,   "ADAUSDT":  0.72,
    "DOGEUSDT": 0.168,   "DOTUSDT":  4.80,    "MATICUSDT":0.42,
    "LTCUSDT":  105.0,   "SHIBUSDT": 0.0000138,"TRXUSDT":  0.235,
    "AVAXUSDT": 22.5,    "LINKUSDT": 13.8,    "UNIUSDT":  7.40,
}

# Approximate ATR values (% of price) for fallback SL/TP
FALLBACK_ATR_PCT: Dict[str, float] = {
    "EURUSD": 0.006, "GBPUSD": 0.008, "USDJPY": 0.007,
    "AUDUSD": 0.007, "USDCAD": 0.007, "NZDUSD": 0.007,
    "USDCHF": 0.006, "EURGBP": 0.005, "EURJPY": 0.008,
    "GBPJPY": 0.010, "AUDJPY": 0.009, "EURCAD": 0.008,
    "AUDCAD": 0.008, "EURAUD": 0.009, "GBPAUD": 0.010,
    "BTCUSDT":  0.030, "ETHUSDT":  0.035, "BNBUSDT":  0.030,
    "XRPUSDT":  0.040, "SOLUSDT":  0.040, "ADAUSDT":  0.040,
    "DOGEUSDT": 0.045, "DOTUSDT":  0.045, "MATICUSDT":0.050,
    "LTCUSDT":  0.035, "SHIBUSDT": 0.050, "TRXUSDT":  0.040,
    "AVAXUSDT": 0.045, "LINKUSDT": 0.040, "UNIUSDT":  0.045,
}


# ─────────────────────────────────────────────────────────────────
#  Crypto helpers
# ─────────────────────────────────────────────────────────────────
CRYPTO_BASES = {
    "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOGE",
    "DOT", "MATIC", "LTC", "SHIB", "TRX", "AVAX", "LINK", "UNI"
}

def extract_crypto_base(symbol: str) -> Optional[str]:
    s = symbol.upper()
    for suffix in ["USDT", "BUSD", "USD", "USDC"]:
        if s.endswith(suffix):
            base = s[: len(s) - len(suffix)]
            if base in CRYPTO_BASES:
                return base
    if s in CRYPTO_BASES:
        return s
    return None

def is_crypto_symbol(symbol: str) -> bool:
    return extract_crypto_base(symbol) is not None

def normalize_symbol(symbol: str) -> str:
    base = extract_crypto_base(symbol)
    if base:
        return f"{base}USDT"
    return symbol.upper()


# ─────────────────────────────────────────────────────────────────
#  Alpha Vantage fetch (handles rate limits gracefully)
# ─────────────────────────────────────────────────────────────────
def _fetch_av(params: dict, timeout: int = 15) -> dict:
    if not ALPHA_VANTAGE_API_KEY:
        return {"_no_key": True}
    # Circuit breaker: if we've been rate-limited recently, skip all AV calls
    if _av_is_rate_limited():
        return {"_limited": True}
    params["apikey"] = ALPHA_VANTAGE_API_KEY
    try:
        r = requests.get(BASE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if "Note" in j or "Information" in j:
            msg = j.get("Note") or j.get("Information", "")
            print(f"[AV] Rate limit detected — engaging circuit breaker for 30 min: {msg[:60]}")
            _mark_av_rate_limited(30)
            return {"_limited": True}
        if "Error Message" in j:
            print(f"[AV] Error: {j['Error Message']}")
            return {"_error": True}
        return j
    except Exception as e:
        print(f"[AV] Request failed: {e}")
        return {"_error": True}


# ─────────────────────────────────────────────────────────────────
#  Real-time quote
# ─────────────────────────────────────────────────────────────────
def fetch_realtime_quote(from_sym: str, to_sym: str = "USD") -> dict:
    cache_key = f"quote_{from_sym}{to_sym}"
    if cache_key in _DATA_CACHE and time.time() - _DATA_CACHE[cache_key]["ts"] < 300:
        return _DATA_CACHE[cache_key]["data"]
    data = _fetch_av({
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_sym,
        "to_currency": to_sym,
    })
    rate_data = data.get("Realtime Currency Exchange Rate", {})
    if not rate_data:
        return {}
    result = {
        "price": float(rate_data.get("5. Exchange Rate") or 0),
        "bid":   float(rate_data.get("8. Bid Price") or 0),
        "ask":   float(rate_data.get("9. Ask Price") or 0),
        "last_refreshed": rate_data.get("6. Last Refreshed", ""),
    }
    _DATA_CACHE[cache_key] = {"data": result, "ts": time.time()}
    return result


# ─────────────────────────────────────────────────────────────────
#  Forex daily OHLCV
# ─────────────────────────────────────────────────────────────────
def fetch_forex_daily(from_sym: str, to_sym: str) -> Optional[pd.DataFrame]:
    cache_key = f"fx_daily_{from_sym}{to_sym}"
    if cache_key in _DATA_CACHE and time.time() - _DATA_CACHE[cache_key]["ts"] < CACHE_TTL:
        return _DATA_CACHE[cache_key]["df"]

    data = _fetch_av({
        "function": "FX_DAILY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "outputsize": "full",
    })

    ts_key = "Time Series FX (Daily)"
    if ts_key not in data:
        return None

    rows = []
    for dt_str, vals in data[ts_key].items():
        rows.append({
            "time":   pd.to_datetime(dt_str),
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["4. close"]),
            "volume": 0.0,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    _DATA_CACHE[cache_key] = {"df": df, "ts": time.time()}
    return df


# ─────────────────────────────────────────────────────────────────
#  Crypto daily OHLCV
# ─────────────────────────────────────────────────────────────────
def fetch_crypto_daily(base_symbol: str, market: str = "USD") -> Optional[pd.DataFrame]:
    cache_key = f"crypto_daily_{base_symbol}{market}"
    if cache_key in _DATA_CACHE and time.time() - _DATA_CACHE[cache_key]["ts"] < CACHE_TTL:
        return _DATA_CACHE[cache_key]["df"]

    data = _fetch_av({
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": base_symbol,
        "market": market,
    })

    ts_key = "Time Series (Digital Currency Daily)"
    if ts_key not in data:
        return None

    rows = []
    for dt_str, vals in data[ts_key].items():
        rows.append({
            "time":   pd.to_datetime(dt_str),
            "open":   float(vals.get("1. open") or vals.get("1a. open (USD)") or 0),
            "high":   float(vals.get("2. high") or vals.get("2a. high (USD)") or 0),
            "low":    float(vals.get("3. low")  or vals.get("3a. low (USD)")  or 0),
            "close":  float(vals.get("4. close") or vals.get("4a. close (USD)") or 0),
            "volume": float(vals.get("5. volume") or 0),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    _DATA_CACHE[cache_key] = {"df": df, "ts": time.time()}
    return df


# ─────────────────────────────────────────────────────────────────
#  Technical Indicators
# ─────────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else 50.0


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[float, float, float]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    dp = 6
    return (round(float(macd_line.iloc[-1]),  dp),
            round(float(signal_line.iloc[-1]), dp),
            round(float(histogram.iloc[-1]),   dp))


def calc_ema(close: pd.Series, period: int) -> Optional[float]:
    if len(close) < period:
        return None
    val = close.ewm(span=period, adjust=False).mean().iloc[-1]
    return round(float(val), 6)


def calc_bollinger(close: pd.Series, period=20, std_dev=2.0) -> Tuple[float, float, float]:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    dp = 6
    return (round(float(upper.iloc[-1]), dp),
            round(float(mid.iloc[-1]),   dp),
            round(float(lower.iloc[-1]), dp))


def calc_atr(df: pd.DataFrame, period=14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 8)


def calc_volume_signal(volume: pd.Series) -> str:
    if volume.sum() == 0:
        return "no volume data (forex daily — typical)"
    if len(volume) < 20:
        return "insufficient data"
    avg  = volume.rolling(20).mean().iloc[-1]
    last = volume.iloc[-1]
    if last > avg * 1.5:
        return f"HIGH ({last:,.0f} vs avg {avg:,.0f}) — strong confirmation"
    elif last > avg:
        return f"ABOVE average ({last:,.0f} vs avg {avg:,.0f})"
    else:
        return f"BELOW average ({last:,.0f} vs avg {avg:,.0f}) — weak confirmation"


def calc_support_resistance(df: pd.DataFrame, lookback=60) -> Tuple[float, float]:
    recent = df.tail(lookback)
    return round(float(recent["low"].min()), 6), round(float(recent["high"].max()), 6)


def calc_trend(close: pd.Series) -> str:
    ema20  = calc_ema(close, 20)
    ema50  = calc_ema(close, 50)
    ema200 = calc_ema(close, 200)
    price  = round(float(close.iloc[-1]), 6)

    parts = [f"price={price}"]
    if ema20:  parts.append(f"EMA20={ema20}")
    if ema50:  parts.append(f"EMA50={ema50}")
    if ema200: parts.append(f"EMA200={ema200}")

    if ema20 and ema50 and ema200:
        if price > ema20 > ema50 > ema200:
            return f"STRONG BULLISH — {' | '.join(parts)}"
        elif price < ema20 < ema50 < ema200:
            return f"STRONG BEARISH — {' | '.join(parts)}"
        elif price > ema50 > ema200:
            return f"BULLISH (above 50+200) — {' | '.join(parts)}"
        elif price < ema50 < ema200:
            return f"BEARISH (below 50+200) — {' | '.join(parts)}"
    if ema20 and ema50:
        if price > ema20 > ema50:
            return f"BULLISH — {' | '.join(parts)}"
        elif price < ema20 < ema50:
            return f"BEARISH — {' | '.join(parts)}"
    return f"RANGING / MIXED — {' | '.join(parts)}"


def calc_stochastic(df: pd.DataFrame, k_period=14, d_period=3) -> Tuple[float, float]:
    """Stochastic %K and %D — overbought >80, oversold <20."""
    low_min  = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return round(float(k.iloc[-1]), 2), round(float(d.iloc[-1]), 2)


def calc_adx(df: pd.DataFrame, period=14) -> Tuple[float, float, float]:
    """
    Average Directional Index — measures trend STRENGTH (not direction).
    ADX > 25 = trending market (tradeable)
    ADX < 20 = ranging market (avoid directional trades)
    Returns (ADX, +DI, -DI)
    """
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out where the other is bigger
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr_s    = tr.rolling(period).sum()
    plus_di  = 100 * plus_dm.rolling(period).sum()  / (atr_s + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).sum() / (atr_s + 1e-10)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx      = dx.rolling(period).mean()

    return (round(float(adx.iloc[-1]), 2),
            round(float(plus_di.iloc[-1]), 2),
            round(float(minus_di.iloc[-1]), 2))


def calc_williams_r(df: pd.DataFrame, period=14) -> float:
    """Williams %R — overbought > -20, oversold < -80."""
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    wr = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
    return round(float(wr.iloc[-1]), 2)


def calc_pivot_points(df: pd.DataFrame) -> dict:
    """Classic pivot points from the most recent completed candle."""
    if len(df) < 2:
        return {}
    prev  = df.iloc[-2]   # use previous completed candle
    H, L, C = prev["high"], prev["low"], prev["close"]
    pivot = (H + L + C) / 3
    r1 = 2 * pivot - L
    r2 = pivot + (H - L)
    r3 = H + 2 * (pivot - L)
    s1 = 2 * pivot - H
    s2 = pivot - (H - L)
    s3 = L - 2 * (H - pivot)
    dp = 6
    return {
        "pivot": round(pivot, dp),
        "R1": round(r1, dp), "R2": round(r2, dp), "R3": round(r3, dp),
        "S1": round(s1, dp), "S2": round(s2, dp), "S3": round(s3, dp),
    }


def calc_confluence_score(
    rsi: float,
    macd_hist: float,
    ema20: Optional[float],
    ema50: Optional[float],
    price: float,
    stoch_k: float,
    stoch_d: float,
    adx: float,
    williams_r: float,
    bb_pct: float,
) -> Tuple[str, int, List[str]]:
    """
    Score how many independent indicators agree on direction.
    Returns (direction, confluence_count, reasons[])
    direction = "BUY" | "SELL" | "NEUTRAL"
    confluence_count = 0–8 (higher = stronger signal)
    """
    bull_points = 0
    bear_points = 0
    reasons = []

    # RSI
    if rsi < 45:
        bull_points += 1
        reasons.append(f"RSI {rsi} (bullish zone)")
    elif rsi > 55:
        bear_points += 1
        reasons.append(f"RSI {rsi} (bearish zone)")

    # MACD histogram
    if macd_hist > 0:
        bull_points += 1
        reasons.append("MACD histogram positive")
    elif macd_hist < 0:
        bear_points += 1
        reasons.append("MACD histogram negative")

    # EMA stack
    if ema20 and ema50:
        if price > ema20 > ema50:
            bull_points += 1
            reasons.append("Price above EMA20 > EMA50 (bullish stack)")
        elif price < ema20 < ema50:
            bear_points += 1
            reasons.append("Price below EMA20 < EMA50 (bearish stack)")

    # Stochastic
    if stoch_k < 25 and stoch_k > stoch_d:
        bull_points += 1
        reasons.append(f"Stoch {stoch_k:.0f} oversold + K crossing above D")
    elif stoch_k > 75 and stoch_k < stoch_d:
        bear_points += 1
        reasons.append(f"Stoch {stoch_k:.0f} overbought + K crossing below D")

    # Williams %R
    if williams_r < -75:
        bull_points += 1
        reasons.append(f"Williams %R {williams_r} oversold")
    elif williams_r > -25:
        bear_points += 1
        reasons.append(f"Williams %R {williams_r} overbought")

    # Bollinger position
    if bb_pct < 0.20:
        bull_points += 1
        reasons.append("Price near lower Bollinger band")
    elif bb_pct > 0.80:
        bear_points += 1
        reasons.append("Price near upper Bollinger band")

    # ADX filter — weak trend reduces confidence
    adx_note = ""
    if adx < 20:
        adx_note = f"ADX {adx} — ranging market, reduced confidence"
    elif adx > 25:
        reasons.append(f"ADX {adx} confirms trending conditions")

    total_bull = bull_points
    total_bear = bear_points
    net = total_bull - total_bear

    if net >= 2:
        direction = "BUY"
        count = total_bull
    elif net <= -2:
        direction = "SELL"
        count = total_bear
    else:
        direction = "NEUTRAL"
        count = max(total_bull, total_bear)

    if adx_note:
        reasons.append(adx_note)

    return direction, count, reasons


def detect_candle_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return "insufficient data"
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body        = abs(last["close"] - last["open"])
    upper_wick  = last["high"]  - max(last["close"], last["open"])
    lower_wick  = min(last["close"], last["open"]) - last["low"]
    total_range = last["high"]  - last["low"]
    if total_range == 0:
        return "doji"
    ratio = body / total_range
    if ratio < 0.1:
        return "doji (indecision — watch for breakout)"
    if lower_wick > 2 * body and upper_wick < body * 0.5:
        return "hammer / pin bar (bullish reversal)"
    if upper_wick > 2 * body and lower_wick < body * 0.5:
        return "shooting star (bearish reversal)"
    if (last["close"] > last["open"] and prev["close"] < prev["open"]
            and last["close"] > prev["open"] and last["open"] < prev["close"]):
        return "bullish engulfing"
    if (last["close"] < last["open"] and prev["close"] > prev["open"]
            and last["close"] < prev["open"] and last["open"] > prev["close"]):
        return "bearish engulfing"
    if last["close"] > last["open"]:
        return "bullish candle"
    return "bearish candle"


# ─────────────────────────────────────────────────────────────────
#  AI-only fallback analysis
#  Used when Alpha Vantage is unavailable. ARIA receives estimated
#  price levels so she can still produce realistic SL/TP numbers.
# ─────────────────────────────────────────────────────────────────
def _build_fallback_analysis(symbol: str) -> dict:
    price = FALLBACK_PRICES.get(symbol, 1.0)
    atr_pct = FALLBACK_ATR_PCT.get(symbol, 0.010)
    atr = round(price * atr_pct, 8)

    sl_buy   = round(price - atr * 1.5, 6)
    sl_sell  = round(price + atr * 1.5, 6)
    tp1_buy  = round(price + atr * 1.5, 6)
    tp2_buy  = round(price + atr * 3.0, 6)
    tp3_buy  = round(price + atr * 4.5, 6)
    tp1_sell = round(price - atr * 1.5, 6)
    tp2_sell = round(price - atr * 3.0, 6)
    tp3_sell = round(price - atr * 4.5, 6)

    return {
        "symbol":            symbol,
        "data_source":       "AI reasoning (live feed paused — rate limit)",
        "live_price_source": "estimated reference price",
        "live_price":        price,
        "candles_used":      0,
        "last_candle_time":  "N/A",
        "ai_only_mode":      True,
        "indicators": {
            "rsi_14":          "estimated ~50 — neutral (AI will assess from market context)",
            "macd_12_26_9":    "AI pattern reasoning (no candle data)",
            "ema_stack":       "AI pattern reasoning (no candle data)",
            "bollinger_bands": "AI pattern reasoning (no candle data)",
            "volume":          "no volume data",
            "atr_14":          atr,
            "atr_pct":         f"{atr_pct*100:.2f}% of price",
            "candle_pattern":  "AI pattern reasoning",
        },
        "key_levels": {
            "support":          sl_buy,
            "resistance":       sl_sell,
            "atr_sl_for_buy":   sl_buy,
            "atr_sl_for_sell":  sl_sell,
            "atr_tp1_buy":      tp1_buy,
            "atr_tp2_buy":      tp2_buy,
            "atr_tp3_buy":      tp3_buy,
            "atr_tp1_sell":     tp1_sell,
            "atr_tp2_sell":     tp2_sell,
            "atr_tp3_sell":     tp3_sell,
        },
        "trend_bias": "AI pattern reasoning — see signal for direction",
    }


# ─────────────────────────────────────────────────────────────────
#  Main analysis entry point
# ─────────────────────────────────────────────────────────────────
def get_symbol_analysis(symbol: str) -> dict:
    """
    Returns full technical analysis.
    Mode 1 (preferred): Real Alpha Vantage OHLCV + computed indicators
    Mode 2 (fallback):  Estimated prices + AI pattern reasoning
    Never returns an empty or broken response — always gives ARIA
    enough context to produce a useful signal.
    """
    symbol = normalize_symbol(symbol)
    cache_key = f"analysis_{symbol}"

    # Serve from cache if fresh
    if cache_key in _DATA_CACHE:
        cached = _DATA_CACHE[cache_key]
        ttl = FALLBACK_TTL if cached["data"].get("ai_only_mode") else CACHE_TTL
        if time.time() - cached["ts"] < ttl:
            return cached["data"]

    crypto_base = extract_crypto_base(symbol)
    df: Optional[pd.DataFrame] = None
    live_price = None
    live_price_source = ""
    data_interval = ""

    if crypto_base:
        df = fetch_crypto_daily(crypto_base, market="USD")
        quote = fetch_realtime_quote(crypto_base, "USD")
        live_price = quote.get("price") or (float(df["close"].iloc[-1]) if df is not None else None)
        live_price_source = "CURRENCY_EXCHANGE_RATE" if quote.get("price") else "DIGITAL_CURRENCY_DAILY"
        data_interval = "daily candles (DIGITAL_CURRENCY_DAILY)"
    else:
        from_sym = symbol[:3]
        to_sym   = symbol[3:]
        df = fetch_forex_daily(from_sym, to_sym)
        quote = fetch_realtime_quote(from_sym, to_sym)
        live_price = quote.get("price") or (float(df["close"].iloc[-1]) if df is not None else None)
        live_price_source = "CURRENCY_EXCHANGE_RATE" if quote.get("price") else "FX_DAILY"
        data_interval = "daily candles (FX_DAILY)"

    # ── FALLBACK MODE: no usable candle data ─────────────────────
    if df is None or len(df) < 30:
        result = _build_fallback_analysis(symbol)
        # If we at least got a live quote, use that real price
        if live_price and live_price > 0:
            result["live_price"] = live_price
            result["live_price_source"] = live_price_source
            result["data_source"] = "Live price only (candle data paused — rate limit)"
            result["ai_only_mode"] = False  # we have a real price, just no candles
        _DATA_CACHE[cache_key] = {"data": result, "ts": time.time()}
        _save_disk_cache()
        return result

    # ── FULL MODE: real OHLCV + indicators ───────────────────────
    # Inject real-time price as last close for indicator accuracy
    if live_price and live_price > 0 and live_price != float(df["close"].iloc[-1]):
        last_row = df.iloc[-1].copy()
        last_row["close"] = live_price
        df = pd.concat([df.iloc[:-1], pd.DataFrame([last_row])], ignore_index=True)

    close = df["close"]
    price = round(float(close.iloc[-1]), 6)

    # ── Core indicators ──────────────────────────────────────────
    rsi                        = calc_rsi(close)
    macd_line, sig_line, hist  = calc_macd(close)
    ema20                      = calc_ema(close, 20)
    ema50                      = calc_ema(close, 50)
    ema200                     = calc_ema(close, 200)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close)
    atr                        = calc_atr(df)
    volume_sig                 = calc_volume_signal(df["volume"])
    support, resistance        = calc_support_resistance(df)
    trend                      = calc_trend(close)
    pattern                    = detect_candle_pattern(df)

    # ── Extra indicators ─────────────────────────────────────────
    stoch_k, stoch_d = calc_stochastic(df)
    adx, plus_di, minus_di = calc_adx(df)
    williams_r = calc_williams_r(df)
    pivots     = calc_pivot_points(df)

    # ── Confluence score ─────────────────────────────────────────
    bb_width = bb_upper - bb_lower
    bb_pct   = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

    conf_direction, conf_count, conf_reasons = calc_confluence_score(
        rsi=rsi, macd_hist=hist, ema20=ema20, ema50=ema50,
        price=price, stoch_k=stoch_k, stoch_d=stoch_d,
        adx=adx, williams_r=williams_r, bb_pct=bb_pct,
    )

    # ── Human-readable readings ──────────────────────────────────
    if rsi < 30:    rsi_reading = f"{rsi} — OVERSOLD (strong buy bias)"
    elif rsi < 40:  rsi_reading = f"{rsi} — approaching oversold (mild buy bias)"
    elif rsi > 70:  rsi_reading = f"{rsi} — OVERBOUGHT (strong sell bias)"
    elif rsi > 60:  rsi_reading = f"{rsi} — approaching overbought (mild sell bias)"
    else:           rsi_reading = f"{rsi} — neutral"

    if hist > 0:    macd_reading = f"BULLISH (line={macd_line:+.6f}, histogram={hist:+.6f})"
    else:           macd_reading = f"BEARISH (line={macd_line:+.6f}, histogram={hist:+.6f})"

    if bb_pct > 0.85:   bb_reading = f"Near UPPER band ({price} ≈ {bb_upper}) — overbought zone"
    elif bb_pct < 0.15: bb_reading = f"Near LOWER band ({price} ≈ {bb_lower}) — oversold zone"
    elif bb_width < close.std() * 0.5:
                        bb_reading = f"SQUEEZE (tight bands, mid={bb_mid}) — breakout imminent"
    else:               bb_reading = f"{bb_pct:.0%} of band (lower={bb_lower}, mid={bb_mid}, upper={bb_upper})"

    if stoch_k > 80:    stoch_reading = f"K={stoch_k} D={stoch_d} — OVERBOUGHT zone"
    elif stoch_k < 20:  stoch_reading = f"K={stoch_k} D={stoch_d} — OVERSOLD zone"
    elif stoch_k > stoch_d:
                        stoch_reading = f"K={stoch_k} D={stoch_d} — bullish crossover"
    else:               stoch_reading = f"K={stoch_k} D={stoch_d} — bearish crossover"

    if adx > 40:        adx_reading = f"ADX={adx} (STRONG trend) | +DI={plus_di} -DI={minus_di}"
    elif adx > 25:      adx_reading = f"ADX={adx} (trending) | +DI={plus_di} -DI={minus_di}"
    elif adx > 20:      adx_reading = f"ADX={adx} (weakening trend) | +DI={plus_di} -DI={minus_di}"
    else:               adx_reading = f"ADX={adx} (RANGING — low directional strength) | +DI={plus_di} -DI={minus_di}"

    if williams_r > -20:   wr_reading = f"{williams_r} — OVERBOUGHT"
    elif williams_r < -80: wr_reading = f"{williams_r} — OVERSOLD"
    else:                  wr_reading = f"{williams_r} — neutral"

    # ── ATR-based SL/TP (1.5R / 3R / 4.5R) ──────────────────────
    sl_buy   = round(price - atr * 1.5, 6)
    sl_sell  = round(price + atr * 1.5, 6)
    tp1_buy  = round(price + atr * 1.5, 6)
    tp2_buy  = round(price + atr * 3.0, 6)
    tp3_buy  = round(price + atr * 4.5, 6)
    tp1_sell = round(price - atr * 1.5, 6)
    tp2_sell = round(price - atr * 3.0, 6)
    tp3_sell = round(price - atr * 4.5, 6)

    result = {
        "symbol":             symbol,
        "data_source":        f"Alpha Vantage ({data_interval})",
        "live_price_source":  live_price_source,
        "live_price":         price,
        "candles_used":       len(df),
        "last_candle_time":   str(df["time"].iloc[-1].date()),
        "ai_only_mode":       False,
        "confluence": {
            "direction":    conf_direction,
            "score":        conf_count,
            "out_of":       6,
            "strength":     "STRONG" if conf_count >= 5 else ("MODERATE" if conf_count >= 3 else "WEAK"),
            "reasons":      conf_reasons,
            "tradeable":    adx >= 20,
            "adx_note":     adx_reading,
        },
        "indicators": {
            "rsi_14":          rsi_reading,
            "macd_12_26_9":    macd_reading,
            "stochastic_14":   stoch_reading,
            "adx_14":          adx_reading,
            "williams_r_14":   wr_reading,
            "ema_stack":       trend,
            "bollinger_bands": bb_reading,
            "volume":          volume_sig,
            "atr_14":          atr,
            "candle_pattern":  pattern,
        },
        "key_levels": {
            "support":          support,
            "resistance":       resistance,
            "pivot_points":     pivots,
            "atr_sl_for_buy":   sl_buy,
            "atr_sl_for_sell":  sl_sell,
            "atr_tp1_buy":      tp1_buy,
            "atr_tp2_buy":      tp2_buy,
            "atr_tp3_buy":      tp3_buy,
            "atr_tp1_sell":     tp1_sell,
            "atr_tp2_sell":     tp2_sell,
            "atr_tp3_sell":     tp3_sell,
        },
        "trend_bias": trend,
    }

    _DATA_CACHE[cache_key] = {"data": result, "ts": time.time()}
    _save_disk_cache()
    return result


# ─────────────────────────────────────────────────────────────────
#  AI prompt formatter
# ─────────────────────────────────────────────────────────────────
def format_for_ai_prompt(symbol_or_analysis, analysis: dict = None) -> str:
    """
    Formats market data into a compact string for ARIA's prompt.
    Accepts either:
      format_for_ai_prompt(symbol, analysis)   — two args
      format_for_ai_prompt(analysis)            — one arg (legacy)
    """
    if analysis is None:
        # Called as format_for_ai_prompt(analysis_dict)
        analysis = symbol_or_analysis

    sym = analysis.get("symbol", "UNKNOWN")
    ai_only = analysis.get("ai_only_mode", False)
    price = analysis.get("live_price", "unknown")
    source = analysis.get("live_price_source", "")

    if ai_only:
        ind = analysis.get("indicators", {})
        lv  = analysis.get("key_levels", {})
        lines = [
            f"=== {sym} | Reference Price: {price} ({source}) ===",
            f"Data: AI reasoning mode — live feed temporarily paused (rate limit)",
            f"ATR(estimated): {ind.get('atr_14')} ({ind.get('atr_pct', '')})",
            f"Reference support: {lv.get('support')} | resistance: {lv.get('resistance')}",
            f"BUY  → SL: {lv.get('atr_sl_for_buy')} | TP1: {lv.get('atr_tp1_buy')} | TP2: {lv.get('atr_tp2_buy')} | TP3: {lv.get('atr_tp3_buy')}",
            f"SELL → SL: {lv.get('atr_sl_for_sell')} | TP1: {lv.get('atr_tp1_sell')} | TP2: {lv.get('atr_tp2_sell')} | TP3: {lv.get('atr_tp3_sell')}",
            f"NOTE: Use your institutional knowledge of {sym} market structure, recent macro events,",
            f"      session behaviour, and technical patterns to assess direction.",
        ]
        return "\n".join(lines)

    ind  = analysis.get("indicators", {})
    lv   = analysis.get("key_levels", {})
    conf = analysis.get("confluence", {})
    piv  = lv.get("pivot_points", {})

    lines = [
        f"=== {sym} | Live Price: {price} ({source}) ===",
        f"Data: {analysis.get('data_source')} | Candles: {analysis.get('candles_used')} | Last: {analysis.get('last_candle_time','')}",
        f"",
        f"── CONFLUENCE: {conf.get('direction','?')} | Score {conf.get('score',0)}/{conf.get('out_of',6)} ({conf.get('strength','?')}) | Tradeable: {conf.get('tradeable','?')}",
        f"   Reasons: {' | '.join(conf.get('reasons', ['insufficient data']))}",
        f"",
        f"── INDICATORS",
        f"RSI(14):        {ind.get('rsi_14')}",
        f"MACD(12,26,9):  {ind.get('macd_12_26_9')}",
        f"Stochastic(14): {ind.get('stochastic_14')}",
        f"Williams %R:    {ind.get('williams_r_14')}",
        f"ADX(14):        {ind.get('adx_14')}",
        f"EMA Stack:      {ind.get('ema_stack')}",
        f"Bollinger(20):  {ind.get('bollinger_bands')}",
        f"ATR(14):        {ind.get('atr_14')} | Pattern: {ind.get('candle_pattern')}",
        f"Volume:         {ind.get('volume')}",
        f"",
        f"── KEY LEVELS",
        f"Support: {lv.get('support')} | Resistance: {lv.get('resistance')}",
        f"Pivots: P={piv.get('pivot')} R1={piv.get('R1')} R2={piv.get('R2')} S1={piv.get('S1')} S2={piv.get('S2')}",
        f"BUY  → SL: {lv.get('atr_sl_for_buy')} | TP1: {lv.get('atr_tp1_buy')} | TP2: {lv.get('atr_tp2_buy')} | TP3: {lv.get('atr_tp3_buy')}",
        f"SELL → SL: {lv.get('atr_sl_for_sell')} | TP1: {lv.get('atr_tp1_sell')} | TP2: {lv.get('atr_tp2_sell')} | TP3: {lv.get('atr_tp3_sell')}",
    ]
    return "\n".join(lines)
