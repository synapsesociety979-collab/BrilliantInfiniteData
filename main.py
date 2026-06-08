# main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, time, requests, random, math, threading, base64
from datetime import datetime, timedelta
from ai_provider import (
    get_ai_response,
    get_ai_response_creative,
    get_ai_response_chat,
    transcribe_audio,
    text_to_speech,
    TTS_VOICES,
)
from sqlalchemy.orm import Session

from fastapi.middleware.cors import CORSMiddleware
from backtest_api import (
    load_history_df,
    run_backtest_from_signals,
    signals_from_sma,
    router as backtest_router,
)
from models import (
    init_db,
    get_db,
    User,
    DemoAccount,
    DemoTrade,
    ChatMessage as DBChatMessage,
    TradeJournalEntry as DBJournalEntry,
    WatchlistItem,
    UserActivity,
    Conversation,
    UserMemory,
    BotConfig,
    BotOrder,
)
from trading_engine import RiskEngine, MarketFilter, TradeManager
from market_data import (
    get_symbol_analysis, format_for_ai_prompt, format_for_ai_prompt_compact, fetch_realtime_quote,
    FALLBACK_PRICES, FALLBACK_ATR_PCT,
)

# ----------------------------
# DB init on startup
# ----------------------------
init_db()

# ----------------------------
# Alpha Vantage Integration
# ----------------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_alpha_vantage_forex(from_symbol: str, to_symbol: str = "USD") -> Dict:
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_symbol}&to_currency={to_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get("Realtime Currency Exchange Rate", {})
        if data:
            return {
                "rate": float(data.get("5. Exchange Rate") or 0),
                "bid": float(data.get("8. Bid Price") or 0),
                "ask": float(data.get("9. Ask Price") or 0),
                "last_refreshed": data.get("6. Last Refreshed"),
            }
    except Exception as e:
        print(f"AV Forex Error: {e}")
    return {}


def fetch_alpha_vantage_crypto(symbol: str) -> Dict:
    base = symbol.replace("USDT", "").replace("BUSD", "")
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base}&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get("Realtime Currency Exchange Rate", {})
        if data:
            return {
                "price": float(data.get("5. Exchange Rate") or 0),
                "last_refreshed": data.get("6. Last Refreshed"),
            }
    except Exception as e:
        print(f"AV Crypto Error: {e}")
    return {}


def get_live_price(symbol: str) -> float:
    # First try to get price from full candle analysis (already cached)
    analysis = get_symbol_analysis(symbol)
    if analysis.get("live_price"):
        return float(analysis["live_price"])
    # Fallback to real-time quote
    if "USDT" in symbol or "BUSD" in symbol:
        base = symbol.replace("USDT", "").replace("BUSD", "")
        d = fetch_realtime_quote(base, "USD")
    else:
        d = fetch_realtime_quote(symbol[:3], symbol[3:] or "USD")
    return float(d.get("price") or 0)


_RATES_CACHE: Dict[str, Any] = {}
_RATES_CACHE_TIMES: Dict[str, float] = {}
_RATES_CACHE_TTL: int = 900  # refresh every 15 min for live rates

EXCHANGE_RATE_API_KEY = os.environ.get("EXCHANGE_RATE_API_KEY", "958b36cefc4f5be763d8a458")

def get_all_exchange_rates(base: str = "USD") -> Dict[str, float]:
    """
    Fetch live exchange rates from ExchangeRate-API v6 (using user's API key).
    Returns a dict like {"NGN": 1610.5, "EUR": 0.918, "GBP": 0.785, ...}
    Cached for 15 minutes for near-real-time accuracy.
    Falls back to free tier if key fails.
    """
    global _RATES_CACHE, _RATES_CACHE_TIMES
    now = time.time()
    cache_key = f"rates_{base.upper()}"

    if _RATES_CACHE.get(cache_key) and (now - _RATES_CACHE_TIMES.get(cache_key, 0)) < _RATES_CACHE_TTL:
        return _RATES_CACHE[cache_key]

    # Primary: ExchangeRate-API v6 with user's API key (1,500 req/month free, updates daily)
    try:
        r = requests.get(
            f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/{base.upper()}",
            timeout=8
        )
        data = r.json()
        if data.get("result") == "success":
            rates = data.get("conversion_rates", {})
            _RATES_CACHE[cache_key] = rates
            _RATES_CACHE_TIMES[cache_key] = now
            print(f"[RATES] ✅ Updated from ExchangeRate-API v6 ({len(rates)} currencies)")
            return rates
    except Exception as e:
        print(f"[RATES] ExchangeRate-API v6 failed: {e}")

    # Fallback: free open.er-api.com (no key, 1500 req/month)
    try:
        r = requests.get(
            f"https://open.er-api.com/v6/latest/{base.upper()}",
            timeout=8
        )
        data = r.json()
        if data.get("result") == "success":
            rates = data.get("rates", {})
            _RATES_CACHE[cache_key] = rates
            _RATES_CACHE_TIMES[cache_key] = now
            return rates
    except Exception:
        pass

    # Last resort: return cached rates even if stale
    if _RATES_CACHE.get(cache_key):
        return _RATES_CACHE[cache_key]

    # Absolute fallback
    return {"NGN": 1610.0, "EUR": 0.87, "GBP": 0.76, "JPY": 150.0,
            "CAD": 1.36, "AUD": 1.52, "CHF": 0.88, "ZAR": 18.5}


def get_ngn_rate() -> float:
    rates = get_all_exchange_rates("USD")
    return float(rates.get("NGN", 1610.0))


def convert_currency(amount: float, from_cur: str, to_cur: str) -> dict:
    """Convert any amount between any two currencies using live rates."""
    from_cur = from_cur.upper()
    to_cur   = to_cur.upper()

    # Get rates based on USD as base
    usd_rates = get_all_exchange_rates("USD")

    if not usd_rates:
        return {"error": "Rate data unavailable"}

    # Convert: amount in from_cur → USD → to_cur
    if from_cur == "USD":
        usd_amount = amount
    elif from_cur in usd_rates:
        usd_amount = amount / usd_rates[from_cur]
    else:
        return {"error": f"Currency {from_cur} not supported"}

    if to_cur == "USD":
        result = usd_amount
    elif to_cur in usd_rates:
        result = usd_amount * usd_rates[to_cur]
    else:
        return {"error": f"Currency {to_cur} not supported"}

    rate = usd_rates.get(to_cur, 1) / usd_rates.get(from_cur, 1) if from_cur != "USD" else usd_rates.get(to_cur, 1)

    return {
        "from": from_cur,
        "to": to_cur,
        "amount": amount,
        "result": round(result, 6),
        "rate": round(rate, 6),
        "display": f"{amount:,.2f} {from_cur} = {result:,.2f} {to_cur}",
    }


# ─── Economic Calendar (ForexFactory free public feed) ───────────────────────
_CALENDAR_CACHE: list = []
_CALENDAR_CACHE_TIME: float = 0
_CALENDAR_CACHE_TTL: int = 1800  # 30-min cache

# Title keywords → currency inference (for when ForexFactory omits the currency field)
_NEWS_TITLE_CURRENCY_MAP = {
    # USD events
    "nfp": "USD", "non-farm": "USD", "non farm": "USD", "fomc": "USD",
    "federal reserve": "USD", "fed ": "USD", "cpi": "USD", "pce": "USD",
    "gdp": "USD", "ism": "USD", "unemployment": "USD", "payroll": "USD",
    "core pce": "USD", "retail sales": "USD", "ppi": "USD",
    "consumer price": "USD", "core cpi": "USD", "employment change": "USD",
    # GBP events
    "boe": "GBP", "bank of england": "GBP", "mpc": "GBP",
    # EUR events
    "ecb": "EUR", "european central": "EUR",
    # JPY events
    "boj": "JPY", "bank of japan": "JPY",
    # CAD events
    "boc": "CAD", "bank of canada": "CAD",
    # AUD events
    "rba": "AUD", "reserve bank of australia": "AUD",
    # NZD events
    "rbnz": "NZD", "reserve bank of new zealand": "NZD",
    "official cash rate": "NZD",
    # CHF events
    "snb": "CHF", "swiss national": "CHF",
    # Oil/global
    "opec": "USD",
}


def _infer_currency(title: str, raw_currency: str) -> str:
    """Return the currency for a news event, inferring from title if raw_currency is empty."""
    if raw_currency and raw_currency.upper() not in ("NONE", ""):
        return raw_currency.upper()
    title_lower = title.lower()
    for keyword, cur in _NEWS_TITLE_CURRENCY_MAP.items():
        if keyword in title_lower:
            return cur
    return ""


SYMBOL_CURRENCIES = {
    "EURUSD": ["EUR","USD"], "GBPUSD": ["GBP","USD"], "USDJPY": ["USD","JPY"],
    "AUDUSD": ["AUD","USD"], "USDCAD": ["USD","CAD"], "NZDUSD": ["NZD","USD"],
    "USDCHF": ["USD","CHF"], "EURGBP": ["EUR","GBP"], "EURJPY": ["EUR","JPY"],
    "GBPJPY": ["GBP","JPY"], "AUDJPY": ["AUD","JPY"], "EURCAD": ["EUR","CAD"],
    "AUDCAD": ["AUD","CAD"], "EURAUD": ["EUR","AUD"], "GBPAUD": ["GBP","AUD"],
    # Crypto: USD-denominated, block on high-impact USD events
    "BTCUSDT": ["USD"], "ETHUSDT": ["USD"], "BNBUSDT": ["USD"],
    "XRPUSDT": ["USD"], "SOLUSDT": ["USD"], "ADAUSDT": ["USD"],
    "DOGEUSDT": ["USD"], "DOTUSDT": ["USD"], "MATICUSDT": ["USD"],
    "LTCUSDT": ["USD"], "SHIBUSDT": ["USD"], "TRXUSDT": ["USD"],
    "AVAXUSDT": ["USD"], "LINKUSDT": ["USD"], "UNIUSDT": ["USD"],
}


def _build_static_calendar() -> list:
    """
    Fallback: a rolling set of known recurring high-impact events for the current week.
    Used when the live feed is rate-limited or unavailable.
    """
    from datetime import timezone as _tz
    now = datetime.utcnow()
    # Use this week's Mon 00:00 UTC as the anchor
    monday = now - timedelta(days=now.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    # Known weekly schedule (day_offset from Monday, hour UTC, title, currency, impact)
    SCHEDULE = [
        # Monday
        (0, 14, 0,  "ISM Manufacturing PMI",      "USD", "High"),
        # Tuesday
        (1,  9, 0,  "RBA Interest Rate Decision",  "AUD", "High"),
        # Wednesday
        (2, 14, 0,  "ISM Services PMI",             "USD", "High"),
        (2, 18, 0,  "FOMC Meeting Minutes",          "USD", "High"),
        (2, 14, 30, "ADP Non-Farm Employment",       "USD", "High"),
        # Thursday
        (3, 12, 0,  "ECB Monetary Policy Decision", "EUR", "High"),
        (3, 12, 30, "Initial Jobless Claims",        "USD", "Medium"),
        (3,  7, 0,  "BOE Interest Rate Decision",   "GBP", "High"),
        # Friday
        (4, 12, 30, "Non-Farm Payrolls (NFP)",       "USD", "High"),
        (4, 12, 30, "Unemployment Rate",              "USD", "High"),
        (4, 12, 30, "Average Hourly Earnings",        "USD", "Medium"),
        (4, 14, 0,  "Consumer Sentiment",             "USD", "Medium"),
        # These happen monthly but we show every week as placeholders
        (2, 12, 30, "CPI m/m",                       "USD", "High"),
        (3, 12, 30, "Core PPI m/m",                  "USD", "Medium"),
    ]

    events = []
    for day_off, hour, minute, title, cur, impact in SCHEDULE:
        ev_time = monday + timedelta(days=day_off, hours=hour, minutes=minute)
        events.append({
            "title":    title,
            "currency": cur,
            "impact":   impact,
            "date":     ev_time.isoformat(),
            "forecast": "",
            "previous": "",
            "actual":   "",
            "_source":  "static_fallback",
        })
    return events


def get_economic_calendar() -> list:
    """
    Fetch this week's economic events from ForexFactory free feed.
    30-min cache prevents rate limiting. Falls back to static schedule if unavailable.
    """
    global _CALENDAR_CACHE, _CALENDAR_CACHE_TIME
    now = time.time()

    # Serve from cache if fresh enough
    if _CALENDAR_CACHE and (now - _CALENDAR_CACHE_TIME) < _CALENDAR_CACHE_TTL:
        return _CALENDAR_CACHE

    try:
        r = requests.get(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            timeout=8,
            headers={"User-Agent": "CLEO-TradingBot/4.0"},
        )
        if r.status_code == 200:
            text = r.text.strip()
            if text and text.startswith("["):
                data = r.json()
                if data:
                    _CALENDAR_CACHE = data
                    _CALENDAR_CACHE_TIME = now
                    print(f"[CALENDAR] ✅ Live feed: {len(data)} events this week")
                    return data
        elif r.status_code == 429:
            print("[CALENDAR] ⚠️ Rate-limited by ForexFactory — using cache/fallback")
        else:
            print(f"[CALENDAR] HTTP {r.status_code} from ForexFactory")
    except Exception as e:
        print(f"[CALENDAR] Fetch failed: {e}")

    # Return stale cache if available
    if _CALENDAR_CACHE:
        print("[CALENDAR] Serving stale cache")
        return _CALENDAR_CACHE

    # Last resort: static fallback
    print("[CALENDAR] Using static fallback schedule")
    fallback = _build_static_calendar()
    _CALENDAR_CACHE = fallback
    _CALENDAR_CACHE_TIME = now - (_CALENDAR_CACHE_TTL - 120)  # re-try in 2 min
    return fallback


def check_news_blackout(symbol: str, hours_before: float = 2.0, hours_after: float = 1.0) -> dict:
    """
    Check if a high-impact news event is within hours_before/after for the symbol's currencies.
    Returns: {blocked, warning, events, message}
    """
    from datetime import timezone as _tz
    symbol = symbol.upper()
    currencies = SYMBOL_CURRENCIES.get(symbol, [])
    if not currencies and "USD" in symbol:
        currencies = ["USD"]

    events = get_economic_calendar()
    now_utc = datetime.utcnow()
    relevant_high = []
    relevant_medium = []

    for event in events:
        raw_cur  = (event.get("currency") or "")
        impact   = (event.get("impact")   or "").lower()
        date_str = (event.get("date")     or "")
        title    = (event.get("title")    or "")
        currency = _infer_currency(title, raw_cur)

        if currency not in [c.upper() for c in currencies]:
            continue
        if impact not in ("high", "medium"):
            continue

        try:
            event_dt = datetime.fromisoformat(date_str)
            if event_dt.tzinfo is not None:
                event_utc = event_dt.astimezone(_tz.utc).replace(tzinfo=None)
            else:
                event_utc = event_dt

            mins_until = (event_utc - now_utc).total_seconds() / 60

            if impact == "high" and (-hours_after * 60) <= mins_until <= (hours_before * 60):
                relevant_high.append({
                    "title": title, "currency": currency, "impact": "High",
                    "time_utc": event_utc.strftime("%Y-%m-%d %H:%M UTC"),
                    "mins_until": round(mins_until),
                    "status": "upcoming" if mins_until > 0 else "just released"
                })
            elif impact == "medium" and 0 < mins_until <= 60:
                relevant_medium.append({
                    "title": title, "currency": currency, "impact": "Medium",
                    "time_utc": event_utc.strftime("%Y-%m-%d %H:%M UTC"),
                    "mins_until": round(mins_until),
                    "status": "upcoming"
                })
        except Exception:
            continue

    if relevant_high:
        parts = []
        for e in relevant_high:
            if e["mins_until"] > 0:
                parts.append(f"{e['title']} ({e['currency']}) in {e['mins_until']} mins")
            else:
                parts.append(f"{e['title']} ({e['currency']}) released {abs(e['mins_until'])} mins ago")
        return {
            "blocked": True, "warning": False, "events": relevant_high,
            "message": f"🚫 News blackout: {'; '.join(parts)} — holding off to protect capital"
        }

    if relevant_medium:
        parts = [f"{e['title']} ({e['currency']}) in {e['mins_until']}m" for e in relevant_medium[:2]]
        return {
            "blocked": False, "warning": True, "events": relevant_medium,
            "message": f"⚠️ Medium-impact news soon: {'; '.join(parts)} — trade with caution"
        }

    return {"blocked": False, "warning": False, "events": [], "message": ""}


def get_upcoming_events_text(currencies: list = None, hours_ahead: float = 6) -> str:
    """Return a readable summary of upcoming high/medium events for CLEO's chat context."""
    from datetime import timezone as _tz
    events = get_economic_calendar()
    now_utc = datetime.utcnow()
    upcoming = []
    for event in events:
        raw_cur  = (event.get("currency") or "")
        impact   = (event.get("impact")   or "").lower()
        date_str = (event.get("date")     or "")
        title    = (event.get("title")    or "")
        currency = _infer_currency(title, raw_cur)
        if impact not in ("high", "medium"):
            continue
        if currencies and currency not in [c.upper() for c in currencies]:
            continue
        try:
            event_dt = datetime.fromisoformat(date_str)
            if event_dt.tzinfo is not None:
                event_utc = event_dt.astimezone(_tz.utc).replace(tzinfo=None)
            else:
                event_utc = event_dt
            mins_until = (event_utc - now_utc).total_seconds() / 60
            if 0 < mins_until <= hours_ahead * 60:
                upcoming.append((mins_until, currency, impact, title, event_utc))
        except Exception:
            continue

    if not upcoming:
        return ""
    upcoming.sort(key=lambda x: x[0])
    lines = []
    for mins, cur, imp, title, dt in upcoming[:8]:
        h = int(mins // 60)
        m = int(mins % 60)
        time_str = f"{h}h {m}m" if h else f"{m}m"
        imp_icon = "🔴" if imp == "high" else "🟡"
        lines.append(f"  {imp_icon} {title} ({cur}) — in {time_str}")
    return "━━━ UPCOMING ECONOMIC EVENTS (next 6 hours) ━━━\n" + "\n".join(lines)


def calc_ngn_trade_details(
    symbol: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    balance_ngn: float,
    ngn_rate: float,
) -> dict:
    """
    Given a signal's price levels and the user's NGN balance, calculate:
    - Recommended lot size (forex) or units (crypto) for exactly 2% risk
    - SL and TP values expressed in Naira (NGN)
    - NGN per pip (forex) so the user knows the exact cost of each pip move
    - Viability check: flags if the account is too small to trade this market
    Returns an empty dict if inputs are invalid.
    """
    try:
        if not all([entry > 0, stop_loss > 0, balance_ngn > 0, ngn_rate > 0]):
            return {}
        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            return {}

        balance_usd = balance_ngn / ngn_rate
        risk_usd    = balance_usd * 0.02          # 2% of account
        risk_ngn    = balance_ngn * 0.02

        is_crypto = "USDT" in symbol.upper()

        if is_crypto:
            # ── Crypto: position in coin units ────────────────────────────
            # Minimum viable: risk_usd must be at least $1 (most exchanges min ~$1-5 order)
            MIN_CRYPTO_RISK_USD = 1.0
            viable = risk_usd >= MIN_CRYPTO_RISK_USD

            units = risk_usd / sl_distance if sl_distance > 0 else 0
            position_value_usd = units * entry

            def _crypto_pnl(tp):
                return units * abs(tp - entry) if tp and tp > 0 else 0

            tp1_usd = _crypto_pnl(tp1)
            tp2_usd = _crypto_pnl(tp2)
            tp3_usd = _crypto_pnl(tp3)

            # Round units sensibly depending on price
            if entry >= 1000:
                units_rounded = round(units, 6)
            elif entry >= 1:
                units_rounded = round(units, 4)
            else:
                units_rounded = round(units, 2)

            min_balance_for_crypto_ngn = (MIN_CRYPTO_RISK_USD / 0.02) * ngn_rate

            result = {
                "type":                "crypto",
                "viable":              viable,
                "account_balance_ngn": f"₦{balance_ngn:,.0f}",
                "risk_per_trade_ngn":  f"₦{risk_ngn:,.0f}",
                "risk_per_trade_usd":  f"${risk_usd:.4f}",
                "recommended_units":   units_rounded,
                "unit_label":          f"{units_rounded} units of {symbol.replace('USDT','')}",
                "position_value_ngn":  f"₦{position_value_usd * ngn_rate:,.0f}",
                "position_value_usd":  f"${position_value_usd:,.4f}",
                "sl_loss_ngn":         f"₦{risk_ngn:,.0f}",
                "sl_loss_usd":         f"${risk_usd:.4f}",
                "tp1_profit_ngn":      f"₦{tp1_usd * ngn_rate:,.0f}" if tp1_usd else "N/A",
                "tp2_profit_ngn":      f"₦{tp2_usd * ngn_rate:,.0f}" if tp2_usd else "N/A",
                "tp3_profit_ngn":      f"₦{tp3_usd * ngn_rate:,.0f}" if tp3_usd else "N/A",
                "tp1_profit_usd":      f"${tp1_usd:.4f}" if tp1_usd else "N/A",
                "tp2_profit_usd":      f"${tp2_usd:.4f}" if tp2_usd else "N/A",
                "tp3_profit_usd":      f"${tp3_usd:.4f}" if tp3_usd else "N/A",
            }
            if not viable:
                result["warning"] = (
                    f"Account too small for real crypto trading. "
                    f"Your 2% risk = ${risk_usd:.4f} which is below the typical $1 minimum order. "
                    f"Use the demo account to practice. "
                    f"Minimum recommended balance for real crypto: "
                    f"₦{min_balance_for_crypto_ngn:,.0f} (~${MIN_CRYPTO_RISK_USD / 0.02:.0f})"
                )
                result["recommendation"] = "demo_only"
            return result

        else:
            # ── Forex: position in standard lots ──────────────────────────
            # Pip size: JPY pairs use 0.01, all others 0.0001
            pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001

            # Pip value per standard lot (100,000 units) in USD
            # For pairs quoted vs USD (EUR/USD, GBP/USD, AUD/USD, NZD/USD): $10/pip exactly
            # For USD-base pairs (USD/JPY, USD/CAD, USD/CHF): ~$10 / entry
            # For cross pairs: approximate with $10
            usd_quoted_pairs = ("EURUSD","GBPUSD","AUDUSD","NZDUSD","XAUUSD","XAGUSD")
            usd_base_pairs   = ("USDJPY","USDCAD","USDCHF","USDSGD","USDHKD")

            sym_upper = symbol.upper()
            if sym_upper in usd_quoted_pairs or sym_upper.endswith("USD"):
                pip_value_per_lot = 10.0
            elif sym_upper in usd_base_pairs or sym_upper.startswith("USD"):
                pip_value_per_lot = round(pip_size * 100_000 / entry, 4)
            else:
                pip_value_per_lot = 10.0

            sl_pips       = sl_distance / pip_size
            ideal_lot     = risk_usd / (sl_pips * pip_value_per_lot)
            MIN_FOREX_LOT = 0.01   # smallest micro lot most brokers offer

            viable = ideal_lot >= MIN_FOREX_LOT

            # If not viable, show what 0.01 lots would ACTUALLY cost so user sees the danger
            lot_size       = round(ideal_lot, 2) if viable else MIN_FOREX_LOT
            actual_risk_usd = sl_pips * pip_value_per_lot * MIN_FOREX_LOT
            actual_risk_pct = (actual_risk_usd / balance_usd * 100) if balance_usd > 0 else 999

            # Minimum account balance to trade this pair safely at 2%
            min_balance_usd = actual_risk_usd / 0.02
            min_balance_ngn = min_balance_usd * ngn_rate

            # Lot size label
            if lot_size >= 1.0:
                lot_label = f"{lot_size} standard lot{'s' if lot_size != 1 else ''}"
            elif lot_size >= 0.1:
                lot_label = f"{lot_size} mini lot{'s' if lot_size != 0.1 else ''}"
            else:
                lot_label = f"{lot_size} micro lot{'s' if lot_size != 0.01 else ''}"

            pnl_per_pip = lot_size * pip_value_per_lot
            ngn_per_pip = pnl_per_pip * ngn_rate

            def _forex_pnl_usd(tp):
                return (abs(tp - entry) / pip_size) * pnl_per_pip if tp and tp > 0 else 0

            tp1_usd = _forex_pnl_usd(tp1)
            tp2_usd = _forex_pnl_usd(tp2)
            tp3_usd = _forex_pnl_usd(tp3)

            result = {
                "type":                 "forex",
                "viable":               viable,
                "account_balance_ngn":  f"₦{balance_ngn:,.0f}",
                "risk_per_trade_ngn":   f"₦{risk_ngn:,.0f}",
                "risk_per_trade_usd":   f"${risk_usd:.4f}",
                "ideal_lot_size":       round(ideal_lot, 4),
                "recommended_lot_size": lot_size,
                "lot_size_label":       lot_label,
                "sl_pips":              round(sl_pips, 1),
                "ngn_per_pip":          f"₦{ngn_per_pip:,.2f}",
                "usd_per_pip":          f"${pnl_per_pip:.2f}",
                "position_value_ngn":   f"₦{lot_size * 100_000 * entry * ngn_rate:,.0f}",
                "position_value_usd":   f"${lot_size * 100_000 * entry:,.2f}",
                "sl_loss_ngn":          f"₦{risk_ngn:,.0f}" if viable else f"₦{actual_risk_usd * ngn_rate:,.0f}",
                "sl_loss_usd":          f"${risk_usd:.4f}" if viable else f"${actual_risk_usd:.2f}",
                "tp1_profit_ngn":       f"₦{tp1_usd * ngn_rate:,.0f}" if tp1_usd else "N/A",
                "tp2_profit_ngn":       f"₦{tp2_usd * ngn_rate:,.0f}" if tp2_usd else "N/A",
                "tp3_profit_ngn":       f"₦{tp3_usd * ngn_rate:,.0f}" if tp3_usd else "N/A",
                "tp1_profit_usd":       f"${tp1_usd:.2f}" if tp1_usd else "N/A",
                "tp2_profit_usd":       f"${tp2_usd:.2f}" if tp2_usd else "N/A",
                "tp3_profit_usd":       f"${tp3_usd:.2f}" if tp3_usd else "N/A",
                "minimum_balance_for_forex_ngn": f"₦{min_balance_ngn:,.0f}",
                "minimum_balance_for_forex_usd": f"${min_balance_usd:.0f}",
            }

            if not viable:
                result["warning"] = (
                    f"⚠️ Account too small for live forex trading. "
                    f"Your 2% risk = ${risk_usd:.4f} but the smallest forex lot (0.01 micro) "
                    f"would risk ${actual_risk_usd:.2f} ({actual_risk_pct:.0f}% of your account) "
                    f"on this {sl_pips:.0f}-pip stop — that is {actual_risk_pct:.0f}x your intended risk. "
                    f"Minimum account needed for this trade at 2% risk: "
                    f"₦{min_balance_ngn:,.0f} (${min_balance_usd:.0f}). "
                    f"Use the demo account until you reach this balance."
                )
                result["actual_risk_if_forced_pct"] = round(actual_risk_pct, 1)
                result["recommendation"] = "demo_only"

            return result
    except Exception:
        return {}


def get_budget_trading_plan(balance_ngn: float, ngn_rate: float) -> str:
    """
    Given any NGN balance, return a realistic, honest trading plan.
    Uses accurate minimum account sizes based on real lot-size mathematics.

    REAL MINIMUMS (derived from lot size math, not guesses):
    - Forex 0.01 micro lot, EURUSD, 20-pip SL → $2 risk → need $100 account for 2%
    - Forex 0.01 micro lot, EURUSD, 10-pip SL → $1 risk → need $50 account for 2%
    - Crypto spot: $1 minimum order → need $50 account for 2%
    - Safe forex minimum: ~$100 (₦160,000 at ₦1,600/$)
    - Minimum crypto real money: ~$50 (₦80,000)
    """
    balance_usd  = balance_ngn / ngn_rate
    risk_ngn     = balance_ngn * 0.02
    risk_usd     = balance_usd * 0.02

    # ── Tier thresholds (USD) based on real lot-size math ──────────────────
    # $0-5      → demo only (2% = $0.10 — can't open any real position)
    # $5-50     → demo + crypto spot watch (2% = $0.10-1.00 — barely viable for crypto)
    # $50-100   → crypto real money only (2% = $1-2 — enough for crypto, NOT forex)
    # $100-300  → forex micro OK with TIGHT SL (2% = $2-6 — covers 0.01 lots barely)
    # $300-1000 → forex micro comfortable (2% = $6-20)
    # $1000+    → standard forex approach

    if balance_usd < 5:
        tier = "demo_only"
    elif balance_usd < 50:
        tier = "crypto_watch"
    elif balance_usd < 100:
        tier = "crypto_only"
    elif balance_usd < 300:
        tier = "forex_entry"
    elif balance_usd < 1000:
        tier = "forex_micro"
    else:
        tier = "forex_standard"

    # ── Calculate what 0.01 lots ACTUALLY risks on EURUSD with a 20-pip SL ──
    # $0.10/pip × 20 pips = $2.00 actual risk per trade at minimum lot
    actual_min_lot_risk_usd = 2.00   # 0.01 lots, 20-pip SL on EURUSD
    min_forex_balance_usd   = actual_min_lot_risk_usd / 0.02   # = $100
    min_forex_balance_ngn   = min_forex_balance_usd * ngn_rate
    min_crypto_balance_usd  = 1.00 / 0.02  # $1 min order / 2% = $50
    min_crypto_balance_ngn  = min_crypto_balance_usd * ngn_rate

    plan = f"""━━━ YOUR HONEST TRADING PLAN ━━━
Balance: ₦{balance_ngn:,.0f} = ${balance_usd:.2f} USD
2% Risk amount: ₦{risk_ngn:,.0f} = ${risk_usd:.4f} USD
"""

    if tier == "demo_only":
        plan += f"""
⚠️  REALITY CHECK — Your 2% risk is only ${risk_usd:.4f}.
    No real broker or exchange accepts orders this small.
    The right move here is DEMO trading — not real money yet.

WHAT TO DO NOW:
• Practice on CLEO's demo account (virtual ₦50,000 — free)
• Study signals daily, paper-trade every entry/exit, track your win rate
• Goal: Hit 65%+ win rate on demo for 30 consecutive trades
• Then deposit your ₦{balance_ngn:,.0f} PLUS save more to reach ₦{min_crypto_balance_ngn:,.0f} for crypto

MINIMUM TO START REAL TRADING:
• Crypto (Binance/Bybit spot): ₦{min_crypto_balance_ngn:,.0f} (~${min_crypto_balance_usd:.0f})
• Forex micro (0.01 lots):     ₦{min_forex_balance_ngn:,.0f} (~${min_forex_balance_usd:.0f})"""

    elif tier == "crypto_watch":
        plan += f"""
⚠️  ALMOST THERE — Your 2% risk = ${risk_usd:.4f}.
    Crypto spot needs $1+ per order (your 2% is below this).
    You CAN buy and hold crypto as spot investment (not leveraged trading).

WHAT TO DO NOW:
• Demo trade forex and crypto on CLEO's demo system — sharpen your entries
• You can BUY (not trade) fractional crypto: ${balance_usd:.2f} of BTC/ETH as long-term hold
• Grow your account to ₦{min_crypto_balance_ngn:,.0f} before trading with SL/TP
• Target: save ₦{max(0, min_crypto_balance_ngn - balance_ngn):,.0f} more to reach crypto trading minimum

MINIMUM TO START REAL TRADING:
• Crypto (Binance/Bybit spot): ₦{min_crypto_balance_ngn:,.0f} (~${min_crypto_balance_usd:.0f})
• Forex micro (0.01 lots):     ₦{min_forex_balance_ngn:,.0f} (~${min_forex_balance_usd:.0f})"""

    elif tier == "crypto_only":
        plan += f"""
✅  CRYPTO READY — Your 2% risk = ${risk_usd:.2f} (above $1 minimum for crypto).
⚠️  NOT FOREX READY — Forex 0.01 micro lot risks $2 minimum. Your 2% = ${risk_usd:.2f}.
    Opening 0.01 forex lots now would risk {actual_min_lot_risk_usd / balance_usd * 100:.0f}% of your account per trade.

WHAT TO DO NOW (CRYPTO):
• Platform: Binance, Bybit, or Kraken — spot trading only (no leverage)
• Risk per trade: ${risk_usd:.2f} (buy this USD value of the asset, set SL/TP)
• Best assets for your size: BTCUSDT, ETHUSDT, SOLUSDT (liquid, tight spread)
• 3 trades max open at once

WHAT TO DO FOR FOREX:
• Demo only — practice on CLEO's demo system (free, no real money)
• Save to ₦{min_forex_balance_ngn:,.0f} (~${min_forex_balance_usd:.0f}) before opening real forex

GROWTH TARGET:
• At 70% win rate trading crypto: expect +15-30% per month on your ${balance_usd:.2f}
• Reinvest profits until you hit ₦{min_forex_balance_ngn:,.0f} for forex"""

    elif tier == "forex_entry":
        plan += f"""
✅  FOREX MICRO READY — Your 2% risk = ${risk_usd:.2f}.
    You can trade 0.01 micro lots. Keep SL tight (≤20 pips) to stay within 2%.

FOREX PLAN (0.01 micro lots):
• Broker: Exness, XM, or FBS (low minimum deposit, tight spreads)
• Lot size: 0.01 lots on every trade (fixed — do NOT increase yet)
• Max SL: 20 pips (at 0.01 lots, 20 pips = $2 = your ~2% risk)
• Best pairs: EURUSD, GBPUSD (tightest spreads, most liquidity)
• Max 3 trades open at once

CRYPTO PLAN (parallel):
• Risk ${risk_usd:.2f} per signal on BTCUSDT, ETHUSDT, SOLUSDT
• Run both forex and crypto simultaneously to diversify

GROWTH TARGET:
• At 70% win rate: +10-20% per month on your ₦{balance_ngn:,.0f}
• Scale lot size to 0.02 when account hits ${balance_usd * 1.5:.0f}"""

    elif tier == "forex_micro":
        ideal_lots = round(risk_usd / 2.0 * 0.01, 2)   # $2 per 0.01 lot
        plan += f"""
✅  COMFORTABLE FOREX — Your 2% risk = ${risk_usd:.2f}.
    You can comfortably scale beyond the minimum micro lot.

FOREX PLAN:
• Lot size: {ideal_lots} lots (scaled for your 2% risk with 20-pip SL)
• Broker: Exness, XM, Pepperstone, or FXTM
• Best pairs: EURUSD, GBPUSD, USDJPY, AUDUSD
• Max 3 trades open at once — diversify across uncorrelated pairs

CRYPTO PLAN:
• Risk ${risk_usd:.2f} per signal on BTCUSDT, ETHUSDT, SOLUSDT
• Use spot trading — no leverage needed at this stage

GROWTH TARGET:
• Conservative: +8-15% per month
• Aggressive: +20-30% per month (3 trades/week, TP2 targets)"""

    else:  # forex_standard
        ideal_lots = round(risk_usd / 2.0 * 0.01, 2)
        plan += f"""
✅  FULL FOREX ACCESS — Your 2% risk = ${risk_usd:.2f}.
    All 30 pairs available. Scale properly.

FOREX PLAN:
• Lot size: {ideal_lots} lots (2% risk, 20-pip SL baseline)
• All 15 forex pairs in CLEO's universe available
• Pyramid into winning trades (add at 1R, move SL to entry)
• Consider prop firm challenge at this level (FTMO, MyFundedFX)

CRYPTO PLAN:
• Risk ${risk_usd:.2f} per crypto signal
• Can diversify across 3-4 crypto assets simultaneously

GROWTH TARGET:
• Conservative: +8-12% per month
• Compound effect: ₦{balance_ngn:,.0f} → ₦{balance_ngn * 2:,.0f} in 6-9 months at 10%/month"""

    return plan


def get_market_context() -> str:
    context = []
    for f in ["EUR", "GBP", "JPY", "AUD", "CAD"]:
        d = fetch_alpha_vantage_forex(f)
        if d.get("rate"):
            context.append(f"{f}/USD={d['rate']:.5f}")
    for c in ["BTC", "ETH", "SOL", "BNB"]:
        d = fetch_alpha_vantage_crypto(c)
        if d.get("price"):
            context.append(f"{c}=${d['price']:,.2f}")
    return " | ".join(context) if context else "Live data unavailable"


# ----------------------------
# Cache
# ----------------------------
PREDICTIONS_CACHE: Dict[str, Any] = {}
ADVICE_CACHE: Dict[str, Any] = {}
CACHE_TTL = 3600          # 60 min prediction cache (preserves Groq's 100K TPD free tier)
PRED_DISK_PATH = "/tmp/aria_pred_cache.json"
_PRED_GENERATING = False   # lock to prevent simultaneous Groq calls

def _load_pred_disk_cache() -> None:
    """Load persisted predictions from disk — survives server restarts."""
    global PREDICTIONS_CACHE
    try:
        with open(PRED_DISK_PATH, "r") as f:
            raw = json.load(f)
        now = time.time()
        for k, v in raw.items():
            age = now - v.get("ts", 0)
            if age < CACHE_TTL * 6:   # keep up to 1 h stale — better than no signals
                PREDICTIONS_CACHE[k] = v
        if PREDICTIONS_CACHE:
            print(f"[PRED-CACHE] Loaded {len(PREDICTIONS_CACHE)} prediction sets from disk")
    except (FileNotFoundError, Exception):
        pass

def _save_pred_disk_cache() -> None:
    """Persist predictions to disk after every successful generation."""
    try:
        with open(PRED_DISK_PATH, "w") as f:
            json.dump(PREDICTIONS_CACHE, f, default=str)
    except Exception as e:
        print(f"[PRED-CACHE] Disk save failed: {e}")

_load_pred_disk_cache()

# ----------------------------
# Symbol config
# ----------------------------
TRADING_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "USDCHF",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "EURCAD",
    "AUDCAD",
    "EURAUD",
    "GBPAUD",
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "SHIBUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "UNIUSDT",
]

SYMBOL_BASE_PRICES = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2700,
    "USDJPY": 149.50,
    "AUDUSD": 0.6530,
    "USDCAD": 1.3600,
    "NZDUSD": 0.6050,
    "USDCHF": 0.8950,
    "EURGBP": 0.8550,
    "EURJPY": 162.0,
    "GBPJPY": 189.5,
    "AUDJPY": 97.5,
    "EURCAD": 1.4750,
    "AUDCAD": 0.8880,
    "EURAUD": 1.6600,
    "GBPAUD": 1.9450,
    "BTCUSDT": 82000.0,
    "ETHUSDT": 1900.0,
    "BNBUSDT": 580.0,
    "XRPUSDT": 2.1,
    "SOLUSDT": 130.0,
    "ADAUSDT": 0.75,
    "DOGEUSDT": 0.16,
    "DOTUSDT": 6.5,
    "MATICUSDT": 0.55,
    "LTCUSDT": 95.0,
    "SHIBUSDT": 0.000013,
    "TRXUSDT": 0.22,
    "AVAXUSDT": 25.0,
    "LINKUSDT": 13.5,
    "UNIUSDT": 8.5,
}


# ----------------------------
# DB helpers
# ----------------------------
def get_or_create_user(username: str, db: Session) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_active = datetime.utcnow()
        db.commit()
    return user


def log_activity(
    user: User,
    action: str,
    symbol: str = None,
    details: dict = None,
    db: Session = None,
):
    if db:
        entry = UserActivity(
            user_id=user.id, action=action, symbol=symbol, details=details
        )
        db.add(entry)
        db.commit()


def get_user_profile_summary(user: User, db: Session) -> str:
    """Build a text summary of user behavior for AI personalization."""
    journal = (
        db.query(DBJournalEntry)
        .filter(DBJournalEntry.user_id == user.id)
        .order_by(DBJournalEntry.logged_at.desc())
        .limit(20)
        .all()
    )
    watchlist = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    activity = (
        db.query(UserActivity)
        .filter(UserActivity.user_id == user.id)
        .order_by(UserActivity.created_at.desc())
        .limit(30)
        .all()
    )

    wins = [j for j in journal if j.result == "WIN"]
    losses = [j for j in journal if j.result == "LOSS"]
    win_rate = (len(wins) / len(journal) * 100) if journal else 0
    most_traded = {}
    for j in journal:
        most_traded[j.symbol] = most_traded.get(j.symbol, 0) + 1
    top_symbol = max(most_traded, key=most_traded.get) if most_traded else "N/A"
    recent_actions = [f"{a.action} {a.symbol or ''}" for a in activity[:10]]
    watched = [w.symbol for w in watchlist]

    return (
        f"Username: {user.username} | Balance: {user.balance_ngn:,.0f} NGN | "
        f"Risk Tolerance: {user.risk_tolerance} | Style: {user.trading_style} | "
        f"Preferred Pairs: {user.preferred_pairs or watched or ['any']} | "
        f"Total Trades: {len(journal)} | Win Rate: {win_rate:.1f}% | Top Symbol: {top_symbol} | "
        f"Recent Activity: {', '.join(recent_actions[:5])}"
    )


# ----------------------------
# AI — Enhanced Predictions
# ----------------------------
def _get_session_priority_symbols(h: int) -> List[str]:
    """Return symbols ordered by current market session — most active first."""
    if 7 <= h < 12:
        priority = [
            "EURUSD",
            "GBPUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
            "EURCAD",
            "EURAUD",
            "GBPAUD",
            "BTCUSDT",
            "ETHUSDT",
            "USDCHF",
            "AUDUSD",
            "USDJPY",
            "SOLUSDT",
            "BNBUSDT",
        ]
    elif 12 <= h < 17:
        priority = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "USDCAD",
            "USDCHF",
            "EURGBP",
            "GBPJPY",
            "EURJPY",
            "XRPUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "LINKUSDT",
        ]
    elif 17 <= h < 21:
        priority = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "BNBUSDT",
            "AVAXUSDT",
            "LINKUSDT",
            "UNIUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "EURUSD",
            "GBPUSD",
            "USDCAD",
            "MATICUSDT",
            "TRXUSDT",
        ]
    else:
        priority = [
            "USDJPY",
            "GBPJPY",
            "AUDJPY",
            "EURJPY",
            "AUDCAD",
            "AUDUSD",
            "NZDUSD",
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "LTCUSDT",
            "XRPUSDT",
            "SHIBUSDT",
            "TRXUSDT",
        ]
    # Add remaining symbols not already in list
    remaining = [s for s in TRADING_SYMBOLS if s not in priority]
    return priority + remaining


def get_timing_context() -> dict:
    """
    Returns current trading session info, precise UTC entry windows, time exits,
    and recommended chart timeframes — all derived from the current UTC clock.
    Inject this into every AI prompt so CLEO always gives time-specific guidance.
    """
    from datetime import timedelta
    utc = datetime.utcnow()
    h   = utc.hour

    if 13 <= h < 16:
        session_name    = "London/NY Overlap"
        session_end_h   = 16
        volatility      = "PEAK — highest institutional orderflow of the day"
        best_pairs      = "EURUSD, GBPUSD, USDJPY, USDCAD, USDCHF, BTCUSDT, ETHUSDT"
        typical_range   = "80-150 pips (forex) / 1-3% (crypto)"
        rec_tf          = "H1"
        entry_valid_hrs = 1.5
    elif 7 <= h < 13:
        session_name    = "London"
        session_end_h   = 16
        volatility      = "HIGH — European institutional volume"
        best_pairs      = "EURUSD, GBPUSD, EURGBP, EURJPY, GBPJPY, EURAUD, GBPAUD"
        typical_range   = "60-100 pips (EUR/GBP pairs)"
        rec_tf          = "H1" if h < 10 else "H4"
        entry_valid_hrs = 2.0
    elif 16 <= h < 21:
        session_name    = "New York"
        session_end_h   = 21
        volatility      = "HIGH for USD pairs and crypto"
        best_pairs      = "USDCAD, USDCHF, USDJPY, BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT"
        typical_range   = "50-90 pips (USD pairs) / 1-4% (crypto)"
        rec_tf          = "H1"
        entry_valid_hrs = 2.0
    elif 21 <= h or h < 1:
        session_name    = "NY Close / Transition"
        session_end_h   = 1
        volatility      = "LOW — avoid new forex positions; crypto still active"
        best_pairs      = "BTCUSDT, ETHUSDT, SOLUSDT (crypto 24/7 only)"
        typical_range   = "Low forex (20-40 pips). Crypto: 1-2% moves possible"
        rec_tf          = "H4"
        entry_valid_hrs = 4.0
    elif 1 <= h < 4:
        session_name    = "Asian Early (Tokyo)"
        session_end_h   = 9
        volatility      = "LOW-MEDIUM — Tokyo open; JPY pairs most active"
        best_pairs      = "USDJPY, AUDJPY, GBPJPY, AUDUSD, NZDUSD, crypto"
        typical_range   = "30-60 pips (JPY pairs)"
        rec_tf          = "H4"
        entry_valid_hrs = 4.0
    else:
        session_name    = "Asian / Pre-London"
        session_end_h   = 9
        volatility      = "MEDIUM — building toward London Open breakout"
        best_pairs      = "USDJPY, AUDJPY, AUDUSD, NZDUSD, crypto"
        typical_range   = "40-70 pips (JPY/AUD pairs)"
        rec_tf          = "H1"
        entry_valid_hrs = 2.0

    session_end_dt = utc.replace(hour=session_end_h, minute=0, second=0, microsecond=0)
    if session_end_dt <= utc:
        session_end_dt += timedelta(days=1)
    mins_left = max(0, int((session_end_dt - utc).total_seconds() / 60))

    entry_win_end  = utc + timedelta(minutes=30)
    entry_win_utc  = f"{utc.strftime('%H:%M')}–{entry_win_end.strftime('%H:%M')} UTC"

    deadline_dt    = min(
        session_end_dt - timedelta(hours=2),
        utc + timedelta(hours=entry_valid_hrs),
    )
    if deadline_dt <= utc:
        deadline_dt = utc + timedelta(minutes=30)
    entry_deadline = deadline_dt.strftime("%H:%M UTC")

    time_exit_dt   = session_end_dt - timedelta(hours=1, minutes=30)
    if time_exit_dt <= utc:
        time_exit_dt = utc + timedelta(hours=2)
    time_exit      = time_exit_dt.strftime("%H:%M UTC")

    return {
        "session_name":       session_name,
        "session_end_utc":    session_end_dt.strftime("%H:%M UTC"),
        "volatility":         volatility,
        "best_pairs":         best_pairs,
        "typical_range":      typical_range,
        "recommended_tf":     rec_tf,
        "entry_window_utc":   entry_win_utc,
        "entry_deadline_utc": entry_deadline,
        "time_exit_utc":      time_exit,
        "minutes_left":       mins_left,
        "current_utc":        utc.strftime("%H:%M UTC, %a %d %b %Y"),
    }


def enrich_signal_timing(signal: dict, timing: dict) -> dict:
    """
    Post-process a signal to guarantee timing fields are accurate.
    Fills missing fields from the session context and validates AI-provided ones.
    Runs AFTER the AI returns so timing is always based on real clock data.
    """
    try:
        tf = signal.get("timeframe_chart", "")
        if tf not in ("M15", "H1", "H4", "D1"):
            tf = timing["recommended_tf"]
        signal["timeframe_chart"]     = tf
        signal["entry_window_utc"]    = signal.get("entry_window_utc")    or timing["entry_window_utc"]
        signal["entry_deadline_utc"]  = signal.get("entry_deadline_utc")  or timing["entry_deadline_utc"]
        signal["time_exit_utc"]       = signal.get("time_exit_utc")       or timing["time_exit_utc"]
        signal["session_context"]     = signal.get("session_context")     or timing["session_name"]

        if not signal.get("expected_tp1_hours"):
            hold = str(signal.get("hold_time", "")).lower()
            m_match = re.search(r"(\d+\.?\d*)", hold)
            raw = float(m_match.group(1)) if m_match else 4.0
            if "min" in hold:
                hrs = raw / 60
            elif "d" in hold:
                hrs = raw * 24
            else:
                hrs = raw
            fallback = {"M15": 0.5, "H1": 2.0, "H4": 8.0, "D1": 24.0}.get(tf, 4.0)
            signal["expected_tp1_hours"] = round(min(hrs * 0.6, fallback), 1)

    except Exception:
        pass
    return signal


def generate_market_predictions(
    investment_amount_ngn: float = 0.0, user_profile: str = ""
) -> Dict[str, Any]:
    global PREDICTIONS_CACHE, _PRED_GENERATING
    now = time.time()
    cache_key = f"preds_{int(investment_amount_ngn)}"
    if cache_key in PREDICTIONS_CACHE and (
        now - PREDICTIONS_CACHE[cache_key]["ts"] < CACHE_TTL
    ):
        return PREDICTIONS_CACHE[cache_key]["data"]

    # Thundering-herd guard: if another request is already generating,
    # wait up to 90 s for it to populate the cache, then serve stale or fail
    if _PRED_GENERATING:
        for _ in range(18):          # wait up to 90 s (18 × 5 s polls)
            time.sleep(5)
            if cache_key in PREDICTIONS_CACHE:
                return PREDICTIONS_CACHE[cache_key]["data"]
        # Timed out waiting — serve stale disk cache if available
        if cache_key in PREDICTIONS_CACHE:
            return PREDICTIONS_CACHE[cache_key]["data"]
        return {"success": False, "error": "Signal generation in progress — please retry shortly."}

    _PRED_GENERATING = True

    ngn_rate = get_ngn_rate()
    inv_usd = investment_amount_ngn / ngn_rate if investment_amount_ngn > 0 else 0
    utc = datetime.utcnow()
    h = utc.hour

    if 7 <= h < 12:
        session = "London Session — EURUSD, GBPUSD, EURGBP most active"
    elif 12 <= h < 17:
        session = "London/NY Overlap — PEAK liquidity, all pairs active"
    elif 17 <= h < 21:
        session = "New York Session — USD pairs and crypto most active"
    else:
        session = "Asian Session — JPY pairs: USDJPY, GBPJPY, AUDJPY most active"

    personalization = f"\nUSER PROFILE: {user_profile}" if user_profile else ""

    # Fetch REAL live data for session-priority symbols (up to 12 to respect rate limits)
    priority_symbols = _get_session_priority_symbols(h)
    real_data_symbols = priority_symbols[:12]
    live_data_blocks = []
    has_real_data = False

    print(f"[CLEO] Fetching real market data for {len(real_data_symbols)} symbols...")
    analyses_by_symbol: Dict[str, Dict] = {}   # ← store for post-AI validation
    for sym in real_data_symbols:
        # fetch_realtime_price=False: use OHLCV last close during batch to save TD quota
        analysis = get_symbol_analysis(sym, fetch_realtime_price=False)
        # Use compact single-line format to stay well within Groq's 6000 TPM limit
        block = format_for_ai_prompt_compact(analysis)
        live_data_blocks.append(block)
        analyses_by_symbol[sym] = analysis
        if analysis.get("live_price"):
            has_real_data = True

    # Remaining symbols — build price + ATR context so AI can do real analysis
    remaining_symbols = [s for s in TRADING_SYMBOLS if s not in real_data_symbols]
    remaining_lines = []
    for sym in remaining_symbols:
        ref_price = FALLBACK_PRICES.get(sym, 0)
        atr_pct   = FALLBACK_ATR_PCT.get(sym, 0.01)
        atr_abs   = round(ref_price * atr_pct, 6)
        sl_buy    = round(ref_price - atr_abs * 1.5, 6)
        tp1_buy   = round(ref_price + atr_abs * 1.5, 6)
        tp2_buy   = round(ref_price + atr_abs * 3.0, 6)
        sl_sell   = round(ref_price + atr_abs * 1.5, 6)
        tp1_sell  = round(ref_price - atr_abs * 1.5, 6)
        tp2_sell  = round(ref_price - atr_abs * 3.0, 6)
        cat = "crypto" if "USDT" in sym else "forex"
        remaining_lines.append(
            f"{sym} ({cat}) | ref_price={ref_price} | ATR≈{atr_abs} | "
            f"BUY SL={sl_buy} TP1={tp1_buy} TP2={tp2_buy} | "
            f"SELL SL={sl_sell} TP1={tp1_sell} TP2={tp2_sell}"
        )
    remaining_text = "\n".join(remaining_lines)

    live_data_section = "\n".join(live_data_blocks)   # compact: newline not double

    # Detect which API actually provided data (for honest reporting)
    _api_used = "Alpha Vantage"
    for _a in analyses_by_symbol.values():
        _src = _a.get("data_source", "")
        if "Twelve Data" in _src or "twelve" in _src.lower():
            _api_used = "Twelve Data"
            break

    data_source_note = (
        f"LIVE {_api_used} data ({len(real_data_symbols)} symbols)"
        if has_real_data
        else "AI pattern reasoning (live data unavailable)"
    )

    timing = get_timing_context()

    prompt = f"""━━━ CLEO MARKET ANALYSIS ENGINE ━━━
SESSION: {session} | UTC: {utc.strftime("%H:%M")} | Data: {data_source_note}
Budget: {investment_amount_ngn:,.0f} NGN (~${inv_usd:,.2f} USD) | Risk/trade: {investment_amount_ngn*0.02:,.0f} NGN (~${investment_amount_ngn*0.02/ngn_rate:.2f} USD){personalization}

━━━ TIMING INTELLIGENCE (use in every signal) ━━━
Active Session : {timing["session_name"]} | Volatility: {timing["volatility"]}
Entry Window   : {timing["entry_window_utc"]} (enter within this window for best fills)
Entry Deadline : {timing["entry_deadline_utc"]} — do NOT take new trades after this
Time Exit Rule : {timing["time_exit_utc"]} — if TP1 not reached by then, exit at BE or small loss
Recommended TF : {timing["recommended_tf"]} chart (matches current session liquidity)
Best Pairs Now : {timing["best_pairs"]}
Typical Range  : {timing["typical_range"]}
Session Ends   : {timing["session_end_utc"]} ({timing["minutes_left"]} min remaining)

GOAL: Analyze ALL 30 symbols and return every symbol with a genuine tradeable edge.
Target: 5-12 signals. Quality over quantity — only include setups with real conviction.

━━━ YOUR REASONING PROCESS (apply to EVERY symbol) ━━━
Before assigning any signal, mentally work through these 6 steps + timing check:

1. TREND: Is price making HH+HL (uptrend), LH+LL (downtrend), or choppy range?
2. MOMENTUM: Do RSI, MACD histogram, and Stochastic all agree on direction?
3. CONFIRMATION: Is EMA stack aligned? Does ADX confirm a trending market (>20)?
4. VOLATILITY: Is ATR healthy (not compressed)? Is this the right session for this pair?
5. RISK: Can I place a stop at 1.5×ATR with at least 1:1.5 R:R to TP1?
6. CONVICTION: How many of steps 1-5 agree? 5-6 = STRONG. 3-4 = BUY/SELL. 1-2 = skip.
7. TIMING: Is this pair active in the {timing["session_name"]} session? Set the exact UTC entry window.

━━━ SECTION A — LIVE INDICATOR DATA (12 symbols with full OHLCV + indicators) ━━━
{"No live data this cycle — see Section B." if not has_real_data else ""}
{live_data_section if has_real_data else ""}

SECTION A RULES:
✓ For each symbol above: run all 6 reasoning steps explicitly in your head
✓ Use the EXACT indicator values shown — never invent or adjust them
✓ Confluence score 2/6+ required. ADX must be > 20. R:R must be ≥ 1:1.5
✓ Confidence mapping: 6/6→92-97 | 5/6→86-91 | 4/6→80-85 | 3/6→75-79 | 2/6→73-74
✗ Skip if: confluence 0-1/6, ADX<20, R:R<1:1.5, or AI direction ≠ confluence direction

━━━ SECTION B — AI PATTERN REASONING ({len(remaining_symbols)} symbols, ATR-derived levels provided) ━━━
For these symbols: use intermarket correlations, macro drivers, session behaviour,
and your knowledge of current USD strength/weakness and global risk appetite.
The ATR-derived SL/TP levels below are pre-calculated — USE THEM EXACTLY.

{remaining_text if remaining_text else "(all symbols covered in Section A)"}

SECTION B RULES:
✓ Run your 6-step reasoning even without live indicators
✓ Use macro context: Is USD strong or weak right now? Risk-on or risk-off?
✓ Use session context: {session.split("—")[0].strip()} session — which pairs have liquidity NOW?
✓ Use correlations: EURUSD ↑ often means GBPUSD ↑, USDCHF ↓
✓ Include if 3+ of: macro bias, EMA estimate, RSI estimate, session activity, S/R level agree
✗ Confidence cap: 78 for ai_reasoning. Confluence format: "X/3"

━━━ SIGNAL CONSTRUCTION RULES ━━━
- entry_price: current live price or best pullback entry (within 0.1% of live price)
- stop_loss: entry ± (1.5 × ATR). Never tighter than 0.5 ATR.
- take_profit_1: 1.5R from entry. take_profit_2: 2.5R. take_profit_3: 4.0R.
- back_out_trigger: the price level that COMPLETELY invalidates the trade thesis
- timeframe_chart: the ACTUAL chart timeframe to use — M15, H1, H4, or D1
  • M15 = scalp (<90 min hold), only during London Open (07-09 UTC) or NY Open (13-15 UTC)
  • H1  = intraday (2-8h hold), any active session — THIS IS THE DEFAULT
  • H4  = swing (8-48h hold), low volatility or Asian session setups
  • D1  = major swing (2-5 days), strong macro trend setups only
- entry_window_utc: "HH:MM–HH:MM UTC" — the valid window to enter this trade TODAY
- entry_deadline_utc: "HH:MM UTC" — latest time to enter; skip if price hasn't triggered by then
- time_exit_utc: "HH:MM UTC" — close trade at BE/small loss if TP1 not reached by this time
- expected_tp1_hours: estimated hours to reach TP1 based on ATR and timeframe (number)
- session_context: which session is driving this and why (e.g. "London Open breakout — EUR volume spike")
- hold_time: human-readable (e.g. "4 hours", "2 days") — must match timeframe_chart
- rationale: 1 crisp sentence — name the 2 strongest indicator signals + the setup logic
- data_source: "live_data" for Section A symbols, "ai_reasoning" for Section B

━━━ WHAT NOT TO DO ━━━
✗ Do NOT output HOLD signals — only output actual trade signals
✗ Do NOT adjust ATR-derived SL/TP numbers (they are pre-calculated for correct R:R)
✗ Do NOT give a signal if indicators conflict — skip that symbol
✗ Do NOT output any text outside the JSON array
✗ Do NOT set entry_deadline_utc after the current session end ({timing["session_end_utc"]})

Return ONLY a valid JSON array:
[
  {{
    "symbol": "SYMBOL",
    "signal": "STRONG_BUY|BUY|SELL|STRONG_SELL",
    "confidence": 73-97,
    "confluence_score": "X/6 or X/3",
    "data_source": "live_data|ai_reasoning",
    "category": "forex|crypto",
    "timeframe_chart": "M15|H1|H4|D1",
    "session_fit": "excellent|good|fair|poor",
    "live_price": 0.0,
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "take_profit_1": 0.0,
    "take_profit_2": 0.0,
    "take_profit_3": 0.0,
    "risk_reward": "1:X.X",
    "hold_time": "X hours or X days",
    "expected_tp1_hours": 2.5,
    "entry_window_utc": "HH:MM–HH:MM UTC",
    "entry_deadline_utc": "HH:MM UTC",
    "time_exit_utc": "HH:MM UTC",
    "session_context": "which session + what is driving this move",
    "back_out_trigger": 0.0,
    "rationale": "2 indicators + setup in one sentence."
  }}
]"""

    try:
        content = get_ai_response(prompt, max_tokens=2048)
        if not content or content.startswith("ERROR:"):
            # If stale cache exists, serve it rather than failing completely
            if cache_key in PREDICTIONS_CACHE:
                stale = PREDICTIONS_CACHE[cache_key]["data"]
                stale_age_min = int((now - PREDICTIONS_CACHE[cache_key]["ts"]) / 60)
                stale["_stale_cache"] = True
                stale["_stale_age_min"] = stale_age_min
                print(f"[CLEO] Rate limited — serving stale cache ({stale_age_min} min old)")
                return stale
            return {"success": False, "error": content or "Empty AI response"}
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        print(f"[CLEO] AI returned {len(data)} raw signals before Python gates")

        # ── PYTHON-LEVEL QUALITY GATES (hard rules AI cannot override) ──────────
        validated = []
        for s in data:
            if not isinstance(s, dict):
                continue
            sym = s.get("symbol", "")
            sig = s.get("signal", "")
            conf = int(s.get("confidence", 0))
            src  = s.get("data_source", "ai_reasoning")

            # Gate 1: minimum confidence
            if conf < 73:
                continue

            # Gate 2: HOLD signals excluded
            if sig == "HOLD":
                continue

            # Gate 3: live-data signals must pass Python confluence check
            if src == "live_data" and sym in analyses_by_symbol:
                calc = analyses_by_symbol[sym].get("confluence", {})
                calc_dir       = calc.get("direction", "NEUTRAL")
                calc_score     = int(calc.get("score", 0))
                calc_tradeable = calc.get("tradeable", True)

                # Need at least 2 indicators agreeing (was 3 — lowered for real data
                # where markets are often mixed; auto-trader still requires 3)
                if calc_score < 2:
                    print(f"[GATE] Rejected {sym} {sig}: confluence {calc_score}/6 < 2")
                    continue

                # Direction must not be opposite to what Python calculated
                is_buy  = sig in ("BUY", "STRONG_BUY")
                is_sell = sig in ("SELL", "STRONG_SELL")
                if is_buy  and calc_dir == "SELL":
                    print(f"[GATE] Rejected {sym} {sig}: AI says BUY but Python says SELL")
                    continue
                if is_sell and calc_dir == "BUY":
                    print(f"[GATE] Rejected {sym} {sig}: AI says SELL but Python says BUY")
                    continue

                # Fully neutral market with weak score → reject (was < 3, now < 2)
                if calc_dir == "NEUTRAL" and calc_score < 2:
                    print(f"[GATE] Rejected {sym} {sig}: NEUTRAL market, score only {calc_score}/6")
                    continue

                # ADX gate: market must be trending (not ranging)
                if not calc_tradeable:
                    print(f"[GATE] Rejected {sym} {sig}: ADX <20 (ranging market)")
                    continue

                # Confidence scaled to confluence score
                if calc_score >= 5:
                    if conf > 95: s["confidence"] = 93
                elif calc_score == 4:
                    if conf > 87: s["confidence"] = 85
                elif calc_score == 3:
                    if conf > 79: s["confidence"] = 77
                else:  # score == 2
                    if conf > 73: s["confidence"] = 73

                # Signal quality label (for UI — auto-trader uses 3+ only)
                if calc_score >= 4:
                    s["signal_quality"] = "strong"
                elif calc_score == 3:
                    s["signal_quality"] = "moderate"
                else:
                    s["signal_quality"] = "weak — monitor only"

                # Stamp the verified confluence data
                s["confluence_score"]     = f"{calc_score}/6"
                s["confluence_direction"] = calc_dir
                s["auto_trade_eligible"]  = calc_score >= 3 and calc_tradeable

            # ── NGN trade details (injected for every validated signal) ──────
            if investment_amount_ngn > 0:
                try:
                    s["ngn_trade"] = calc_ngn_trade_details(
                        symbol      = s.get("symbol", sym),
                        entry       = float(s.get("entry_price", 0) or 0),
                        stop_loss   = float(s.get("stop_loss", 0) or 0),
                        tp1         = float(s.get("take_profit_1", 0) or 0),
                        tp2         = float(s.get("take_profit_2", 0) or 0),
                        tp3         = float(s.get("take_profit_3", 0) or 0),
                        balance_ngn = investment_amount_ngn,
                        ngn_rate    = ngn_rate,
                    )
                except Exception:
                    pass

            if src == "ai_reasoning":
                # Cap AI-only signals at 78%
                if conf > 78:
                    s["confidence"] = 78
                s["signal_quality"]      = "ai_reasoning"
                s["auto_trade_eligible"] = False
                # Normalise confluence_score string (AI sometimes returns "4/3" etc.)
                raw_cs = str(s.get("confluence_score", "0/3"))
                try:
                    parts = raw_cs.split("/")
                    num   = int(parts[0])
                    denom = int(parts[1]) if len(parts) > 1 else 3
                    num   = min(num, denom)
                    s["confluence_score"] = f"{num}/{denom}"
                except Exception:
                    s["confluence_score"] = raw_cs

            validated.append(s)

        # ── Enrich every signal with accurate timing fields ──────────────────
        data = [enrich_signal_timing(s, timing) for s in validated]
        # ── END PYTHON GATES ────────────────────────────────────────────────────

        strong = [s for s in data if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]]
        live_count = len([s for s in data if s.get("data_source") == "live_data"])

        result = {
            "success": True,
            "generated_at": utc.isoformat(),
            "active_session": session,
            "data_quality": {
                "source": data_source_note,
                "live_data_symbols": real_data_symbols,
                "signals_from_live_data": live_count,
                "signals_from_ai_reasoning": len(data) - live_count,
            },
            "investment_context_ngn": investment_amount_ngn,
            "usd_ngn_rate": ngn_rate,
            "model": "llama-3.3-70b-versatile",
            "signals": data,
            "total_signals": len(data),
            "strong_signals_count": len(strong),
            "top_picks": strong[:5],
            "disclaimer": "Signals are data-driven using live market feeds (Twelve Data / Alpha Vantage). Trading involves substantial risk.",
        }
        PREDICTIONS_CACHE[cache_key] = {"data": result, "ts": now}
        _save_pred_disk_cache()
        return result
    except Exception as e:
        return {"success": False, "error": f"Prediction failed: {str(e)}"}
    finally:
        _PRED_GENERATING = False


# ----------------------------
# Personalized Signal for 1 Symbol (FIXED)
# ----------------------------
def get_personalized_signal(symbol: str, user: User, db: Session) -> Dict:
    """Generate a focused AI signal for a single symbol using REAL live data + user personalization."""
    symbol = symbol.upper()
    ngn_rate = get_ngn_rate()
    inv_usd = user.balance_ngn / ngn_rate if user.balance_ngn > 0 else 0
    profile = get_user_profile_summary(user, db)
    utc = datetime.utcnow()

    # Fetch real live data first
    analysis = get_symbol_analysis(symbol)
    live_data_block = format_for_ai_prompt(analysis)
    live_price = analysis.get("live_price", 0)
    has_live = bool(live_price)

    timing = get_timing_context()

    prompt = f"""━━━ MARKET DATA FOR {symbol} ━━━
{live_data_block}

━━━ USER CONTEXT ━━━
{profile}
Balance: {user.balance_ngn:,.0f} NGN (~${inv_usd:,.2f} USD)
Style: {user.trading_style} | Risk: {user.risk_tolerance}
UTC: {utc.strftime("%Y-%m-%d %H:%M")}
Position size (2% risk): {user.balance_ngn * 0.02:,.0f} NGN = ${user.balance_ngn * 0.02 / ngn_rate:.2f} USD

━━━ SESSION & TIMING INTELLIGENCE ━━━
Active Session  : {timing["session_name"]} | Volatility: {timing["volatility"]}
Recommended TF  : {timing["recommended_tf"]} chart for current session
Entry Window    : {timing["entry_window_utc"]} — optimal entry window right now
Entry Deadline  : {timing["entry_deadline_utc"]} — do NOT enter after this time
Time Exit Rule  : {timing["time_exit_utc"]} — exit at BE if TP1 not reached by then
Best Pairs Now  : {timing["best_pairs"]}
Typical Range   : {timing["typical_range"]}
Session Ends    : {timing["session_end_utc"]} ({timing["minutes_left"]} min remaining)

━━━ CONFLUENCE-BASED CONFIDENCE ━━━
{"Use the confluence score from the data block to calibrate confidence:" if has_live else "AI reasoning mode — cap confidence at 78."}
{"Score 5-6/6 → 88-97 (STRONG) | Score 4/6 → 80-87 | Score 3/6 → 73-79 | Score 0-2 → HOLD" if has_live else ""}

━━━ QUALITY GATES (fail any = return HOLD) ━━━
{"✗ ADX < 20 → HOLD (ranging) | ✗ MACD/RSI conflict → HOLD | ✗ Stoch/Williams disagree → HOLD | ✗ R:R < 1:1.8 → HOLD" if has_live else ""}

━━━ PERSONALISATION ━━━
- Chart timeframe: match {user.trading_style} style (scalper=M15, intraday=H1, swing=H4/D1)
- SL distance: {"tighter (0.8–1.0×ATR)" if user.risk_tolerance == "low" else ("standard (1.5×ATR)" if user.risk_tolerance == "medium" else "wider (2.0×ATR)")} for {user.risk_tolerance} risk
- {"Use EXACT indicator values — do not invent numbers." if has_live else "Conservative analysis only — no live data."}

Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
  "confidence": 0-97,
  "confluence_score": "X/6 or N/A",
  "data_source": "{"live_data" if has_live else "ai_reasoning"}",
  "category": "forex|crypto",
  "timeframe_chart": "M15|H1|H4|D1",
  "session_fit": "excellent|good|fair|poor",
  "live_price": "{live_price if live_price else "unavailable"}",
  "entry_price": "exact number",
  "stop_loss": "exact number (ATR-based)",
  "take_profit_1": "exact number",
  "take_profit_2": "exact number",
  "take_profit_3": "exact number",
  "risk_reward": "1:X.X",
  "hold_time": "X hours or X days (e.g. '4 hours', '2 days')",
  "expected_tp1_hours": 2.5,
  "entry_window_utc": "HH:MM–HH:MM UTC",
  "entry_deadline_utc": "HH:MM UTC",
  "time_exit_utc": "HH:MM UTC",
  "session_context": "which session + what is driving this move right now",
  "position_size_ngn": "{user.balance_ngn * 0.02:,.0f} NGN",
  "position_size_usd": "${user.balance_ngn * 0.02 / ngn_rate:.2f} USD",
  "back_out_trigger": "exact invalidation price",
  "indicators": {{
    "rsi": "exact value + zone",
    "macd": "histogram value + direction",
    "stochastic": "K/D + zone",
    "adx": "value + trend strength",
    "williams_r": "value + zone",
    "ema_bias": "price vs EMA20/50/200",
    "bollinger": "band position",
    "pattern": "detected pattern"
  }},
  "key_levels": {{
    "support": "exact level",
    "resistance": "exact level",
    "pivot": "pivot level if available"
  }},
  "personalized_note": "1 sentence addressing this user's specific style, risk, and history",
  "rationale": "3 sentences: (1) what the confluence/indicators say, (2) why this entry is valid NOW in this session, (3) what the risk is and when to exit"
}}"""

    try:
        content = get_ai_response(prompt)
        if not content or content.startswith("ERROR:"):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "error": "AI unavailable",
                "live_price": live_price,
            }
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)

        # ── Python-level quality gate for single-symbol ──────────────────────
        if has_live:
            calc   = analysis.get("confluence", {})
            c_dir  = calc.get("direction", "NEUTRAL")
            c_score = int(calc.get("score", 0))
            c_trade = calc.get("tradeable", True)
            sig    = data.get("signal", "HOLD")
            is_buy  = sig in ("BUY", "STRONG_BUY")
            is_sell = sig in ("SELL", "STRONG_SELL")

            gate_fail = None
            if c_score < 3:
                gate_fail = f"confluence {c_score}/6 below minimum (need ≥3)"
            elif is_buy and c_dir == "SELL":
                gate_fail = "AI BUY conflicts with calculated SELL confluence"
            elif is_sell and c_dir == "BUY":
                gate_fail = "AI SELL conflicts with calculated BUY confluence"
            elif not c_trade:
                gate_fail = "ADX <20 — ranging market, no directional trade"

            if gate_fail:
                data["signal"]     = "HOLD"
                data["confidence"] = 0
                data["gate_rejection"] = gate_fail
                data["confluence_score"] = f"{c_score}/6"
                data["note"] = "Overridden to HOLD by Python quality gate. Wait for clearer setup."
            else:
                # Stamp verified confluence data
                data["confluence_score"]     = f"{c_score}/6"
                data["confluence_direction"] = c_dir
                # Cap confidence to match confluence score
                conf = int(data.get("confidence", 0))
                if c_score <= 3 and conf > 80:   data["confidence"] = 76
                elif c_score == 4 and conf > 87: data["confidence"] = 85
                elif c_score >= 5 and conf > 97: data["confidence"] = 95
        else:
            # AI-only mode — cap at 78
            if int(data.get("confidence", 0)) > 78:
                data["confidence"] = 78
        # ── End gate ──────────────────────────────────────────────────────────

        data["personalized"] = True
        data["raw_live_analysis"] = analysis

        # ── Enrich with accurate timing fields (Python-guaranteed) ────────
        data = enrich_signal_timing(data, timing)

        # ── Inject NGN trade details ──────────────────────────────────────
        if user.balance_ngn and user.balance_ngn > 0 and data.get("signal") != "HOLD":
            try:
                data["ngn_trade"] = calc_ngn_trade_details(
                    symbol      = symbol,
                    entry       = float(data.get("entry_price", 0) or 0),
                    stop_loss   = float(data.get("stop_loss", 0) or 0),
                    tp1         = float(data.get("take_profit_1", 0) or 0),
                    tp2         = float(data.get("take_profit_2", 0) or 0),
                    tp3         = float(data.get("take_profit_3", 0) or 0),
                    balance_ngn = user.balance_ngn,
                    ngn_rate    = ngn_rate,
                )
            except Exception:
                pass

        return data
    except Exception as e:
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "confidence": 0,
            "error": str(e),
            "live_price": live_price,
        }


# ----------------------------
# Simulator
# ----------------------------
def simulate_ohlcv(
    symbol: str, num_candles: int = 150, interval_minutes: int = 15, seed: int = None
) -> List[Dict]:
    s = symbol.upper()
    base_price = SYMBOL_BASE_PRICES.get(s, 1.0)
    is_crypto = "USDT" in s
    annual_sigma = 0.65 if is_crypto else 0.08
    dt = (interval_minutes / 60) / 8760
    sigma = annual_sigma * math.sqrt(dt)
    rng = random.Random(seed or int(time.time()))
    candles, price = [], base_price
    now = datetime.utcnow() - timedelta(minutes=interval_minutes * num_candles)

    for i in range(num_candles):
        z = rng.gauss(0, 1)
        pct = 0.00005 * dt + sigma * z
        close = price * (1 + pct)
        cv = abs(rng.gauss(0, sigma * 0.4))
        high = max(price, close) * (1 + abs(rng.gauss(0, cv)))
        low = min(price, close) * (1 - abs(rng.gauss(0, cv)))
        bvol = 2000 if is_crypto else 200000
        vol = rng.uniform(bvol * 0.5, bvol * 2.5)
        dp = 4 if is_crypto else 5
        ts = now + timedelta(minutes=interval_minutes * i)
        candles.append(
            {
                "time": int(ts.timestamp()),
                "datetime": ts.strftime("%Y-%m-%d %H:%M"),
                "open": round(price, dp),
                "high": round(high, dp),
                "low": round(low, dp),
                "close": round(close, dp),
                "volume": round(vol, 2),
            }
        )
        price = close
    return candles


def run_strategy_on_candles(
    candles: List[Dict], initial_balance: float = 10000.0
) -> Dict:
    if len(candles) < 51:
        return {"error": "Need 51+ candles"}
    closes = [c["close"] for c in candles]

    def sma(d, p):
        return [None] * (p - 1) + [sum(d[i : i + p]) / p for i in range(len(d) - p + 1)]

    s20, s50 = sma(closes, 20), sma(closes, 50)
    balance, position, trades, equity_curve = initial_balance, None, [], []

    for i in range(50, len(candles)):
        c = candles[i]
        if s20[i] and s50[i] and s20[i - 1] and s50[i - 1]:
            if s20[i - 1] <= s50[i - 1] and s20[i] > s50[i] and position is None:
                position = {
                    "entry": c["close"],
                    "time": c["time"],
                    "datetime": c["datetime"],
                }
            elif s20[i - 1] >= s50[i - 1] and s20[i] < s50[i] and position is not None:
                pnl_pct = (c["close"] - position["entry"]) / position["entry"]
                pnl_usd = balance * 0.1 * pnl_pct
                balance += pnl_usd
                trades.append(
                    {
                        "type": "BUY",
                        "entry_price": position["entry"],
                        "entry_time": position["time"],
                        "entry_datetime": position["datetime"],
                        "exit_price": c["close"],
                        "exit_time": c["time"],
                        "exit_datetime": c["datetime"],
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "pnl_usd": round(pnl_usd, 2),
                        "result": "WIN" if pnl_usd > 0 else "LOSS",
                    }
                )
                position = None
        equity_curve.append(
            {"time": c["time"], "datetime": c["datetime"], "equity": round(balance, 2)}
        )

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    return {
        "initial_balance": initial_balance,
        "final_balance": round(balance, 2),
        "total_pnl_usd": round(balance - initial_balance, 2),
        "total_pnl_pct": round((balance - initial_balance) / initial_balance * 100, 2),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_usd": round(sum(t["pnl_usd"] for t in wins) / len(wins), 2)
        if wins
        else 0,
        "avg_loss_usd": round(sum(t["pnl_usd"] for t in losses) / len(losses), 2)
        if losses
        else 0,
        "strategy": "SMA 20/50 Crossover",
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="AI Trading Bot — CLEO v3.1")
app.include_router(backtest_router, prefix="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _keep_alive_ping():
    """Ping our own health endpoint every 10 minutes to prevent Render free-tier spin-down."""
    import requests as _req
    time.sleep(60)  # wait for server to fully start first
    render_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not render_url:
        # Try to detect Render URL automatically, fall back to localhost
        render_url = "http://localhost:5000"
    ping_url = render_url.rstrip("/") + "/"
    print(f"[KEEPALIVE] Self-ping started — hitting {ping_url} every 10 min")
    while True:
        try:
            r = _req.get(ping_url, timeout=10)
            print(f"[KEEPALIVE] ✅ Ping OK ({r.status_code})")
        except Exception as e:
            print(f"[KEEPALIVE] ⚠️ Ping failed: {e}")
        time.sleep(600)  # 10 minutes


@app.on_event("startup")
def start_auto_scheduler():
    """Launch the 24/7 auto-trading scheduler as a background daemon thread."""
    t = threading.Thread(target=_auto_trade_scheduler, daemon=True)
    t.start()
    print("[SCHEDULER] 24/7 auto-trade scheduler started")

    ka = threading.Thread(target=_keep_alive_ping, daemon=True)
    ka.start()
    print("[KEEPALIVE] Keep-alive ping thread started")


# ----------------------------
# Pydantic Models
# ----------------------------
class ChatMessage(BaseModel):
    message: str
    username: Optional[str] = "guest"
    conversation_id: Optional[int] = None  # omit to auto-create a new thread


class RiskCalcRequest(BaseModel):
    balance_ngn: float
    risk_percent: float = 2.0
    entry_price: float
    stop_loss_price: float
    symbol: str


class TradeIdeaRequest(BaseModel):
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    rationale: Optional[str] = ""
    username: Optional[str] = "guest"


class TradeJournalEntryRequest(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    result: str
    pnl_usd: float
    notes: Optional[str] = ""


class UserProfileUpdate(BaseModel):
    balance_ngn: Optional[float] = None
    risk_tolerance: Optional[str] = None
    trading_style: Optional[str] = None
    preferred_pairs: Optional[List[str]] = None


class DemoTradeRequest(BaseModel):
    symbol: str
    volume: float
    trade_type: str


class UserAccountRequest(BaseModel):
    username: str
    mt5_login: Optional[str] = None
    mt5_server: Optional[str] = None


# ----------------------------
# Health
# ----------------------------
@app.get("/mt5_bridge_download")
def download_bridge():
    """Serve the latest mt5_bridge.py for easy update on Windows."""
    from fastapi.responses import FileResponse
    import os
    bridge_path = os.path.join(os.path.dirname(__file__), "mt5_bridge.py")
    return FileResponse(bridge_path, media_type="text/plain", filename="mt5_bridge.py")


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "CLEO — AI Trading Bot",
        "version": "3.1",
        "pairs_monitored": len(TRADING_SYMBOLS),
        "persistence": "PostgreSQL",
        "features": [
            "30 Pairs (15 Forex + 15 Crypto)",
            "Personalized Signals per User",
            "NGN Position Sizing",
            "Risk Calculator",
            "Visual Simulator",
            "Trade Journal (DB)",
            "Watchlist (DB)",
            "Chat History (DB)",
            "User Profiles + Activity Tracking",
            "Trade Idea Scorer",
            "Demo Account (DB)",
        ],
    }


# ----------------------------
# User Profile
# ----------------------------
@app.post("/user/{username}")
def create_or_update_user(
    username: str, profile: UserProfileUpdate, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    if profile.balance_ngn is not None:
        user.balance_ngn = profile.balance_ngn
    if profile.risk_tolerance is not None:
        user.risk_tolerance = profile.risk_tolerance
    if profile.trading_style is not None:
        user.trading_style = profile.trading_style
    if profile.preferred_pairs is not None:
        user.preferred_pairs = profile.preferred_pairs
    db.commit()
    db.refresh(user)
    return {
        "status": "success",
        "user": {
            "username": user.username,
            "balance_ngn": user.balance_ngn,
            "risk_tolerance": user.risk_tolerance,
            "trading_style": user.trading_style,
            "preferred_pairs": user.preferred_pairs,
            "created_at": user.created_at.isoformat(),
        },
    }


@app.get("/user/{username}")
def get_user_profile(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    wl = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).all()
    wins = [j for j in journal if j.result == "WIN"]
    return {
        "username": user.username,
        "balance_ngn": user.balance_ngn,
        "risk_tolerance": user.risk_tolerance,
        "trading_style": user.trading_style,
        "preferred_pairs": user.preferred_pairs,
        "watchlist": [w.symbol for w in wl],
        "stats": {
            "total_trades": len(journal),
            "wins": len(wins),
            "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(sum(j.pnl_usd for j in journal), 2),
        },
        "last_active": user.last_active.isoformat(),
        "member_since": user.created_at.isoformat(),
    }


# ----------------------------
# Predictions
# ----------------------------
@app.get("/predictions")
def get_predictions_public(
    amount_ngn: float = 0.0,
    username: Optional[str] = None,
    db: Session = Depends(get_db),
):
    profile = ""
    if username:
        user = get_or_create_user(username, db)
        profile = get_user_profile_summary(user, db)
        log_activity(user, "viewed_predictions", db=db)
    return generate_market_predictions(amount_ngn, user_profile=profile)


@app.get("/get_predictions")
def get_personalized_predictions(
    username: str, symbol: str, db: Session = Depends(get_db)
):
    """FIXED: Personalized signal for a specific symbol without needing a CSV file."""
    user = get_or_create_user(username, db)
    log_activity(user, "viewed_signal", symbol=symbol.upper(), db=db)
    signal = get_personalized_signal(symbol, user, db)
    return {
        "username": username,
        "symbol": symbol.upper(),
        "signal": signal,
        "ngn_rate": get_ngn_rate(),
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.get("/market_analysis")
def market_analysis(db: Session = Depends(get_db)):
    preds = generate_market_predictions()
    if not preds.get("success"):
        return {"error": "Analysis failed", "details": preds.get("error")}
    signals = preds.get("signals", [])
    forex = [s for s in signals if s.get("category") == "forex"]
    crypto = [s for s in signals if s.get("category") == "crypto"]
    strong = [s for s in signals if s.get("signal") in ["STRONG_BUY", "STRONG_SELL"]]
    return {
        "market_health": "healthy",
        "active_session": preds.get("active_session"),
        "total_signals": len(signals),
        "forex_signals": len(forex),
        "crypto_signals": len(crypto),
        "strong_signals": len(strong),
        "best_opportunities": strong[:5],
        "usd_ngn_rate": preds.get("usd_ngn_rate"),
        "generated_at": preds.get("generated_at"),
    }


@app.get("/advice/{symbol}")
def get_trading_advice(
    symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)
):
    symbol = symbol.upper()
    now = time.time()
    if symbol in ADVICE_CACHE and (now - ADVICE_CACHE[symbol]["ts"] < 3600):
        return ADVICE_CACHE[symbol]["data"]
    if username:
        user = get_or_create_user(username, db)
        log_activity(user, "viewed_advice", symbol=symbol, db=db)

    # Fetch full live indicator analysis (not just price)
    analysis  = get_symbol_analysis(symbol)
    live_block = format_for_ai_prompt(analysis)
    live_price = analysis.get("live_price", "unavailable")
    conf       = analysis.get("confluence", {})
    has_live   = not analysis.get("ai_only_mode", True)

    mom   = analysis.get("momentum", {})
    mstr  = analysis.get("market_structure", "unknown")

    prompt = f"""━━━ CLEO INSTITUTIONAL DEEP ANALYSIS: {symbol} ━━━

{"LIVE MARKET DATA — use EXACT values below, do NOT invent numbers:" if has_live else "AI REASONING MODE — no live candle feed:"}
{live_block}

You are CLEO, an institutional-grade AI trading analyst. Apply the full 6-step framework:

STEP 1 — TREND: What is the primary trend? Use EMA stack (9/20/50/200), ADX, and market structure ({mstr}).
STEP 2 — MOMENTUM: Score = {mom.get("score", "?")}/10 ({mom.get("label", "?")}). Analyse RSI, MACD histogram, Stochastic crossover.
STEP 3 — CONFIRMATION: Do oscillators confirm the trend? Check Williams %R, Stochastic zone, RSI divergence.
STEP 4 — VOLATILITY: ATR tells you trade size. Bollinger squeeze = breakout pending. Pattern = {analysis.get("indicators", {}).get("candle_pattern", "unknown")}.
STEP 5 — RISK: Use pivot points and Fibonacci levels for precise SL/TP placement. Confluence score = {conf.get("score", 0)}/6.
STEP 6 — CONVICTION: Only if Steps 1–5 align give BUY or SELL. If conflicting, give HOLD with clear reason.

Rules:
- Use EXACT numbers from the live data block above — never round or estimate
- {"Do NOT say you lack data — the full live data is above" if has_live else "State clearly this is AI pattern reasoning without live candles"}
- SL must be beyond the nearest pivot/Fibonacci level, not just ATR-based
- TP1 = 1:1.5 R, TP2 = 1:3 R, TP3 = 1:4.5 R
- Confidence max 95 — never 100

Return ONLY valid JSON (no markdown, no explanation outside the JSON):
{{
  "symbol": "{symbol}",
  "live_price": "{live_price}",
  "data_source": "{"live_data" if has_live else "ai_reasoning"}",
  "momentum_score": "{mom.get("score", "?")} / 10 — {mom.get("label", "?")}",
  "market_structure": "{mstr}",
  "confluence_score": "{conf.get("score", 0)}/6 — {conf.get("strength", "?")}",
  "recommendation": "BUY or SELL or HOLD",
  "confidence": 73,
  "reasoning_summary": {{
    "step1_trend": "What EMA9/20/50/200 and ADX say about trend direction",
    "step2_momentum": "RSI + MACD + Stochastic reading with exact values",
    "step3_confirmation": "Williams %R + oscillator confluence verdict",
    "step4_volatility": "ATR size, Bollinger state, candle pattern signal",
    "step5_risk": "Best SL and TP based on pivots + Fibonacci levels",
    "step6_conviction": "Final verdict — why BUY/SELL/HOLD and confidence level"
  }},
  "trade_setup": {{
    "entry": "exact current price",
    "stop_loss": "level beyond nearest pivot/Fibonacci support or resistance",
    "take_profit_1": "1:1.5 R target",
    "take_profit_2": "1:3 R target",
    "take_profit_3": "1:4.5 R target",
    "risk_reward": "1:3",
    "hold_time": "estimated hours or days",
    "position_size_note": "2% risk per trade recommended"
  }},
  "technical_analysis": {{
    "trend": "bullish / bearish / ranging",
    "market_structure": "HH+HL / LH+LL / ranging",
    "ema9": "exact EMA9 value and relationship to price",
    "ema_stack": "exact EMA20/50/200 alignment",
    "rsi": "exact RSI value and zone (oversold/neutral/overbought)",
    "macd": "exact histogram value and direction",
    "stochastic": "exact K/D values and crossover signal",
    "adx": "exact ADX value and trend strength classification",
    "williams_r": "exact value and zone",
    "bollinger": "band position and squeeze status",
    "fibonacci": "nearest Fibonacci level and its significance",
    "pivot_points": "P / R1 / R2 / S1 / S2 from live data",
    "key_support": "exact support level from data",
    "key_resistance": "exact resistance level from data",
    "candle_pattern": "detected pattern and what it signals",
    "volume": "volume reading and what it confirms"
  }},
  "market_context": "2 sentences: what macro/session/news factors are driving {symbol} right now and what the smart money positioning looks like",
  "risk_assessment": {{
    "risk_level": "low / medium / high",
    "max_position_pct": "2%",
    "invalidation_price": "exact price where thesis is dead",
    "news_risk": "upcoming high-impact events for {symbol} currencies/assets",
    "session_note": "which trading session is active and how it affects liquidity"
  }},
  "rationale": "3 detailed sentences: cite exact indicator values, explain why they align or conflict, state the final edge that justifies this signal"
}}"""
    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        result = {
            "success": True,
            "generated_at": datetime.utcnow().isoformat(),
            "live_data_used": has_live,
            "advice": data,
        }
        ADVICE_CACHE[symbol] = {"data": result, "ts": now}
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════
#  MEMORY HELPERS — CLEO remembers facts about users
# ═══════════════════════════════════════════════════


def get_user_memories(user: User, db: Session) -> str:
    """Return a formatted block of everything CLEO remembers about this user."""
    memories = db.query(UserMemory).filter(UserMemory.user_id == user.id).all()
    if not memories:
        return "No memories yet — learn about the user from this conversation."
    lines = [f"- {m.key}: {m.value}" for m in memories]
    return "\n".join(lines)


def extract_and_save_memories(
    user: User, user_message: str, aria_response: str, db: Session
):
    """
    Ask the AI to extract any personal facts from the exchange and save them.
    Only runs if the message might contain personal info (cheap heuristic).
    """
    keywords = [
        "my name",
        "i am",
        "i'm",
        "i work",
        "i live",
        "i want",
        "i trade",
        "my goal",
        "my balance",
        "i have",
        "call me",
        "know that",
        "remember",
    ]
    if not any(k in user_message.lower() for k in keywords):
        return

    existing = get_user_memories(user, db)
    extract_prompt = f"""You are a memory extraction assistant.
Extract ONLY clear personal facts the user stated about themselves.

USER MESSAGE: {user_message}
CLEO RESPONSE: {aria_response}
ALREADY KNOWN: {existing}

Return a JSON array of new or updated facts ONLY (skip anything already known):
[
  {{"key": "name", "value": "Simeon"}},
  {{"key": "occupation", "value": "software engineer"}},
  {{"key": "trading_goal", "value": "grow ₦500,000 to ₦2,000,000"}},
  {{"key": "location", "value": "Lagos"}},
  {{"key": "preferred_pairs", "value": "BTCUSDT, EURUSD"}},
  {{"key": "experience_level", "value": "beginner"}}
]
Return [] if nothing new was stated. Output ONLY valid JSON. No explanation."""

    try:
        content = get_ai_response(extract_prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\[.*\]", content, re.DOTALL)
        if not m:
            return
        facts = json.loads(m.group(0))
        for fact in facts:
            key = str(fact.get("key", "")).strip().lower().replace(" ", "_")
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            existing_mem = (
                db.query(UserMemory)
                .filter(UserMemory.user_id == user.id, UserMemory.key == key)
                .first()
            )
            if existing_mem:
                existing_mem.value = value
                existing_mem.updated_at = datetime.utcnow()
            else:
                db.add(
                    UserMemory(
                        user_id=user.id, key=key, value=value, source="user_stated"
                    )
                )
        db.commit()

        # Also update display_name if "name" was found
        name_fact = next((f for f in facts if f.get("key") == "name"), None)
        if name_fact:
            user.display_name = name_fact["value"]
            db.commit()
    except Exception as e:
        print(f"[MEMORY] extraction error: {e}")


def auto_title_conversation(first_message: str) -> str:
    """Generate a short title from the first message (max 50 chars)."""
    clean = first_message.strip().replace("\n", " ")
    if len(clean) <= 50:
        return clean
    words = clean.split()
    title = ""
    for w in words:
        if len(title) + len(w) + 1 > 47:
            break
        title = (title + " " + w).strip()
    return title + "..."


# ═══════════════════════════════════════════════════
#  CONVERSATION MANAGEMENT
# ═══════════════════════════════════════════════════


@app.post("/conversations/{username}")
def create_conversation(
    username: str, title: Optional[str] = None, db: Session = Depends(get_db)
):
    """Start a new named conversation thread (like a new chat in ChatGPT sidebar)."""
    user = get_or_create_user(username, db)
    conv = Conversation(user_id=user.id, title=title or "New Chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {
        "conversation_id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
    }


@app.get("/conversations/{username}")
def list_conversations(username: str, db: Session = Depends(get_db)):
    """List all conversations for a user — like the ChatGPT sidebar."""
    user = get_or_create_user(username, db)
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    result = []
    for c in convs:
        last_msg = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.conversation_id == c.id)
            .order_by(DBChatMessage.created_at.desc())
            .first()
        )
        msg_count = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.conversation_id == c.id)
            .count()
        )
        result.append(
            {
                "conversation_id": c.id,
                "title": c.title,
                "message_count": msg_count,
                "last_message": last_msg.content[:80] + "..."
                if last_msg and len(last_msg.content) > 80
                else (last_msg.content if last_msg else ""),
                "last_role": last_msg.role if last_msg else None,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
        )
    return {"username": username, "total_chats": len(result), "conversations": result}


@app.get("/conversations/{username}/{conv_id}")
def get_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Get full message history of a specific conversation."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.conversation_id == conv_id)
        .order_by(DBChatMessage.created_at.asc())
        .all()
    )
    return {
        "conversation_id": conv.id,
        "title": conv.title,
        "username": username,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "time": m.created_at.isoformat(),
            }
            for m in msgs
        ],
        "total_messages": len(msgs),
        "created_at": conv.created_at.isoformat(),
    }


@app.patch("/conversations/{username}/{conv_id}/rename")
def rename_conversation(
    username: str, conv_id: int, title: str, db: Session = Depends(get_db)
):
    """Rename a conversation."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv.title = title
    db.commit()
    return {"status": "success", "conversation_id": conv_id, "new_title": title}


@app.delete("/conversations/{username}/{conv_id}")
def delete_conversation(username: str, conv_id: int, db: Session = Depends(get_db)):
    """Delete a conversation and all its messages."""
    user = get_or_create_user(username, db)
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return {"status": "deleted", "conversation_id": conv_id}


@app.delete("/conversations/{username}")
def delete_all_conversations(username: str, db: Session = Depends(get_db)):
    """Delete ALL conversations for a user."""
    user = get_or_create_user(username, db)
    db.query(Conversation).filter(Conversation.user_id == user.id).delete()
    db.commit()
    return {"status": "success", "message": f"All conversations deleted for {username}"}


# ═══════════════════════════════════════════════════
#  MAIN CHAT ENDPOINT — Full memory + conversation
# ═══════════════════════════════════════════════════


@app.post("/chat")
def chat_with_aria(chat: ChatMessage, db: Session = Depends(get_db)):
    """
    Send a message to CLEO.
    - Pass conversation_id to continue an existing thread.
    - Omit conversation_id to auto-create a new thread.
    CLEO remembers facts about the user across ALL conversations.
    """
    user = get_or_create_user(chat.username, db)
    log_activity(user, "sent_chat", db=db)

    # ── resolve / create conversation ──
    conv_id = getattr(chat, "conversation_id", None)
    conv = None
    if conv_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conv_id, Conversation.user_id == user.id)
            .first()
        )

    if not conv:
        # Auto-create a new conversation
        conv = Conversation(
            user_id=user.id, title=auto_title_conversation(chat.message)
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    # ── load this conversation's history (last 40 messages = 20 turns) ──
    history = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.conversation_id == conv_id)
        .order_by(DBChatMessage.created_at.desc())
        .limit(40)
        .all()
    )
    history = list(reversed(history))

    # Build native Groq messages array (proper multi-turn memory)
    history_messages = []
    for m in history:
        role = "assistant" if m.role in ("aria", "assistant", "cleo") else "user"
        history_messages.append({"role": role, "content": m.content})

    # Compact history text for system prompt reference
    history_text = "\n".join(
        [f"{m.role.upper()}: {m.content[:200]}" for m in history[-6:]]
    ) if history else ""

    # ── load long-term memories ──
    memories = get_user_memories(user, db)
    name = user.display_name or user.username

    # ── market context ──
    ngn_rate = get_ngn_rate()
    preds = generate_market_predictions(user.balance_ngn)
    signals = preds.get("signals", [])
    profile = get_user_profile_summary(user, db)

    # ── detect symbols mentioned in the message and fetch live data ──
    SYMBOL_ALIASES = {
        # Standard slash format
        "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
        "AUD/USD": "AUDUSD", "USD/CAD": "USDCAD", "NZD/USD": "NZDUSD",
        "USD/CHF": "USDCHF", "EUR/GBP": "EURGBP", "EUR/JPY": "EURJPY",
        "GBP/JPY": "GBPJPY", "AUD/JPY": "AUDJPY", "EUR/CAD": "EURCAD",
        "AUD/CAD": "AUDCAD", "EUR/AUD": "EURAUD", "GBP/AUD": "GBPAUD",
        # Reversed slash format (users often type these)
        "USD/EUR": "EURUSD", "USD/GBP": "GBPUSD", "JPY/USD": "USDJPY",
        "CAD/USD": "USDCAD", "CHF/USD": "USDCHF", "GBP/EUR": "EURGBP",
        "JPY/EUR": "EURJPY", "JPY/GBP": "GBPJPY", "JPY/AUD": "AUDJPY",
        "CAD/EUR": "EURCAD", "CAD/AUD": "AUDCAD", "AUD/EUR": "EURAUD",
        "AUD/GBP": "GBPAUD",
        # Dot format
        "EUR.USD": "EURUSD", "GBP.USD": "GBPUSD", "USD.JPY": "USDJPY",
        # Short names / natural language
        "CABLE": "GBPUSD", "FIBER": "EURUSD", "LOONIE": "USDCAD",
        "SWISSY": "USDCHF", "KIWI": "NZDUSD", "AUSSIE": "AUDUSD",
        "EURO": "EURUSD",
        # Crypto names & tickers
        "BITCOIN": "BTCUSDT", "BTC": "BTCUSDT",
        "ETHEREUM": "ETHUSDT", "ETH": "ETHUSDT",
        "BNB": "BNBUSDT", "BINANCE COIN": "BNBUSDT",
        "XRP": "XRPUSDT", "RIPPLE": "XRPUSDT",
        "SOL": "SOLUSDT", "SOLANA": "SOLUSDT",
        "ADA": "ADAUSDT", "CARDANO": "ADAUSDT",
        "DOGE": "DOGEUSDT", "DOGECOIN": "DOGEUSDT",
        "DOT": "DOTUSDT", "POLKADOT": "DOTUSDT",
        "MATIC": "MATICUSDT", "POLYGON": "MATICUSDT",
        "LTC": "LTCUSDT", "LITECOIN": "LTCUSDT",
        "SHIB": "SHIBUSDT", "SHIBA": "SHIBUSDT",
        "TRX": "TRXUSDT", "TRON": "TRXUSDT",
        "AVAX": "AVAXUSDT", "AVALANCHE": "AVAXUSDT",
        "LINK": "LINKUSDT", "CHAINLINK": "LINKUSDT",
        "UNI": "UNIUSDT", "UNISWAP": "UNIUSDT",
    }
    msg_upper = chat.message.upper()
    # ── symbol detection: track both canonical symbol AND the alias the user typed ──
    # mentioned_map: { canonical_symbol -> alias_as_typed_by_user }
    mentioned_map: dict = {}

    # 1. Direct canonical symbol names in the message
    for sym in TRADING_SYMBOLS:
        if sym in msg_upper:
            mentioned_map[sym] = sym
        elif sym.endswith("USDT") and sym[:-4] in msg_upper.split():
            # standalone word match only (avoids "ADA" hitting "CANADA")
            if sym not in mentioned_map:
                mentioned_map[sym] = sym[:-4]

    # 2. Alias map (longer aliases checked first to avoid partial matches)
    for alias in sorted(SYMBOL_ALIASES.keys(), key=len, reverse=True):
        sym = SYMBOL_ALIASES[alias]
        if alias in msg_upper and sym not in mentioned_map:
            mentioned_map[sym] = alias   # remember the exact text the user used

    # limit to 3 symbols per message
    mentioned_pairs = list(mentioned_map.items())[:3]   # [(canonical, alias_used), ...]

    # Fetch live indicator data for each mentioned symbol
    live_data_blocks_chat = []
    for sym, alias_used in mentioned_pairs:
        try:
            analysis = get_symbol_analysis(sym)
            block = format_for_ai_prompt(analysis)
            conf = analysis.get("confluence", {})
            mode = "AI-reasoning — no live candles" if analysis.get("ai_only_mode") else "LIVE candle data"
            # Header clarifies the notation mapping so CLEO never confuses them
            alias_note = f" [user typed: '{alias_used}' — same instrument]" if alias_used != sym else ""
            live_data_blocks_chat.append(
                f"── DATA FOR {sym}{alias_note} ({mode}) ──\n"
                f"NOTE: {sym} = {alias_used} = same tradeable pair. Use the price/indicators below directly.\n"
                f"{block}\n"
                f"CONFLUENCE: {conf.get('direction','?')} | Score {conf.get('score',0)}/6 | "
                f"Tradeable: {conf.get('tradeable','?')}"
            )
        except Exception as ex:
            live_data_blocks_chat.append(f"── {sym}: data fetch error ({ex})")

    live_data_section_chat = "\n\n".join(live_data_blocks_chat) if live_data_blocks_chat else ""

    # Which symbols have injected data (for CLEO's instructions)
    covered_syms = [sym for sym, _ in mentioned_pairs]

    # ── detect currency conversion questions and inject live rates ──
    CONVERSION_KEYWORDS = ["convert", "rate", "exchange", "how much", "worth",
                           "equals", "ngn", "naira", "dollar", "pound", "euro",
                           "yen", "usd", "gbp", "eur", "jpy", "cad", "aud",
                           "zar", "ghs", "kes", "egp", "aed", "inr", "cny"]
    is_conversion_question = any(kw in msg_upper for kw in [k.upper() for k in CONVERSION_KEYWORDS])

    exchange_rates_section = ""
    if is_conversion_question:
        try:
            all_rates = get_all_exchange_rates("USD")
            key_currencies = ["NGN", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF",
                              "NZD", "ZAR", "GHS", "KES", "EGP", "AED", "INR",
                              "CNY", "XAF", "XOF", "MAD", "TZS", "UGX"]
            rate_lines = [f"  1 USD = {all_rates[c]:,.4f} {c}"
                         for c in key_currencies if c in all_rates]
            if rate_lines:
                exchange_rates_section = (
                    "━━━ LIVE EXCHANGE RATES (fetched now) ━━━\n"
                    + "\n".join(rate_lines)
                    + "\n• Use these exact rates for any conversions the user asks about."
                    + "\n• All rates are live and update every hour."
                )
        except Exception:
            pass

    # ── upcoming economic events context ──
    try:
        news_section = get_upcoming_events_text(hours_ahead=6)
    except Exception:
        news_section = ""

    # ── detect balance/budget mentions → inject personalised trading plan ──
    import re as _re
    budget_plan_section = ""
    BUDGET_KEYWORDS = ["have", "naira", "ngn", "₦", "budget", "capital",
                       "balance", "invest", "deposit", "start with", "money",
                       "fund", "save", "dollar", "usd", "$"]
    msg_lower = chat.message.lower()
    has_budget_keyword = any(kw in msg_lower for kw in BUDGET_KEYWORDS)
    if has_budget_keyword:
        # Try to extract a number from the message (could be NGN or USD)
        nums = _re.findall(r"[\d,]+\.?\d*", chat.message.replace(",", ""))
        nums = [float(n) for n in nums if float(n) > 0] if nums else []
        if nums:
            raw_amount = max(nums)   # take the largest number as the balance
            # Heuristic: if amount > 500 and no "$" sign, assume NGN
            if "$" in chat.message or "usd" in msg_lower or "dollar" in msg_lower:
                detected_ngn = raw_amount * ngn_rate
            elif raw_amount > 500 or "ngn" in msg_lower or "naira" in msg_lower or "₦" in msg_lower:
                detected_ngn = raw_amount
            else:
                detected_ngn = raw_amount * ngn_rate  # assume USD for small numbers
            try:
                budget_plan_section = get_budget_trading_plan(detected_ngn, ngn_rate)
            except Exception:
                pass
        elif any(kw in msg_lower for kw in ["what can i do", "how to start",
                                             "how much do i need", "can i trade",
                                             "how do i start", "where do i start"]):
            # General "how do I start" question — use their stored balance if any
            try:
                stored_bal = float(user.balance_ngn or 0)
                if stored_bal > 0:
                    budget_plan_section = get_budget_trading_plan(stored_bal, ngn_rate)
            except Exception:
                pass

    # ── build system prompt ──
    balance_usd  = (user.balance_ngn or 0) / ngn_rate
    risk_ngn_2pct = (user.balance_ngn or 0) * 0.02
    risk_usd_2pct = balance_usd * 0.02

    system_prompt = f"""You are CLEO — Creative Loop Expert Optimizer.
You are a world-class AI trading strategist with a warm, confident personality.

━━━ FOREX NOTATION GUIDE (READ FIRST) ━━━
• EUR/USD = USD/EUR = EURUSD = Fiber — all ONE pair. Price ~1.08
• GBP/USD = USD/GBP = GBPUSD = Cable — all ONE pair. Price ~1.29
• USD/JPY = JPY/USD = USDJPY — ONE pair. Price ~150
• USD/CAD = CAD/USD = USDCAD = Loonie — ONE pair. Price ~1.37
• When the data block below says "EURUSD" and the user wrote "USD/EUR" — it is THE SAME pair.
• NEVER invent a price of ~0.92 for "USD/EUR". That is NOT a real market price.
  EUR/USD is ~1.08 whether written as EUR/USD or USD/EUR.
• NEVER say "I don't have data for this pair" if a data block is shown below for its canonical symbol.

━━━ WHO YOU ARE TALKING TO ━━━
Name: {name}
{profile}
Live USD/NGN Rate: {ngn_rate:.2f}
Time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC

{exchange_rates_section}

{news_section}

{budget_plan_section}

━━━ WHAT YOU REMEMBER ABOUT {name.upper()} ━━━
{memories}

{"━━━ LIVE MARKET DATA FOR: " + ", ".join(covered_syms) + " ━━━" if live_data_section_chat else ""}
{live_data_section_chat}
{("CRITICAL — Data is shown above for " + ", ".join(covered_syms) + ". "
  "Apply your 6-step reasoning framework to this data NOW. "
  "Generate a precise signal. State the data mode (AI-reasoning / live candles).") if live_data_section_chat else ""}

━━━ CURRENT TOP SIGNALS ━━━
{json.dumps([{
    "symbol": s.get("symbol"), "signal": s.get("signal"),
    "confidence": s.get("confidence"), "entry": s.get("entry_price"),
    "sl": s.get("stop_loss"), "tp1": s.get("take_profit_1"),
    "data_source": s.get("data_source"),
} for s in signals[:6]], indent=2)}

━━━ THIS CONVERSATION ━━━
{history_text if history_text else "New conversation."}

━━━ YOUR OPERATING RULES ━━━
1. REASONING FIRST — before answering any market question, silently apply the 6-step framework:
   Trend → Momentum → Confirmation → Volatility → Risk → Conviction.
   Then give your answer with that reasoning embedded naturally in your response.
2. LIVE DATA — if live data is shown above, USE it. Never say "I don't have data" when data is shown.
3. HONEST BUDGET GUIDANCE — when {name} mentions a balance, apply these real rules:
   • Under $5 (₦{5 * ngn_rate:,.0f}): Demo ONLY. Say it directly. 2% = $0.10 — no broker accepts this.
   • $5-$50 (₦{5 * ngn_rate:,.0f}-₦{50 * ngn_rate:,.0f}): Hold crypto as investment only. NO active trading.
   • $50-$100 (₦{50 * ngn_rate:,.0f}-₦{100 * ngn_rate:,.0f}): Crypto real money ✅. Forex demo only.
   • $100+ (₦{100 * ngn_rate:,.0f}+): Forex micro (0.01 lots) now viable. Both markets open.
   ⚠️ NEVER recommend forex to anyone below $100. 0.01 lots with 20-pip SL = $2 risk minimum.
   Always tell them the exact NGN amount they need to reach the next tier.
4. NGN ↔ USD — convert instantly: ₦X = ${{X / {ngn_rate:.2f}:.2f}} USD. Always show both.
5. RISK DISCIPLINE — recommend 2% risk per trade always. For {user.trading_style} / {user.risk_tolerance} risk.
6. MATHS — show exact calculations. Entry ± ATR×1.5 = SL. R×1.5 = TP1. R×2.5 = TP2. R×4 = TP3.
7. NGN-FIRST SIGNALS WITH TIMING — every signal in chat MUST use this exact format.
   Calculate from {name}'s ₦{user.balance_ngn:,.0f} balance at ₦{ngn_rate:.2f}/$ rate:

   ━━━ SIGNAL: [PAIR] — [STRONG BUY / BUY / SELL / STRONG SELL] ━━━
   Chart:          [M15 / H1 / H4 / D1] — set alerts on this timeframe
   Session:        [London / NY Overlap / New York / Asian] — [why this pair is active NOW]
   Entry:          [exact price] | Entry Window: [HH:MM–HH:MM UTC] — enter within this window
   Entry Deadline: [HH:MM UTC] — if price hasn't triggered by then, skip this trade today
   Stop Loss:      [level] — [X pips] | If SL hits → lose ₦{risk_ngn_2pct:,.0f} (${risk_usd_2pct:.2f})
   TP1:            [level] — Expected in [X hours] | Profit → ₦[NGN] (~$[USD])
   TP2:            [level] — Expected in [X hours] | Profit → ₦[NGN] (~$[USD])
   TP3:            [level] — Expected in [X days]  | Profit → ₦[NGN] (~$[USD])
   Time Exit:      [HH:MM UTC] — if TP1 not reached by this time, close at BE or small loss
   Lot Size:       [calculate for 2% risk] lots (forex) OR [units] (crypto)
   NGN/pip:        ₦[lot size × pip value × NGN rate] per pip (forex only)
   Risk:           ₦{risk_ngn_2pct:,.0f} = ${risk_usd_2pct:.2f} USD (2% of account)
   Momentum Score: [X/10] — [label]
   Confluence:     [X/6] — [strength]
   Market Struct:  [UPTREND / DOWNTREND / RANGING]
   Invalidation:   [exact price — if hit, thesis is wrong, exit immediately]
   Data:           [live candles / AI-reasoning]
   ⚠️ Risk: [one-sentence disclaimer]

8. TIMEFRAME RULES — always specify the chart timeframe and session:
   • M15 = scalp, hold 15-60 min, only during London Open (07-09 UTC) or NY Open (13-15 UTC)
   • H1  = intraday, hold 2-8 hours, any active session (DEFAULT for most signals)
   • H4  = swing, hold 8-48 hours, Asian session or low-vol setups
   • D1  = major swing, hold 2-5 days, strong macro trend only
   Entry window always in UTC. Time exit always in UTC. NEVER say "short-term" without a clock time.
9. NEWS AWARENESS — if upcoming events above show 🔴 High-impact news within 2 hours for the pair,
   warn: "⚠️ [Event] in X minutes — I'd wait for the dust to settle before entering."
10. CONTINUOUS IMPROVEMENT — reference the user's trade history, risk tolerance, and past wins/losses
    when giving advice. Tailor every recommendation to {name}'s specific situation.
11. TONE — warm, institutional, direct. Like a senior trader mentoring a student. No hype.
12. NEVER guarantee profits. End every trade recommendation with ⚠️ Risk: [one-sentence disclaimer]."""

    # ── Build the messages array: history + current user message ──
    # This gives CLEO native multi-turn memory rather than text-embedded history.
    chat_messages = history_messages + [{"role": "user", "content": chat.message}]
    response = get_ai_response_chat(chat_messages, system_prompt, max_tokens=2048)

    # ── save messages to DB ──
    db.add(
        DBChatMessage(
            user_id=user.id, conversation_id=conv_id, role="user", content=chat.message
        )
    )
    db.add(
        DBChatMessage(
            user_id=user.id, conversation_id=conv_id, role="aria", content=response
        )
    )

    # ── update conversation timestamp ──
    conv.updated_at = datetime.utcnow()

    # ── auto-title if this is the first message ──
    if conv.title == "New Chat" and not history:
        conv.title = auto_title_conversation(chat.message)

    db.commit()

    # ── extract + save memories in background (non-blocking) ──
    try:
        extract_and_save_memories(user, chat.message, response, db)
    except Exception as e:
        print(f"[MEMORY] non-critical error: {e}")

    return {
        "success": True,
        "username": chat.username,
        "display_name": name,
        "conversation_id": conv_id,
        "conversation_title": conv.title,
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── legacy single-user history endpoint (kept for backwards compat) ──
@app.get("/chat/history/{username}")
def get_chat_history(username: str, limit: int = 30, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    msgs = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.user_id == user.id)
        .order_by(DBChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return {
        "username": username,
        "history": [
            {
                "role": m.role,
                "content": m.content,
                "conversation_id": m.conversation_id,
                "time": m.created_at.isoformat(),
            }
            for m in msgs
        ],
        "total": len(msgs),
    }


@app.delete("/chat/history/{username}")
def clear_chat_history(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    db.query(DBChatMessage).filter(DBChatMessage.user_id == user.id).delete()
    db.commit()
    return {"status": "success", "message": f"All chat history cleared for {username}"}


# ═══════════════════════════════════════════════════
#  MEMORY ENDPOINTS — See / manage what CLEO remembers
# ═══════════════════════════════════════════════════


@app.get("/memory/{username}")
def get_memory(username: str, db: Session = Depends(get_db)):
    """See everything CLEO currently remembers about you."""
    user = get_or_create_user(username, db)
    memories = db.query(UserMemory).filter(UserMemory.user_id == user.id).all()
    return {
        "username": username,
        "display_name": user.display_name,
        "memory_count": len(memories),
        "memories": [
            {
                "key": m.key,
                "value": m.value,
                "source": m.source,
                "updated_at": m.updated_at.isoformat(),
            }
            for m in memories
        ],
        "note": "CLEO uses these facts to personalise every conversation.",
    }


@app.post("/memory/{username}")
def add_memory(username: str, key: str, value: str, db: Session = Depends(get_db)):
    """Manually tell CLEO something to remember about you."""
    user = get_or_create_user(username, db)
    key = key.strip().lower().replace(" ", "_")
    existing = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user.id, UserMemory.key == key)
        .first()
    )
    if existing:
        existing.value = value
        existing.updated_at = datetime.utcnow()
    else:
        db.add(UserMemory(user_id=user.id, key=key, value=value, source="user_stated"))
    if key == "name":
        user.display_name = value
    db.commit()
    return {"status": "saved", "key": key, "value": value}


@app.delete("/memory/{username}/{key}")
def delete_memory(username: str, key: str, db: Session = Depends(get_db)):
    """Tell CLEO to forget a specific fact."""
    user = get_or_create_user(username, db)
    deleted = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user.id, UserMemory.key == key.lower())
        .delete()
    )
    db.commit()
    if deleted:
        return {"status": "forgotten", "key": key}
    raise HTTPException(status_code=404, detail=f"No memory found with key '{key}'")


@app.delete("/memory/{username}")
def clear_all_memory(username: str, db: Session = Depends(get_db)):
    """Clear ALL of CLEO's memories about this user."""
    user = get_or_create_user(username, db)
    db.query(UserMemory).filter(UserMemory.user_id == user.id).delete()
    user.display_name = None
    db.commit()
    return {"status": "success", "message": f"All memories cleared for {username}"}


# ----------------------------
# Risk Calculator
# ----------------------------
@app.get("/exchange_rates")
def exchange_rates_endpoint(base: str = "USD"):
    """
    Get live exchange rates for any base currency.
    Example: /exchange_rates?base=USD  → returns all currencies vs USD
    Example: /exchange_rates?base=EUR  → returns all currencies vs EUR
    Free, no API key, updates every hour.
    """
    rates = get_all_exchange_rates(base.upper())
    if not rates:
        raise HTTPException(status_code=503, detail="Exchange rate data unavailable")

    # Return the most useful currencies prominently
    highlight = ["NGN", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF",
                 "NZD", "ZAR", "GHS", "KES", "EGP", "INR", "CNY", "AED"]
    highlighted = {k: rates[k] for k in highlight if k in rates}

    cache_key = f"rates_{base.upper()}"
    last_updated = _RATES_CACHE_TIMES.get(cache_key)
    last_updated_str = (
        datetime.utcfromtimestamp(last_updated).strftime("%Y-%m-%d %H:%M UTC")
        if last_updated else "just now"
    )

    return {
        "base": base.upper(),
        "rates": rates,
        "highlighted": highlighted,
        "total_currencies": len(rates),
        "last_updated": last_updated_str,
        "next_update_in_minutes": max(0, round((_RATES_CACHE_TIMES.get(cache_key, 0) + _RATES_CACHE_TTL - time.time()) / 60, 1)),
        "source": "ExchangeRate-API v6 (live)",
    }


@app.get("/convert")
def convert_endpoint(amount: float, from_currency: str, to_currency: str):
    """
    Convert any amount between any two currencies using live rates.
    Example: /convert?amount=100&from_currency=USD&to_currency=NGN
    Example: /convert?amount=50000&from_currency=NGN&to_currency=EUR
    Supports 160+ currencies worldwide.
    """
    result = convert_currency(amount, from_currency, to_currency)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/economic_calendar")
def economic_calendar_endpoint(
    impact: str = "all",
    symbol: str = None,
    hours_ahead: float = 24,
):
    """
    Return upcoming economic events from ForexFactory (free feed, updated every 30 min).
    - impact: 'high', 'medium', 'low', or 'all' (default: all)
    - symbol: optional — filter to events relevant to this trading pair (e.g. GBPUSD)
    - hours_ahead: how many hours ahead to look (default 24)

    Example: /economic_calendar?impact=high&symbol=GBPUSD&hours_ahead=6
    """
    from datetime import timezone as _tz
    events = get_economic_calendar()
    now_utc = datetime.utcnow()

    currencies_filter = None
    if symbol:
        currencies_filter = [c.upper() for c in SYMBOL_CURRENCIES.get(symbol.upper(), [])]

    result = []
    for event in events:
        ev_raw_cur  = (event.get("currency") or "")
        ev_impact   = (event.get("impact")   or "").lower()
        ev_date     = (event.get("date")     or "")
        ev_title    = (event.get("title")    or "")
        ev_forecast = event.get("forecast", "")
        ev_previous = event.get("previous", "")
        ev_actual   = event.get("actual", "")
        ev_currency = _infer_currency(ev_title, ev_raw_cur)

        if impact != "all" and ev_impact != impact.lower():
            continue
        if currencies_filter and ev_currency not in currencies_filter:
            continue

        try:
            event_dt = datetime.fromisoformat(ev_date)
            if event_dt.tzinfo is not None:
                event_utc = event_dt.astimezone(_tz.utc).replace(tzinfo=None)
            else:
                event_utc = event_dt

            mins_until = (event_utc - now_utc).total_seconds() / 60
            if mins_until < -(60 * 24):  # skip events more than 24h in the past
                continue
            if mins_until > hours_ahead * 60:
                continue

            result.append({
                "title":    ev_title,
                "currency": ev_currency,
                "impact":   ev_impact.capitalize(),
                "time_utc": event_utc.strftime("%Y-%m-%d %H:%M UTC"),
                "mins_until": round(mins_until),
                "status": (
                    "upcoming" if mins_until > 0
                    else "in_progress" if mins_until > -30
                    else "released"
                ),
                "forecast": ev_forecast,
                "previous": ev_previous,
                "actual":   ev_actual,
            })
        except Exception:
            continue

    result.sort(key=lambda x: x["mins_until"])

    high_count   = sum(1 for e in result if e["impact"] == "High")
    medium_count = sum(1 for e in result if e["impact"] == "Medium")

    return {
        "events": result,
        "total": len(result),
        "high_impact": high_count,
        "medium_impact": medium_count,
        "symbol_filter": symbol,
        "hours_ahead": hours_ahead,
        "source": "ForexFactory (free feed)",
        "cache_age_mins": round((time.time() - _CALENDAR_CACHE_TIME) / 60, 1) if _CALENDAR_CACHE_TIME else None,
    }


@app.post("/risk_calculator")
def calculate_risk(req: RiskCalcRequest):
    ngn_rate = get_ngn_rate()
    balance_usd = req.balance_ngn / ngn_rate
    risk_ngn = req.balance_ngn * (req.risk_percent / 100)
    risk_usd = risk_ngn / ngn_rate
    dist = abs(req.entry_price - req.stop_loss_price)
    pct_dist = dist / req.entry_price if req.entry_price > 0 else 0
    pos_usd = risk_usd / pct_dist if pct_dist > 0 else 0
    units = pos_usd / req.entry_price if req.entry_price > 0 else 0
    is_crypto = "USDT" in req.symbol.upper()
    pip_dist = dist if is_crypto else dist * 10000

    return {
        "symbol": req.symbol.upper(),
        "balance_ngn": req.balance_ngn,
        "balance_usd": round(balance_usd, 2),
        "risk_percent": req.risk_percent,
        "risk_amount_ngn": round(risk_ngn, 2),
        "risk_amount_usd": round(risk_usd, 4),
        "entry_price": req.entry_price,
        "stop_loss_price": req.stop_loss_price,
        "stop_distance_pct": round(pct_dist * 100, 4),
        "recommended_position_usd": round(pos_usd, 2),
        "units": round(units, 6),
        "pip_distance": round(pip_dist, 1) if not is_crypto else None,
        "note": f"Risking {req.risk_percent}% of your NGN balance per this trade.",
    }


# ----------------------------
# Trade Idea Scorer
# ----------------------------
@app.post("/score_trade")
def score_trade_idea(idea: TradeIdeaRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(idea.username, db)
    log_activity(user, "scored_trade", symbol=idea.symbol, db=db)
    dist_sl = abs(idea.entry - idea.stop_loss)
    dist_tp = abs(idea.take_profit - idea.entry)
    rr = dist_tp / dist_sl if dist_sl > 0 else 0
    profile = get_user_profile_summary(user, db)

    prompt = f"""Elite institutional trade reviewer. Evaluate this trade idea for user {idea.username}.
USER PROFILE: {profile}

TRADE:
Symbol: {idea.symbol.upper()} | Direction: {idea.direction.upper()}
Entry: {idea.entry} | Stop Loss: {idea.stop_loss} | Take Profit: {idea.take_profit}
Calculated R/R: 1:{round(rr, 2)} | Rationale: {idea.rationale}

Return ONLY valid JSON:
{{
  "score": 0-100, "grade": "A+|A|B|C|D|F", "verdict": "TAKE IT|RISKY BUT CONSIDER|AVOID",
  "rr_ratio": "1:{round(rr, 2)}", "rr_assessment": "excellent(>1:3)|good(1:2-3)|weak(<1:2)",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "improvements": ["improvement1", "improvement2"],
  "adjusted_entry": "better entry if applicable",
  "adjusted_sl": "better SL if applicable",
  "adjusted_tp": "better TP if applicable",
  "personalized_feedback": "feedback specific to this user's style and history",
  "summary": "2-sentence professional verdict"
}}"""
    try:
        content = get_ai_response(prompt)
        content = re.sub(r"```json|```", "", content.strip()).strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            content = m.group(0)
        data = json.loads(content)
        return {"success": True, "trade_idea": idea.dict(), "analysis": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# Watchlist — DB persistent
# ----------------------------
@app.post("/watchlist/{username}")
def add_to_watchlist(username: str, symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.upper()
    if symbol not in TRADING_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"{symbol} not supported")
    user = get_or_create_user(username, db)
    exists = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol)
        .first()
    )
    if not exists:
        db.add(WatchlistItem(user_id=user.id, symbol=symbol))
        db.commit()
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    return {"status": "success", "watchlist": [w.symbol for w in items]}


@app.delete("/watchlist/{username}/{symbol}")
def remove_from_watchlist(username: str, symbol: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user.id, WatchlistItem.symbol == symbol.upper()
    ).delete()
    db.commit()
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    return {"status": "success", "watchlist": [w.symbol for w in items]}


@app.get("/watchlist/{username}")
def get_watchlist(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    items = db.query(WatchlistItem).filter(WatchlistItem.user_id == user.id).all()
    preds = generate_market_predictions(user.balance_ngn)
    sig_map = {s.get("symbol"): s for s in preds.get("signals", [])}
    result = []
    for w in items:
        sig = sig_map.get(w.symbol, {})
        price = get_live_price(w.symbol)
        result.append(
            {
                "symbol": w.symbol,
                "live_price": price,
                "signal": sig.get("signal", "N/A"),
                "confidence": sig.get("confidence", "N/A"),
                "entry_price": sig.get("entry_price", "N/A"),
                "stop_loss": sig.get("stop_loss", "N/A"),
                "hold_time": sig.get("hold_time", "N/A"),
                "added_at": w.added_at.isoformat(),
            }
        )
    return {"username": username, "watchlist": result, "total": len(result)}


# ----------------------------
# Trade Journal — DB persistent
# ----------------------------
@app.post("/journal/{username}")
def add_journal_entry(
    username: str, entry: TradeJournalEntryRequest, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    record = DBJournalEntry(
        user_id=user.id,
        symbol=entry.symbol.upper(),
        direction=entry.direction.upper(),
        entry_price=entry.entry_price,
        exit_price=entry.exit_price,
        volume=entry.volume,
        result=entry.result.upper(),
        pnl_usd=entry.pnl_usd,
        notes=entry.notes,
    )
    db.add(record)
    log_activity(
        user,
        "logged_trade",
        symbol=entry.symbol.upper(),
        details={"result": entry.result, "pnl": entry.pnl_usd},
        db=db,
    )
    db.commit()
    db.refresh(record)
    return {
        "status": "success",
        "entry_id": record.id,
        "logged_at": record.logged_at.isoformat(),
    }


@app.get("/journal/{username}")
def get_journal(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    entries = (
        db.query(DBJournalEntry)
        .filter(DBJournalEntry.user_id == user.id)
        .order_by(DBJournalEntry.logged_at.desc())
        .all()
    )
    if not entries:
        return {"username": username, "entries": [], "stats": {}}
    wins = [e for e in entries if e.result == "WIN"]
    losses = [e for e in entries if e.result == "LOSS"]
    total_pnl = sum(e.pnl_usd for e in entries)
    sym_freq = {}
    for e in entries:
        sym_freq[e.symbol] = sym_freq.get(e.symbol, 0) + 1
    best = max(entries, key=lambda e: e.pnl_usd)
    worst = min(entries, key=lambda e: e.pnl_usd)

    return {
        "username": username,
        "entries": [
            {
                "id": e.id,
                "symbol": e.symbol,
                "direction": e.direction,
                "entry_price": e.entry_price,
                "exit_price": e.exit_price,
                "volume": e.volume,
                "result": e.result,
                "pnl_usd": e.pnl_usd,
                "notes": e.notes,
                "logged_at": e.logged_at.isoformat(),
            }
            for e in entries
        ],
        "stats": {
            "total_trades": len(entries),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(entries) * 100, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(sum(e.pnl_usd for e in wins) / len(wins), 2)
            if wins
            else 0,
            "avg_loss_usd": round(sum(e.pnl_usd for e in losses) / len(losses), 2)
            if losses
            else 0,
            "most_traded_symbol": max(sym_freq, key=sym_freq.get),
            "best_trade": {"symbol": best.symbol, "pnl_usd": best.pnl_usd},
            "worst_trade": {"symbol": worst.symbol, "pnl_usd": worst.pnl_usd},
        },
    }


# ----------------------------
# Demo Account — DB persistent
# ----------------------------
@app.post("/demo/open_account/{username}")
def open_demo_account(
    username: str, initial_balance: float = 10000.0, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    existing = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if existing:
        return {
            "status": "exists",
            "message": f"Demo account already exists. Balance: ${existing.balance:,.2f}",
            "balance": existing.balance,
        }
    acct = DemoAccount(user_id=user.id, balance=initial_balance)
    db.add(acct)
    db.commit()
    db.refresh(acct)
    return {
        "status": "success",
        "message": f"Demo account opened for {username}",
        "balance": acct.balance,
        "account_id": acct.id,
    }


@app.post("/demo/execute_trade/{username}")
def execute_demo_trade(
    username: str, req: DemoTradeRequest, db: Session = Depends(get_db)
):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        acct = DemoAccount(user_id=user.id, balance=10000.0)
        db.add(acct)
        db.commit()
        db.refresh(acct)

    symbol = req.symbol.upper()
    entry_price = get_live_price(symbol)
    if entry_price == 0.0:
        raise HTTPException(
            status_code=400, detail=f"Could not fetch live price for {symbol}"
        )

    trade = DemoTrade(
        demo_account_id=acct.id,
        symbol=symbol,
        entry_price=entry_price,
        current_price=entry_price,
        volume=req.volume,
        trade_type=req.trade_type.upper(),
    )
    db.add(trade)
    log_activity(
        user,
        "opened_demo_trade",
        symbol=symbol,
        details={"entry": entry_price, "volume": req.volume},
        db=db,
    )
    db.commit()
    db.refresh(trade)
    return {
        "status": "success",
        "trade_id": trade.id,
        "symbol": symbol,
        "entry_price": entry_price,
        "volume": req.volume,
        "type": trade.trade_type,
    }


@app.post("/demo/close_trade/{username}/{trade_id}")
def close_demo_trade(username: str, trade_id: int, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        raise HTTPException(status_code=404, detail="No demo account found")
    trade = (
        db.query(DemoTrade)
        .filter(DemoTrade.id == trade_id, DemoTrade.demo_account_id == acct.id)
        .first()
    )
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    exit_price = get_live_price(trade.symbol)
    if exit_price == 0:
        exit_price = trade.current_price
    pnl = (
        (exit_price - trade.entry_price) * trade.volume
        if trade.trade_type == "BUY"
        else (trade.entry_price - exit_price) * trade.volume
    )
    trade.exit_price = exit_price
    trade.current_price = exit_price
    trade.pnl = pnl
    trade.is_active = False
    trade.closed_at = datetime.utcnow()
    acct.balance += pnl
    db.commit()
    return {
        "status": "closed",
        "symbol": trade.symbol,
        "entry": trade.entry_price,
        "exit": exit_price,
        "pnl_usd": round(pnl, 2),
        "new_balance": round(acct.balance, 2),
    }


@app.get("/demo/account/{username}")
def get_demo_account(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {
            "error": "No demo account found. POST /demo/open_account/{username} to create one."
        }

    active = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True)
        .all()
    )
    closed = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == False)
        .all()
    )

    # Refresh PnL for active trades
    total_floating_pnl = 0
    active_list = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (
            (price - t.entry_price) * t.volume
            if t.trade_type == "BUY"
            else (t.entry_price - price) * t.volume
        )
        t.current_price = price
        t.pnl = pnl
        total_floating_pnl += pnl
        active_list.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "type": t.trade_type,
                "entry_price": t.entry_price,
                "current_price": price,
                "volume": t.volume,
                "pnl_usd": round(pnl, 2),
                "opened_at": t.opened_at.isoformat(),
            }
        )
    db.commit()

    wins = [t for t in closed if t.pnl > 0]
    return {
        "username": username,
        "balance_usd": round(acct.balance, 2),
        "floating_pnl_usd": round(total_floating_pnl, 2),
        "equity_usd": round(acct.balance + total_floating_pnl, 2),
        "active_trades": active_list,
        "trade_history": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "type": t.trade_type,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl_usd": round(t.pnl, 2),
                "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            }
            for t in closed[-20:]
        ],
        "stats": {
            "total_closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "realized_pnl_usd": round(sum(t.pnl for t in closed), 2),
        },
    }


@app.get("/demo/ai_feedback/{username}")
def get_demo_ai_feedback(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    if not acct:
        return {"error": "No demo account found"}
    active = (
        db.query(DemoTrade)
        .filter(DemoTrade.demo_account_id == acct.id, DemoTrade.is_active == True)
        .all()
    )
    if not active:
        return {"error": "No active trades to analyze"}

    summary = []
    for t in active:
        price = get_live_price(t.symbol) or t.current_price
        pnl = (
            (price - t.entry_price) * t.volume
            if t.trade_type == "BUY"
            else (t.entry_price - price) * t.volume
        )
        summary.append(
            f"{t.trade_type} {t.volume} {t.symbol} @ {t.entry_price} | Now: {price} | PnL: ${pnl:.2f}"
        )

    profile = get_user_profile_summary(user, db)
    prompt = f"""Elite trading risk manager reviewing DEMO TRADES.
USER: {profile}
ACTIVE TRADES: {json.dumps(summary)}
DEMO BALANCE: ${acct.balance:,.2f}

Provide a direct, calculative review:
1. Which trades are profitable and which are at risk?
2. Should any trade be closed NOW? Justify with specific levels.
3. Adjust SL/TP for active trades with exact new levels.
4. Is the overall portfolio over-leveraged or well-managed?
5. One specific improvement for this trader's style."""

    feedback = get_ai_response(prompt)
    return {
        "username": username,
        "balance": acct.balance,
        "active_trades": summary,
        "ai_strategic_advice": feedback,
    }


# ----------------------------
# Account Monitor
# ----------------------------
@app.get("/account/monitor/{username}")
def monitor_account(username: str, db: Session = Depends(get_db)):
    user = get_or_create_user(username, db)
    ngn_rate = get_ngn_rate()
    demo_acct = db.query(DemoAccount).filter(DemoAccount.user_id == user.id).first()
    journal = db.query(DBJournalEntry).filter(DBJournalEntry.user_id == user.id).all()
    wins = [j for j in journal if j.result == "WIN"]
    total_pnl = sum(j.pnl_usd for j in journal)

    return {
        "username": username,
        "balance_ngn": user.balance_ngn,
        "balance_usd": round(user.balance_ngn / ngn_rate, 2),
        "usd_ngn_rate": ngn_rate,
        "demo_account": {
            "balance_usd": demo_acct.balance if demo_acct else None,
            "exists": demo_acct is not None,
        },
        "trading_stats": {
            "total_trades": len(journal),
            "wins": len(wins),
            "losses": len(journal) - len(wins),
            "win_rate_pct": round(len(wins) / len(journal) * 100, 1) if journal else 0,
            "total_pnl_usd": round(total_pnl, 2),
            "total_pnl_ngn": round(total_pnl * ngn_rate, 2),
        },
        "profile": {
            "risk_tolerance": user.risk_tolerance,
            "trading_style": user.trading_style,
            "preferred_pairs": user.preferred_pairs,
            "member_since": user.created_at.isoformat(),
        },
    }


# ----------------------------
# Simulator
# ----------------------------
@app.get("/live_data/{symbol}")
def get_live_technical_data(
    symbol: str, username: Optional[str] = None, db: Session = Depends(get_db)
):
    """Returns real live OHLCV-based technical indicators for any symbol."""
    symbol = symbol.upper()
    if username:
        user = get_or_create_user(username, db)
        log_activity(user, "viewed_live_data", symbol=symbol, db=db)
    analysis = get_symbol_analysis(symbol)
    return {
        "symbol": symbol,
        "success": bool(analysis.get("live_price")),
        "data_source": analysis.get("data_source", "unavailable"),
        "live_price": analysis.get("live_price"),
        "last_candle_time": analysis.get("last_candle_time"),
        "candles_used": analysis.get("candles_used"),
        "confluence": analysis.get("confluence", {}),
        "indicators": analysis.get("indicators", {}),
        "key_levels": analysis.get("key_levels", {}),
        "trend_bias": analysis.get("trend_bias"),
        "fetched_at": datetime.utcnow().isoformat(),
        "cache_ttl_minutes": 15,
        "note": "Data from Alpha Vantage daily candles. Cached 15 min to respect rate limits.",
    }


@app.get("/simulator/candles/{symbol}")
def get_simulator_candles(
    symbol: str, candles: int = 120, interval: int = 15, seed: int = None
):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(
        symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed
    )
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "total_candles": len(candle_data),
        "chart_type": "candlestick",
        "candles": candle_data,
        "meta": {
            "base_price": SYMBOL_BASE_PRICES.get(symbol),
            "note": "Simulated via GBM — demo only.",
        },
    }


@app.get("/simulator/run/{symbol}")
def run_simulator(
    symbol: str,
    candles: int = 150,
    interval: int = 15,
    balance: float = 10000.0,
    seed: int = None,
):
    symbol = symbol.upper()
    candle_data = simulate_ohlcv(
        symbol, num_candles=min(candles, 300), interval_minutes=interval, seed=seed
    )
    results = run_strategy_on_candles(candle_data, initial_balance=balance)
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    entry_markers = [
        {
            "time": t["entry_time"],
            "datetime": t["entry_datetime"],
            "price": t["entry_price"],
            "label": "BUY",
            "color": "#00ff88",
        }
        for t in results["trades"]
    ]
    exit_markers = [
        {
            "time": t["exit_time"],
            "datetime": t["exit_datetime"],
            "price": t["exit_price"],
            "label": t["result"],
            "color": "#00bfff" if t["result"] == "WIN" else "#ff4444",
        }
        for t in results["trades"]
    ]
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "strategy": results["strategy"],
        "performance": {
            "initial_balance_usd": results["initial_balance"],
            "final_balance_usd": results["final_balance"],
            "total_pnl_usd": results["total_pnl_usd"],
            "total_pnl_pct": results["total_pnl_pct"],
            "total_trades": results["total_trades"],
            "wins": results["wins"],
            "losses": results["losses"],
            "win_rate_pct": results["win_rate_pct"],
        },
        "chart_data": {
            "candles": candle_data,
            "equity_curve": results["equity_curve"],
            "entry_markers": entry_markers,
            "exit_markers": exit_markers,
        },
        "trades": results["trades"],
        "disclaimer": "Simulated results. Past performance is not indicative of future results.",
    }


# ----------------------------
# Connect Account
# ----------------------------
@app.post("/connect_account")
def connect_account(req: UserAccountRequest, db: Session = Depends(get_db)):
    user = get_or_create_user(req.username, db)
    return {
        "status": "success",
        "message": f"Account connected for {req.username}",
        "user_id": user.id,
    }


# ════════════════════════════════════════════════════════════════
#  CLEO AUTO-TRADER  —  Bot Config · Risk Engine · Market Filter
#                        Trade Manager · MT5 Bridge endpoints
# ════════════════════════════════════════════════════════════════

import secrets as _secrets


# ── Pydantic models ──────────────────────────────────────────────
class BotConfigRequest(BaseModel):
    risk_percent: Optional[float] = 1.0
    max_concurrent_trades: Optional[int] = 3
    max_daily_loss_pct: Optional[float] = 5.0
    max_weekly_loss_pct: Optional[float] = 10.0
    max_lot_size: Optional[float] = 1.0
    min_lot_size: Optional[float] = 0.01
    min_confidence: Optional[float] = 75.0
    max_spread_pips: Optional[float] = 3.0
    avoid_news_minutes: Optional[int] = 30
    min_atr_percentile: Optional[float] = 30.0
    allowed_sessions: Optional[List[str]] = None
    allowed_pairs: Optional[List[str]] = None
    use_break_even: Optional[bool] = True
    break_even_trigger_rr: Optional[float] = 1.0
    use_trailing_stop: Optional[bool] = True
    trail_trigger_rr: Optional[float] = 1.5
    trail_step_atr: Optional[float] = 0.5
    use_partial_close: Optional[bool] = True
    partial_close_pct: Optional[float] = 50.0
    max_hold_hours: Optional[int] = 48
    mt5_account_number: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_broker: Optional[str] = None


class ManualSignalRequest(BaseModel):
    symbol: str
    direction: str  # BUY | SELL
    entry_price: Optional[float] = None
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    account_balance_usd: float
    signal_confidence: Optional[float] = 80.0
    timeframe: Optional[str] = "H1"


class BridgeExecutedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    filled_price: float
    spread_pips: Optional[float] = None
    executed_at: Optional[str] = None


class BridgeRejectedRequest(BaseModel):
    order_id: int
    reason: str
    rejected_at: Optional[str] = None


class PositionUpdateItem(BaseModel):
    mt5_ticket: int
    symbol: str
    current_price: float
    floating_pnl: float
    volume: float
    current_sl: Optional[float] = None
    current_tp: Optional[float] = None


class PositionUpdateRequest(BaseModel):
    positions: List[PositionUpdateItem]
    account: Optional[Dict[str, Any]] = None


class PartialClosedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    close_price: Optional[float] = None
    success: bool


class ClosedRequest(BaseModel):
    order_id: int
    mt5_ticket: int
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    closed_at: Optional[str] = None


class BridgeConnectRequest(BaseModel):
    account_info: Optional[Dict[str, Any]] = None
    bridge_version: Optional[str] = "1.0"
    bridge_key: Optional[str] = None
    mt5_account: Optional[int] = None
    mt5_server: Optional[str] = None
    balance: Optional[float] = None
    equity: Optional[float] = None
    currency: Optional[str] = "USD"


# ── Helper: verify bridge key ─────────────────────────────────────
def _verify_bridge_key(username: str, provided_key: str, db: Session) -> BotConfig:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Bot not configured for this user")
    if cfg.bridge_api_key and cfg.bridge_api_key != provided_key:
        raise HTTPException(status_code=403, detail="Invalid bridge API key")
    return cfg


# ════════════════════════════════════════════════════════════════
#  1. BOT CONFIGURATION
# ════════════════════════════════════════════════════════════════


@app.post("/bot/configure/{username}")
def configure_bot(username: str, req: BotConfigRequest, db: Session = Depends(get_db)):
    """
    Create or update the trading bot configuration for a user.
    Also generates a unique bridge API key if one doesn't exist yet.
    """
    user = get_or_create_user(username, db)
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()

    if not cfg:
        cfg = BotConfig(user_id=user.id, bridge_api_key=_secrets.token_urlsafe(32))
        db.add(cfg)

    fields = req.dict(exclude_none=True)
    for k, v in fields.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if cfg.allowed_sessions is None:
        cfg.allowed_sessions = ["london", "overlap", "newyork"]

    cfg.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(cfg)

    return {
        "success": True,
        "message": "Bot configured successfully",
        "bridge_api_key": cfg.bridge_api_key,
        "note": "Keep your bridge_api_key secret — paste it in the .env of your mt5_bridge.py",
        "config": {
            "is_active": cfg.is_active,
            "risk_percent": cfg.risk_percent,
            "max_concurrent_trades": cfg.max_concurrent_trades,
            "max_daily_loss_pct": cfg.max_daily_loss_pct,
            "max_weekly_loss_pct": cfg.max_weekly_loss_pct,
            "min_confidence": cfg.min_confidence,
            "max_spread_pips": cfg.max_spread_pips,
            "allowed_sessions": cfg.allowed_sessions,
            "allowed_pairs": cfg.allowed_pairs or "All 30 pairs",
            "use_trailing_stop": cfg.use_trailing_stop,
            "use_break_even": cfg.use_break_even,
            "use_partial_close": cfg.use_partial_close,
            "mt5_account_number": cfg.mt5_account_number,
            "mt5_server": cfg.mt5_server,
        },
    }


@app.post("/bot/start/{username}")
def start_bot(username: str, db: Session = Depends(get_db)):
    """Activate the bot — it will start processing signals."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail="Configure the bot first via POST /bot/configure/{username}",
        )
    cfg.is_active = True
    cfg.updated_at = datetime.utcnow()
    db.commit()
    return {
        "success": True,
        "message": "CLEO bot is now ACTIVE — monitoring markets 24/7",
    }


@app.post("/bot/stop/{username}")
def stop_bot(username: str, db: Session = Depends(get_db)):
    """Pause the bot — no new orders will be queued. Open trades are unaffected."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=404, detail="Bot not configured")
    cfg.is_active = False
    cfg.updated_at = datetime.utcnow()
    db.commit()
    return {
        "success": True,
        "message": "CLEO bot PAUSED — no new orders will be placed",
    }


@app.get("/bot/status/{username}")
def bot_status(username: str, db: Session = Depends(get_db)):
    """Full bot status: config, active orders, today's P&L."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        return {
            "configured": False,
            "message": "Bot not configured. Call POST /bot/configure/{username} first.",
        }

    active_orders = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status.in_(["QUEUED", "SENT", "EXECUTED", "ACTIVE"]),
        )
        .all()
    )

    today = datetime.utcnow().date()
    closed_today = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
            BotOrder.closed_at >= datetime.combine(today, datetime.min.time()),
        )
        .all()
    )

    daily_pnl = sum(o.realised_pnl_usd or 0 for o in closed_today)
    total_closed = (
        db.query(BotOrder)
        .filter(BotOrder.user_id == user.id, BotOrder.status == "CLOSED")
        .count()
    )
    total_wins = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
            BotOrder.realised_pnl_usd > 0,
        )
        .count()
    )
    win_rate = round(total_wins / total_closed * 100, 1) if total_closed > 0 else 0.0

    return {
        "configured": True,
        "is_active": cfg.is_active,
        "status_label": "🟢 ACTIVE" if cfg.is_active else "⏸ PAUSED",
        "config": {
            "risk_percent": cfg.risk_percent,
            "max_concurrent_trades": cfg.max_concurrent_trades,
            "max_daily_loss_pct": cfg.max_daily_loss_pct,
            "min_confidence": cfg.min_confidence,
            "max_spread_pips": cfg.max_spread_pips,
            "allowed_sessions": cfg.allowed_sessions,
            "use_trailing_stop": cfg.use_trailing_stop,
            "use_break_even": cfg.use_break_even,
        },
        "active_orders": [
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry": o.filled_price or o.requested_entry,
                "sl": o.current_sl or o.stop_loss,
                "tp1": o.take_profit_1,
                "status": o.status,
                "floating_pnl": o.floating_pnl_usd,
                "ticket": o.mt5_ticket,
            }
            for o in active_orders
        ],
        "todays_performance": {
            "trades_closed": len(closed_today),
            "realised_pnl": round(daily_pnl, 2),
            "pnl_sign": "+" if daily_pnl >= 0 else "",
        },
        "all_time_performance": {
            "total_closed": total_closed,
            "wins": total_wins,
            "losses": total_closed - total_wins,
            "win_rate_pct": win_rate,
        },
    }


# ════════════════════════════════════════════════════════════════
#  2. PLACE A TRADE  (passes through Risk Engine + Market Filter)
# ════════════════════════════════════════════════════════════════


@app.post("/bot/trade/{username}")
def place_bot_trade(
    username: str, req: ManualSignalRequest, db: Session = Depends(get_db)
):
    """
    Queue a trade after passing it through:
      1. Market Filter  — spread, session, news, volatility, confidence
      2. Risk Engine    — lot size, daily loss limit, concurrent position cap
    If everything passes, an order is written to DB with status=QUEUED.
    The MT5 bridge will pick it up within seconds.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=400, detail="Configure bot first")
    if not cfg.is_active:
        raise HTTPException(
            status_code=400, detail="Bot is paused. Call /bot/start/{username} first."
        )

    symbol = req.symbol.upper()
    direction = req.direction.upper()
    confidence = req.signal_confidence or 80.0

    # ── Get live market data for filter checks ──────────────────
    analysis = get_symbol_analysis(symbol)
    indicators = analysis.get("indicators", {})
    atr = indicators.get("atr")
    atr_avg = indicators.get("atr_avg") or (atr * 1.2 if atr else None)

    # ── 1. Market Filter ────────────────────────────────────────
    filter_result = MarketFilter.run_all_checks(
        symbol=symbol,
        signal_confidence=confidence,
        config=cfg,
        current_spread_pips=None,  # bridge will re-check live spread at execution
        atr=atr,
        atr_avg=atr_avg,
    )

    if not filter_result["approved"]:
        return {
            "success": False,
            "approved": False,
            "blocks": filter_result["blocks"],
            "warnings": filter_result.get("warnings", []),
            "message": "Trade BLOCKED by Market Filter — no order queued",
        }

    # ── 1b. News Blackout Filter ──────────────────────────────────
    news_check = check_news_blackout(symbol)
    if news_check["blocked"]:
        print(f"[NEWS-FILTER] {symbol} BLOCKED — {news_check['message']}")
        return {
            "success": False,
            "approved": False,
            "blocks": [news_check["message"]],
            "news_events": news_check["events"],
            "message": f"Trade BLOCKED by News Filter — {news_check['message']}",
        }
    warnings = filter_result.get("warnings", [])
    if news_check["warning"]:
        warnings.append(news_check["message"])

    # ── 2. Risk Engine — daily loss check ───────────────────────
    loss_check = RiskEngine.check_daily_loss(user.id, cfg, db)
    if not loss_check["allowed"]:
        return {
            "success": False,
            "approved": False,
            "blocks": [loss_check["reason"]],
            "message": "Trade BLOCKED by Risk Engine — daily/weekly loss limit reached",
        }

    # ── 3. Risk Engine — concurrent positions check ─────────────
    conc_check = RiskEngine.check_concurrent_positions(user.id, symbol, cfg, db)
    if not conc_check["allowed"]:
        return {
            "success": False,
            "approved": False,
            "blocks": [conc_check["reason"]],
            "message": "Trade BLOCKED — too many concurrent positions",
        }

    # ── 4. Risk Engine — calculate lot size ─────────────────────
    entry = req.entry_price or analysis.get("live_price") or 0
    if not entry:
        raise HTTPException(
            status_code=400,
            detail="entry_price not provided and live price unavailable",
        )

    sizing = RiskEngine.calculate_lot_size(
        account_balance_usd=req.account_balance_usd,
        risk_percent=cfg.risk_percent,
        entry=entry,
        stop_loss=req.stop_loss,
        symbol=symbol,
        min_lot=cfg.min_lot_size,
        max_lot=cfg.max_lot_size,
    )
    if "error" in sizing:
        raise HTTPException(status_code=400, detail=sizing["error"])

    # ── 5. Queue the order ───────────────────────────────────────
    order = BotOrder(
        user_id=user.id,
        symbol=symbol,
        direction=direction,
        signal_confidence=confidence,
        signal_source="aria",
        timeframe=req.timeframe,
        requested_entry=entry,
        stop_loss=req.stop_loss,
        take_profit_1=req.take_profit_1,
        take_profit_2=req.take_profit_2,
        take_profit_3=req.take_profit_3,
        lot_size=sizing["lot_size"],
        risk_percent=sizing["risk_percent"],
        risk_usd=sizing["risk_usd"],
        sl_pips=sizing["sl_pips"],
        current_sl=req.stop_loss,
        filter_passed=True,
        filter_block_reasons=filter_result.get("warnings"),
        status="QUEUED",
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    log_activity(user, "bot_order_queued", symbol=symbol, db=db)

    return {
        "success": True,
        "approved": True,
        "order_id": order.id,
        "status": "QUEUED",
        "message": f"✅ Order queued — MT5 bridge will execute within seconds",
        "trade_details": {
            "symbol": symbol,
            "direction": direction,
            "lot_size": sizing["lot_size"],
            "entry": entry,
            "stop_loss": req.stop_loss,
            "take_profit_1": req.take_profit_1,
            "take_profit_2": req.take_profit_2,
        },
        "risk_details": {
            "risk_percent": sizing["risk_percent"],
            "risk_usd": sizing["risk_usd"],
            "sl_pips": sizing["sl_pips"],
        },
        "filter_warnings": filter_result.get("warnings", []),
    }


@app.post("/bot/auto_signal/{username}")
def auto_signal(
    username: str,
    symbol: str,
    account_balance_usd: float,
    db: Session = Depends(get_db),
):
    """
    CLEO generates a signal for the symbol, runs it through all filters,
    and queues it automatically if it passes.  No manual entry/SL/TP needed —
    CLEO calculates everything from live indicator data.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg or not cfg.is_active:
        return {"success": False, "message": "Bot not configured or not active"}

    sym = symbol.upper()
    analysis = get_symbol_analysis(sym)
    if not analysis.get("live_price"):
        return {"success": False, "message": f"No live data available for {sym}"}

    indicators = analysis.get("indicators", {})
    prompt = (
        f"You are CLEO. Give an auto-trade signal for {sym}.\n"
        f"Live data: {format_for_ai_prompt(sym, analysis)}\n\n"
        f"Reply ONLY with this JSON (no text before/after):\n"
        f'{{"direction":"BUY or SELL","confidence":0-100,"entry":{analysis["live_price"]},'
        f'"stop_loss":float,"take_profit_1":float,"take_profit_2":float,"reasoning":"one sentence"}}'
    )
    try:
        raw = get_ai_response(prompt)
        # Extract JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"success": False, "message": "AI did not return valid JSON"}
        signal = json.loads(match.group())
    except Exception as e:
        return {"success": False, "message": f"AI signal failed: {e}"}

    req = ManualSignalRequest(
        symbol=sym,
        direction=signal["direction"],
        entry_price=float(signal["entry"]),
        stop_loss=float(signal["stop_loss"]),
        take_profit_1=float(signal["take_profit_1"]),
        take_profit_2=signal.get("take_profit_2"),
        account_balance_usd=account_balance_usd,
        signal_confidence=float(signal.get("confidence", 78)),
        timeframe="H1",
    )
    result = place_bot_trade(username=username, req=req, db=db)
    result["ai_reasoning"] = signal.get("reasoning", "")
    return result


# ════════════════════════════════════════════════════════════════
#  3. MT5 BRIDGE ENDPOINTS  (called by mt5_bridge.py on Windows)
# ════════════════════════════════════════════════════════════════


@app.post("/bot/bridge_connect/{username}")
def bridge_connect(
    username: str,
    req: BridgeConnectRequest,
    x_bridge_key: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """MT5 bridge calls this on startup to register itself."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if not cfg:
        raise HTTPException(status_code=400, detail="Bot not configured. Call /bot/configure first.")

    # Validate bridge key at registration
    provided_key = req.bridge_key or x_bridge_key
    if cfg.bridge_api_key and provided_key and cfg.bridge_api_key != provided_key:
        raise HTTPException(status_code=403, detail="Invalid bridge API key")

    # Convert balance to USD for correct lot sizing
    raw_balance  = float(req.balance or 0)
    raw_equity   = float(req.equity  or 0)
    currency     = (req.currency or "USD").upper()

    if currency == "USD" or raw_balance == 0:
        balance_usd = raw_balance
        equity_usd  = raw_equity
    else:
        # Convert foreign currency → USD using live rates
        try:
            rates = get_all_exchange_rates("USD")
            fx = rates.get(currency, 1.0)
            balance_usd = round(raw_balance / fx, 2) if fx else raw_balance
            equity_usd  = round(raw_equity  / fx, 2) if fx else raw_equity
        except Exception:
            balance_usd = raw_balance
            equity_usd  = raw_equity

    # Persist
    cfg.mt5_account_balance  = balance_usd
    cfg.mt5_account_equity   = equity_usd
    cfg.mt5_account_currency = currency
    cfg.mt5_account_number   = str(req.mt5_account) if req.mt5_account else cfg.mt5_account_number
    cfg.mt5_server           = req.mt5_server or cfg.mt5_server
    cfg.bridge_last_seen     = datetime.utcnow()
    db.commit()

    print(f"[BRIDGE] {username} connected | {currency} {raw_balance:,.2f} → ${balance_usd:,.2f} USD | MT5 #{req.mt5_account}")

    return {
        "success": True,
        "status": "connected",
        "message": f"Bridge registered for {username}",
        "balance_usd": balance_usd,
        "equity_usd": equity_usd,
        "currency": currency,
        "server_time": datetime.utcnow().isoformat(),
    }


@app.get("/bot/queue/{username}")
def get_order_queue(username: str, bridge_key: str, db: Session = Depends(get_db)):
    """
    MT5 bridge polls this endpoint every 5 seconds.
    Returns orders with status=QUEUED and marks them SENT.
    """
    cfg = _verify_bridge_key(username, bridge_key, db)
    user = db.query(User).filter(User.username == username).first()

    queued = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "QUEUED",
        )
        .all()
    )

    orders_out = []
    for o in queued:
        orders_out.append(
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry_price": o.requested_entry,
                "stop_loss": o.stop_loss,
                "take_profit_1": o.take_profit_1,
                "take_profit_2": o.take_profit_2,
            }
        )
        o.status = "SENT"
        o.sent_at = datetime.utcnow()

    db.commit()
    return {"orders": orders_out, "count": len(orders_out)}


@app.post("/bot/executed/{username}")
def order_executed(
    username: str,
    bridge_key: str,
    req: BridgeExecutedRequest,
    db: Session = Depends(get_db),
):
    """Bridge reports a successful MT5 execution."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.mt5_ticket = req.mt5_ticket
    order.filled_price = req.filled_price
    order.current_sl = order.stop_loss
    order.status = "ACTIVE"
    order.executed_at = (
        datetime.fromisoformat(req.executed_at)
        if req.executed_at
        else datetime.utcnow()
    )

    db.commit()
    return {"success": True, "message": f"Order {req.order_id} marked ACTIVE"}


@app.post("/bot/rejected/{username}")
def order_rejected(
    username: str,
    bridge_key: str,
    req: BridgeRejectedRequest,
    db: Session = Depends(get_db),
):
    """Bridge reports a failed MT5 execution."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    order.status = "REJECTED"
    order.reject_reason = req.reason
    db.commit()
    return {
        "success": True,
        "message": f"Order {req.order_id} marked REJECTED: {req.reason}",
    }


@app.post("/bot/position_update/{username}")
def position_update(
    username: str,
    bridge_key: str,
    req: PositionUpdateRequest,
    db: Session = Depends(get_db),
):
    """
    Bridge sends current position state every 10 seconds.
    We run TradeManager on each position and return instructions.
    """
    cfg = _verify_bridge_key(username, bridge_key, db)
    instructions = []

    for pos in req.positions:
        order = db.query(BotOrder).filter(BotOrder.mt5_ticket == pos.mt5_ticket).first()
        if not order:
            continue

        # Update floating state
        order.current_price = pos.current_price
        order.floating_pnl_usd = pos.floating_pnl
        if pos.current_sl:
            order.current_sl = pos.current_sl

        # Run Trade Manager
        decision = TradeManager.evaluate_position(order, pos.current_price, cfg)
        action = decision.get("action", "HOLD")

        if action == "HOLD":
            pass

        elif action in ("MOVE_SL_BREAKEVEN", "TRAIL_SL"):
            new_sl = decision.get("new_sl")
            if new_sl:
                order.current_sl = new_sl
                if action == "MOVE_SL_BREAKEVEN":
                    order.sl_at_breakeven = True
                else:
                    order.trailing_active = True
                instructions.append(
                    {
                        "order_id": order.id,
                        "mt5_ticket": pos.mt5_ticket,
                        "symbol": order.symbol,
                        "action": action,
                        "new_sl": new_sl,
                        "details": decision.get("details"),
                    }
                )

        elif action == "PARTIAL_CLOSE":
            order.tp1_closed = True
            order.sl_at_breakeven = True
            order.current_sl = decision.get("new_sl") or order.current_sl
            instructions.append(
                {
                    "order_id": order.id,
                    "mt5_ticket": pos.mt5_ticket,
                    "symbol": order.symbol,
                    "action": "PARTIAL_CLOSE",
                    "close_pct": cfg.partial_close_pct,
                    "new_sl": decision.get("new_sl"),
                    "details": decision.get("details"),
                }
            )

        elif action == "CLOSE":
            instructions.append(
                {
                    "order_id": order.id,
                    "mt5_ticket": pos.mt5_ticket,
                    "symbol": order.symbol,
                    "action": "CLOSE",
                    "close_reason": decision.get("close_reason"),
                    "details": decision.get("details"),
                }
            )

    db.commit()
    return {"instructions": instructions, "processed": len(req.positions)}


@app.post("/bot/partial_closed/{username}")
def partial_closed(
    username: str,
    bridge_key: str,
    req: PartialClosedRequest,
    db: Session = Depends(get_db),
):
    """Bridge confirms a partial close was executed."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order.tp1_closed = True
    db.commit()
    return {
        "success": True,
        "message": f"TP1 partial close confirmed for order {req.order_id}",
    }


@app.post("/bot/closed/{username}")
def position_closed(
    username: str, bridge_key: str, req: ClosedRequest, db: Session = Depends(get_db)
):
    """Bridge reports a position was fully closed."""
    cfg = _verify_bridge_key(username, bridge_key, db)
    order = db.query(BotOrder).filter(BotOrder.id == req.order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    entry = order.filled_price or order.requested_entry or 0
    close = req.close_price or 0

    pnl = (
        TradeManager.calculate_pnl(
            direction=order.direction,
            entry=entry,
            close=close,
            lot_size=order.lot_size,
            symbol=order.symbol,
        )
        if (entry and close)
        else None
    )

    order.status = "CLOSED"
    order.close_price = close
    order.close_reason = req.close_reason
    order.realised_pnl_usd = pnl
    order.closed_at = (
        datetime.fromisoformat(req.closed_at) if req.closed_at else datetime.utcnow()
    )

    # Also log to trade journal
    user = db.query(User).filter(User.id == order.user_id).first()
    if user and entry and close:
        journal = DBJournalEntry(
            user_id=user.id,
            symbol=order.symbol,
            direction=order.direction,
            entry_price=entry,
            exit_price=close,
            volume=order.lot_size,
            result="WIN"
            if (pnl or 0) > 0
            else ("LOSS" if (pnl or 0) < 0 else "BREAK_EVEN"),
            pnl_usd=pnl or 0,
            notes=f"Auto-trade | Reason: {req.close_reason} | Ticket: {req.mt5_ticket}",
        )
        db.add(journal)

    db.commit()
    return {
        "success": True,
        "order_id": order.id,
        "symbol": order.symbol,
        "pnl_usd": pnl,
        "result": "WIN" if (pnl or 0) > 0 else "LOSS",
        "message": f"Order {req.order_id} closed — {'WIN' if (pnl or 0) > 0 else 'LOSS'} ${abs(pnl or 0):.2f}",
    }


# ════════════════════════════════════════════════════════════════
#  4. PERFORMANCE & HISTORY
# ════════════════════════════════════════════════════════════════


@app.get("/bot/history/{username}")
def bot_trade_history(username: str, limit: int = 50, db: Session = Depends(get_db)):
    """Full closed trade history for the bot."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    orders = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "CLOSED",
        )
        .order_by(BotOrder.closed_at.desc())
        .limit(limit)
        .all()
    )

    trades = []
    for o in orders:
        trades.append(
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "lot_size": o.lot_size,
                "entry": o.filled_price,
                "close": o.close_price,
                "pnl_usd": o.realised_pnl_usd,
                "result": "WIN" if (o.realised_pnl_usd or 0) > 0 else "LOSS",
                "close_reason": o.close_reason,
                "risk_percent": o.risk_percent,
                "risk_usd": o.risk_usd,
                "mt5_ticket": o.mt5_ticket,
                "opened_at": o.executed_at.isoformat() if o.executed_at else None,
                "closed_at": o.closed_at.isoformat() if o.closed_at else None,
            }
        )

    total = len(orders)
    wins = sum(1 for o in orders if (o.realised_pnl_usd or 0) > 0)
    total_pnl = sum(o.realised_pnl_usd or 0 for o in orders)

    return {
        "username": username,
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "trades": trades,
    }


@app.get("/bot/performance/{username}")
def bot_performance(username: str, db: Session = Depends(get_db)):
    """Aggregated performance stats: all-time, monthly, weekly breakdown."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    all_closed = (
        db.query(BotOrder)
        .filter(BotOrder.user_id == user.id, BotOrder.status == "CLOSED")
        .order_by(BotOrder.closed_at)
        .all()
    )

    def stats(orders):
        if not orders:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0, "pnl_usd": 0}
        wins = sum(1 for o in orders if (o.realised_pnl_usd or 0) > 0)
        pnl = sum(o.realised_pnl_usd or 0 for o in orders)
        return {
            "trades": len(orders),
            "wins": wins,
            "losses": len(orders) - wins,
            "win_rate": round(wins / len(orders) * 100, 1),
            "pnl_usd": round(pnl, 2),
        }

    now = datetime.utcnow()
    today = now.date()
    this_week_start = today - timedelta(days=today.weekday())
    this_month_start = today.replace(day=1)

    today_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() == today
    ]
    week_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() >= this_week_start
    ]
    month_orders = [
        o for o in all_closed if o.closed_at and o.closed_at.date() >= this_month_start
    ]

    # Symbol breakdown
    by_symbol = {}
    for o in all_closed:
        if o.symbol not in by_symbol:
            by_symbol[o.symbol] = []
        by_symbol[o.symbol].append(o)
    symbol_stats = {sym: stats(orders) for sym, orders in by_symbol.items()}

    return {
        "username": username,
        "all_time": stats(all_closed),
        "this_month": stats(month_orders),
        "this_week": stats(week_orders),
        "today": stats(today_orders),
        "by_symbol": symbol_stats,
        "generated_at": now.isoformat(),
    }


@app.post("/bot/emergency_stop/{username}")
def emergency_stop(username: str, db: Session = Depends(get_db)):
    """
    EMERGENCY — immediately:
    1. Pauses the bot (no new orders)
    2. Cancels all QUEUED orders
    3. Returns a list of ACTIVE positions the user must manually close in MT5
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    cfg = db.query(BotConfig).filter(BotConfig.user_id == user.id).first()
    if cfg:
        cfg.is_active = False

    queued = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status == "QUEUED",
        )
        .all()
    )
    cancelled_count = len(queued)
    for o in queued:
        o.status = "CANCELLED"
        o.reject_reason = "EMERGENCY_STOP"

    active = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status.in_(["SENT", "EXECUTED", "ACTIVE"]),
        )
        .all()
    )

    db.commit()

    return {
        "success": True,
        "bot_paused": True,
        "cancelled_queued": cancelled_count,
        "active_positions": [
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "direction": o.direction,
                "ticket": o.mt5_ticket,
                "lot_size": o.lot_size,
            }
            for o in active
        ],
        "action_required": (
            f"⚠️ Bot stopped. {len(active)} live position(s) still open in MT5. "
            "Close them manually in your MT5 terminal or restart the bridge to allow auto-close."
        )
        if active
        else "✅ All clear. No open positions.",
    }


@app.post("/bot/reset_stuck/{username}")
def reset_stuck_orders(username: str, db: Session = Depends(get_db)):
    """
    Resets SENT orders that have no MT5 ticket (stuck, never executed)
    back to QUEUED so the bridge can retry them.
    Also marks REJECTED orders as cleaned up.
    Call this if the bridge was restarted after fixing a connection issue.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    stuck = (
        db.query(BotOrder)
        .filter(
            BotOrder.user_id == user.id,
            BotOrder.status.in_(["SENT", "REJECTED"]),
            BotOrder.mt5_ticket == None,
        )
        .all()
    )

    reset_count = 0
    for o in stuck:
        o.status = "CANCELLED"
        o.reject_reason = "RESET_STUCK — bridge reconnected"
        reset_count += 1

    db.commit()
    return {
        "success": True,
        "reset_count": reset_count,
        "message": f"✅ {reset_count} stuck order(s) cleared. Bot is ready for fresh trades.",
    }


# ════════════════════════════════════════════════════════════════
#  24/7 AUTO-TRADE SCHEDULER
#  Runs as a background daemon thread.
#  Every 60 min it generates predictions (uses cache — no extra
#  Groq calls) and queues all auto-trade eligible signals for
#  every user who has an active bot configured.
# ════════════════════════════════════════════════════════════════

SCHEDULER_INTERVAL = 60 * 60      # 1 hour between full signal runs
SCHEDULER_WARMUP   = 30           # seconds to wait after server starts


def _auto_trade_scheduler():
    """Background thread: generate signals + queue trades every hour."""
    from models import SessionLocal

    print(f"[SCHEDULER] Warming up — first run in {SCHEDULER_WARMUP}s")
    time.sleep(SCHEDULER_WARMUP)

    while True:
        cycle_start = time.time()
        print(f"[SCHEDULER] ⏰ Auto-trade cycle starting at {datetime.utcnow().strftime('%H:%M UTC')}")

        try:
            db = SessionLocal()
            try:
                _run_scheduler_cycle(db)
            finally:
                db.close()
        except Exception as e:
            print(f"[SCHEDULER] ❌ Unhandled error in cycle: {e}")

        elapsed   = time.time() - cycle_start
        sleep_for = max(60, SCHEDULER_INTERVAL - elapsed)
        print(f"[SCHEDULER] Cycle done in {elapsed:.0f}s — next run in {sleep_for/60:.0f} min")
        time.sleep(sleep_for)


def _run_scheduler_cycle(db):
    """
    Core scheduler logic:
    1. Generate fresh predictions (uses 60-min cache — usually free)
    2. For each user with an active bot, queue signals that pass all filters
    """
    # ── Step 1: Generate / refresh predictions ──────────────────────────────
    preds = generate_market_predictions(investment_amount_ngn=0, user_profile="")
    if not preds.get("success"):
        print(f"[SCHEDULER] Predictions not available: {preds.get('error', 'unknown error')}")
        return

    signals = preds.get("signals", [])

    # Only exclude HOLD signals and signals with no direction
    tradeable_signals = [
        s for s in signals
        if s.get("signal") in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL")
        and s.get("confidence", 0) > 0
    ]

    if not tradeable_signals:
        print("[SCHEDULER] No tradeable signals this cycle — nothing queued")
        return

    print(f"[SCHEDULER] {len(tradeable_signals)} candidate signal(s) this cycle")

    # ── Step 2: Find all users with an active bot ────────────────────────────
    active_cfgs = db.query(BotConfig).filter(BotConfig.is_active == True).all()
    if not active_cfgs:
        print("[SCHEDULER] No active bots configured — nothing queued")
        return

    print(f"[SCHEDULER] {len(active_cfgs)} active bot(s) found")

    queued_total = 0
    blocked_total = 0

    for cfg in active_cfgs:
        user = db.query(User).filter(User.id == cfg.user_id).first()
        if not user:
            continue

        # Filter signals per this user's confidence setting
        user_min_conf = float(cfg.min_confidence or 73)
        auto_signals = [
            s for s in tradeable_signals
            if float(s.get("confidence", 0)) >= user_min_conf
        ]

        if not auto_signals:
            print(f"[SCHEDULER] No signals above {user_min_conf}% for {user.username}")
            continue

        print(f"[SCHEDULER] {len(auto_signals)} signal(s) >= {user_min_conf}% for {user.username}: "
              + ", ".join(f"{s['symbol']} {s['confidence']}%" for s in auto_signals))

        # Get the user's MT5 account balance (fall back to a safe default)
        account_balance_usd = float(cfg.mt5_account_balance or 1000.0)

        for sig in auto_signals:
            symbol     = sig["symbol"]
            direction  = "BUY"  if "BUY"  in sig["signal"] else "SELL"
            confidence = float(sig.get("confidence", 75))
            entry      = float(sig.get("entry_price") or sig.get("live_price") or 0)
            sl         = float(sig.get("stop_loss")    or 0)
            tp1        = float(sig.get("take_profit_1") or 0)
            tp2        = sig.get("take_profit_2")

            # Skip if no price data
            if not entry or not sl:
                continue

            # Skip if below this user's configured minimum confidence
            if confidence < float(cfg.min_confidence or 75):
                continue

            try:
                req = ManualSignalRequest(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=float(tp2) if tp2 else None,
                    account_balance_usd=account_balance_usd,
                    signal_confidence=confidence,
                    timeframe="H1",
                )
                result = place_bot_trade(username=user.username, req=req, db=db)

                if result.get("success"):
                    queued_total += 1
                    print(f"[SCHEDULER] ✅ Queued: {user.username} | {direction} {symbol} "
                          f"conf={confidence:.0f}% lot={result.get('lot_size', '?')}")
                else:
                    blocked_total += 1
                    blocks = result.get("blocks", [])
                    print(f"[SCHEDULER] ⛔ Blocked: {user.username} | {symbol} — {blocks}")

            except Exception as e:
                print(f"[SCHEDULER] Error queuing {symbol} for {user.username}: {e}")

    print(f"[SCHEDULER] Cycle summary — queued: {queued_total} | blocked/filtered: {blocked_total}")


# ── Status endpoint for the scheduler ────────────────────────────────────────
@app.get("/scheduler/status")
def scheduler_status():
    """Check whether the auto-trade scheduler is running and when it last fired."""
    return {
        "scheduler": "running",
        "interval_minutes": SCHEDULER_INTERVAL // 60,
        "description": (
            "The scheduler automatically generates signals every hour and queues "
            "auto-trade eligible ones (confidence >= min_confidence) for all active bots."
        ),
    }


# ═══════════════════════════════════════════════════
#  VOICE CHAT ENDPOINTS
#  STT: Groq Whisper large-v3 (free)
#  TTS: Groq PlayAI (free)
# ═══════════════════════════════════════════════════


@app.post("/voice/transcribe")
async def voice_transcribe(
    file: UploadFile = File(...),
):
    """
    Convert an audio recording to text using Groq Whisper.

    Send a multipart/form-data POST with field name `file`.
    Accepts: webm, mp3, mp4, wav, ogg, m4a, flac (max ~25 MB).

    Returns:
      { "text": "transcription here", "filename": "...", "duration_hint": "..." }
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    filename = file.filename or "audio.webm"
    try:
        text = transcribe_audio(audio_bytes, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {str(e)[:200]}")

    return {
        "success": True,
        "text": text,
        "filename": filename,
        "bytes_received": len(audio_bytes),
    }


@app.post("/voice/speak")
async def voice_speak(
    body: dict,
):
    """
    Convert text to speech using Groq PlayAI TTS.

    POST JSON body:
      {
        "text": "Hello, this is CLEO speaking.",
        "voice": "cleo"          // optional: cleo | female | male | warm | deep
      }

    Returns raw MP3 audio bytes (Content-Type: audio/mpeg).
    Frontend can play this directly:
      const audio = new Audio(URL.createObjectURL(new Blob([response.data], {type:'audio/mpeg'})))
      audio.play()
    """
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` field is required.")

    voice = body.get("voice", "cleo")
    try:
        audio_bytes = await text_to_speech(text, voice=voice)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {str(e)[:200]}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=cleo_response.mp3",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    username: str    = Form(...),
    conversation_id: Optional[int] = Form(None),
    voice: str       = Form("cleo"),
    db: Session      = Depends(get_db),
):
    """
    Full voice chat round-trip:
      1. Transcribe audio → text   (Groq Whisper)
      2. Send text to CLEO         (same as POST /chat)
      3. Convert CLEO response → MP3  (Groq PlayAI TTS)

    Send as multipart/form-data:
      - file           : audio blob (webm / mp3 / wav / m4a)
      - username       : your username
      - conversation_id: (optional) continue existing conversation
      - voice          : (optional) cleo | female | male | warm | deep

    Returns JSON:
      {
        "success": true,
        "transcription": "what the user said",
        "response": "CLEO's text response",
        "audio_base64": "base64-encoded MP3 — play directly in browser",
        "audio_content_type": "audio/mpeg",
        "conversation_id": 42,
        "conversation_title": "EURUSD Strategy"
      }

    To play audio on the frontend:
      const bytes = Uint8Array.from(atob(data.audio_base64), c => c.charCodeAt(0))
      const blob  = new Blob([bytes], { type: 'audio/mpeg' })
      new Audio(URL.createObjectURL(blob)).play()
    """
    # ── Step 1: Transcribe audio ──
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    filename = file.filename or "audio.webm"
    try:
        transcription = transcribe_audio(audio_bytes, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {str(e)[:200]}")

    if not transcription:
        raise HTTPException(status_code=422, detail="Could not transcribe audio — please speak clearly.")

    # ── Step 2: Send transcription to CLEO (reuse chat endpoint logic) ──
    chat_req = ChatMessage(
        username        = username,
        message         = transcription,
        conversation_id = conversation_id,
    )
    try:
        chat_result = chat_with_aria(chat_req, db=db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)[:200]}")

    cleo_text = chat_result.get("response", "")

    # ── Step 3: Convert CLEO's response to audio ──
    # Strip markdown for cleaner TTS (remove **, ##, ━━, emojis lists, etc.)
    tts_text = re.sub(r"[*#━─│╔╗╚╝╠╣╦╩╬]+", "", cleo_text)
    tts_text = re.sub(r"\n{3,}", "\n\n", tts_text).strip()
    tts_text = tts_text[:3500]   # keep within TTS limit

    audio_mp3: bytes = b""
    audio_b64: str   = ""
    try:
        audio_mp3 = await text_to_speech(tts_text, voice=voice)
        audio_b64 = base64.b64encode(audio_mp3).decode("utf-8")
    except Exception as e:
        print(f"[VOICE/TTS] non-critical error: {e}")
        # Return text response even if TTS fails

    return {
        "success":              True,
        "transcription":        transcription,
        "response":             cleo_text,
        "audio_base64":         audio_b64,
        "audio_content_type":   "audio/mpeg",
        "conversation_id":      chat_result.get("conversation_id"),
        "conversation_title":   chat_result.get("conversation_title"),
        "username":             username,
        "tts_available":        bool(audio_b64),
    }


@app.get("/voice/voices")
def list_voices():
    """List all available TTS voice options for the voice chat endpoints."""
    return {
        "voices": [
            {"id": "cleo",   "edge_voice": "en-US-AriaNeural",  "gender": "female", "style": "warm professional — CLEO default"},
            {"id": "female", "edge_voice": "en-US-JennyNeural", "gender": "female", "style": "clear, friendly"},
            {"id": "male",   "edge_voice": "en-US-GuyNeural",   "gender": "male",   "style": "clear, authoritative"},
            {"id": "warm",   "edge_voice": "en-GB-SoniaNeural", "gender": "female", "style": "warm British female"},
            {"id": "deep",   "edge_voice": "en-US-EricNeural",  "gender": "male",   "style": "deeper male voice"},
        ],
        "tts_engine": "Microsoft Edge Neural TTS (edge-tts, free)",
        "stt_model":  "whisper-large-v3 via Groq (free)",
        "usage": {
            "speak":     "POST /voice/speak  — JSON body: {text, voice}  → returns MP3 bytes",
            "transcribe": "POST /voice/transcribe  — multipart file  → returns {text}",
            "chat":      "POST /voice/chat  — multipart (file, username, conversation_id?, voice?) → returns {transcription, response, audio_base64}",
        },
    }


import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
