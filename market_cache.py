import time
from market_data import get_symbol_analysis

CACHE = {}
LAST_FETCH = 0
FETCH_INTERVAL = 900  # 15 minutes


def get_market(symbol):
    global LAST_FETCH

    symbol = symbol.replace("/", "").upper()
    now = time.time()

    # use cache if fresh
    if CACHE and now - LAST_FETCH < FETCH_INTERVAL:
        return CACHE.get(symbol)

    print("🌍 Fetching market data...")

    data = get_symbol_analysis(symbol)

    if data and data.get("live_price"):
        CACHE[symbol] = data
        LAST_FETCH = now
        return data

    return CACHE.get(symbol)