    import time
    from market_data import get_symbol_analysis

    CACHE = {}
    LAST_FETCH = 0

    FETCH_INTERVAL = 900  # 15 minutes


    def update_market(symbol="EURUSD"):
        global LAST_FETCH

        now = time.time()

        # ⛔ prevent spam
        if now - LAST_FETCH < FETCH_INTERVAL:
            return CACHE.get(symbol)

        print(f"🌍 Fetching market data for {symbol}")

        data = get_symbol_analysis(symbol)

        if data and data.get("live_price"):
            CACHE[symbol] = data
            LAST_FETCH = now
            print("✅ Market data updated")

        return CACHE.get(symbol)


    def get_cached(symbol="EURUSD"):
        return CACHE.get(symbol)