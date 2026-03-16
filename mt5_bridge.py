"""
mt5_bridge.py  —  Run this file on your Windows machine / VPS that has
                   MetaTrader 5 terminal installed.

HOW IT WORKS:
  1. This script connects to your MT5 terminal via the official MetaTrader5
     Python library (free, from MetaQuotes).
  2. Every few seconds it polls your ARIA backend for pending orders.
  3. When it finds one, it executes the trade directly in MT5.
  4. It then continuously sends position updates back to the backend,
     which runs the TradeManager (trailing stop, break-even, partial close).
  5. When the backend says CLOSE / TRAIL_SL / PARTIAL_CLOSE, this script
     acts immediately in MT5.

SETUP INSTRUCTIONS:
  1. Install Python 3.10+ for Windows:  https://www.python.org/downloads/
  2. Install dependencies:
       pip install MetaTrader5 requests python-dotenv
  3. Install MetaTrader 5 terminal from your broker's website.
  4. Log into your MT5 account in the terminal and keep it open.
  5. Create a .env file next to this script:
       API_URL=https://your-replit-deployment-url
       USERNAME=your_username
       BRIDGE_API_KEY=your_secret_key_from_bot_config
       MT5_LOGIN=12345678
       MT5_PASSWORD=your_mt5_password
       MT5_SERVER=BrokerName-Live
  6. Run:  python mt5_bridge.py

RECOMMENDED VPS FOR 24/7:
  - Contabo VPS S (Windows): ~$7/month  https://contabo.com
  - Vultr Cloud Compute (Windows): ~$10/month  https://vultr.com
  - Azure B1s VM (Windows): ~$15/month — free tier available
  - IMPORTANT: Choose a VPS in London or New York for lowest latency.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── CONFIG FROM ENVIRONMENT ──────────────────────────────────────
API_URL       = os.getenv("API_URL",        "https://your-backend-url.replit.app")
USERNAME      = os.getenv("USERNAME",       "")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")
MT5_LOGIN     = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD  = os.getenv("MT5_PASSWORD",  "")
MT5_SERVER    = os.getenv("MT5_SERVER",    "")

POLL_INTERVAL_SEC  = 5    # how often to poll for new orders
UPDATE_INTERVAL_SEC = 10  # how often to send position updates to backend
MAX_RETRIES        = 3    # MT5 order retry attempts on requote

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aria_bridge.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ARIA-Bridge")

# ── IMPORT MT5 ───────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    log.error("MetaTrader5 library not installed. Run: pip install MetaTrader5")
    log.error("If you are on Linux/Mac — MT5 only works on Windows.")


# ════════════════════════════════════════════════════════════════
#  MT5 Connection
# ════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    if not MT5_AVAILABLE:
        return False
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if info is None:
        log.error(f"MT5 account_info failed: {mt5.last_error()}")
        return False
    log.info(f"✅ MT5 Connected | Account: {info.login} | Balance: {info.balance:.2f} {info.currency} | Broker: {info.company}")
    return True


def get_account_info() -> dict:
    if not MT5_AVAILABLE:
        return {}
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login":    info.login,
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "free_margin": info.margin_free,
        "profit":   info.profit,
        "currency": info.currency,
        "leverage": info.leverage,
    }


# ════════════════════════════════════════════════════════════════
#  API calls to ARIA backend
# ════════════════════════════════════════════════════════════════

HEADERS = {
    "Content-Type": "application/json",
    "X-Bridge-Key": BRIDGE_API_KEY,
}


def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET {path} failed: {e}")
        return {}


def api_post(path: str, data: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}{path}", headers=HEADERS, json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"POST {path} failed: {e}")
        return {}


# ════════════════════════════════════════════════════════════════
#  Order Execution
# ════════════════════════════════════════════════════════════════

def normalize_symbol_for_mt5(symbol: str) -> str:
    """
    MT5 brokers use different symbol names.
    Common mappings — adjust to match YOUR broker.
    """
    mappings = {
        "BTCUSDT": "BTCUSD",    # Most brokers drop the T
        "ETHUSDT": "ETHUSD",
        "BNBUSDT": "BNBUSD",
        "XRPUSDT": "XRPUSD",
        "SOLUSDT": "SOLUSD",
        "ADAUSDT": "ADAUSD",
        "DOGEUSDT": "DOGEUSD",
        "DOTUSDT": "DOTUSD",
        "MATICUSDT": "MATICUSD",
        "LTCUSDT": "LTCUSD",
        "AVAXUSDT": "AVAXUSD",
        "LINKUSDT": "LINKUSD",
        "UNIUSDT": "UNIUSD",
        "SHIBUSDT": "SHIBUSD",
        "TRXUSDT": "TRXUSD",
    }
    return mappings.get(symbol.upper(), symbol.upper())


def get_current_price(symbol: str) -> tuple:
    """Returns (bid, ask) for the symbol."""
    mt5_sym = normalize_symbol_for_mt5(symbol)
    tick = mt5.symbol_info_tick(mt5_sym)
    if tick is None:
        return None, None
    return tick.bid, tick.ask


def get_spread_pips(symbol: str) -> float:
    """Returns current spread in pips."""
    bid, ask = get_current_price(symbol)
    if bid is None:
        return 0.0
    from trading_engine import SYMBOL_META
    meta = SYMBOL_META.get(symbol.upper(), {"pip": 0.0001})
    return round((ask - bid) / meta["pip"], 1)


def execute_order(order: dict) -> dict:
    """
    Places an order in MT5.
    Returns {"success": bool, "ticket": int, "filled_price": float, "error": str}
    """
    symbol    = normalize_symbol_for_mt5(order["symbol"])
    direction = order["direction"].upper()
    lot_size  = float(order["lot_size"])
    sl        = float(order["stop_loss"])
    tp        = float(order["take_profit_1"])

    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        return {"success": False, "error": f"Cannot select symbol {symbol} in MT5"}

    # Get current prices
    bid, ask = get_current_price(order["symbol"])
    if bid is None:
        return {"success": False, "error": f"Cannot get price for {symbol}"}

    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price      = ask if direction == "BUY" else bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,          # max price deviation (points)
        "magic":        202600,      # ARIA magic number — identifies our orders
        "comment":      f"ARIA#{order['order_id']}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        result = mt5.order_send(request)
        if result is None:
            log.error(f"order_send returned None (attempt {attempt}): {mt5.last_error()}")
            time.sleep(1)
            continue

        log.info(f"order_send retcode={result.retcode} (attempt {attempt})")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"✅ Order FILLED | Ticket={result.order} | Price={result.price} | {direction} {lot_size} {symbol}")
            return {
                "success":      True,
                "ticket":       result.order,
                "filled_price": result.price,
                "error":        None,
            }
        elif result.retcode in (mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED):
            log.warning(f"Requote on attempt {attempt}, retrying...")
            # Update price and retry
            bid, ask = get_current_price(order["symbol"])
            request["price"] = ask if direction == "BUY" else bid
            time.sleep(0.5)
        else:
            error_msg = f"MT5 retcode {result.retcode}: {result.comment}"
            log.error(f"Order REJECTED: {error_msg}")
            return {"success": False, "ticket": None, "filled_price": None, "error": error_msg}

    return {"success": False, "error": "Max retries exceeded (requote loop)"}


def modify_sl(ticket: int, new_sl: float, symbol: str) -> bool:
    """Modify the stop-loss of an open position."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        log.warning(f"Position {ticket} not found in MT5")
        return False

    pos    = positions[0]
    mt5_sym = normalize_symbol_for_mt5(symbol)
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol":   mt5_sym,
        "sl":       new_sl,
        "tp":       pos.tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"✅ SL modified | Ticket={ticket} | New SL={new_sl}")
        return True
    log.error(f"SL modify failed: {result.comment if result else mt5.last_error()}")
    return False


def close_position(ticket: int, symbol: str, volume: float, direction: str) -> dict:
    """Close an open position at market."""
    mt5_sym = normalize_symbol_for_mt5(symbol)
    bid, ask = get_current_price(symbol)
    if bid is None:
        return {"success": False, "error": "Cannot get price"}

    # Closing reverses direction
    close_type  = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
    close_price = bid if direction == "BUY" else ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       mt5_sym,
        "volume":       volume,
        "type":         close_type,
        "position":     ticket,
        "price":        close_price,
        "deviation":    20,
        "magic":        202600,
        "comment":      "ARIA-close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"✅ Position closed | Ticket={ticket} | Close price={result.price}")
        return {"success": True, "close_price": result.price}
    return {"success": False, "error": result.comment if result else str(mt5.last_error())}


def partial_close_position(ticket: int, symbol: str, direction: str, close_pct: float, total_volume: float) -> dict:
    """Close a percentage of an open position."""
    from trading_engine import LOT_STEP
    close_vol = round(total_volume * (close_pct / 100) / LOT_STEP) * LOT_STEP
    close_vol = max(0.01, close_vol)
    if close_vol >= total_volume:
        return close_position(ticket, symbol, total_volume, direction)
    return close_position(ticket, symbol, close_vol, direction)


# ════════════════════════════════════════════════════════════════
#  Main polling loop
# ════════════════════════════════════════════════════════════════

def main_loop():
    log.info("=" * 60)
    log.info("  ARIA MT5 Bridge  |  Starting up...")
    log.info("=" * 60)

    if not connect_mt5():
        log.critical("Cannot connect to MT5. Exiting.")
        return

    account = get_account_info()
    log.info(f"Account balance: {account.get('balance', '?')} {account.get('currency', '')}")

    # Register bridge with backend
    api_post(f"/bot/bridge_connect/{USERNAME}", {
        "account_info":  account,
        "bridge_version": "1.0",
    })

    last_update_time = 0

    while True:
        try:
            # ── Poll for pending orders ──────────────────────────
            pending = api_get(f"/bot/queue/{USERNAME}")
            orders  = pending.get("orders", [])

            for order in orders:
                order_id = order["order_id"]
                symbol   = order["symbol"]
                log.info(f"[ORDER #{order_id}] Processing {order['direction']} {symbol} @ {order.get('lot_size')} lots")

                # Get current spread for filter reporting
                spread = get_spread_pips(symbol) if MT5_AVAILABLE else None

                # Execute in MT5
                result = execute_order(order)

                if result["success"]:
                    bid, ask = get_current_price(symbol)
                    api_post(f"/bot/executed/{USERNAME}", {
                        "order_id":     order_id,
                        "mt5_ticket":   result["ticket"],
                        "filled_price": result["filled_price"],
                        "spread_pips":  spread,
                        "executed_at":  datetime.utcnow().isoformat(),
                    })
                    log.info(f"[ORDER #{order_id}] ✅ Reported execution to backend")
                else:
                    api_post(f"/bot/rejected/{USERNAME}", {
                        "order_id":    order_id,
                        "reason":      result["error"],
                        "rejected_at": datetime.utcnow().isoformat(),
                    })
                    log.error(f"[ORDER #{order_id}] ❌ Rejected: {result['error']}")

            # ── Send position updates to backend ─────────────────
            now = time.time()
            if now - last_update_time >= UPDATE_INTERVAL_SEC:
                last_update_time = now

                if MT5_AVAILABLE:
                    positions = mt5.positions_get(magic=202600)
                    if positions:
                        updates = []
                        for pos in positions:
                            updates.append({
                                "mt5_ticket":   pos.ticket,
                                "symbol":       pos.symbol,
                                "current_price": pos.price_current,
                                "floating_pnl": pos.profit,
                                "volume":        pos.volume,
                                "current_sl":    pos.sl,
                                "current_tp":    pos.tp,
                            })
                        instructions = api_post(f"/bot/position_update/{USERNAME}", {
                            "positions": updates,
                            "account":   get_account_info(),
                        })

                        # Process trade manager instructions from backend
                        for instr in instructions.get("instructions", []):
                            ticket  = instr.get("mt5_ticket")
                            action  = instr.get("action")
                            new_sl  = instr.get("new_sl")
                            order_id = instr.get("order_id")

                            if action in ("MOVE_SL_BREAKEVEN", "TRAIL_SL") and new_sl and ticket:
                                log.info(f"[{action}] Ticket={ticket} → New SL={new_sl}")
                                # Find position volume for symbol
                                pos_list = mt5.positions_get(ticket=ticket)
                                if pos_list:
                                    modify_sl(ticket, new_sl, instr.get("symbol", ""))

                            elif action == "PARTIAL_CLOSE" and ticket:
                                log.info(f"[PARTIAL_CLOSE] Ticket={ticket} {instr.get('close_pct')}%")
                                pos_list = mt5.positions_get(ticket=ticket)
                                if pos_list:
                                    pos = pos_list[0]
                                    result = partial_close_position(
                                        ticket, instr.get("symbol", ""),
                                        "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                                        instr.get("close_pct", 50),
                                        pos.volume
                                    )
                                    api_post(f"/bot/partial_closed/{USERNAME}", {
                                        "order_id": order_id, "mt5_ticket": ticket,
                                        "close_price": result.get("close_price"),
                                        "success": result.get("success"),
                                    })

                            elif action == "CLOSE" and ticket:
                                log.info(f"[CLOSE] Ticket={ticket} reason={instr.get('close_reason')}")
                                pos_list = mt5.positions_get(ticket=ticket)
                                if pos_list:
                                    pos = pos_list[0]
                                    result = close_position(
                                        ticket, instr.get("symbol", ""),
                                        pos.volume,
                                        "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                                    )
                                    api_post(f"/bot/closed/{USERNAME}", {
                                        "order_id":    order_id,
                                        "mt5_ticket":  ticket,
                                        "close_price": result.get("close_price"),
                                        "close_reason": instr.get("close_reason"),
                                        "closed_at":   datetime.utcnow().isoformat(),
                                    })

        except KeyboardInterrupt:
            log.info("Shutting down bridge (keyboard interrupt)")
            if MT5_AVAILABLE:
                mt5.shutdown()
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    if not USERNAME or not BRIDGE_API_KEY:
        log.critical("USERNAME and BRIDGE_API_KEY must be set in .env file")
        sys.exit(1)
    main_loop()
