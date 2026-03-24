"""
ARIA MT5 Bridge — Windows-side script
======================================
Runs on your Windows PC/VPS alongside MetaTrader 5.
Polls the ARIA backend for queued orders, executes them in MT5,
and manages open positions (break-even, trailing stop, partial close).

SETUP:
  1. Install Python 3.10+ for Windows  (python.org)
  2. pip install MetaTrader5 requests python-dotenv
  3. Install MT5 from HFM:  https://www.hfm.com/en/platforms/metatrader5
  4. Log in to MT5 with your HFM account
  5. Create a file called  .env  in the same folder as this script:

        API_URL=https://your-replit-url.replit.app
        USERNAME=your_aria_username
        BRIDGE_API_KEY=paste_key_from_bot_configure_endpoint
        MT5_LOGIN=12345678
        MT5_PASSWORD=your_hfm_password
        MT5_SERVER=HFMarkets-Live

  6. Run:  python mt5_bridge.py
  7. For 24/7 use a Windows VPS (Contabo / Vultr / Azure ~ $7-15/month)

HOW TO GET YOUR BRIDGE_API_KEY:
  POST /bot/configure/<username>  with your bot settings.
  The response contains  "bridge_api_key".  Paste it into .env.
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime

import requests
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

API_URL       = os.getenv("API_URL", "").rstrip("/")
USERNAME      = os.getenv("USERNAME", "")
BRIDGE_KEY    = os.getenv("BRIDGE_API_KEY", "")
MT5_LOGIN     = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD  = os.getenv("MT5_PASSWORD", "")
MT5_SERVER    = os.getenv("MT5_SERVER", "HFMarkets-Live")
MAGIC         = int(os.getenv("MAGIC_NUMBER", "202600"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))    # seconds between polls

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("aria_bridge.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ARIA-Bridge")

# ── Validate config ───────────────────────────────────────────────────────────
missing = [k for k, v in {
    "API_URL": API_URL, "USERNAME": USERNAME,
    "BRIDGE_API_KEY": BRIDGE_KEY, "MT5_LOGIN": MT5_LOGIN,
    "MT5_PASSWORD": MT5_PASSWORD,
}.items() if not v]
if missing:
    log.error(f"Missing .env values: {', '.join(missing)}")
    log.error("Create a .env file — see the setup instructions at the top of this script.")
    sys.exit(1)

# ── Import MT5 ───────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
except ImportError:
    log.error("MetaTrader5 package not installed.")
    log.error("Run:  pip install MetaTrader5 requests python-dotenv")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# MT5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def mt5_connect() -> bool:
    """Initialize and log into MT5. Returns True on success."""
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    info = mt5.account_info()
    log.info(f"MT5 connected — Account: {info.login} | Balance: {info.balance:.2f} {info.currency} | Server: {MT5_SERVER}")
    return True


def symbol_select(symbol: str) -> bool:
    """Ensure the symbol is visible in MarketWatch."""
    mt5.symbol_select(symbol, True)
    return mt5.symbol_info(symbol) is not None


def get_pip_value(symbol: str) -> float:
    """Return the pip size for a symbol (0.0001 for most forex, 0.01 for JPY pairs)."""
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0001
    # For crypto USDT pairs point == 1; for forex point is the smallest move
    return info.point * (10 if info.digits in (3, 5) else 1)


def normalize_symbol(symbol: str) -> str:
    """
    Convert ARIA symbol format to MT5 format.
    ARIA uses:  EURUSD, BTCUSDT
    HFM uses:   EURUSD, BTCUSD (no T) or XAUUSD etc.
    Adjust the mapping below to match your HFM account.
    """
    mapping = {
        "BTCUSDT":  "BTCUSD",
        "ETHUSDT":  "ETHUSD",
        "BNBUSDT":  "BNBUSD",
        "XRPUSDT":  "XRPUSD",
        "SOLUSDT":  "SOLUSD",
        "ADAUSDT":  "ADAUSD",
        "DOGEUSDT": "DOGEUSD",
        "DOTUSDT":  "DOTUSD",
        "MATICUSDT":"MATICUSD",
        "LTCUSDT":  "LTCUSD",
        "SHIBUSDT": "SHIBUSD",
        "TRXUSDT":  "TRXUSD",
        "AVAXUSDT": "AVAXUSD",
        "LINKUSDT": "LINKUSD",
        "UNIUSDT":  "UNIUSD",
    }
    return mapping.get(symbol, symbol)


def place_order(order: dict) -> tuple[bool, str, int]:
    """
    Execute a market order in MT5.
    Returns (success, message, ticket_number).
    """
    symbol   = normalize_symbol(order["symbol"])
    direction = order["direction"].upper()
    lot_size  = float(order.get("lot_size", 0.01))
    sl        = float(order.get("stop_loss", 0))
    tp1       = float(order.get("take_profit_1", 0))

    if not symbol_select(symbol):
        return False, f"Symbol {symbol} not found in MT5", 0

    info  = mt5.symbol_info(symbol)
    tick  = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, f"No tick data for {symbol}", 0

    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price      = tick.ask if direction == "BUY" else tick.bid
    deviation  = 20   # max slippage in points

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       round(lot_size, 2),
        "type":         order_type,
        "price":        price,
        "sl":           round(sl, info.digits)    if sl  else 0,
        "tp":           round(tp1, info.digits)   if tp1 else 0,
        "deviation":    deviation,
        "magic":        MAGIC,
        "comment":      f"ARIA-{order.get('id', '')[:8]}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        return False, f"order_send returned None: {mt5.last_error()}", 0
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"MT5 error {result.retcode}: {result.comment}", 0

    log.info(f"✅ {direction} {lot_size} {symbol} @ {result.price} | Ticket: {result.order}")
    return True, "executed", result.order


# ─────────────────────────────────────────────────────────────────────────────
# Backend API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Content-Type": "application/json"}


def register_bridge() -> bool:
    """Tell the backend this bridge is online."""
    try:
        acc = mt5.account_info()
        r = requests.post(
            f"{API_URL}/bot/bridge_connect/{USERNAME}",
            json={
                "bridge_key":       BRIDGE_KEY,
                "mt5_account":      MT5_LOGIN,
                "mt5_server":       MT5_SERVER,
                "balance":          acc.balance if acc else 0,
                "equity":           acc.equity  if acc else 0,
            },
            headers=_headers(), timeout=10,
        )
        data = r.json()
        if data.get("success"):
            log.info("✅ Bridge registered with ARIA backend")
            return True
        log.warning(f"Bridge register: {data.get('message', data)}")
        return False
    except Exception as e:
        log.warning(f"Bridge register failed: {e}")
        return False


def fetch_queue() -> list:
    """Fetch orders with status QUEUED from the backend."""
    try:
        r = requests.get(
            f"{API_URL}/bot/queue/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            timeout=10,
        )
        if r.status_code == 403:
            log.error("❌ BRIDGE KEY REJECTED — update BRIDGE_API_KEY in your .env file")
            log.error(f"   Current key in .env: {BRIDGE_KEY[:20]}...")
            return []
        if r.status_code != 200:
            log.warning(f"Queue poll returned HTTP {r.status_code}: {r.text[:100]}")
            return []
        data = r.json()
        orders = data.get("orders", [])
        if orders:
            log.info(f"📥 {len(orders)} order(s) received from backend")
        return orders
    except Exception as e:
        log.warning(f"fetch_queue error: {e}")
        return []


def report_executed(order_id: str, ticket: int, fill_price: float) -> None:
    try:
        requests.post(
            f"{API_URL}/bot/executed/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            json={"order_id": order_id, "ticket": ticket, "fill_price": fill_price},
            headers=_headers(), timeout=10,
        )
    except Exception as e:
        log.warning(f"report_executed error: {e}")


def report_rejected(order_id: str, reason: str) -> None:
    try:
        requests.post(
            f"{API_URL}/bot/rejected/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            json={"order_id": order_id, "reason": reason},
            headers=_headers(), timeout=10,
        )
    except Exception as e:
        log.warning(f"report_rejected error: {e}")


def send_position_update(position: dict) -> dict:
    """
    Send position state to backend Trade Manager.
    Returns instructions: {move_sl, close_partial, close_full}.
    """
    try:
        r = requests.post(
            f"{API_URL}/bot/position_update/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            json=position,
            headers=_headers(), timeout=10,
        )
        return r.json().get("instructions", {})
    except Exception:
        return {}


def report_partial_close(order_id: str, closed_lots: float, close_price: float) -> None:
    try:
        requests.post(
            f"{API_URL}/bot/partial_closed/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            json={"order_id": order_id, "closed_lots": closed_lots, "close_price": close_price},
            headers=_headers(), timeout=10,
        )
    except Exception as e:
        log.warning(f"report_partial_close error: {e}")


def report_closed(order_id: str, close_price: float, pnl: float, reason: str) -> None:
    try:
        requests.post(
            f"{API_URL}/bot/closed/{USERNAME}",
            params={"bridge_key": BRIDGE_KEY},
            json={"order_id": order_id, "close_price": close_price, "pnl_usd": pnl, "reason": reason},
            headers=_headers(), timeout=10,
        )
    except Exception as e:
        log.warning(f"report_closed error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MT5 position management (Trade Manager)
# ─────────────────────────────────────────────────────────────────────────────

def manage_positions() -> None:
    """
    For every open position with ARIA's magic number:
    1. Send state to backend Trade Manager
    2. Apply any instructions received (move SL, partial close, full close)
    """
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        if pos.magic != MAGIC:
            continue

        symbol  = pos.symbol
        ticket  = pos.ticket
        direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        tick    = mt5.symbol_info_tick(symbol)
        if not tick:
            continue

        current_price = tick.bid if direction == "BUY" else tick.ask
        floating_pnl  = pos.profit

        state = {
            "ticket":        ticket,
            "symbol":        symbol,
            "direction":     direction,
            "open_price":    pos.price_open,
            "current_price": current_price,
            "current_sl":    pos.sl,
            "current_tp":    pos.tp,
            "lots":          pos.volume,
            "floating_pnl":  floating_pnl,
            "open_time":     pos.time,
            "comment":       pos.comment,
        }

        instructions = send_position_update(state)
        if not instructions:
            continue

        info = mt5.symbol_info(symbol)

        # ── Move stop loss (break-even or trailing) ──────────────────────────
        new_sl = instructions.get("move_sl")
        if new_sl and abs(new_sl - pos.sl) > info.point:
            req = {
                "action":   mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl":       round(float(new_sl), info.digits),
                "tp":       pos.tp,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"🔒 SL moved to {new_sl} for {symbol} #{ticket}")
            else:
                log.warning(f"SL move failed for #{ticket}: {res.retcode if res else mt5.last_error()}")

        # ── Partial close ─────────────────────────────────────────────────────
        if instructions.get("close_partial"):
            close_lots = round(pos.volume / 2, 2)
            close_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if direction == "BUY" else tick.ask
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "position":     ticket,
                "symbol":       symbol,
                "volume":       close_lots,
                "type":         close_type,
                "price":        close_price,
                "deviation":    20,
                "magic":        MAGIC,
                "comment":      "ARIA-partial",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"💰 Partial close {close_lots} lots {symbol} #{ticket}")
                report_partial_close(
                    instructions.get("order_id", ""), close_lots, close_price
                )

        # ── Full close ────────────────────────────────────────────────────────
        if instructions.get("close_full"):
            close_type  = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if direction == "BUY" else tick.ask
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "position":     ticket,
                "symbol":       symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "price":        close_price,
                "deviation":    20,
                "magic":        MAGIC,
                "comment":      f"ARIA-{instructions.get('close_reason','exit')}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"🚪 Full close {symbol} #{ticket} PnL={floating_pnl:.2f}")
                report_closed(
                    instructions.get("order_id", ""),
                    close_price,
                    floating_pnl,
                    instructions.get("close_reason", "trade_manager"),
                )


# ─────────────────────────────────────────────────────────────────────────────
# Emergency stop
# ─────────────────────────────────────────────────────────────────────────────

def emergency_close_all() -> None:
    """Close all ARIA-managed positions immediately."""
    positions = mt5.positions_get()
    if not positions:
        return
    log.warning("⚠️  EMERGENCY STOP — closing all ARIA positions")
    for pos in positions:
        if pos.magic != MAGIC:
            continue
        tick  = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            continue
        close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "position":     pos.ticket,
            "symbol":       pos.symbol,
            "volume":       pos.volume,
            "type":         close_type,
            "price":        close_price,
            "deviation":    20,
            "magic":        MAGIC,
            "comment":      "ARIA-emergency",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)
        log.warning(f"  Closed {pos.symbol} #{pos.ticket}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  ARIA MT5 Bridge  —  Starting up")
    log.info(f"  Backend : {API_URL}")
    log.info(f"  Username: {USERNAME}")
    log.info(f"  MT5     : {MT5_SERVER}  account {MT5_LOGIN}")
    log.info("=" * 60)

    # Connect to MT5
    if not mt5_connect():
        log.error("Cannot connect to MT5 — make sure MT5 is open and logged in.")
        sys.exit(1)

    # Register with backend
    register_bridge()

    consecutive_errors = 0
    pos_update_counter = 0

    while True:
        try:
            # ── Reconnect if MT5 disconnected ─────────────────────────────────
            if not mt5.terminal_info():
                log.warning("MT5 disconnected — reconnecting…")
                if not mt5_connect():
                    log.error("Reconnect failed — retrying in 30 s")
                    time.sleep(30)
                    continue
                register_bridge()

            # ── Check for emergency stop flag ─────────────────────────────────
            try:
                r = requests.get(
                    f"{API_URL}/bot/status/{USERNAME}",
                    timeout=8
                )
                status = r.json()
                if status.get("emergency_stop"):
                    emergency_close_all()
            except Exception:
                pass

            # ── Process queued orders ─────────────────────────────────────────
            orders = fetch_queue()
            for order in orders:
                order_id = order.get("order_id", order.get("id", ""))
                log.info(f"📋 Order received: {order.get('direction')} {order.get('symbol')} "
                         f"lot={order.get('lot_size')} conf={order.get('signal_confidence')}")

                success, message, ticket = place_order(order)

                if success:
                    # Get actual fill price
                    pos = mt5.positions_get(ticket=ticket)
                    fill_price = pos[0].price_open if pos else float(order.get("requested_entry", 0))
                    report_executed(order_id, ticket, fill_price)
                else:
                    log.warning(f"❌ Order rejected: {message}")
                    report_rejected(order_id, message)

            # ── Manage open positions (every 3 polls = ~15 s) ─────────────────
            pos_update_counter += 1
            if pos_update_counter >= 3:
                manage_positions()
                pos_update_counter = 0

            consecutive_errors = 0
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down bridge.")
            mt5.shutdown()
            sys.exit(0)

        except Exception as e:
            consecutive_errors += 1
            log.error(f"Unhandled error (#{consecutive_errors}): {e}")
            log.debug(traceback.format_exc())
            if consecutive_errors > 10:
                log.error("Too many consecutive errors — sleeping 5 min before retry")
                time.sleep(300)
                consecutive_errors = 0
            else:
                time.sleep(POLL_INTERVAL * 2)


if __name__ == "__main__":
    main()
