"""
metaapi_client.py — CLEO MetaApi REST client
Handles provisioning, balance fetching, and trade execution via MetaApi cloud.
No Windows machine or bridge needed — MetaApi runs the MT5 terminal in their cloud.

Docs: https://metaapi.cloud/docs/client/restApi/
"""
import os, requests, time
from typing import Optional, Dict, Any

METAAPI_TOKEN = os.getenv("METAAPI_TOKEN", "")

_PROV_BASE  = "https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai"
_API_BASE   = "https://mt-api-v1.agiliumtrade.agiliumtrade.ai"
_TIMEOUT    = 20  # seconds per HTTP call


def _headers() -> dict:
    return {
        "auth-token"  : METAAPI_TOKEN,
        "Content-Type": "application/json",
    }


# ─────────────────────────────────────────────────────────────
#  Account provisioning
# ─────────────────────────────────────────────────────────────

def provision_account(
    login: str,
    password: str,
    server: str,
    name: str,
    platform: str = "mt5",
) -> Dict[str, Any]:
    """
    Create a new MetaApi cloud account for this MT5 login.
    Returns the full account object including 'id'.
    Raises on failure.
    """
    payload = {
        "login"      : str(login),
        "password"   : password,
        "name"       : name,
        "server"     : server,
        "platform"   : platform.lower(),   # "mt5" or "mt4"
        "type"       : "cloud",
        "magic"      : 202600,
        "application": "MetaApi",
        "region"     : "new-york",
        "reliability": "regular",
    }
    r = requests.post(
        f"{_PROV_BASE}/users/current/accounts",
        json    = payload,
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"MetaApi provision failed {r.status_code}: {r.text[:400]}")
    return r.json()


def get_account(account_id: str) -> Dict[str, Any]:
    """Fetch the provisioning record — includes state (DEPLOYING / DEPLOYED / etc.)"""
    r = requests.get(
        f"{_PROV_BASE}/users/current/accounts/{account_id}",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(f"MetaApi get_account failed {r.status_code}: {r.text[:300]}")
    return r.json()


def delete_account(account_id: str) -> bool:
    """Remove a MetaApi account (called when user disconnects)."""
    r = requests.delete(
        f"{_PROV_BASE}/users/current/accounts/{account_id}",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    return r.status_code in (200, 204)


def list_accounts() -> list:
    """List all MetaApi accounts for this token."""
    r = requests.get(
        f"{_PROV_BASE}/users/current/accounts",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code != 200:
        return []
    return r.json()


def find_account_by_login(login: str) -> Optional[Dict[str, Any]]:
    """Return the first provisioned account matching the MT5 login number."""
    try:
        accounts = list_accounts()
        for acc in accounts:
            if str(acc.get("login")) == str(login):
                return acc
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────
#  Live trading data
# ─────────────────────────────────────────────────────────────

def get_account_info(account_id: str) -> Dict[str, Any]:
    """
    Returns live balance, equity, margin, free margin, profit.
    Raises if account is not yet DEPLOYED or not connected to broker.
    """
    r = requests.get(
        f"{_API_BASE}/users/current/accounts/{account_id}/account-information",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code == 404:
        raise RuntimeError("Account not deployed yet — still connecting to broker")
    if r.status_code != 200:
        raise RuntimeError(f"MetaApi account-information failed {r.status_code}: {r.text[:300]}")
    return r.json()


def get_positions(account_id: str) -> list:
    """Returns list of currently open positions."""
    r = requests.get(
        f"{_API_BASE}/users/current/accounts/{account_id}/positions",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code != 200:
        return []
    return r.json()


def get_orders(account_id: str) -> list:
    """Returns list of pending orders."""
    r = requests.get(
        f"{_API_BASE}/users/current/accounts/{account_id}/orders",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code != 200:
        return []
    return r.json()


def get_history_deals(account_id: str, days: int = 7) -> list:
    """Returns closed trades from the last N days."""
    import datetime as _dt
    start = (_dt.datetime.utcnow() - _dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end   = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    r = requests.get(
        f"{_API_BASE}/users/current/accounts/{account_id}/history-deals/time/{start}/{end}",
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code != 200:
        return []
    return r.json()


# ─────────────────────────────────────────────────────────────
#  Trade execution
# ─────────────────────────────────────────────────────────────

def place_trade(
    account_id : str,
    symbol     : str,
    direction  : str,        # "BUY" or "SELL"
    volume     : float,
    stop_loss  : Optional[float] = None,
    take_profit: Optional[float] = None,
    comment    : str = "CLEO",
    magic      : int = 202600,
) -> Dict[str, Any]:
    """
    Place a market order via MetaApi.
    direction: "BUY" → ORDER_TYPE_BUY | "SELL" → ORDER_TYPE_SELL
    """
    action = "ORDER_TYPE_BUY" if direction.upper() == "BUY" else "ORDER_TYPE_SELL"
    payload: Dict[str, Any] = {
        "actionType": action,
        "symbol"    : symbol,
        "volume"    : round(volume, 2),
        "comment"   : comment,
        "magic"     : magic,
    }
    if stop_loss   is not None: payload["stopLoss"]   = stop_loss
    if take_profit is not None: payload["takeProfit"] = take_profit

    r = requests.post(
        f"{_API_BASE}/users/current/accounts/{account_id}/trade",
        json    = payload,
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"MetaApi trade failed {r.status_code}: {r.text[:400]}")
    return r.json()


def close_position(account_id: str, position_id: str) -> Dict[str, Any]:
    """Close an open position by its position ID."""
    payload = {
        "actionType": "POSITION_CLOSE_ID",
        "positionId": position_id,
    }
    r = requests.post(
        f"{_API_BASE}/users/current/accounts/{account_id}/trade",
        json    = payload,
        headers = _headers(),
        timeout = _TIMEOUT,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"MetaApi close failed {r.status_code}: {r.text[:300]}")
    return r.json()


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def wait_for_deployment(account_id: str, timeout_seconds: int = 180) -> bool:
    """
    Poll until the account reaches DEPLOYED state or timeout.
    Returns True if deployed, False if timed out.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            acc   = get_account(account_id)
            state = acc.get("state", "")
            conn  = acc.get("connectionStatus", "")
            print(f"[MetaApi] Account {account_id[:8]}… state={state} conn={conn}")
            if state == "DEPLOYED" and conn == "CONNECTED":
                return True
            if state in ("DEPLOY_FAILED", "DELETING"):
                return False
        except Exception as e:
            print(f"[MetaApi] Poll error: {e}")
        time.sleep(10)
    return False


def is_token_valid() -> bool:
    """Quick check that METAAPI_TOKEN is set and the API is reachable."""
    if not METAAPI_TOKEN:
        return False
    try:
        r = requests.get(
            f"{_PROV_BASE}/users/current/accounts",
            headers = _headers(),
            timeout = 8,
        )
        return r.status_code == 200
    except Exception:
        return False
