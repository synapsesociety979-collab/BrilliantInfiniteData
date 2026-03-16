# trading_engine.py
# ─────────────────────────────────────────────────────────────────
#  Three independent components that MUST all pass before any
#  real money trade is placed:
#
#  1. RiskEngine    – lot size, exposure, daily/weekly loss limits
#  2. MarketFilter  – spread, session, volatility, concurrent cap
#  3. TradeManager  – break-even, trailing stop, partial close,
#                     time-based exit decisions
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session


# ═══════════════════════════════════════════════════════════════
#  Symbol metadata — pip size, pip value per standard lot (USD)
# ═══════════════════════════════════════════════════════════════
SYMBOL_META: Dict[str, Dict] = {
    # Major Forex  (pip = 0.0001, $10 per pip per 1.0 lot)
    "EURUSD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 0.8},
    "GBPUSD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.0},
    "AUDUSD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.0},
    "NZDUSD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.5},
    "USDCAD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.2},
    "USDCHF": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.2},
    # JPY pairs  (pip = 0.01, ~$6.67 per pip per lot — approx @ 150 JPY/USD)
    "USDJPY": {"pip": 0.01,   "pip_value": 6.67,  "category": "forex", "typical_spread": 0.8},
    "EURJPY": {"pip": 0.01,   "pip_value": 6.67,  "category": "forex", "typical_spread": 1.2},
    "GBPJPY": {"pip": 0.01,   "pip_value": 6.67,  "category": "forex", "typical_spread": 1.8},
    "AUDJPY": {"pip": 0.01,   "pip_value": 6.67,  "category": "forex", "typical_spread": 1.5},
    # Crosses
    "EURGBP": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 1.2},
    "EURCAD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 2.0},
    "EURAUD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 2.0},
    "GBPAUD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 2.5},
    "AUDCAD": {"pip": 0.0001, "pip_value": 10.0,  "category": "forex", "typical_spread": 2.0},
    # Crypto  (treat 1 USD move as 1 "pip", $1 per lot per USD move)
    "BTCUSDT":  {"pip": 1.0, "pip_value": 1.0,  "category": "crypto", "typical_spread": 15.0},
    "ETHUSDT":  {"pip": 0.1, "pip_value": 1.0,  "category": "crypto", "typical_spread": 1.0},
    "BNBUSDT":  {"pip": 0.1, "pip_value": 1.0,  "category": "crypto", "typical_spread": 0.5},
    "XRPUSDT":  {"pip": 0.0001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.001},
    "SOLUSDT":  {"pip": 0.01, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.1},
    "ADAUSDT":  {"pip": 0.0001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.001},
    "DOGEUSDT": {"pip": 0.00001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.0001},
    "DOTUSDT":  {"pip": 0.01, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.05},
    "MATICUSDT":{"pip": 0.0001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.001},
    "LTCUSDT":  {"pip": 0.01, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.2},
    "AVAXUSDT": {"pip": 0.01, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.1},
    "LINKUSDT": {"pip": 0.001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.02},
    "UNIUSDT":  {"pip": 0.001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.02},
    "SHIBUSDT": {"pip": 0.000001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.0000001},
    "TRXUSDT":  {"pip": 0.0001, "pip_value": 1.0, "category": "crypto", "typical_spread": 0.0001},
}

LOT_STEP    = 0.01   # MT5 minimum lot step
LOT_MIN     = 0.01
LOT_MAX_CAP = 10.0   # absolute hard cap regardless of config


def _round_lot(raw: float, min_lot: float, max_lot: float) -> float:
    """Round to nearest MT5 lot step and clamp within limits."""
    stepped = math.floor(raw / LOT_STEP) * LOT_STEP
    return round(max(min_lot, min(max_lot, stepped)), 2)


def _pip_distance(entry: float, sl: float, symbol: str) -> float:
    """Return SL distance in pips."""
    meta = SYMBOL_META.get(symbol.upper(), {"pip": 0.0001})
    return abs(entry - sl) / meta["pip"]


# ═══════════════════════════════════════════════════════════════
#  1. RISK ENGINE
# ═══════════════════════════════════════════════════════════════
class RiskEngine:
    """
    Calculates safe lot size and checks whether account exposure
    limits allow a new trade to be placed.
    """

    @staticmethod
    def calculate_lot_size(
        account_balance_usd: float,
        risk_percent: float,
        entry: float,
        stop_loss: float,
        symbol: str,
        min_lot: float = LOT_MIN,
        max_lot: float = 1.0,
    ) -> Dict:
        """
        ATR-style position sizing:
          lot_size = risk_amount_usd / (sl_pips × pip_value_per_lot)
        Returns dict with lot_size, risk_usd, sl_pips, rr details.
        """
        symbol = symbol.upper()
        meta = SYMBOL_META.get(symbol, {"pip": 0.0001, "pip_value": 10.0})
        pip_size  = meta["pip"]
        pip_value = meta["pip_value"]   # USD per pip per 1.0 lot

        risk_usd = account_balance_usd * (risk_percent / 100)
        sl_pips  = abs(entry - stop_loss) / pip_size

        if sl_pips == 0:
            return {"error": "Stop loss cannot equal entry price", "lot_size": 0}

        raw_lot  = risk_usd / (sl_pips * pip_value)
        lot_size = _round_lot(raw_lot, min_lot, min(max_lot, LOT_MAX_CAP))

        # Re-derive actual risk with snapped lot size
        actual_risk_usd = lot_size * sl_pips * pip_value

        return {
            "lot_size":          lot_size,
            "risk_usd":          round(actual_risk_usd, 2),
            "risk_percent":      round(actual_risk_usd / account_balance_usd * 100, 3),
            "sl_pips":           round(sl_pips, 1),
            "pip_value_per_lot": pip_value,
            "symbol":            symbol,
            "account_balance":   account_balance_usd,
        }

    @staticmethod
    def check_daily_loss(
        user_id: int,
        config,
        db: Session
    ) -> Dict:
        """
        Check if today's realised losses have hit the daily/weekly halt threshold.
        Returns {"allowed": bool, "reason": str, "daily_loss_usd": float}
        """
        from models import BotOrder
        today  = datetime.utcnow().date()
        monday = today - timedelta(days=today.weekday())

        closed_today = db.query(BotOrder).filter(
            BotOrder.user_id == user_id,
            BotOrder.status  == "CLOSED",
            BotOrder.closed_at >= datetime.combine(today, datetime.min.time()),
        ).all()

        closed_week = db.query(BotOrder).filter(
            BotOrder.user_id == user_id,
            BotOrder.status  == "CLOSED",
            BotOrder.closed_at >= datetime.combine(monday, datetime.min.time()),
        ).all()

        daily_pnl  = sum(o.realised_pnl_usd or 0 for o in closed_today)
        weekly_pnl = sum(o.realised_pnl_usd or 0 for o in closed_week)

        # Estimate account balance (fallback if unknown: use config's implicit assumption)
        # We use the first order's risk_usd / risk_percent to back-calculate balance
        all_orders = db.query(BotOrder).filter(BotOrder.user_id == user_id).first()
        est_balance = 1000.0
        if all_orders and all_orders.risk_usd and all_orders.risk_percent:
            est_balance = all_orders.risk_usd / (all_orders.risk_percent / 100)

        max_daily_loss  = est_balance * (config.max_daily_loss_pct  / 100)
        max_weekly_loss = est_balance * (config.max_weekly_loss_pct / 100)

        if daily_pnl < -max_daily_loss:
            return {
                "allowed": False,
                "reason":  f"Daily loss limit hit: ${-daily_pnl:.2f} lost today (limit ${max_daily_loss:.2f})",
                "daily_loss_usd": round(daily_pnl, 2),
                "weekly_loss_usd": round(weekly_pnl, 2),
            }
        if weekly_pnl < -max_weekly_loss:
            return {
                "allowed": False,
                "reason":  f"Weekly loss limit hit: ${-weekly_pnl:.2f} lost this week (limit ${max_weekly_loss:.2f})",
                "daily_loss_usd": round(daily_pnl, 2),
                "weekly_loss_usd": round(weekly_pnl, 2),
            }
        return {
            "allowed": True,
            "daily_loss_usd": round(daily_pnl, 2),
            "weekly_loss_usd": round(weekly_pnl, 2),
        }

    @staticmethod
    def check_concurrent_positions(user_id: int, symbol: str, config, db: Session) -> Dict:
        """Check if adding this trade would exceed concurrent position limits."""
        from models import BotOrder
        active = db.query(BotOrder).filter(
            BotOrder.user_id == user_id,
            BotOrder.status.in_(["QUEUED", "SENT", "EXECUTED", "ACTIVE"]),
        ).all()

        if len(active) >= config.max_concurrent_trades:
            return {
                "allowed": False,
                "reason": f"Max concurrent trades reached ({len(active)}/{config.max_concurrent_trades})",
                "active_count": len(active),
            }

        # Also block same symbol if already in active position
        same_symbol = [o for o in active if o.symbol == symbol.upper()]
        if same_symbol:
            return {
                "allowed": False,
                "reason": f"Already have an active position on {symbol}",
                "active_count": len(active),
            }

        return {"allowed": True, "active_count": len(active)}


# ═══════════════════════════════════════════════════════════════
#  2. MARKET FILTER
# ═══════════════════════════════════════════════════════════════

# High-impact news windows (UTC hour, minute) — hardcoded common ones
# Key: (weekday 0=Mon, hour) → description
HIGH_IMPACT_NEWS_UTC: List[Tuple] = [
    # NFP — first Friday of month 13:30 UTC
    (4, 13, 30, "US Non-Farm Payrolls"),
    # FOMC — varies but roughly Wednesday 18:00 UTC
    (2, 18,  0, "FOMC Rate Decision"),
    # CPI releases — variable but often Tuesday/Wednesday 13:30 UTC
    (1, 13, 30, "US CPI"),
    (2, 13, 30, "US CPI"),
    # ECB Press Conference — Thursday 13:45 UTC
    (3, 13, 45, "ECB Press Conference"),
    # BOE — Thursday 12:00 UTC
    (3, 12,  0, "BOE Rate Decision"),
]


class MarketFilter:
    """
    Checks conditions that make a market dangerous to trade.
    ALL checks must pass before an order is queued.
    """

    @staticmethod
    def check_spread(symbol: str, current_spread_pips: Optional[float], max_spread_pips: float) -> Dict:
        if current_spread_pips is None:
            return {"passed": True, "note": "Spread unknown — assumed acceptable"}
        meta    = SYMBOL_META.get(symbol.upper(), {})
        typical = meta.get("typical_spread", 2.0)
        limit   = max(max_spread_pips, typical * 3)   # allow up to 3× typical even if config is tight

        if current_spread_pips > limit:
            return {
                "passed": False,
                "reason": f"Spread too wide: {current_spread_pips:.1f} pips (limit {limit:.1f} pips)",
            }
        return {"passed": True, "spread_pips": current_spread_pips}

    @staticmethod
    def check_session(symbol: str, utc_hour: int, allowed_sessions: List[str]) -> Dict:
        """Verify the current market session is in the allowed list for this symbol."""
        if 7 <= utc_hour < 12:
            session = "london"
        elif 12 <= utc_hour < 17:
            session = "overlap"
        elif 17 <= utc_hour < 21:
            session = "newyork"
        elif 21 <= utc_hour or utc_hour < 7:
            session = "asian"
        else:
            session = "transition"

        if "all" in [s.lower() for s in allowed_sessions]:
            return {"passed": True, "session": session}

        # Session–symbol fitness
        sym = symbol.upper()
        poor_fit = {
            "asian":   ["EURUSD", "GBPUSD", "EURGBP", "EURAUD", "GBPAUD"],  # low liquidity Asian
            "london":  ["BTCUSDT", "ETHUSDT"],                                # crypto less liquid London open
        }
        if session not in [s.lower() for s in allowed_sessions]:
            return {
                "passed": False,
                "reason": f"Session '{session}' not in allowed sessions {allowed_sessions}",
                "session": session,
            }
        if sym in poor_fit.get(session, []):
            return {
                "passed": True,   # allow but warn
                "warning": f"{sym} has reduced liquidity during {session} session",
                "session": session,
            }
        return {"passed": True, "session": session}

    @staticmethod
    def check_news_window(avoid_minutes: int) -> Dict:
        """Block trading within N minutes of known high-impact news."""
        now = datetime.utcnow()
        for weekday, hour, minute, label in HIGH_IMPACT_NEWS_UTC:
            if now.weekday() != weekday:
                continue
            news_time  = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            delta_min  = abs((now - news_time).total_seconds() / 60)
            if delta_min <= avoid_minutes:
                return {
                    "passed": False,
                    "reason": f"Within {avoid_minutes}min of high-impact news: {label} at {hour:02d}:{minute:02d} UTC",
                    "news_event": label,
                    "minutes_away": round(delta_min, 1),
                }
        return {"passed": True}

    @staticmethod
    def check_volatility(atr: Optional[float], atr_avg: Optional[float], min_percentile: float) -> Dict:
        """
        Skip trades in ranging / dead markets.
        Require ATR > min_percentile% of its rolling average.
        """
        if atr is None or atr_avg is None or atr_avg == 0:
            return {"passed": True, "note": "ATR data unavailable — assumed acceptable"}
        threshold = atr_avg * (min_percentile / 100)
        if atr < threshold:
            return {
                "passed": False,
                "reason": f"Low volatility: ATR {atr:.6f} < {min_percentile}% of avg {atr_avg:.6f} — market is ranging",
                "atr": atr,
                "atr_avg": atr_avg,
            }
        return {"passed": True, "atr": atr, "atr_avg": atr_avg}

    @staticmethod
    def check_confidence(signal_confidence: float, min_confidence: float) -> Dict:
        if signal_confidence < min_confidence:
            return {
                "passed": False,
                "reason": f"Signal confidence {signal_confidence}% below minimum {min_confidence}%",
            }
        return {"passed": True}

    @classmethod
    def run_all_checks(
        cls,
        symbol: str,
        signal_confidence: float,
        config,
        current_spread_pips: Optional[float] = None,
        atr: Optional[float] = None,
        atr_avg: Optional[float] = None,
    ) -> Dict:
        """
        Run every filter in sequence.
        Returns {"approved": bool, "blocks": [...], "warnings": [...]}
        """
        now     = datetime.utcnow()
        blocks  = []
        warnings = []

        checks = [
            cls.check_confidence(signal_confidence, config.min_confidence),
            cls.check_spread(symbol, current_spread_pips, config.max_spread_pips),
            cls.check_session(symbol, now.hour, config.allowed_sessions or ["london", "overlap", "newyork"]),
            cls.check_news_window(config.avoid_news_minutes),
            cls.check_volatility(atr, atr_avg, config.min_atr_percentile),
        ]

        # Allowed-pairs whitelist
        if config.allowed_pairs:
            if symbol.upper() not in [p.upper() for p in config.allowed_pairs]:
                blocks.append(f"{symbol} is not in your allowed pairs list")

        for result in checks:
            if not result.get("passed"):
                blocks.append(result.get("reason", "Unknown filter block"))
            elif result.get("warning"):
                warnings.append(result["warning"])

        return {
            "approved":  len(blocks) == 0,
            "blocks":    blocks,
            "warnings":  warnings,
            "checked_at": now.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════
#  3. TRADE MANAGER
# ═══════════════════════════════════════════════════════════════
class TradeManager:
    """
    Given a live position update, decides what action to take:
    HOLD | MOVE_SL_BREAKEVEN | TRAIL_SL | PARTIAL_CLOSE | CLOSE
    """

    @staticmethod
    def evaluate_position(order, current_price: float, config) -> Dict:
        """
        Evaluate a single open position and return the recommended action.

        order: BotOrder instance
        current_price: latest market price from MT5 bridge
        config: BotConfig instance

        Returns:
          {
            "action": "HOLD" | "MOVE_SL_BREAKEVEN" | "TRAIL_SL" | "PARTIAL_CLOSE" | "CLOSE",
            "new_sl": float | None,
            "close_reason": str | None,
            "details": str
          }
        """
        entry    = order.filled_price or order.requested_entry
        sl       = order.current_sl or order.stop_loss
        tp1      = order.take_profit_1
        tp2      = order.take_profit_2
        direction = order.direction.upper()  # BUY | SELL

        if not entry:
            return {"action": "HOLD", "details": "No entry price yet"}

        # Calculate R (initial risk distance in price units)
        R = abs(entry - order.stop_loss)
        if R == 0:
            return {"action": "HOLD", "details": "R = 0, cannot calculate position"}

        # Current P&L in R multiples
        if direction == "BUY":
            price_delta = current_price - entry
        else:
            price_delta = entry - current_price
        current_r = price_delta / R

        details_parts = [f"Entry={entry}, Price={current_price}, R={R:.6f}, Current={current_r:.2f}R"]

        # ── Time-based exit ──────────────────────────────────────
        if order.executed_at:
            hours_open = (datetime.utcnow() - order.executed_at).total_seconds() / 3600
            if hours_open > config.max_hold_hours and current_r < 0.5:
                return {
                    "action": "CLOSE",
                    "new_sl": None,
                    "close_reason": f"TIME_EXIT ({hours_open:.1f}h open, only {current_r:.2f}R profit)",
                    "details": " | ".join(details_parts),
                }

        # ── Stop-loss was hit (broker should have closed it, but safety check) ──
        if direction == "BUY" and current_price <= order.stop_loss:
            return {
                "action": "CLOSE",
                "new_sl": None,
                "close_reason": "SL_HIT",
                "details": " | ".join(details_parts),
            }
        if direction == "SELL" and current_price >= order.stop_loss:
            return {
                "action": "CLOSE",
                "new_sl": None,
                "close_reason": "SL_HIT",
                "details": " | ".join(details_parts),
            }

        # ── TP1 partial close ────────────────────────────────────
        if config.use_partial_close and not order.tp1_closed and tp1:
            tp1_hit = (direction == "BUY" and current_price >= tp1) or \
                      (direction == "SELL" and current_price <= tp1)
            if tp1_hit:
                return {
                    "action":       "PARTIAL_CLOSE",
                    "close_pct":    config.partial_close_pct,
                    "new_sl":       entry,   # move SL to break-even simultaneously
                    "close_reason": "TP1",
                    "details":      " | ".join(details_parts + ["TP1 reached"]),
                }

        # ── Break-even ───────────────────────────────────────────
        if config.use_break_even and not order.sl_at_breakeven:
            if current_r >= config.break_even_trigger_rr:
                # Move SL to entry + small buffer (2 pips)
                meta   = SYMBOL_META.get(order.symbol.upper(), {"pip": 0.0001})
                buffer = meta["pip"] * 2
                if direction == "BUY":
                    new_sl = round(entry + buffer, 6)
                else:
                    new_sl = round(entry - buffer, 6)
                # Only move if new_sl is better than current SL
                sl_improved = (direction == "BUY" and new_sl > sl) or \
                              (direction == "SELL" and new_sl < sl)
                if sl_improved:
                    details_parts.append(f"Moving SL to break-even @ {new_sl}")
                    return {
                        "action": "MOVE_SL_BREAKEVEN",
                        "new_sl": new_sl,
                        "close_reason": None,
                        "details": " | ".join(details_parts),
                    }

        # ── Trailing stop ────────────────────────────────────────
        if config.use_trailing_stop and current_r >= config.trail_trigger_rr:
            meta    = SYMBOL_META.get(order.symbol.upper(), {"pip": 0.0001})
            trail_distance = R * config.trail_step_atr   # trail by fraction of R

            if direction == "BUY":
                candidate_sl = round(current_price - trail_distance, 6)
                if candidate_sl > sl:
                    details_parts.append(f"Trailing SL up to {candidate_sl}")
                    return {
                        "action": "TRAIL_SL",
                        "new_sl": candidate_sl,
                        "close_reason": None,
                        "details": " | ".join(details_parts),
                    }
            else:
                candidate_sl = round(current_price + trail_distance, 6)
                if candidate_sl < sl:
                    details_parts.append(f"Trailing SL down to {candidate_sl}")
                    return {
                        "action": "TRAIL_SL",
                        "new_sl": candidate_sl,
                        "close_reason": None,
                        "details": " | ".join(details_parts),
                    }

        # ── TP2 / TP3 full close ─────────────────────────────────
        if tp2:
            tp2_hit = (direction == "BUY" and current_price >= tp2) or \
                      (direction == "SELL" and current_price <= tp2)
            if tp2_hit:
                return {
                    "action": "CLOSE",
                    "new_sl": None,
                    "close_reason": "TP2",
                    "details": " | ".join(details_parts + ["TP2 reached"]),
                }

        if tp1 and not tp2:
            tp_hit = (direction == "BUY" and current_price >= tp1) or \
                     (direction == "SELL" and current_price <= tp1)
            if tp_hit:
                return {
                    "action": "CLOSE",
                    "new_sl": None,
                    "close_reason": "TP1_FULL",
                    "details": " | ".join(details_parts + ["TP1 full close"]),
                }

        return {
            "action":       "HOLD",
            "new_sl":       None,
            "close_reason": None,
            "details":      " | ".join(details_parts + [f"No action at {current_r:.2f}R"]),
        }

    @staticmethod
    def calculate_pnl(direction: str, entry: float, close: float,
                      lot_size: float, symbol: str) -> float:
        """Estimate realised P&L in USD."""
        meta      = SYMBOL_META.get(symbol.upper(), {"pip": 0.0001, "pip_value": 10.0})
        pip_size  = meta["pip"]
        pip_value = meta["pip_value"]
        pips = (close - entry) / pip_size if direction == "BUY" else (entry - close) / pip_size
        return round(pips * pip_value * lot_size, 2)
