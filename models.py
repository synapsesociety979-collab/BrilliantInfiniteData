from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    ForeignKey,
    DateTime,
    Text,
    JSON,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
#  User
# ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    display_name = Column(String, nullable=True)
    balance_ngn = Column(Float, default=0.0)
    risk_tolerance = Column(String, default="medium")
    trading_style = Column(String, default="intraday")
    preferred_pairs = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    demo_account = relationship(
        "DemoAccount",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    memories = relationship(
        "UserMemory", back_populates="user", cascade="all, delete-orphan"
    )
    trade_journal = relationship(
        "TradeJournalEntry", back_populates="user", cascade="all, delete-orphan"
    )
    watchlist = relationship(
        "WatchlistItem", back_populates="user", cascade="all, delete-orphan"
    )
    activity_log = relationship(
        "UserActivity", back_populates="user", cascade="all, delete-orphan"
    )
    bot_config = relationship(
        "BotConfig", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    bot_orders = relationship(
        "BotOrder", back_populates="user", cascade="all, delete-orphan"
    )


# ─────────────────────────────────────────────
#  Conversation
# ─────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


# ─────────────────────────────────────────────
#  ChatMessage
# ─────────────────────────────────────────────
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# ─────────────────────────────────────────────
#  UserMemory
# ─────────────────────────────────────────────
class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    source = Column(String, default="user_stated")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")


# ─────────────────────────────────────────────
#  Demo Trading
# ─────────────────────────────────────────────
class DemoAccount(Base):
    __tablename__ = "demo_accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    balance = Column(Float, default=10000.0)
    currency = Column(String, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="demo_account")
    trades = relationship(
        "DemoTrade", back_populates="demo_account", cascade="all, delete-orphan"
    )


class DemoTrade(Base):
    __tablename__ = "demo_trades"

    id = Column(Integer, primary_key=True, index=True)
    demo_account_id = Column(Integer, ForeignKey("demo_accounts.id"), nullable=False)
    symbol = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=False)
    trade_type = Column(String, nullable=False)
    pnl = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    demo_account = relationship("DemoAccount", back_populates="trades")


# ─────────────────────────────────────────────
#  Trade Journal
# ─────────────────────────────────────────────
class TradeJournalEntry(Base):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    result = Column(String, nullable=False)
    pnl_usd = Column(Float, nullable=False)
    notes = Column(Text, default="")
    logged_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="trade_journal")


# ─────────────────────────────────────────────
#  Watchlist
# ─────────────────────────────────────────────
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="watchlist")


# ─────────────────────────────────────────────
#  Activity log
# ─────────────────────────────────────────────
class UserActivity(Base):
    __tablename__ = "user_activity"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False)
    symbol = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="activity_log")


# ─────────────────────────────────────────────
#  Bot Config  (per-user trading bot settings)
# ─────────────────────────────────────────────
class BotConfig(Base):
    __tablename__ = "bot_configs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    is_active = Column(Boolean, default=False)

    # Risk Engine settings
    risk_percent = Column(Float, default=1.0)  # % of balance risked per trade
    max_concurrent_trades = Column(Integer, default=3)  # max open positions at once
    max_daily_loss_pct = Column(Float, default=5.0)  # halt trading if daily loss > X%
    max_weekly_loss_pct = Column(Float, default=10.0)
    max_lot_size = Column(Float, default=1.0)  # hard cap on single trade size
    min_lot_size = Column(Float, default=0.01)
    min_confidence = Column(Float, default=75.0)  # only trade signals above this

    # Market Filter settings
    max_spread_pips = Column(Float, default=3.0)
    avoid_news_minutes = Column(
        Integer, default=30
    )  # minutes before/after high-impact news
    min_atr_percentile = Column(
        Float, default=30.0
    )  # skip if ATR below 30th percentile
    allowed_sessions = Column(JSON, default=lambda: ["london", "newyork", "overlap"])
    allowed_pairs = Column(JSON, default=list)  # empty = all 30 pairs

    # Trade Manager settings
    use_break_even = Column(Boolean, default=True)
    break_even_trigger_rr = Column(
        Float, default=1.0
    )  # move SL to BE when price = entry + 1R
    use_trailing_stop = Column(Boolean, default=True)
    trail_trigger_rr = Column(Float, default=1.5)  # start trailing after 1.5R
    trail_step_atr = Column(Float, default=0.5)  # trail by 0.5 ATR steps
    use_partial_close = Column(Boolean, default=True)
    partial_close_pct = Column(Float, default=50.0)  # close 50% at TP1
    max_hold_hours = Column(Integer, default=48)  # force close if open > 48h

    # MT5 bridge authentication
    bridge_api_key = Column(String, nullable=True)  # secret key bridge must send
    mt5_account_number = Column(String, nullable=True)
    mt5_server = Column(String, nullable=True)
    mt5_broker = Column(String, nullable=True)

    # Live account state (updated each time the MT5 bridge connects)
    mt5_account_balance = Column(Float, nullable=True)   # last known balance in USD
    mt5_account_equity  = Column(Float, nullable=True)   # last known equity in USD
    bridge_last_seen    = Column(DateTime, nullable=True) # when bridge last pinged

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="bot_config")


# ─────────────────────────────────────────────
#  Bot Order  (order queue: pending → sent → executed → closed)
# ─────────────────────────────────────────────
class BotOrder(Base):
    __tablename__ = "bot_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Signal details
    symbol = Column(String, nullable=False)
    direction = Column(String, nullable=False)  # BUY | SELL
    signal_confidence = Column(Float, nullable=True)
    signal_source = Column(String, default="aria")
    timeframe = Column(String, nullable=True)

    # Pricing (set by risk engine at time of order creation)
    requested_entry = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=True)
    take_profit_3 = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=False)

    # Risk details (logged for audit)
    risk_percent = Column(Float, nullable=True)
    risk_usd = Column(Float, nullable=True)
    sl_pips = Column(Float, nullable=True)
    rr_ratio = Column(String, nullable=True)

    # MT5 execution results (filled by bridge)
    mt5_ticket = Column(Integer, nullable=True, index=True)
    filled_price = Column(Float, nullable=True)
    filled_at = Column(DateTime, nullable=True)

    # Trade management state (updated by bridge during life of trade)
    current_sl = Column(Float, nullable=True)  # latest SL (may have trailed)
    current_price = Column(Float, nullable=True)  # latest price from bridge
    floating_pnl_usd = Column(Float, nullable=True)
    sl_at_breakeven = Column(Boolean, default=False)
    tp1_closed = Column(Boolean, default=False)  # partial close at TP1 done
    trailing_active = Column(Boolean, default=False)

    # Close details
    close_price = Column(Float, nullable=True)
    close_reason = Column(
        String, nullable=True
    )  # TP1/TP2/SL/MANUAL/TIME_EXIT/EMERGENCY
    realised_pnl_usd = Column(Float, nullable=True)

    # Status lifecycle
    # PENDING → QUEUED → SENT → EXECUTED → ACTIVE → CLOSED | REJECTED | CANCELLED
    status = Column(String, default="PENDING", index=True)
    reject_reason = Column(String, nullable=True)

    # Market filter audit trail
    filter_passed = Column(Boolean, default=True)
    filter_block_reasons = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="bot_orders")


# ─────────────────────────────────────────────
#  DB helpers
# ─────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)
    # Safe column migrations — add new columns if they don't exist yet
    migrations = [
        "ALTER TABLE bot_configs ADD COLUMN IF NOT EXISTS mt5_account_balance FLOAT",
        "ALTER TABLE bot_configs ADD COLUMN IF NOT EXISTS mt5_account_equity FLOAT",
        "ALTER TABLE bot_configs ADD COLUMN IF NOT EXISTS bridge_last_seen TIMESTAMP",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                conn.rollback()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
