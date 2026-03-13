from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    ForeignKey, DateTime, Text, JSON
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

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    display_name    = Column(String, nullable=True)          # real / preferred name ARIA uses
    balance_ngn     = Column(Float, default=0.0)
    risk_tolerance  = Column(String, default="medium")       # low / medium / high
    trading_style   = Column(String, default="intraday")     # scalp / intraday / swing
    preferred_pairs = Column(JSON,   default=list)
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_active     = Column(DateTime, default=datetime.utcnow)

    demo_account  = relationship("DemoAccount",       back_populates="user", uselist=False, cascade="all, delete-orphan")
    conversations = relationship("Conversation",       back_populates="user", cascade="all, delete-orphan")
    memories      = relationship("UserMemory",         back_populates="user", cascade="all, delete-orphan")
    trade_journal = relationship("TradeJournalEntry",  back_populates="user", cascade="all, delete-orphan")
    watchlist     = relationship("WatchlistItem",      back_populates="user", cascade="all, delete-orphan")
    activity_log  = relationship("UserActivity",       back_populates="user", cascade="all, delete-orphan")


# ─────────────────────────────────────────────
#  Conversation  (one thread = one sidebar item)
# ─────────────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String, default="New Chat")          # auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user     = relationship("User",        back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation",
                            cascade="all, delete-orphan", order_by="ChatMessage.created_at")


# ─────────────────────────────────────────────
#  ChatMessage  (belongs to a Conversation)
# ─────────────────────────────────────────────
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    role            = Column(String,  nullable=False)        # user | aria
    content         = Column(Text,    nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# ─────────────────────────────────────────────
#  UserMemory  (facts ARIA remembers about user)
# ─────────────────────────────────────────────
class UserMemory(Base):
    __tablename__ = "user_memories"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    key        = Column(String, nullable=False)              # e.g. "name", "occupation", "goal"
    value      = Column(Text,   nullable=False)              # e.g. "Simeon", "software engineer"
    source     = Column(String, default="user_stated")       # user_stated | inferred
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")


# ─────────────────────────────────────────────
#  Demo Trading
# ─────────────────────────────────────────────
class DemoAccount(Base):
    __tablename__ = "demo_accounts"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    balance    = Column(Float, default=10000.0)
    currency   = Column(String, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User",      back_populates="demo_account")
    trades = relationship("DemoTrade", back_populates="demo_account", cascade="all, delete-orphan")


class DemoTrade(Base):
    __tablename__ = "demo_trades"

    id              = Column(Integer, primary_key=True, index=True)
    demo_account_id = Column(Integer, ForeignKey("demo_accounts.id"), nullable=False)
    symbol          = Column(String,  nullable=False)
    entry_price     = Column(Float,   nullable=False)
    current_price   = Column(Float,   nullable=False)
    exit_price      = Column(Float,   nullable=True)
    volume          = Column(Float,   nullable=False)
    trade_type      = Column(String,  nullable=False)
    pnl             = Column(Float,   default=0.0)
    is_active       = Column(Boolean, default=True)
    opened_at       = Column(DateTime, default=datetime.utcnow)
    closed_at       = Column(DateTime, nullable=True)

    demo_account = relationship("DemoAccount", back_populates="trades")


# ─────────────────────────────────────────────
#  Trade Journal
# ─────────────────────────────────────────────
class TradeJournalEntry(Base):
    __tablename__ = "trade_journal"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol      = Column(String,  nullable=False)
    direction   = Column(String,  nullable=False)
    entry_price = Column(Float,   nullable=False)
    exit_price  = Column(Float,   nullable=False)
    volume      = Column(Float,   nullable=False)
    result      = Column(String,  nullable=False)   # WIN | LOSS | BREAK_EVEN
    pnl_usd     = Column(Float,   nullable=False)
    notes       = Column(Text,    default="")
    logged_at   = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="trade_journal")


# ─────────────────────────────────────────────
#  Watchlist
# ─────────────────────────────────────────────
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id       = Column(Integer, primary_key=True, index=True)
    user_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol   = Column(String,  nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="watchlist")


# ─────────────────────────────────────────────
#  Activity log
# ─────────────────────────────────────────────
class UserActivity(Base):
    __tablename__ = "user_activity"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    action     = Column(String, nullable=False)
    symbol     = Column(String, nullable=True)
    details    = Column(JSON,   nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="activity_log")


# ─────────────────────────────────────────────
#  DB helpers
# ─────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
