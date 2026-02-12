from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # demo relationship
    demo_account = relationship("DemoAccount", back_populates="user", uselist=False)

class DemoAccount(Base):
    __tablename__ = "demo_accounts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    balance = Column(Float, default=10000.0)
    currency = Column(String, default="USD")
    
    user = relationship("User", back_populates="demo_account")
    trades = relationship("Trade", back_populates="demo_account")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    demo_account_id = Column(Integer, ForeignKey("demo_accounts.id"))
    symbol = Column(String)
    entry_price = Column(Float)
    current_price = Column(Float)
    volume = Column(Float)
    trade_type = Column(String)
    pnl = Column(Float, default=0.0)
    is_demo = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    demo_account = relationship("DemoAccount", back_populates="trades")

def init_db():
    Base.metadata.create_all(bind=engine)
