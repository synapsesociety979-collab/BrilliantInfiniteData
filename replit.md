# AI Multi-User Trading Bot Backend — ARIA v3.1

## Project Overview
FastAPI backend for ARIA — an AI-powered institutional-grade trading bot analyzing 30 Forex and crypto markets. Provides real-time signal generation from live Alpha Vantage OHLCV data, NGN/USD conversion, demo trading, persistent PostgreSQL user data, personalized advice, and a visual trading simulator.

## Current Status (March 2026)
- **Backend**: Running on port 5000
- **Data Source**: Alpha Vantage real-time OHLCV + local indicator calculation
- **AI Model**: llama-3.3-70b-versatile via Groq
- **Persistence**: Full PostgreSQL (users, trades, chat, journal, watchlist, activity)
- **Accuracy Target**: 70%+ via real indicator data fed to AI

## Architecture

### Key Components
- **FastAPI Server** (`main.py`): All API endpoints, prediction engine, demo trading
- **Market Data Engine** (`market_data.py`): Live OHLCV fetching + full indicator computation
- **AI Provider** (`ai_provider.py`): Groq API wrapper (llama-3.3-70b-versatile)
- **Database Models** (`models.py`): Full PostgreSQL schema via SQLAlchemy
- **Backtest Engine** (`backtest_api.py`): Historical backtesting via CSV upload

### Tech Stack
- Python 3 with FastAPI, Uvicorn
- Groq AI (llama-3.3-70b-versatile) — ARIA persona
- Alpha Vantage API — live OHLCV candles + exchange rates
- PostgreSQL — full persistent storage
- SQLAlchemy ORM
- pandas + numpy — local indicator computation
- CORS middleware for Lovable frontend

## How Live Data + AI Works (Accuracy Pipeline)

### Old approach (inaccurate):
AI was asked to "analyze EURUSD" with no actual data → it would hallucinate indicator values

### New approach (70%+ accuracy):
1. `market_data.py` fetches real 5-min OHLCV candles from Alpha Vantage
2. Computes ALL indicators locally from real price data:
   - RSI(14), MACD(12,26,9), EMA(20/50/200), Bollinger Bands(20,2), ATR(14)
   - Volume analysis, Support/Resistance, Candle pattern detection
3. Real values injected into AI prompt: "RSI=34.2 (oversold), MACD histogram=-0.0023..."
4. AI INTERPRETS real data instead of guessing
5. 15-minute cache per symbol to respect API rate limits
6. Session-priority system: top 12 most relevant symbols get real data each cycle

### Rate Limit Management
- Alpha Vantage free tier: ~25 req/day, 5 req/min
- Solution: 15-minute cache per symbol, session-priority ordering
- Real data for top 12 session-relevant symbols; AI reasoning for the rest
- Single-symbol lookups (`/get_predictions`, `/live_data`) always get real data

## API Endpoints

### Market Data & Signals
- `GET /` — Health check
- `GET /predictions?amount_ngn=X&username=Y` — All 30 symbols with real data
- `GET /get_predictions?username=X&symbol=Y` — Personalized single-symbol (FIXED + real data)
- `GET /advice/{symbol}` — Deep institutional analysis
- `GET /market_analysis` — Overall market summary
- `GET /live_data/{symbol}` — Raw real-time indicators (RSI, MACD, EMA, BB, ATR, etc.)

### User Management
- `POST /user/{username}` — Create/update user profile (balance NGN, style, risk)
- `GET /user/{username}` — Full profile with stats
- `GET /account/monitor/{username}` — Account overview with PnL

### AI Chat (Persistent)
- `POST /chat` — Chat with ARIA (DB-persistent per user)
- `GET /chat/history/{username}` — Full chat history
- `DELETE /chat/history/{username}` — Clear history

### Trade Journal (DB-persistent)
- `POST /journal/{username}` — Log a trade
- `GET /journal/{username}` — Journal + win rate stats

### Watchlist (DB-persistent)
- `POST /watchlist/{username}?symbol=X` — Add symbol
- `DELETE /watchlist/{username}/{symbol}` — Remove symbol
- `GET /watchlist/{username}` — List with live signals

### Demo Trading (DB-persistent)
- `POST /demo/open_account/{username}` — Open demo account
- `POST /demo/execute_trade/{username}` — Open a demo trade
- `POST /demo/close_trade/{username}/{trade_id}` — Close trade with real PnL
- `GET /demo/account/{username}` — Account balance, active trades, history
- `GET /demo/ai_feedback/{username}` — ARIA reviews your open demo trades

### Utilities
- `POST /risk_calculator` — Position size in NGN + USD
- `POST /score_trade` — AI grades your trade idea (A+ to F)
- `GET /connect_account` — Register user account

### Visual Simulator
- `GET /simulator/candles/{symbol}` — Simulated OHLCV candles (GBM)
- `GET /simulator/run/{symbol}` — Full backtest with equity curve + entry/exit markers

### Backtesting (CSV-based)
- `POST /api/upload_history` — Upload CSV history
- `POST /api/backtest` — Run SMA or AI backtest
- `GET /api/history/{symbol}` — View uploaded history

## Database Schema (PostgreSQL)

| Table | Purpose |
|---|---|
| `users` | Username, balance NGN, trading style, risk tolerance, preferences |
| `demo_accounts` | Demo balance per user |
| `demo_trades` | Active and closed demo trades with PnL |
| `chat_messages` | Full ARIA conversation history per user |
| `trade_journal` | Logged real trades with result + PnL |
| `watchlist` | User watchlist items |
| `user_activity` | Every action logged for AI personalization |

## Assets Analyzed (30 total)
**Forex (15)**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF, EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, AUDCAD, EURAUD, GBPAUD
**Crypto (15)**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, DOTUSDT, MATICUSDT, LTCUSDT, SHIBUSDT, TRXUSDT, AVAXUSDT, LINKUSDT, UNIUSDT

## Key Files
- `main.py` — All endpoints + prediction engine (use `write` tool to edit, not `edit`)
- `market_data.py` — Live data fetching + indicator computation
- `models.py` — PostgreSQL schema
- `ai_provider.py` — Groq AI wrapper
- `backtest_api.py` — Backtest engine + CSV routes

## Environment Variables / Secrets
- `GROQ_API_KEY` — AI model access
- `ALPHA_VANTAGE_API_KEY` — Live market data
- `DATABASE_URL` — PostgreSQL connection

## Deployment
- **Dev**: `uvicorn main:app --host 0.0.0.0 --port 5000 --reload`
- **Prod**: `uvicorn main:app --host 0.0.0.0 --port 5000`

## User Preferences
- ARIA persona for all AI chat interactions
- NGN-first amounts (always convert to USD alongside)
- 2% risk per trade as default position sizing
- Only signals with confidence >= 73% are returned
- No "guaranteed profit" language — always include risk disclaimer
