# AI Multi-User Trading Bot Backend — ARIA v4.0

## Project Overview
FastAPI backend for ARIA — an AI-powered institutional-grade trading bot analyzing 30 Forex and crypto markets. Provides real-time signal generation from live Alpha Vantage OHLCV data, NGN/USD conversion, demo trading, persistent PostgreSQL user data, personalized advice, a visual trading simulator, and a full MT5 auto-execution engine with Risk Engine, Market Filter, and Trade Manager.

## Current Status (March 2026)
- **Backend**: Running on port 5000
- **Data Source**: Twelve Data (primary, 800 req/day) → Alpha Vantage (fallback, 25/day) → AI reasoning
- **AI Model**: llama-3.3-70b-versatile via Groq (100,000 TPD free tier)
- **Persistence**: Full PostgreSQL (users, trades, chat, journal, watchlist, activity, bot orders)
- **Accuracy Target**: 70%+ via real indicator data fed to AI
- **Auto-Trading**: Full MT5 bridge system with Risk Engine, Market Filter, Trade Manager
- **Prediction Cache**: 60-minute disk-backed cache (survives restarts; keeps daily Groq usage ~84K/100K TPD)

## Architecture

### Key Components
- **FastAPI Server** (`main.py`): All API endpoints, prediction engine, demo trading, bot endpoints
- **Market Data Engine** (`market_data.py`): Live OHLCV fetching + full indicator computation
- **AI Provider** (`ai_provider.py`): Groq API wrapper (llama-3.3-70b-versatile)
- **Database Models** (`models.py`): Full PostgreSQL schema via SQLAlchemy
- **Trading Engine** (`trading_engine.py`): Risk Engine, Market Filter, Trade Manager
- **MT5 Bridge** (`mt5_bridge.py`): Windows-side script that connects to MT5 terminal
- **Backtest Engine** (`backtest_api.py`): Historical backtesting via CSV upload

### Tech Stack
- Python 3 with FastAPI, Uvicorn
- Groq AI (llama-3.3-70b-versatile) — ARIA persona
- Alpha Vantage API — live OHLCV candles + exchange rates
- PostgreSQL — full persistent storage
- SQLAlchemy ORM
- pandas + numpy — local indicator computation
- MetaTrader5 Python library (on Windows bridge only)
- CORS middleware for Lovable frontend

## MT5 Auto-Trading System

### Architecture
```
[Replit Backend] ←→ HTTP polling ←→ [mt5_bridge.py on Windows VPS] ←→ [MT5 Terminal]
```

### Three Protection Layers (prevents gambling)

**1. Market Filter** (`trading_engine.py → MarketFilter`)
Blocks trades when:
- Signal confidence below minimum (default 75%)
- Spread too wide (>3× typical or configurable limit)
- Outside allowed trading sessions (London/NY/Overlap by default)
- Within N minutes of major news events (NFP, FOMC, CPI, ECB, BOE)
- ATR below 30th percentile of its average (ranging/dead market)
- Symbol not in allowed pairs whitelist

**2. Risk Engine** (`trading_engine.py → RiskEngine`)
- Calculates exact lot size: `risk_usd / (sl_pips × pip_value_per_lot)`
- Checks daily loss limit (default 5% of account)
- Checks weekly loss limit (default 10% of account)
- Enforces max concurrent positions (default 3)
- Blocks duplicate positions on same symbol
- Hard cap on lot size regardless of balance

**3. Trade Manager** (`trading_engine.py → TradeManager`)
Runs every 10 seconds on all open positions:
- **Break-even**: Move SL to entry+2 pips when trade reaches 1:1 R/R
- **Trailing stop**: Activates at 1.5R, trails by 0.5R steps
- **Partial close**: Close 50% at TP1, let rest run to TP2
- **Time exit**: Close if open >48h and profit <0.5R
- **Safety close**: Close if SL is hit (broker should do this but double-checked)

### Order Lifecycle
```
PENDING → QUEUED → SENT → ACTIVE → CLOSED
                        ↘ REJECTED
```

### MT5 Bridge Setup (Windows)
1. Install Python 3.10+ for Windows
2. `pip install MetaTrader5 requests python-dotenv`
3. Install MT5 terminal from broker
4. Create `.env` file:
   ```
   API_URL=https://your-replit-url.replit.app
   USERNAME=your_username
   BRIDGE_API_KEY=key_from_bot_configure_endpoint
   MT5_LOGIN=12345678
   MT5_PASSWORD=your_password
   MT5_SERVER=BrokerName-Live
   ```
5. Run: `python mt5_bridge.py`
6. For 24/7: use Contabo/Vultr/Azure Windows VPS (~$7-15/month)

## API Endpoints

### Market Data & Signals
- `GET /` — Health check
- `GET /predictions?amount_ngn=X&username=Y` — All 30 symbols
- `GET /get_predictions?username=X&symbol=Y` — Single symbol with real data
- `GET /advice/{symbol}` — Deep institutional analysis
- `GET /market_analysis` — Overall market summary
- `GET /live_data/{symbol}` — Raw real-time indicators

### User Management
- `POST /user/{username}` — Create/update profile
- `GET /user/{username}` — Full profile with stats
- `GET /account/monitor/{username}` — Account overview

### AI Chat (Persistent with Memory)
- `POST /chat` — Chat with ARIA
- `GET /chat/history/{username}` — Full history
- `DELETE /chat/history/{username}` — Clear history
- `POST /conversations/{username}` — Start new conversation thread
- `GET /conversations/{username}` — List all threads
- `GET /conversations/{username}/{id}` — Open specific thread
- `PATCH /conversations/{username}/{id}/rename` — Rename thread
- `DELETE /conversations/{username}/{id}` — Delete thread
- `GET /memory/{username}` — What ARIA remembers
- `POST /memory/{username}` — Add memory
- `DELETE /memory/{username}/{key}` — Forget one thing
- `DELETE /memory/{username}` — Wipe all memories

### Auto-Trader (Bot)
- `POST /bot/configure/{username}` — Set bot config, get bridge API key
- `POST /bot/start/{username}` — Activate bot
- `POST /bot/stop/{username}` — Pause bot
- `GET /bot/status/{username}` — Full status + active orders + today's P&L
- `POST /bot/trade/{username}` — Queue a manual signal (passes all filters)
- `POST /bot/auto_signal/{username}` — ARIA generates + queues signal automatically
- `GET /bot/history/{username}` — Closed trade history
- `GET /bot/performance/{username}` — Stats: all-time, monthly, weekly, by symbol
- `POST /bot/emergency_stop/{username}` — Immediately halt + cancel all queued orders

### MT5 Bridge Endpoints (called by mt5_bridge.py)
- `POST /bot/bridge_connect/{username}` — Bridge registers on startup
- `GET /bot/queue/{username}?bridge_key=X` — Poll for pending orders
- `POST /bot/executed/{username}?bridge_key=X` — Report successful execution
- `POST /bot/rejected/{username}?bridge_key=X` — Report failed execution
- `POST /bot/position_update/{username}?bridge_key=X` — Send position state, receive instructions
- `POST /bot/partial_closed/{username}?bridge_key=X` — Confirm partial close
- `POST /bot/closed/{username}?bridge_key=X` — Confirm full close

### Demo Trading
- `POST /demo/open_account/{username}`
- `POST /demo/execute_trade/{username}`
- `POST /demo/close_trade/{username}/{trade_id}`
- `GET /demo/account/{username}`
- `GET /demo/ai_feedback/{username}`

### Utilities
- `POST /risk_calculator`
- `POST /score_trade`
- `GET /connect_account`
- `POST /journal/{username}` / `GET /journal/{username}`
- `POST /watchlist/{username}` / `DELETE` / `GET`

### Visual Simulator
- `GET /simulator/candles/{symbol}`
- `GET /simulator/run/{symbol}`

### Backtesting
- `POST /api/upload_history`
- `POST /api/backtest`
- `GET /api/history/{symbol}`

## Database Schema (PostgreSQL)

| Table | Purpose |
|---|---|
| `users` | Username, balance NGN, trading style, risk tolerance |
| `bot_configs` | Per-user bot settings: risk%, filters, trade manager config, bridge key |
| `bot_orders` | Full order lifecycle from QUEUED to CLOSED, all risk/fill/PnL details |
| `demo_accounts` | Demo balance per user |
| `demo_trades` | Active and closed demo trades |
| `conversations` | Named chat threads (like ChatGPT sidebar) |
| `chat_messages` | Full ARIA conversation history |
| `user_memories` | Facts ARIA remembers about each user |
| `trade_journal` | Logged trades (also auto-populated from bot closes) |
| `watchlist` | User watchlist items |
| `user_activity` | Every action logged for AI personalization |

## Assets Analyzed (30 total)
**Forex (15)**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF, EURGBP, EURJPY, GBPJPY, AUDJPY, EURCAD, AUDCAD, EURAUD, GBPAUD
**Crypto (15)**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT, DOTUSDT, MATICUSDT, LTCUSDT, SHIBUSDT, TRXUSDT, AVAXUSDT, LINKUSDT, UNIUSDT

## Key Files
- `main.py` — All endpoints (use `write` tool to edit, not `edit` for large changes)
- `trading_engine.py` — RiskEngine, MarketFilter, TradeManager classes
- `mt5_bridge.py` — Windows-side MT5 execution script (copy to Windows VPS)
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
- Deploy this backend to Replit for 24/7 uptime
- Run `mt5_bridge.py` on a Windows VPS for 24/7 MT5 execution

## User Preferences
- ARIA persona for all AI chat interactions
- NGN-first amounts (always convert to USD alongside)
- 2% risk per trade as default position sizing
- Only signals with confidence >= 73% are returned to frontend
- Only signals with confidence >= 75% are auto-traded (configurable)
- No "guaranteed profit" language — always include risk disclaimer
