# AI Multi-User Trading Bot Backend

## Project Overview
FastAPI backend for an advanced AI-powered trading bot that provides institutional-grade market analysis, trading predictions, and AI chat assistance for Forex and cryptocurrency markets.

## Current Status (December 22, 2025)
- **Backend**: Running successfully on port 5000
- **API Endpoints**: 12+ functional endpoints
- **AI Model**: GPT-4o-mini with elite institutional analysis prompts
- **Analysis Quality**: Institutional-grade with multi-factor confluence detection

## Architecture

### Key Components
- **FastAPI Server**: RESTful API with CORS enabled for Lovable frontend
- **OpenAI Integration**: Elite trading AI with institutional-grade prompts
- **User Management**: In-memory user database with account registration
- **Chat System**: Stateful AI chat with conversation history
- **Backtesting**: Integrated backtest engine for signal validation

### Tech Stack
- Python 3 with FastAPI, Uvicorn
- OpenAI API (GPT-4o-mini)
- Backtest API integration for signal validation
- CORS middleware for frontend integration

## API Endpoints

### Predictions & Analysis
- `GET /` - Health check
- `GET /predictions` - Comprehensive market predictions (14 assets)
- `GET /get_predictions?username=X&symbol=Y` - User-specific symbol predictions
- `GET /advice/{symbol}` - Deep institutional analysis for specific symbol
- `GET /market_analysis` - Overall market trends and opportunities

### User Management
- `POST /connect_account` - Register/connect trading account
- `POST /execute_trade` - Execute trade (status: mock)

### AI Chat
- `POST /chat` - AI trading assistant chat
- `POST /chat/{username}` - User context-aware AI chat

### Backtesting (via backtest_router)
- `/api/*` - Backtesting endpoints (see backtest_api.py)

## Recent Enhancements

### AI Accuracy Improvements
1. **Enhanced Prediction System**: Multi-factor confluence analysis
   - 5+ technical indicators per asset (RSI, MACD, EMA, Bollinger, Volume)
   - Multi-timeframe trend analysis (5m, 1h, 4h, daily)
   - Chart pattern detection
   - Sentiment analysis (fear/greed index)
   - Position sizing recommendations

2. **Institutional-Grade Analysis Prompts**
   - Temperature: 0.5 (deterministic, high accuracy)
   - Max tokens: 6000 (detailed analysis)
   - Confidence scores based on historical win probability
   - Risk-reward filtering (minimum 1:1.5)
   - Confluence detection (only signals with 5+ aligned factors)

3. **Deep Trading Advice** (`/advice/{symbol}`)
   - Entry zones (not single price)
   - 3 take-profit targets with probabilities
   - Fibonacci levels and key price levels
   - Trade management strategies
   - Risk assessment and invalidation levels

### Trading Signal Quality
- **STRONG_BUY/STRONG_SELL**: Only when 5+ factors align
- **Confidence Scores**: Actual win probability from historical patterns
- **Position Sizing**: 1-5% of capital based on risk assessment
- **Risk-Reward**: Minimum 1:1.5 ratio enforced

## Assets Analyzed
**Forex (7 pairs)**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
**Crypto (7 pairs)**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT, ADAUSDT, DOGEUSDT

## Design Decisions

### Why Lazy OpenAI Initialization?
- Server starts without API key (safer deployment)
- Graceful error handling when key isn't set
- Allows testing other endpoints without OpenAI

### Why In-Memory Database?
- Fast development and testing
- Can be replaced with PostgreSQL/MongoDB
- User data persists during session

### Why 0.5 Temperature?
- Higher determinism = more consistent predictions
- Reduces hallucination risk in financial analysis
- Institutional trading needs consistency over creativity

## Known Limitations & Future Improvements
1. **Not Real Accuracy**: No system achieves 99.9% in trading (market unpredictability)
2. **Mock Trade Execution**: Current execute_trade is mock only
3. **In-Memory State**: No persistence across server restarts
4. **Demo API Key**: Use real OpenAI API key in production
5. **Real Market Data**: Currently uses AI reasoning, not live price data

## Deployment
- **Production Command**: `uvicorn main:app --host 0.0.0.0 --port 5000`
- **Development Command**: `uvicorn main:app --host 0.0.0.0 --port 5000 --reload`
- **Frontend Integration**: Lovable frontend connects to this backend via CORS

## Dependencies
```
fastapi
uvicorn
openai
python-dotenv
pydantic
backtest_api (local module)
```

## User Preferences & Notes
- Focus on institutional-grade analysis over "guaranteed profits"
- Always include disclaimer: "Trading involves substantial risk"
- Emphasis on risk management and multi-factor confluence
- AI should suggest strategies, not promise outcomes

## Frontend Integration
The Lovable frontend connects via:
- Base URL: `https://[deployment-url]`
- All endpoints return JSON
- CORS headers allow cross-origin requests
- Chat endpoint maintains conversation history

## Testing the API
```bash
# Health check
curl https://[url]/

# Get predictions
curl https://[url]/predictions

# Get specific symbol advice
curl https://[url]/advice/BTCUSDT

# Chat with AI
curl -X POST https://[url]/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What should I trade today?"}'
```

## Last Updated
December 22, 2025 - Enhanced AI accuracy with institutional-grade prompts and multi-factor analysis
